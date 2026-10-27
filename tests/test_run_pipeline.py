# SPDX-FileCopyrightText: Copyright (c) 2026 Backblaze, Inc.
# SPDX-License-Identifier: Apache-2.0
import argparse
import ast
import io
import json
import os
import shlex
from pathlib import Path

import pytest
import requests
import run_pipeline
from botocore.exceptions import ClientError, EndpointConnectionError

from aistore_b2_curation.artifacts import manifest_sha256
from aistore_b2_curation.config import Config


@pytest.fixture
def cfg(b2_env) -> Config:
    return Config.from_env(b2_env, require_b2_credentials=False)


def curator_argparser() -> argparse.ArgumentParser:
    """Build the parser from the pinned Curator script without importing nemo_curator.

    Executes only `create_video_splitting_argparser`, so a flag renamed upstream fails here
    instead of on a GPU node.
    """
    if not run_pipeline.CURATOR_SCRIPT.exists():
        pytest.skip("pinned Curator script missing; run `make setup`")
    tree = ast.parse(run_pipeline.CURATOR_SCRIPT.read_text())
    fn = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "create_video_splitting_argparser"
    )
    namespace: dict = {"argparse": argparse}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "curator", "exec"), namespace)  # noqa: S102
    return namespace["create_video_splitting_argparser"]()


@pytest.mark.parametrize(
    ("requested", "gpus", "expected"),
    [
        ("auto", [], "h264_nvenc"),
        ("auto", ["NVIDIA L40S"], "h264_nvenc"),
        ("auto", ["NVIDIA A100-SXM4-80GB"], "libvpx-vp9"),
        ("auto", ["NVIDIA H100 80GB HBM3"], "libvpx-vp9"),
        ("h264_nvenc", ["NVIDIA H100 80GB HBM3"], "h264_nvenc"),  # explicit choice wins
    ],
)
def test_pick_encoder(requested, gpus, expected):
    assert run_pipeline.pick_encoder(requested, gpus) == expected


def test_build_command_defaults(cfg):
    args, extra = run_pipeline.parse_args(["--run-id", "r1"])
    cmd = run_pipeline.build_command(cfg, args, "h264_nvenc", extra)
    assert cmd[:2] == [str(run_pipeline.CURATOR_PYTHON), str(run_pipeline.CURATOR_ENTRYPOINT)]
    flags = dict(zip(cmd[2::2], cmd[3::2], strict=True))
    assert flags["--video-dir"] == "s3://my-video-corpus/raw"
    assert flags["--output-path"].endswith("work/r1")
    assert flags["--embedding-algorithm"] == "cosmos-embed1-224p"
    assert "--generate-captions" not in cmd


def test_build_command_options_and_passthrough(cfg):
    args, extra = run_pipeline.parse_args(
        [
            "--input-prefix",
            "raw/",
            "--video-limit",
            "3",
            "--motion-filter",
            "enable",
        ]
    )
    cmd = run_pipeline.build_command(cfg, args, "libvpx-vp9", extra)
    assert "s3://my-video-corpus/raw" in cmd  # trailing slash normalized
    assert cmd[cmd.index("--video-limit") + 1] == "3"
    assert "--generate-captions" not in cmd
    assert cmd[-2:] == ["--motion-filter", "enable"]


def test_command_is_accepted_by_pinned_curator_parser(cfg):
    parser = curator_argparser()
    args, extra = run_pipeline.parse_args(["--video-limit", "2"])
    cmd = run_pipeline.build_command(cfg, args, "libvpx-vp9", extra)
    parsed = parser.parse_args(cmd[2:])
    assert parsed.video_dir == "s3://my-video-corpus/raw"
    assert parsed.transcode_encoder == "libvpx-vp9"
    assert parsed.generate_embeddings and not parsed.generate_captions
    assert parsed.video_limit == 2


def test_dry_run_prints_command_and_touches_nothing(b2_env, monkeypatch, capsys):
    for k, v in b2_env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("B2_APP_KEY")  # dry-run must not need B2 secrets
    monkeypatch.setattr(run_pipeline, "detect_gpu_names", lambda: [])
    for name in ("prefetch", "upload_dir", "check_media_tools", "run_curator", "ais_backend_gets"):
        monkeypatch.setattr(run_pipeline, name, lambda *a, **k: pytest.fail("side effect"))

    assert run_pipeline.main(["--dry-run", "--run-id", "r1"]) == 0
    out = capsys.readouterr().out.strip()
    assert out.startswith("AWS_ENDPOINT_URL=http://127.0.0.1:51080/s3 ")
    assert "--video-dir s3://my-video-corpus/raw" in out
    assert "fake-key-id" not in out
    shlex.split(out)  # copy-pasteable


def test_missing_curator_venv_is_a_clear_error(b2_env, monkeypatch, tmp_path, capsys):
    for k, v in b2_env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(run_pipeline, "CURATOR_PYTHON", tmp_path / "nope" / "python")
    assert run_pipeline.main(["--encoder", "h264_nvenc"]) == 2
    assert "make setup-curator" in capsys.readouterr().err


@pytest.mark.parametrize("name", ["../outside", "/absolute", "a/b", "."])
def test_run_id_is_a_single_identifier(name):
    with pytest.raises(SystemExit):
        run_pipeline.parse_args(["--run-id", name])


def test_existing_output_is_not_reused(b2_env, monkeypatch, tmp_path, capsys):
    for name, value in b2_env.items():
        monkeypatch.setenv(name, value)
    (tmp_path / "r1").mkdir()
    (tmp_path / "r1" / "old.bin").write_bytes(b"old")
    monkeypatch.setattr(run_pipeline, "CURATOR_PYTHON", tmp_path)
    monkeypatch.setattr(run_pipeline, "CURATOR_SCRIPT", tmp_path)
    rc = run_pipeline.main(
        ["--work-dir", str(tmp_path), "--run-id", "r1", "--encoder", "libvpx-vp9"]
    )
    assert rc == 2
    assert "--run-id" in capsys.readouterr().err
    assert (tmp_path / "r1" / "old.bin").read_bytes() == b"old"


class InputBucket:
    """AIS S3 client over the input bucket: object sizes plus an optional `raw/_SUCCESS.json`."""

    def __init__(self):
        self.manifest = None
        self.objects: dict[str, int] = {}

    def get_object(self, Bucket, Key):
        if self.manifest is None or Key != "raw/_SUCCESS.json":
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        return {"Body": io.BytesIO(json.dumps(self.manifest).encode())}

    def get_paginator(self, _name):
        return self

    def paginate(self, Bucket, Prefix):
        keys = dict(self.objects)
        if self.manifest is not None:
            keys["raw/_SUCCESS.json"] = 1
        yield {"Contents": [{"Key": k, "Size": n} for k, n in keys.items() if k.startswith(Prefix)]}


@pytest.fixture
def inputs() -> InputBucket:
    return InputBucket()


@pytest.fixture
def prepared_pipeline(b2_env, monkeypatch, tmp_path, inputs):
    for name, value in b2_env.items():
        monkeypatch.setenv(name, value)
    script = tmp_path / "curator.py"
    script.write_text("")
    monkeypatch.setattr(run_pipeline, "CURATOR_PYTHON", Path(run_pipeline.sys.executable))
    monkeypatch.setattr(run_pipeline, "CURATOR_SCRIPT", script)
    calls = []
    monkeypatch.setattr(run_pipeline, "check_media_tools", lambda _encoder: calls.append("media"))
    monkeypatch.setattr(run_pipeline, "prefetch", lambda *args: calls.append("prefetch"))
    monkeypatch.setattr(run_pipeline, "ais_s3_client", lambda _cfg: inputs)
    monkeypatch.setattr(run_pipeline, "ais_backend_gets", lambda _cfg: (0, 0))

    def run(*args, **kwargs):
        calls.append("curator")
        return 0

    def upload(*args, **kwargs):
        calls.append("upload")
        return 1, 0

    monkeypatch.setattr(run_pipeline, "run_curator", run)
    monkeypatch.setattr(run_pipeline, "upload_dir", upload)
    argv = ["--encoder", "libvpx-vp9", "--work-dir", str(tmp_path / "work"), "--run-id", "r1"]
    return argv, calls, tmp_path / "work" / "r1"


COMPLETE = {
    "schema_version": 1,
    "status": "complete",
    "files": {"a.mp4": {"size": 1, "sha256": "0" * 64}},
}


@pytest.mark.parametrize(
    "manifest", [{**COMPLETE, "status": "pending"}, {"schema_version": 1, "status": "complete"}]
)
def test_pending_or_invalid_input_publication_is_never_curated(
    prepared_pipeline, inputs, manifest, capsys
):
    argv, calls, out_dir = prepared_pipeline
    inputs.manifest = manifest
    assert run_pipeline.main(argv) == 2
    assert calls == ["media"]
    assert not out_dir.exists()
    assert "raw/_SUCCESS.json" in capsys.readouterr().err


def test_complete_input_publication_is_curated(prepared_pipeline, inputs):
    argv, calls, _ = prepared_pipeline
    inputs.manifest = COMPLETE
    inputs.objects = {"raw/a.mp4": 1, "raw2/other.mp4": 5}  # sibling prefix is not input
    assert run_pipeline.main(argv) == 0
    assert calls == ["media", "prefetch", "curator", "upload"]


def test_curator_workers_keep_gpu_environment_under_uv(prepared_pipeline, monkeypatch):
    argv, _, _ = prepared_pipeline
    monkeypatch.setenv("VIRTUAL_ENV", "/wrapper/.venv")
    monkeypatch.setenv("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "1")
    monkeypatch.setenv("PATH", "/wrapper/.venv/bin:/usr/bin")
    monkeypatch.setenv("HF_TOKEN", "test-hf-token")
    launched = []

    def run(cmd, env):
        launched.append((cmd, env))
        return 0

    monkeypatch.setattr(run_pipeline, "run_curator", run)
    assert run_pipeline.main(argv) == 0
    cmd, env = launched[0]
    python = Path(cmd[0])
    assert env["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] == "0"
    assert env["VIRTUAL_ENV"] == str(python.parent.parent)
    assert env["PATH"].split(os.pathsep)[0] == str(python.parent)
    assert env["HF_TOKEN"] == "test-hf-token"
    assert "B2_KEY_ID" not in env and "B2_APP_KEY" not in env


@pytest.mark.parametrize(
    ("manifest", "expected"),
    [(COMPLETE, manifest_sha256(COMPLETE)), (None, None)],
    ids=["manifested", "hand-uploaded"],
)
def test_output_provenance_pins_input_manifest(
    prepared_pipeline, inputs, monkeypatch, manifest, expected
):
    argv, _, _ = prepared_pipeline
    inputs.manifest = manifest
    inputs.objects = {"raw/a.mp4": 1}
    uploads = []
    monkeypatch.setattr(run_pipeline, "upload_dir", lambda *a, **k: uploads.append(k) or (1, 0))
    assert run_pipeline.main(argv) == 0
    assert uploads[0]["provenance"]["input_manifest_sha256"] == expected


@pytest.mark.parametrize(
    "objects",
    [
        {"raw/a.mp4": 1, "raw/leftover.mp4": 9},  # unlisted object Curator would read
        {},  # listed object missing
        {"raw/a.mp4": 2},  # resized
    ],
)
def test_input_prefix_must_match_its_manifest_exactly(prepared_pipeline, inputs, objects, capsys):
    argv, calls, out_dir = prepared_pipeline
    inputs.manifest = COMPLETE
    inputs.objects = objects
    assert run_pipeline.main(argv) == 2
    assert calls == ["media"]
    assert not out_dir.exists()
    assert "does not match the prefix" in capsys.readouterr().err


def test_unreachable_ais_at_input_check_is_a_clear_error(prepared_pipeline, monkeypatch, capsys):
    argv, calls, _ = prepared_pipeline

    def unreachable(*args):
        raise EndpointConnectionError(endpoint_url="http://127.0.0.1:51080/s3")

    monkeypatch.setattr(run_pipeline, "read_manifest", unreachable)
    assert run_pipeline.main(argv) == 2
    assert calls == ["media"]
    assert "make up" in capsys.readouterr().err


def test_symlinked_work_dir_runs_on_its_target(prepared_pipeline, tmp_path):
    # Cloud GPU hosts often link the work dir to local NVMe.
    argv, _, _ = prepared_pipeline
    nvme = tmp_path / "nvme"
    nvme.mkdir()
    (tmp_path / "linked").symlink_to(nvme, target_is_directory=True)
    assert run_pipeline.main([*argv, "--work-dir", str(tmp_path / "linked")]) == 0
    assert (nvme / "r1").is_dir()


@pytest.mark.parametrize("extra", [[], ["--video-limit", "5"], ["--skip-prefetch"]])
def test_limited_and_explicit_cold_reads_avoid_whole_prefix_prefetch(prepared_pipeline, extra):
    argv, calls, _ = prepared_pipeline
    assert run_pipeline.main(argv + extra) == 0
    expected = ["media"] + (["prefetch"] if not extra else []) + ["curator", "upload"]
    assert calls == expected


def test_media_failure_precedes_all_storage_activity(prepared_pipeline, monkeypatch, capsys):
    argv, calls, out_dir = prepared_pipeline

    def fail(_encoder):
        raise run_pipeline.MediaPrerequisiteError("ffmpeg requires software h264 decoder")

    monkeypatch.setattr(run_pipeline, "check_media_tools", fail)
    assert run_pipeline.main(argv) == 2
    assert calls == []
    assert not out_dir.exists()
    assert "software h264 decoder" in capsys.readouterr().err


def test_failed_prefetch_does_not_launch_curator_or_claim_output(
    prepared_pipeline, monkeypatch, capsys
):
    argv, calls, out_dir = prepared_pipeline

    def fail(*args):
        raise RuntimeError("AIS prefetch prefetch-job failed: backend unavailable")

    monkeypatch.setattr(run_pipeline, "prefetch", fail)
    assert run_pipeline.main(argv) == 1
    assert calls == ["media"]
    assert not out_dir.exists()
    assert "--skip-prefetch" in capsys.readouterr().err


@pytest.mark.parametrize("duration", ["nan", "inf", "-inf", "0", "-1"])
def test_clip_duration_must_be_finite_and_positive(duration):
    with pytest.raises(SystemExit):
        run_pipeline.parse_args([f"--clip-seconds={duration}"])


@pytest.mark.parametrize(
    "flag",
    [
        "--transcode-encoder",
        "--transcode-enc",
        "--fixed-stride-split-duration",
        "--fixed-stride-split-dur",
    ],
)
def test_passthrough_cannot_override_preflighted_encoder_or_validated_duration(flag):
    with pytest.raises(SystemExit):
        run_pipeline.parse_args([f"{flag}=unexpected"])


def test_empty_input_prefix_prefetches_bucket_root(prepared_pipeline, monkeypatch):
    argv, _, _ = prepared_pipeline
    prefixes = []
    monkeypatch.setattr(run_pipeline, "prefetch", lambda _cfg, prefix: prefixes.append(prefix))
    assert run_pipeline.main([*argv, "--input-prefix", ""]) == 0
    assert prefixes == [""]


def test_curator_stage_reports_b2_reads_made_by_ais(prepared_pipeline, monkeypatch, capsys):
    argv, _, _ = prepared_pipeline
    counters = iter([(7, 1000), (9, 3048)])  # cumulative, sampled around Curator only
    monkeypatch.setattr(run_pipeline, "ais_backend_gets", lambda _cfg: next(counters))
    assert run_pipeline.main(argv) == 0
    assert "Curator stage: 2 B2 GETs, 2048 bytes" in capsys.readouterr().err


@pytest.mark.parametrize("failing_read", [0, 1])
def test_unreadable_stats_never_block_upload(prepared_pipeline, monkeypatch, capsys, failing_read):
    argv, calls, _ = prepared_pipeline
    reads = iter(range(2))

    def gets(_cfg):
        if next(reads) == failing_read:
            raise requests.ConnectionError("stats endpoint down")
        return (0, 0)

    monkeypatch.setattr(run_pipeline, "ais_backend_gets", gets)
    assert run_pipeline.main(argv) == 0
    assert calls[-1] == "upload"
    err = capsys.readouterr().err
    assert "Curator stage: AIS backend counters unavailable: stats endpoint down" in err
    assert "B2 GETs" not in err


def test_unreachable_ais_before_curator_is_a_clear_error(prepared_pipeline, monkeypatch, capsys):
    argv, calls, out_dir = prepared_pipeline

    def unreachable(*args):
        raise requests.ConnectionError("connection refused")

    monkeypatch.setattr(run_pipeline, "prefetch", unreachable)
    assert run_pipeline.main(argv) == 2
    assert calls == ["media"]
    assert not out_dir.exists()
    assert "make up" in capsys.readouterr().err


def test_failed_upload_keeps_output_and_says_where(prepared_pipeline, monkeypatch, capsys):
    argv, _, out_dir = prepared_pipeline

    def changed(*args, **kwargs):
        raise OSError(f"artifact changed during upload: {out_dir} (clip.mp4)")

    monkeypatch.setattr(run_pipeline, "upload_dir", changed)
    assert run_pipeline.main(argv) == 1
    assert out_dir.is_dir()
    err = capsys.readouterr().err
    assert "clip.mp4" in err and f"local output kept in {out_dir}" in err


def test_unreachable_ais_at_upload_keeps_output(prepared_pipeline, monkeypatch, capsys):
    argv, _, out_dir = prepared_pipeline

    def unreachable(*args, **kwargs):
        raise EndpointConnectionError(endpoint_url="http://127.0.0.1:51080/s3")

    monkeypatch.setattr(run_pipeline, "upload_dir", unreachable)
    assert run_pipeline.main(argv) == 2
    assert out_dir.is_dir()
    err = capsys.readouterr().err
    assert "make up" in err
    assert str(out_dir) in err
