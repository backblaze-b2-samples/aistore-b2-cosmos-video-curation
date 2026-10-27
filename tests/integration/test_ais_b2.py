# SPDX-FileCopyrightText: Copyright (c) 2026 Backblaze, Inc.
# SPDX-License-Identifier: Apache-2.0
"""AIS + real Backblaze B2 end-to-end checks. Run with `make test-integration`.

Needs `.env` with B2_IT=1 and a key for a dedicated test bucket. `make test-integration`
starts a fresh AIS cluster on port 51280, so every run begins with a cold cache.

Every object is written under a unique it/<timestamp>-<hex>/ prefix and nothing is deleted
(the sample's key has no deleteFiles); a B2 lifecycle rule on `it/` expires them.

Cache hits are measured with the AIS target's cumulative `aws.get.*` counters, which
are cluster-wide: run these tests serially and against no other workload.
"""

from __future__ import annotations

import json
import os
import secrets
import subprocess
import sys
import textwrap
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

import pytest
from aistore.sdk import Client
from boto3.s3.transfer import TransferConfig
from botocore.exceptions import ClientError

from aistore_b2_curation import storage
from aistore_b2_curation.artifacts import MANIFEST, verify_remote
from aistore_b2_curation.config import Config

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[2]
IT_AIS_PORT = 51280  # the disposable cluster from `make test-integration`, never `make up`'s
RUN = f"it/{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{secrets.token_hex(4)}/"
RAW = {f"{RUN}raw/robot/episode_{i:06d}.mp4": os.urandom(256 * 1024 + i) for i in range(3)}
RAW_BYTES = sum(len(v) for v in RAW.values())


@pytest.fixture(scope="module")
def cfg() -> Config:
    if os.environ.get("B2_IT") != "1":
        pytest.fail("integration tests write to real B2; set B2_IT=1 in .env (see README)")
    cfg = Config.from_env()
    # attach_b2_bucket.sh changes cluster features, so refuse the user's `make up` cluster.
    if urlparse(cfg.ais_endpoint).port != IT_AIS_PORT:
        pytest.fail(f"AIS_ENDPOINT must use port {IT_AIS_PORT}; use `make test-integration`")
    try:
        urllib.request.urlopen(f"{cfg.ais_endpoint}/v1/health", timeout=3)  # noqa: S310
    except OSError:
        pytest.fail(f"AIS not reachable at {cfg.ais_endpoint}; use `make test-integration`")
    return cfg


@pytest.fixture(scope="module")
def b2(cfg: Config):
    """Direct B2 client, with this run's corpus seeded and the bucket attached to AIS."""
    s3 = storage.b2_client(cfg)
    try:
        s3.head_bucket(Bucket=cfg.b2_bucket)
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        if code not in ("403", "404"):
            raise
        pytest.fail(
            f"B2 bucket {cfg.b2_bucket!r} not accessible (HTTP {code}); use an existing test "
            "bucket and a key with listBuckets, listFiles, readFiles, writeFiles"
        )
    for key, data in RAW.items():
        s3.put_object(Bucket=cfg.b2_bucket, Key=key, Body=data)
    env = {**os.environ, "AIS_ENDPOINT": cfg.ais_endpoint}
    subprocess.run([str(ROOT / "scripts" / "attach_b2_bucket.sh")], env=env, check=True)
    return s3


@pytest.fixture
def ais(cfg: Config, b2):
    return storage.ais_s3_client(cfg)


def b2_gets_since(cfg: Config, before: tuple[int, int]) -> tuple[int, int]:
    """(GET count, bytes) AIS fetched from B2 since `before` was sampled."""
    count, size = storage.ais_backend_gets(cfg)
    return count - before[0], size - before[1]


def test_cold_get_reads_b2_once(cfg, ais):
    storage.evict(cfg, list(RAW))
    assert not storage.cached_keys(cfg, f"{RUN}raw/")
    key, data = next(iter(RAW.items()))
    before = storage.ais_backend_gets(cfg)
    assert ais.get_object(Bucket=cfg.b2_bucket, Key=key)["Body"].read() == data
    assert b2_gets_since(cfg, before) == (1, len(data))
    assert key in storage.cached_keys(cfg, f"{RUN}raw/")
    # AIS omits zero counters, so only now must the name exist; an image bump that renames
    # it would otherwise turn every "zero B2 reads" assertion into a silent pass.
    perf = Client(cfg.ais_endpoint).cluster().get_performance()
    assert any(storage.BACKEND_GET_COUNT in node["tracker"] for node in perf.values())


def test_warm_get_skips_b2(cfg, ais):
    key, data = next(iter(RAW.items()))
    ais.get_object(Bucket=cfg.b2_bucket, Key=key)["Body"].read()  # ensure warm
    before = storage.ais_backend_gets(cfg)
    assert ais.get_object(Bucket=cfg.b2_bucket, Key=key)["Body"].read() == data
    assert b2_gets_since(cfg, before) == (0, 0)


def test_prefetch_reads_each_object_from_b2_once(cfg, b2):
    storage.evict(cfg, list(RAW))
    before = storage.ais_backend_gets(cfg)
    storage.prefetch(cfg, f"{RUN}raw/", timeout_s=120)
    assert b2_gets_since(cfg, before) == (len(RAW), RAW_BYTES)
    assert storage.cached_keys(cfg, f"{RUN}raw/") == set(RAW)


def test_multipart_put_writes_through_to_backend(cfg, ais, b2, tmp_path):
    src = tmp_path / "big.bin"
    src.write_bytes(os.urandom(12 * 1024**2))
    small_parts = TransferConfig(multipart_threshold=5 * 1024**2, multipart_chunksize=5 * 1024**2)
    key = f"{RUN}curated/big.bin"
    ais.upload_file(str(src), cfg.b2_bucket, key, Config=small_parts)
    assert b2.get_object(Bucket=cfg.b2_bucket, Key=key)["Body"].read() == src.read_bytes()


# Replays the calls NeMo Curator 1.4.0 makes (ClientPartitioningStage + FSPath) and writes
# Curator-shaped output, so run_pipeline is exercised end to end without a GPU.
FAKE_CURATOR = textwrap.dedent(
    """
    import argparse, pathlib
    from fsspec.core import url_to_fs
    p = argparse.ArgumentParser()
    p.add_argument("--video-dir"); p.add_argument("--output-path")
    a, _ = p.parse_known_args()
    fs, root = url_to_fs(a.video_dir)              # no storage_options, like VideoReader
    out = pathlib.Path(a.output_path)
    for path in fs.glob(f"{root}/**/*.mp4"):
        size = fs.size(path)
        data = b"".join(fs.cat_ranges([path, path], [0, size // 2], [size // 2, size]))
        dest = out / "clips" / (pathlib.PurePosixPath(path).stem + ".bin")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    """
)


def test_run_pipeline_end_to_end_through_ais(cfg, b2, monkeypatch, tmp_path, capsys):
    import run_pipeline

    fake = tmp_path / "fake_curator.py"
    fake.write_text(FAKE_CURATOR)
    monkeypatch.setattr(run_pipeline, "CURATOR_PYTHON", Path(sys.executable))
    monkeypatch.setattr(run_pipeline, "CURATOR_SCRIPT", fake)
    monkeypatch.setattr(run_pipeline, "check_media_tools", lambda _encoder: None)

    storage.evict(cfg, list(RAW))
    argv = [
        "--encoder", "h264_nvenc", "--work-dir", str(tmp_path / "work"), "--run-id", "e2e",
        "--input-prefix", f"{RUN}raw", "--output-prefix", f"{RUN}curated",
    ]  # fmt: skip
    before = storage.ais_backend_gets(cfg)
    assert run_pipeline.main(argv) == 0
    # The unlimited run prefetches every input once; Curator's reads then never reach B2.
    # Only GETs count: uploads' HEADs/PUTs are separate counters, and S3 CopyObject does
    # not record backend GET stats, so don't extend this check to copies.
    assert b2_gets_since(cfg, before) == (len(RAW), RAW_BYTES)
    assert "Curator stage: 0 B2 GETs, 0 bytes" in capsys.readouterr().err

    out = f"{RUN}curated/e2e"
    written = dict(storage.list_keys(b2, cfg.b2_bucket, f"{out}/clips/"))
    expected = {f"{out}/clips/{Path(k).stem}.bin": len(v) for k, v in RAW.items()}
    assert written == expected
    for key, data in RAW.items():
        out_key = f"{out}/clips/{Path(key).stem}.bin"
        assert b2.get_object(Bucket=cfg.b2_bucket, Key=out_key)["Body"].read() == data
    marker = b2.get_object(Bucket=cfg.b2_bucket, Key=f"{out}/{MANIFEST}")["Body"]
    assert json.loads(marker.read())["status"] == "complete"
    assert verify_remote(b2, cfg.b2_bucket, out) == len(RAW)


def test_manifest_identity_survives_ais_and_same_size_change_is_written(cfg, ais, b2, tmp_path):
    prefix = f"{RUN}curated/hash"
    path = tmp_path / "model.bin"
    path.write_bytes(b"old")
    assert storage.upload_dir(ais, tmp_path, cfg.b2_bucket, prefix, publish=True) == (1, 0)
    assert storage.upload_dir(ais, tmp_path, cfg.b2_bucket, prefix, publish=True) == (0, 1)
    path.write_bytes(b"new")
    assert storage.upload_dir(ais, tmp_path, cfg.b2_bucket, prefix, publish=True) == (1, 0)
    assert b2.get_object(Bucket=cfg.b2_bucket, Key=f"{prefix}/model.bin")["Body"].read() == b"new"
    assert verify_remote(b2, cfg.b2_bucket, prefix) == 1
