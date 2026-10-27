# SPDX-FileCopyrightText: Copyright (c) 2026 Backblaze, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Curate raw videos in B2 into clips and Cosmos-Embed1 embeddings through AIStore.

Unlimited runs prefetch the input prefix into AIS first; bounded runs and --skip-prefetch
read on demand through the cache. Curator reads `s3://` via
AWS_ENDPOINT_URL; the B2 GETs AIS made during that stage are reported on stderr. Curator
1.4.0 writes clips locally, so output is uploaded through AIS to
s3://$B2_BUCKET/<output-prefix>/<run-id>/.

Unrecognized arguments are forwarded to Curator verbatim, e.g. `--motion-filter enable`.

    uv run python scripts/run_pipeline.py --dry-run         # print the exact Curator command
    uv run python scripts/run_pipeline.py --video-limit 10  # small real run (GPU host)
"""

from __future__ import annotations

import argparse
import math
import os
import re
import shlex
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import requests
from botocore.exceptions import ClientError, EndpointConnectionError
from patch_curator import VERSION as CURATOR_VERSION
from prepare_model import MODEL_REVISION

from aistore_b2_curation.artifacts import (
    MANIFEST,
    manifest_sha256,
    read_manifest,
    reject_symlinks,
)
from aistore_b2_curation.config import Config
from aistore_b2_curation.curator_process import run_curator
from aistore_b2_curation.media import MediaPrerequisiteError, check_media_tools
from aistore_b2_curation.storage import (
    ais_backend_gets,
    ais_s3_client,
    ais_s3_env,
    list_keys,
    prefetch,
    upload_dir,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
# Installed by `make setup-curator` / `make setup` (see Makefile for pins and checksum).
CURATOR_PYTHON = REPO_ROOT / ".venv-curator" / "bin" / "python"
CURATOR_SCRIPT = REPO_ROOT / ".cache" / "curator" / "video_split_clip_example.py"
CURATOR_ENTRYPOINT = REPO_ROOT / "scripts" / "curator_entrypoint.py"

ENCODERS = ("h264_nvenc", "libvpx-vp9")
# GA100/GH100-based GPUs have no NVENC; Curator documents libvpx-vp9 (CPU) for them.
_NO_NVENC_GPUS = ("A100", "A800", "A30", "H100", "H200", "H800")
# Raised by the AIS SDK and boto3 when no AIS gateway answers at the configured endpoint.
_AIS_UNREACHABLE = (requests.ConnectionError, EndpointConnectionError)


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--input-prefix", default="raw", help="B2 key prefix holding source videos")
    p.add_argument("--output-prefix", default="curated", help="B2 key prefix for results")
    p.add_argument(
        "--run-id",
        default=datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ-") + uuid4().hex[:12],
    )
    p.add_argument("--work-dir", type=Path, default=REPO_ROOT / "work", help="local Curator output")
    p.add_argument(
        "--model-dir", type=Path, default=REPO_ROOT / "models", help="model weights cache"
    )
    p.add_argument("--clip-seconds", type=float, default=5.0, help="fixed-stride clip length")
    p.add_argument("--encoder", choices=("auto", *ENCODERS), default="auto")
    p.add_argument("--video-limit", type=int, help="process at most N videos")
    p.add_argument("--skip-prefetch", action="store_true", help="let Curator cold-read via AIS")
    p.add_argument("--dry-run", action="store_true", help="print the Curator command and exit")
    args, extra = p.parse_known_args(argv)
    unsupported = ("--with-captions", "--generate-captions", "--enhance-captions")
    if any(
        flag.startswith(arg.split("=", 1)[0])
        for arg in extra
        if arg.startswith("--")
        for flag in unsupported
    ) or any(arg.startswith(("--captioning-", "--enhance-caption")) for arg in extra):
        p.error(
            "captioning is not included; this sample produces clips and Cosmos-Embed1 embeddings"
        )
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", args.run_id):
        p.error("--run-id must be a single identifier (letters, digits, underscores, hyphens)")
    if not math.isfinite(args.clip_seconds) or args.clip_seconds <= 0:
        p.error("--clip-seconds must be finite and positive")
    if args.video_limit is not None and args.video_limit < 1:
        p.error("--video-limit must be positive")
    reserved = (
        "--video-dir", "--output-path", "--output-clip-path", "--model-dir",
        "--transcode-encoder", "--fixed-stride-split-duration",
        "--embedding-algorithm",
    )  # fmt: skip
    if any(
        flag.startswith(a.split("=", 1)[0])
        for a in extra
        if a.startswith("--") and a != "--"
        for flag in reserved
    ):
        p.error("input, output, model, encoder and clip duration are controlled by the wrapper")
    args.work_dir = args.work_dir.resolve()  # may be a symlink to local NVMe
    return args, extra


def detect_gpu_names() -> list[str]:
    """GPU model names from nvidia-smi; empty when no NVIDIA driver is present."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def pick_encoder(requested: str, gpu_names: list[str]) -> str:
    """Select an encoder from GPU names; preflight verifies actual runtime support."""
    if requested != "auto":
        return requested
    if any(tag in name for name in gpu_names for tag in _NO_NVENC_GPUS):
        return "libvpx-vp9"
    return "h264_nvenc"


def build_command(
    cfg: Config, args: argparse.Namespace, encoder: str, passthrough: list[str]
) -> list[str]:
    """Curator video_split_clip_example.py invocation (flags verified against v1.4.0)."""
    cmd = [
        str(CURATOR_PYTHON),
        str(CURATOR_ENTRYPOINT),
        "--video-dir", f"s3://{cfg.b2_bucket}/{args.input_prefix.strip('/')}",
        "--output-path", str(args.work_dir / args.run_id),
        "--model-dir", str(args.model_dir),
        "--splitting-algorithm", "fixed_stride",
        "--fixed-stride-split-duration", str(args.clip_seconds),
        "--embedding-algorithm", "cosmos-embed1-224p",
        "--transcode-encoder", encoder,
    ]  # fmt: skip
    if args.video_limit is not None:
        cmd += ["--video-limit", str(args.video_limit)]
    return cmd + passthrough


def main(argv: list[str] | None = None) -> int:
    args, passthrough = parse_args(argv)
    cfg = Config.from_env(require_b2_credentials=False)
    encoder = pick_encoder(args.encoder, detect_gpu_names() if args.encoder == "auto" else [])
    cmd = build_command(cfg, args, encoder, passthrough)

    if args.dry_run:
        print(f"AWS_ENDPOINT_URL={cfg.ais_s3_endpoint} {shlex.join(cmd)}")
        return 0

    for path, hint in ((CURATOR_PYTHON, "make setup-curator"), (CURATOR_SCRIPT, "make setup")):
        if not path.exists():
            print(f"error: {path} not found; run `{hint}`", file=sys.stderr)
            return 2

    out_dir = args.work_dir / args.run_id
    reject_symlinks(out_dir)
    if out_dir.exists():
        print(f"error: {out_dir} already exists; choose a new --run-id", file=sys.stderr)
        return 2
    try:
        check_media_tools(encoder)
    except MediaPrerequisiteError as exc:
        print(f"error: {exc}; see README media prerequisites", file=sys.stderr)
        return 2
    try:
        input_manifest_sha256 = check_input_publication(cfg, args.input_prefix)
    except _AIS_UNREACHABLE as exc:
        print(
            f"error: AIS unreachable at {cfg.ais_endpoint} ({exc}); run `make up`", file=sys.stderr
        )
        return 2
    except ValueError as exc:
        print(f"error: {exc}; re-run `make seed` or fix the input prefix", file=sys.stderr)
        return 2
    if not args.skip_prefetch and args.video_limit is None:
        print(f"prefetching s3://{cfg.b2_bucket}/{args.input_prefix} into AIS ...", file=sys.stderr)
        try:
            prefix = args.input_prefix.strip("/")
            prefetch(cfg, prefix + "/" if prefix else "")
        except _AIS_UNREACHABLE as exc:
            print(
                f"error: AIS unreachable at {cfg.ais_endpoint} ({exc}); run `make up`",
                file=sys.stderr,
            )
            return 2
        except RuntimeError as exc:
            print(f"error: {exc}; use --skip-prefetch to request cold reads", file=sys.stderr)
            return 1
    out_dir.mkdir(parents=True, exist_ok=False)

    gets_before = _backend_gets(cfg)
    curator_env = ais_s3_env(cfg)
    # Ray's uv hook detects the wrapper's ancestor process and selects its CPU environment.
    # Workers must instead inherit the separately installed Curator GPU interpreter.
    curator_env["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"
    curator_env["VIRTUAL_ENV"] = str(CURATOR_PYTHON.parent.parent)
    curator_env["PATH"] = (
        str(CURATOR_PYTHON.parent) + os.pathsep + curator_env.get("PATH", os.defpath)
    )
    rc = run_curator(cmd, curator_env)
    if rc != 0:
        print(f"Curator exited with {rc}; local output kept in {out_dir}", file=sys.stderr)
        return rc
    # Cluster-wide counters: concurrent AIS users would be counted too.
    if gets_before is not None and (gets_after := _backend_gets(cfg)) is not None:
        print(
            f"Curator stage: {gets_after[0] - gets_before[0]} B2 GETs, "
            f"{gets_after[1] - gets_before[1]} bytes (AIS backend counters)",
            file=sys.stderr,
        )

    dest = f"{args.output_prefix.strip('/')}/{args.run_id}"
    try:
        uploaded, _ = upload_dir(
            ais_s3_client(cfg),
            out_dir,
            cfg.b2_bucket,
            dest,
            publish=True,
            provenance={
                "curator_version": CURATOR_VERSION,
                "cosmos_embed1_revision": MODEL_REVISION,
                "input_prefix": args.input_prefix,
                # Pins the exact input publication; None for an unmanifested prefix.
                "input_manifest_sha256": input_manifest_sha256,
                "command": cmd[2:],
                "encoder": encoder,
            },
        )
    except _AIS_UNREACHABLE as exc:
        print(
            f"error: AIS unreachable at {cfg.ais_endpoint} ({exc}); local output kept in "
            f"{out_dir}; run `make up` and rerun with a new --run-id",
            file=sys.stderr,
        )
        return 2
    except OSError as exc:  # e.g. the output changed during upload
        print(f"error: {exc}; local output kept in {out_dir}", file=sys.stderr)
        return 1
    print(f"wrote {uploaded} objects to s3://{cfg.b2_bucket}/{dest}/ (via AIS -> B2)")
    return 0


def check_input_publication(cfg: Config, input_prefix: str) -> str | None:
    """Refuse an input prefix that does not hold exactly its complete `_SUCCESS.json` set.

    Curator reads every object under the prefix, so a pending manifest (interrupted seed),
    unlisted leftovers, or missing objects would otherwise be curated and published as
    complete. Sizes are compared, not bytes; one writer owns the prefix, so this point-in-time
    check holds through the run. A prefix with no manifest (hand-uploaded videos) is accepted.

    Returns the manifest's `manifest_sha256` so outputs can record their exact input, or
    None when the prefix has no manifest.
    """
    s3 = ais_s3_client(cfg)
    try:
        manifest = read_manifest(s3, cfg.b2_bucket, input_prefix)
    except ClientError as exc:
        if exc.response["Error"]["Code"] not in ("404", "NoSuchKey", "NotFound"):
            raise
        return None
    prefix = input_prefix.strip("/") + "/" if input_prefix.strip("/") else ""
    listed = {k[len(prefix) :]: n for k, n in list_keys(s3, cfg.b2_bucket, prefix)}
    listed.pop(MANIFEST, None)
    expected = {rel: info["size"] for rel, info in manifest["files"].items()}
    if listed != expected:
        mismatched = sorted(
            k for k in listed.keys() | expected.keys() if listed.get(k) != expected.get(k)
        )
        shown = ", ".join(prefix + k for k in mismatched[:3])
        more = f" (+{len(mismatched) - 3} more)" if len(mismatched) > 3 else ""
        raise ValueError(
            f"{prefix}{MANIFEST} does not match the prefix: unlisted, missing, or resized "
            f"objects {shown}{more}"
        )
    return manifest_sha256(manifest)


def _backend_gets(cfg: Config) -> tuple[int, int] | None:
    """AIS backend GET counters; diagnostic only, so a failed read never blocks publication."""
    try:
        return ais_backend_gets(cfg)
    except Exception as exc:
        print(f"Curator stage: AIS backend counters unavailable: {exc}", file=sys.stderr)
        return None


if __name__ == "__main__":
    sys.exit(main())
