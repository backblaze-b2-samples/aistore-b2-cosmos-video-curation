# SPDX-FileCopyrightText: Copyright (c) 2026 Backblaze, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Seed the B2 bucket with raw videos, simulating a corpus that already lives in B2.

Source is either a pinned Hugging Face dataset revision filtered by `--allow-pattern`, or a
local directory. Uploads go straight to B2 and skip objects with matching SHA-256 metadata, so
re-running is cheap. The full CanSort pattern below is ~1.2 GB.

    uv run python scripts/seed_b2.py                  # default 50-video subset (~60 MB)
    uv run python scripts/seed_b2.py --allow-pattern 'gr1_arms_only.CanSort/videos/**/*.mp4'
    uv run python scripts/seed_b2.py --local-dir ./my_videos
"""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from fnmatch import fnmatch
from pathlib import Path

from aistore_b2_curation.config import Config
from aistore_b2_curation.storage import b2_client, upload_dir

# CC BY 4.0, ungated, H.264 MP4s (256x256, 20 fps, ~16 s each); revision pinned.
DEFAULT_REPO = "nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim"
DEFAULT_REVISION = "ea7ac0b68f87da62f1e726771bba0fe74300802f"
DEFAULT_PATTERN = (
    "gr1_arms_only.CanSort/videos/chunk-000/observation.images.ego_view/episode_0000[0-4]?.mp4"
)
DEFAULT_PREFIX = "raw"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--local-dir", type=Path, help="upload this directory instead of HF data")
    src.add_argument("--repo", default=DEFAULT_REPO, help="Hugging Face dataset repo id")
    p.add_argument("--revision", default=DEFAULT_REVISION, help="dataset commit SHA")
    p.add_argument(
        "--allow-pattern",
        action="append",
        help=f"fnmatch glob over repo paths (repeatable; default: {DEFAULT_PATTERN})",
    )
    p.add_argument("--prefix", default=DEFAULT_PREFIX, help="destination key prefix in B2")
    args = p.parse_args(argv)
    if not args.local_dir and not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        p.error("--revision must be a full 40-character commit SHA")
    if args.local_dir is not None:
        args.local_dir = args.local_dir.resolve()  # the root may be a symlink; its contents may not
    return args


def listing_root(pattern: str) -> str | None:
    """Deepest directory in `pattern` without glob characters.

    Listing only that subtree keeps seeding fast; the full repo tree is very large.
    """
    literal = []
    for part in pattern.split("/")[:-1]:
        if any(c in part for c in "*?["):
            break
        literal.append(part)
    return "/".join(literal) or None


def download_hf(
    repo: str,
    revision: str,
    patterns: list[str],
    dest: Path,
    token: str | None,
) -> int:
    """Download files matching any fnmatch `patterns` (where `*` also matches `/`).

    Resolve every selected file at the pinned revision. Hugging Face's cache avoids
    redundant downloads.
    """
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.hf_api import RepoFile

    api = HfApi(token=token)
    matched = {
        entry.path: entry.size
        for pattern in patterns
        for entry in api.list_repo_tree(
            repo, listing_root(pattern), recursive=True, revision=revision, repo_type="dataset"
        )
        if isinstance(entry, RepoFile) and fnmatch(entry.path, pattern)
    }
    if not matched:
        raise SystemExit(f"no files matched {patterns} in {repo}@{revision}")

    def fetch(filename: str) -> str:
        return hf_hub_download(
            repo, filename, repo_type="dataset", revision=revision, local_dir=dest, token=token
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(fetch, sorted(matched)))
    return len(matched)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = Config.from_env()
    s3 = b2_client(cfg)

    with tempfile.TemporaryDirectory(prefix="seed-b2-") as tmp:
        if args.local_dir:
            src = args.local_dir
            if not src.is_dir():
                raise SystemExit(f"{src} is not a directory")
        else:
            patterns = args.allow_pattern or [DEFAULT_PATTERN]
            print(f"downloading {args.repo}@{args.revision[:12]} {patterns}", file=sys.stderr)
            src = Path(tmp).resolve()
            matched = download_hf(
                args.repo,
                args.revision,
                patterns,
                src,
                cfg.hf_token,
            )
            print(f"{matched} files matched and fetched", file=sys.stderr)
        # HF writes bookkeeping under .cache/; only upload dataset files.
        uploaded, skipped = upload_dir(
            s3,
            src,
            cfg.b2_bucket,
            args.prefix,
            exclude=(".cache",),
            publish=True,
            provenance={
                "source": args.repo if not args.local_dir else "local",
                "revision": args.revision if not args.local_dir else None,
            },
        )

    print(f"s3://{cfg.b2_bucket}/{args.prefix}/: {uploaded} uploaded, {skipped} unchanged")
    return 0


if __name__ == "__main__":
    sys.exit(main())
