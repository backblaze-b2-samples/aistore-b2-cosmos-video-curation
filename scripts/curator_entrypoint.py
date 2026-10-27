# SPDX-FileCopyrightText: Copyright (c) 2026 Backblaze, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Run the verified Curator video tutorial without optional captioning imports."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

from prepare_model import prepare_model

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / ".cache/curator/video_split_clip_example.py"
SCRIPT_SHA256 = "e6fec820b74911cd3b9a0eadc88be1c01b47dc06b49e90683a88fd584570307d"
CAPTION_IMPORTS = (
    "from nemo_curator.stages.video.caption.caption_enhancement import CaptionEnhancementStage\n",
    "from nemo_curator.stages.video.caption.caption_generation import CaptionGenerationStage\n",
    "from nemo_curator.stages.video.caption.caption_preparation import CaptionPreparationStage\n",
)


def video_source(path: Path) -> str:
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != SCRIPT_SHA256:
        raise ValueError("unexpected Curator tutorial checksum; run make setup")
    text = source.decode()
    for statement in CAPTION_IMPORTS:
        text = text.replace(statement, "")
    return text


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--model-dir", type=Path, required=True)
    args, _ = parser.parse_known_args(argv)
    source = video_source(SCRIPT)
    prepare_model(args.model_dir)
    sys.argv = [str(SCRIPT), *argv]
    # Execute only the checksum-verified upstream CLI; its original license is retained.
    exec(compile(source, str(SCRIPT), "exec"), {"__name__": "__main__", "__file__": str(SCRIPT)})  # noqa: S102
    return 0


if __name__ == "__main__":
    sys.exit(main())
