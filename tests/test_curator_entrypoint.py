# SPDX-FileCopyrightText: Copyright (c) 2026 Backblaze, Inc.
# SPDX-License-Identifier: Apache-2.0

import ast

import curator_entrypoint
import pytest
import run_pipeline


def test_verified_tutorial_has_no_captioning_imports():
    source = curator_entrypoint.video_source(run_pipeline.CURATOR_SCRIPT)
    imports = [n.module for n in ast.walk(ast.parse(source)) if isinstance(n, ast.ImportFrom)]
    assert not any(m and m.startswith("nemo_curator.stages.video.caption") for m in imports)
    assert "CosmosEmbed1EmbeddingStage" in source
    assert "ClipTranscodingStage" in source


def test_modified_tutorial_is_rejected(tmp_path):
    path = tmp_path / "tutorial.py"
    path.write_text("raise RuntimeError('unverified code')")
    with pytest.raises(ValueError, match="unexpected Curator tutorial checksum"):
        curator_entrypoint.video_source(path)


@pytest.mark.parametrize(
    "flag",
    [
        "--with-captions",
        "--with-c",
        "--generate-captions",
        "--generate-c",
        "--enhance-captions",
        "--captioning-algorithm=qwen2.5",
        "--embedding-algorithm=other",
    ],
)
def test_unsupported_model_paths_fail_before_storage_or_gpu_access(flag):
    with pytest.raises(SystemExit) as exc:
        run_pipeline.parse_args([flag, "--dry-run"])
    assert exc.value.code == 2
