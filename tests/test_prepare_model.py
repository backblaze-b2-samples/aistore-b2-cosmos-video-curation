# SPDX-FileCopyrightText: Copyright (c) 2026 Backblaze, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Model preparation must reject unexpected artifacts before executing model code."""

import hashlib
from pathlib import Path

import huggingface_hub
import prepare_model
import pytest


@pytest.fixture
def model(tmp_path, monkeypatch):
    sources = {
        "modeling_qformer.py": (
            b"from transformers.pytorch_utils import (\n"
            b"    apply_chunking_to_forward,\n"
            b"    find_pruneable_heads_and_indices,\n)\n"
            b"logger = getLogger(__file__)\n"
            b"class BertPreTrainedModel(PreTrainedModel):\n"
            b"    def __init__(self):\n        self.init_weights()\n"
        ),
        "modeling_embed1.py": (
            b"from .configuration_embed1 import CosmosEmbed1Config\n"
            b"class Model:\n    def __init__(self):\n"
            b"        self.logit_bias = nn.Parameter(torch.tensor(-10.0))\n"
            b"    def load(self):\n"
            b"        return super().from_pretrained(pretrained_model_name_or_path, **kwargs)\n"
        ),
        "config.json": b'{"model_type":"cosmos-embed1"}',
        "model.safetensors": b"test-weights",
    }
    directory = tmp_path / prepare_model.MODEL_ID
    directory.mkdir(parents=True)
    for name, content in sources.items():
        (directory / name).write_bytes(content)
    monkeypatch.setattr(
        prepare_model,
        "SHA256",
        {name: hashlib.sha256(content).hexdigest() for name, content in sources.items()},
    )
    return tmp_path, directory, sources


def test_verified_cache_is_patched_without_network_and_is_idempotent(model, monkeypatch):
    root, directory, _ = model
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **_: pytest.fail("network"))
    prepare_model.prepare_model(root)
    first = {p.name: p.read_bytes() for p in directory.iterdir()}
    assert b"BertMaskMixin" in first["modeling_qformer.py"]
    assert b"self.post_init()" in first["modeling_qformer.py"]
    assert b"restore_video_normalization" in first["modeling_embed1.py"]
    assert first[prepare_model.COMPAT.name] == prepare_model.COMPAT.read_bytes()
    prepare_model.prepare_model(root)
    assert first == {p.name: p.read_bytes() for p in directory.iterdir()}


@pytest.mark.parametrize("name", ["modeling_qformer.py", "config.json", "model.safetensors"])
def test_tampering_is_rejected_before_any_patch(model, name):
    root, directory, sources = model
    path = directory / name
    path.write_bytes(path.read_bytes() + b"tampered")
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    with pytest.raises(ValueError, match=r"checksum mismatch|unexpected model source"):
        prepare_model.prepare_model(root)
    assert before == {p.name: p.read_bytes() for p in directory.iterdir()}
    assert not (directory / prepare_model.COMPAT.name).exists()
    assert sources["modeling_embed1.py"] == (directory / "modeling_embed1.py").read_bytes()


def test_modified_patched_model_is_also_rejected(model):
    root, directory, _ = model
    prepare_model.prepare_model(root)
    path = directory / "modeling_qformer.py"
    path.write_bytes(path.read_bytes() + b"# unknown change\n")
    with pytest.raises(ValueError, match="unexpected model source"):
        prepare_model.prepare_model(root)


def test_download_uses_exact_revision_and_expected_artifacts(model, monkeypatch):
    root, directory, sources = model
    missing = directory / "model.safetensors"
    missing.unlink()
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        missing.write_bytes(sources[missing.name])

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    prepare_model.prepare_model(root)
    assert calls == [
        {
            "repo_id": prepare_model.MODEL_ID,
            "revision": prepare_model.MODEL_REVISION,
            "local_dir": directory,
            "allow_patterns": list(prepare_model.SHA256),
        }
    ]


def test_symlinked_model_artifact_is_rejected(model):
    root, directory, _ = model
    path = directory / "config.json"
    target = root / "external.json"
    path.rename(target)
    path.symlink_to(target)
    with pytest.raises(ValueError, match="must not be symlinks"):
        prepare_model.prepare_model(root)


def test_missing_symlink_target_is_rejected_before_download(model, monkeypatch):
    root, directory, _ = model
    path = directory / "config.json"
    path.unlink()
    path.symlink_to(root / "missing.json")
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **_: pytest.fail("network"))
    with pytest.raises(ValueError, match="must not be symlinks"):
        prepare_model.prepare_model(root)
    assert not (root / "missing.json").exists()


def test_all_pinned_files_use_sha256():
    assert len(prepare_model.MODEL_REVISION) == 40
    assert len(prepare_model.SHA256) == 20
    assert all(len(bytes.fromhex(digest)) == 32 for digest in prepare_model.SHA256.values())


def test_interrupted_replace_preserves_original_and_cleans_temporary_file(tmp_path, monkeypatch):
    path = tmp_path / "model.py"
    path.write_bytes(b"original")

    def interrupted(*_):
        raise OSError("interrupted replace")

    monkeypatch.setattr(Path, "replace", interrupted)
    with pytest.raises(OSError, match="interrupted replace"):
        prepare_model.write_model_file(path, b"updated")
    assert path.read_bytes() == b"original"
    assert list(tmp_path.iterdir()) == [path]
