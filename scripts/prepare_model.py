# SPDX-FileCopyrightText: Copyright (c) 2026 Backblaze, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Verify the pinned Cosmos-Embed1 artifacts and adapt its model code to Transformers 5."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

MODEL_ID = "nvidia/Cosmos-Embed1-224p"
MODEL_REVISION = "787e0b996f5260a71ad474a283c90539a2e12986"
COMPAT = Path(__file__).resolve().parents[1] / "deploy/curator/compat_transformers5.py"
SHA256 = {
    "__init__.py": "dd7d5f5e544e1d2923bf4f1aa548504255938f58a6e74a119356573e6ce9e36e",
    "configuration_embed1.py": "28b2b7220eff2334a984b8719c147e1ebe34cadd497b4ec0664f55710607fddb",
    "modeling_embed1.py": "6da9cca3b8b3432e9137440985f94f8b66fce005cffbc2c78e202b677f540644",
    "modeling_outputs.py": "aa0517eded364e26a0272285ac856074872990cbc322659ad5158e805605556f",
    "modeling_qformer.py": "05498a61aa8a459153770474973e1739b23f1571f0e66fb36e18156103d624ea",
    "modeling_utils.py": "d56f3857ab66943276eeb5c1253e04300958bba87b084be4309e16e10b599a6b",
    "modeling_vit.py": "8225681e7946066eae08e6cafa695bd76bdcad1724edc2828134e077bcecba1b",
    "preprocessing_embed1.py": "e5b6bdc0f2de529edb58e3b5b7d856077f2473e8a81f3a43a89394939f23804c",
    "config.json": "e743a553c42c8dfdfab523ebfb2a2731dce10204fb52ae47968a66366b63cfc2",
    "processor_config.json": "b16edd8ed3dae8b6db1d08580f33b781286f1358f28ede835800c98a54495ca9",
    "added_tokens.json": "f62bb3d9726a553158ae4c0ab64e5b4ece7c4439108d60352722c4cb7af78b44",
    "special_tokens_map.json": "1d8f6c4a10208c351c229d1d7b017ac26354c2ee51580f1d75bc3112a5745ce8",
    "tokenizer_config.json": "eff5d18b71c09817683e2ec5dcabb9de1249288fda799d5ab7eb69c47e127792",
    "model.safetensors.index.json": (
        "5dab0fcbd05d0f03f09ac74843e4fdfa791fa5bd0fe9e2534bdc0c827304ee55"
    ),
    "vocab.txt": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
    "model-00001-of-00005.safetensors": (
        "95d087b31593acfc8fe1fe0f655a97e45431218852eaea228f12503396b0d15e"
    ),
    "model-00002-of-00005.safetensors": (
        "311d7844f5f01c51d49b9fef03469582cac8f2257904634e9fc7a87dc894c45f"
    ),
    "model-00003-of-00005.safetensors": (
        "a4b39f5d9ff3fdbf1fbafd5ac31663be581d0917ebb2152acec5cdf9a9fd7820"
    ),
    "model-00004-of-00005.safetensors": (
        "ed32f4d48984a6a52043a7eb30671cc4816a8bec36ac9d4bb268d49fa8110ad2"
    ),
    "model-00005-of-00005.safetensors": (
        "2af664d97d8d73f4c91ee5a19556b76873b9264fdbc7778919e392e7ff7e12f1"
    ),
}
PATCHES = {
    "modeling_qformer.py": (
        ("    find_pruneable_heads_and_indices,\n", ""),
        (
            "logger = getLogger(__file__)",
            "# Modified by Backblaze for Transformers 5 tensor helpers.\n"
            "from .compat_transformers5 import BertMaskMixin, find_pruneable_heads_and_indices\n\n"
            "logger = getLogger(__file__)",
        ),
        (
            "class BertPreTrainedModel(PreTrainedModel):",
            "class BertPreTrainedModel(BertMaskMixin, PreTrainedModel):",
        ),
        ("        self.init_weights()", "        self.post_init()"),
    ),
    "modeling_embed1.py": (
        (
            "from .configuration_embed1 import CosmosEmbed1Config",
            "# Modified by Backblaze for Transformers 5 initialization.\n"
            "from .compat_transformers5 import restore_video_normalization\n"
            "from .configuration_embed1 import CosmosEmbed1Config",
        ),
        (
            "        self.logit_bias = nn.Parameter(torch.tensor(-10.0))",
            "        self.logit_bias = nn.Parameter(torch.tensor(-10.0))\n        self.post_init()",
        ),
        ("return model_with_te", "return restore_video_normalization(model_with_te)"),
        (
            "return super().from_pretrained(pretrained_model_name_or_path, **kwargs)",
            "return restore_video_normalization(\n"
            "                super().from_pretrained(pretrained_model_name_or_path, **kwargs)\n"
            "            )",
        ),
    ),
}


def patched_source(name: str, source: bytes) -> bytes:
    """Accept only the original source or the exact compatibility adaptation."""
    replacements = PATCHES[name]
    original = source
    # The deleted import is restored separately because an empty replacement is ambiguous.
    for old, new in reversed(replacements):
        if new:
            original = original.replace(new.encode(), old.encode())
    if name == "modeling_qformer.py" and b"    find_pruneable_heads_and_indices,\n" not in original:
        original = original.replace(
            b"    apply_chunking_to_forward,\n",
            b"    apply_chunking_to_forward,\n    find_pruneable_heads_and_indices,\n",
        )
    if hashlib.sha256(original).hexdigest() != SHA256[name]:
        raise ValueError(f"unexpected model source: {name}; refusing to patch")
    result = original
    for old, new in replacements:
        result = result.replace(old.encode(), new.encode())
    return result


def verify_and_patch(directory: Path) -> None:
    """Validate every artifact before modifying any model code."""
    changes = {}
    for name, expected in SHA256.items():
        path = directory / name
        if path.is_symlink():
            raise ValueError(f"model artifact must not be a symlink: {name}")
        if name in PATCHES:
            changes[path] = patched_source(name, path.read_bytes())
        else:
            with path.open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != expected:
                raise ValueError(f"model checksum mismatch: {name}")
    for path, source in changes.items():
        if path.read_bytes() != source:
            write_model_file(path, source)
    helper = directory / COMPAT.name
    if helper.is_symlink():
        raise ValueError("model compatibility helper must not be a symlink")
    content = COMPAT.read_bytes()
    if not helper.exists() or helper.read_bytes() != content:
        write_model_file(helper, content)


def write_model_file(path: Path, content: bytes) -> None:
    """Replace source atomically so an interrupted setup cannot truncate the cache."""
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(content)
            stream.flush()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def prepare_model(model_dir: Path) -> None:
    from filelock import FileLock

    model_dir.mkdir(parents=True, exist_ok=True)
    with FileLock(str(model_dir / ".cosmos_embed1.lock")):
        directory = model_dir / MODEL_ID
        if any((directory / name).is_symlink() for name in (*SHA256, COMPAT.name)):
            raise ValueError("model artifacts must not be symlinks")
        if not all((directory / name).is_file() for name in SHA256):
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=MODEL_ID,
                revision=MODEL_REVISION,
                local_dir=directory,
                allow_patterns=list(SHA256),
            )
        verify_and_patch(directory)
