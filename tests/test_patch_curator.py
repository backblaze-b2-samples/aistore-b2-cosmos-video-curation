# SPDX-FileCopyrightText: Copyright (c) 2026 Backblaze, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the pinned upstream task method without importing GPU dependencies."""

import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace

import patch_curator
import pytest

UPSTREAM = Path(__file__).parent / "fixtures" / "curator_file_group.py.txt"


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "file_group.py"
    path.write_bytes(UPSTREAM.read_bytes())
    return path


class RemotePath:
    def __init__(self, path):
        self.path = path

    def __str__(self):
        return self.path


def task_id(path, data):
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "get_deterministic_id"
    )
    namespace = {
        "get_deterministic_hash": lambda paths: hashlib.sha256(
            ("|".join(sorted(paths)) + "|").encode()
        ).hexdigest()[:12]
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)  # noqa: S102
    return namespace["get_deterministic_id"](SimpleNamespace(data=data))


def test_remote_task_ids_preserve_reads_and_string_ids(source):
    paths = ["bucket/raw/b.mp4", "bucket/raw/a.mp4"]
    expected = task_id(source, paths)
    remote = [RemotePath(path) for path in paths]
    with pytest.raises(TypeError):
        task_id(source, remote[:1])
    assert patch_curator.patch_file(source)
    assert task_id(source, paths) == expected
    assert task_id(source, remote) == expected
    assert task_id(source, remote[::-1]) == expected
    assert task_id(source, [remote[0], paths[1]]) == expected
    assert all(isinstance(path, RemotePath) for path in remote)


def test_patch_is_idempotent(source):
    assert patch_curator.patch_file(source)
    patched = source.read_bytes()
    assert not patch_curator.patch_file(source)
    assert source.read_bytes() == patched


@pytest.mark.parametrize("already_patched", [False, True])
def test_unknown_source_is_rejected_without_changes(source, already_patched):
    if already_patched:
        patch_curator.patch_file(source)
    changed = source.read_bytes() + b"# unexpected upstream change\n"
    source.write_bytes(changed)
    with pytest.raises(ValueError, match="unexpected Curator source"):
        patch_curator.patch_file(source)
    assert source.read_bytes() == changed


def test_setup_checks_distribution_version_before_patching(monkeypatch, capsys):
    monkeypatch.setattr(patch_curator, "distribution", lambda _: SimpleNamespace(version="1.4.0"))
    assert patch_curator.main() == 1
    assert "expected nemo-curator 1.3.0; found 1.4.0" in capsys.readouterr().err


def test_setup_patches_installed_distribution(monkeypatch, source, capsys):
    dist = SimpleNamespace(version="1.3.0", locate_file=lambda _: source)
    monkeypatch.setattr(patch_curator, "distribution", lambda _: dist)
    assert patch_curator.main() == 0
    assert "fix: applied" in capsys.readouterr().out
    assert patch_curator.main() == 0
    assert "fix: already applied" in capsys.readouterr().out
