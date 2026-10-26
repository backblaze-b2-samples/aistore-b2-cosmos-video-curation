# SPDX-FileCopyrightText: Copyright (c) 2026 Backblaze, Inc.
# SPDX-License-Identifier: Apache-2.0
from fnmatch import fnmatch

import seed_b2

EGO = "gr1_arms_only.CanSort/videos/chunk-000/observation.images.ego_view"


def test_default_pattern_selects_first_50_episodes():
    matched = [
        i for i in range(1000) if fnmatch(f"{EGO}/episode_{i:06d}.mp4", seed_b2.DEFAULT_PATTERN)
    ]
    assert matched == list(range(50))


def test_parse_args_defaults_and_local_dir(tmp_path):
    args = seed_b2.parse_args([])
    assert (args.repo, args.revision, args.prefix) == (
        seed_b2.DEFAULT_REPO,
        seed_b2.DEFAULT_REVISION,
        "raw",
    )
    assert len(args.revision) == 40  # pinned commit, never a branch
    assert seed_b2.parse_args(["--local-dir", str(tmp_path)]).local_dir == tmp_path


def test_main_uploads_local_dir_to_b2(b2_env, monkeypatch, tmp_path, capsys):
    for k, v in b2_env.items():
        monkeypatch.setenv(k, v)
    (tmp_path / "v.mp4").write_bytes(b"x")
    calls = []
    monkeypatch.setattr(seed_b2, "b2_client", lambda cfg: "client")
    monkeypatch.setattr(seed_b2, "upload_dir", lambda *a, **k: calls.append((a, k)) or (1, 0))
    assert seed_b2.main(["--local-dir", str(tmp_path)]) == 0
    (client, src, bucket, prefix), kwargs = calls[0]
    assert (client, src, bucket, prefix) == ("client", tmp_path, "my-video-corpus", "raw")
    assert kwargs["exclude"] == (".cache",)
    assert "1 uploaded" in capsys.readouterr().out


def test_listing_root_stops_at_first_glob():
    assert seed_b2.listing_root(seed_b2.DEFAULT_PATTERN) == EGO
    assert seed_b2.listing_root("gr1_arms_only.CanSort/videos/**/*.mp4") == (
        "gr1_arms_only.CanSort/videos"
    )
    assert seed_b2.listing_root("*.mp4") is None
