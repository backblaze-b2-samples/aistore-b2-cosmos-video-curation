# SPDX-FileCopyrightText: Copyright (c) 2026 Backblaze, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures. Entry-point scripts live outside the package, so expose them for import."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for sub in ("scripts", "benchmarks"):
    sys.path.insert(0, str(ROOT / sub))


@pytest.fixture
def b2_env() -> dict[str, str]:
    """A complete, valid environment with obviously fake values."""
    return {
        "B2_ENDPOINT": "https://s3.us-west-004.backblazeb2.com",
        "B2_REGION": "us-west-004",
        "B2_BUCKET": "my-video-corpus",
        "B2_KEY_ID": "fake-key-id",
        "B2_APP_KEY": "fake-app-key",
    }
