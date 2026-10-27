# SPDX-FileCopyrightText: Copyright (c) 2026 Backblaze, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Guard the model's known dependency requirement without importing GPU packages."""

import re
from pathlib import Path

from packaging.version import Version


def test_cosmos_embed1_gpu_lock_retains_transformers_four():
    lock = Path(__file__).resolve().parents[1] / "deploy" / "curator" / "uv.lock"
    match = re.search(r'^name = "transformers"\nversion = "([^"]+)"$', lock.read_text(), re.M)
    assert match is not None, "Transformers must be present in the GPU lock"
    assert Version(match[1]).major == 4, (
        "Cosmos-Embed1 imports Transformers 4 APIs removed in version 5; "
        "validate updated model code before changing the major version"
    )
