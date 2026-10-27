# SPDX-FileCopyrightText: Copyright (c) 2026 Backblaze, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Apply the remote file-group ID fix to the pinned NeMo Curator GPU environment."""

from __future__ import annotations

import hashlib
import sys
import tempfile
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

VERSION = "1.4.0"
SOURCE_SHA256 = "190224b7ff697f3c6dbfe497f2fbbb4f59d430734fe34af6afb136a6ba275f96"
OLD = b"        return get_deterministic_hash(sorted(self.data))\n"
NEW = (
    b"        # Modified by Backblaze: normalize remote FSPath values before hashing.\n"
    b"        return get_deterministic_hash(sorted(str(path) for path in self.data))\n"
)


def patch_file(path: Path) -> bool:
    """Patch only the known upstream source; an already patched file is unchanged."""
    source = path.read_bytes()
    original = source.replace(NEW, OLD)
    if hashlib.sha256(original).hexdigest() != SOURCE_SHA256:
        raise ValueError(f"unexpected Curator source: {path}; refusing to patch")
    if original != source:
        return False
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(source.replace(OLD, NEW))
            stream.flush()
            temporary.chmod(path.stat().st_mode & 0o777)
            # Replace the file instead of modifying uv's shared wheel-cache hard link.
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    return True


def main() -> int:
    try:
        dist = distribution("nemo-curator")
        if dist.version != VERSION:
            raise ValueError(f"expected nemo-curator {VERSION}; found {dist.version}")
        changed = patch_file(Path(dist.locate_file("nemo_curator/tasks/file_group.py")))
    except (OSError, ValueError, PackageNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Curator remote file-group ID fix: {'applied' if changed else 'already applied'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
