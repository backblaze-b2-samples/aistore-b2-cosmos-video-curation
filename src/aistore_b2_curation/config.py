# SPDX-FileCopyrightText: Copyright (c) 2026 Backblaze, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Environment-driven settings shared by the seed, run, and benchmark entrypoints."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field

DEFAULT_AIS_ENDPOINT = "http://127.0.0.1:51080"

# B2 bucket names: 6-63 chars, letters, digits, and hyphens.
_BUCKET_RE = re.compile(r"^[A-Za-z0-9-]{6,63}$")


class ConfigError(ValueError):
    """Raised when required settings are missing or malformed."""


@dataclass(frozen=True)
class Config:
    b2_endpoint: str
    b2_region: str
    b2_bucket: str
    b2_key_id: str = field(default="", repr=False)
    b2_app_key: str = field(default="", repr=False)
    ais_endpoint: str = DEFAULT_AIS_ENDPOINT
    hf_token: str | None = field(default=None, repr=False)

    @property
    def ais_s3_endpoint(self) -> str:
        """AIS S3-compatible API root, the endpoint S3 clients (boto3, s3fs) target."""
        return f"{self.ais_endpoint}/s3"

    @property
    def has_b2_credentials(self) -> bool:
        return bool(self.b2_key_id and self.b2_app_key)

    @classmethod
    def from_env(
        cls, env: Mapping[str, str] | None = None, *, require_b2_credentials: bool = True
    ) -> Config:
        """Build from environment variables.

        Only code that talks to B2 directly (seeding, the B2-direct benchmark) needs the key.
        Everything that goes through AIS runs with `require_b2_credentials=False`.
        """
        env = os.environ if env is None else env

        def get(name: str, default: str = "") -> str:
            return env.get(name, default).strip()

        missing = [n for n in ("B2_ENDPOINT", "B2_REGION", "B2_BUCKET") if not get(n)]
        if require_b2_credentials:
            missing += [n for n in ("B2_KEY_ID", "B2_APP_KEY") if not get(n)]
        if missing:
            raise ConfigError(f"missing required env vars: {', '.join(missing)} (see .env.example)")

        cfg = cls(
            b2_endpoint=get("B2_ENDPOINT").rstrip("/"),
            b2_region=get("B2_REGION"),
            b2_bucket=get("B2_BUCKET"),
            b2_key_id=get("B2_KEY_ID"),
            b2_app_key=get("B2_APP_KEY"),
            ais_endpoint=(get("AIS_ENDPOINT") or DEFAULT_AIS_ENDPOINT).rstrip("/"),
            hf_token=get("HF_TOKEN") or None,
        )
        cfg._validate()
        return cfg

    def _validate(self) -> None:
        for name in ("b2_endpoint", "ais_endpoint"):
            value = getattr(self, name)
            if not value.startswith(("http://", "https://")):
                raise ConfigError(
                    f"{name.upper()} must start with http:// or https://, got {value!r}"
                )
        if not _BUCKET_RE.match(self.b2_bucket):
            raise ConfigError(f"B2_BUCKET {self.b2_bucket!r} is not a valid bucket name")
