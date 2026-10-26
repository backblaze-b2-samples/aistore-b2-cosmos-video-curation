# SPDX-FileCopyrightText: Copyright (c) 2026 Backblaze, Inc.
# SPDX-License-Identifier: Apache-2.0
import dataclasses

import pytest

from aistore_b2_curation.config import DEFAULT_AIS_ENDPOINT, Config, ConfigError


def test_from_env_defaults_and_normalization(b2_env):
    b2_env["B2_ENDPOINT"] += "/"
    cfg = Config.from_env(b2_env)
    assert cfg.b2_endpoint == "https://s3.us-west-004.backblazeb2.com"
    assert cfg.ais_endpoint == DEFAULT_AIS_ENDPOINT
    assert cfg.ais_s3_endpoint == "http://127.0.0.1:51080/s3"
    assert cfg.hf_token is None
    assert cfg.has_b2_credentials


def test_custom_ais_endpoint_trailing_slash(b2_env):
    b2_env["AIS_ENDPOINT"] = "http://10.0.0.5:51080/"
    assert Config.from_env(b2_env).ais_s3_endpoint == "http://10.0.0.5:51080/s3"


def test_missing_vars_are_all_reported(b2_env):
    del b2_env["B2_BUCKET"], b2_env["B2_APP_KEY"]
    with pytest.raises(ConfigError, match="B2_BUCKET, B2_APP_KEY"):
        Config.from_env(b2_env)


def test_credentials_optional_for_ais_only_paths(b2_env):
    del b2_env["B2_KEY_ID"], b2_env["B2_APP_KEY"]
    cfg = Config.from_env(b2_env, require_b2_credentials=False)
    assert not cfg.has_b2_credentials


@pytest.mark.parametrize(
    ("var", "value"),
    [("B2_ENDPOINT", "s3.us-west-004.backblazeb2.com"), ("AIS_ENDPOINT", "localhost:51080")],
)
def test_rejects_endpoint_without_scheme(b2_env, var, value):
    b2_env[var] = value
    with pytest.raises(ConfigError, match="http"):
        Config.from_env(b2_env)


@pytest.mark.parametrize("bucket", ["short", "has_underscore", "x" * 64])
def test_rejects_invalid_bucket(b2_env, bucket):
    b2_env["B2_BUCKET"] = bucket
    with pytest.raises(ConfigError, match="bucket"):
        Config.from_env(b2_env)


def test_secrets_never_in_repr(b2_env):
    b2_env["HF_TOKEN"] = "hf_fake_token"
    text = repr(Config.from_env(b2_env))
    for secret in ("fake-key-id", "fake-app-key", "hf_fake_token"):
        assert secret not in text


def test_is_immutable(b2_env):
    cfg = Config.from_env(b2_env)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.b2_bucket = "other-bucket"
