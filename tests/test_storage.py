# SPDX-FileCopyrightText: Copyright (c) 2026 Backblaze, Inc.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from types import SimpleNamespace

import pytest
from aistore.sdk.errors import Timeout as AISTimeout
from aistore.sdk.wait_result import WaitResult
from botocore.exceptions import ClientError

from aistore_b2_curation import storage
from aistore_b2_curation.config import Config


class FakeS3:
    """Minimal boto3 S3 client fake for list_keys/upload_dir."""

    def __init__(self, objects: dict[str, int] | None = None):
        self.objects = dict(objects or {})
        self.uploads: list[str] = []
        self.metadata = {}

    def get_paginator(self, _name):
        return self

    def paginate(self, Bucket, Prefix):
        yield {
            "Contents": [
                {"Key": k, "Size": v} for k, v in self.objects.items() if k.startswith(Prefix)
            ]
        }

    def head_object(self, Bucket, Key):
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "404"}}, "HeadObject")
        return {"ContentLength": self.objects[Key], "Metadata": self.metadata.get(Key, {})}

    def upload_file(self, path, bucket, key, Config=None, ExtraArgs=None):
        self.uploads.append(key)
        self.objects[key] = Path(path).stat().st_size
        self.metadata[key] = ExtraArgs["Metadata"]


@pytest.fixture
def cfg(b2_env) -> Config:
    return Config.from_env(b2_env)


def test_ais_s3_env_targets_ais_and_drops_b2_secrets(cfg):
    base = {"PATH": "/usr/bin", "B2_KEY_ID": "k", "B2_APP_KEY": "s", "AWS_PROFILE": "prod"}
    env = storage.ais_s3_env(cfg, base)
    assert env["AWS_ENDPOINT_URL"] == "http://127.0.0.1:51080/s3"
    assert env["AWS_REGION"] == "us-west-004"
    assert env["PATH"] == "/usr/bin"
    for name in ("B2_KEY_ID", "B2_APP_KEY", "AWS_PROFILE"):
        assert name not in env
    assert cfg.b2_app_key not in env.values()


def test_clients_point_at_the_right_endpoints(cfg):
    assert storage.b2_client(cfg).meta.endpoint_url == cfg.b2_endpoint
    assert storage.ais_s3_client(cfg).meta.endpoint_url == cfg.ais_s3_endpoint


def test_b2_client_requires_credentials(b2_env):
    del b2_env["B2_APP_KEY"]
    cfg = Config.from_env(b2_env, require_b2_credentials=False)
    with pytest.raises(ValueError, match="B2_APP_KEY"):
        storage.b2_client(cfg)


def test_upload_dir_is_idempotent_and_honors_exclude(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "one.mp4").write_bytes(b"1" * 10)
    (tmp_path / "two.mp4").write_bytes(b"2" * 20)
    (tmp_path / ".cache").mkdir()
    (tmp_path / ".cache" / "lock").write_bytes(b"x")

    s3 = FakeS3({"raw/two.mp4": 20})  # same-size object without hash metadata
    assert storage.upload_dir(s3, tmp_path, "b", "/raw/", exclude=(".cache",)) == (2, 0)
    assert sorted(s3.uploads) == ["raw/a/one.mp4", "raw/two.mp4"]
    assert storage.upload_dir(s3, tmp_path, "b", "raw", exclude=(".cache",)) == (0, 2)


def test_upload_dir_reuploads_on_size_change(tmp_path):
    (tmp_path / "v.mp4").write_bytes(b"new-content")
    s3 = FakeS3({"v.mp4": 3})
    assert storage.upload_dir(s3, tmp_path, "b", "") == (1, 0)
    assert s3.uploads == ["v.mp4"]


def test_service_specific_endpoint_cannot_override_ais(cfg):
    base = {
        "AWS_ENDPOINT_URL_S3": "https://other.example",
        "AWS_IGNORE_CONFIGURED_ENDPOINT_URLS": "true",
    }
    env = storage.ais_s3_env(cfg, base)
    assert env["AWS_ENDPOINT_URL_S3"] == cfg.ais_s3_endpoint
    assert env["AWS_IGNORE_CONFIGURED_ENDPOINT_URLS"] == "false"


def test_same_size_changed_content_is_uploaded(tmp_path):
    path = tmp_path / "v.mp4"
    path.write_bytes(b"old")
    s3 = FakeS3()
    storage.upload_dir(s3, tmp_path, "b", "raw")
    path.write_bytes(b"new")
    assert storage.upload_dir(s3, tmp_path, "b", "raw") == (1, 0)


def mock_prefetch(monkeypatch, result):
    calls = []

    def wait(**kwargs):
        calls.append(kwargs)
        if isinstance(result, Exception):
            raise result
        return result

    group = SimpleNamespace(prefetch=lambda: "prefetch-job")
    bucket = SimpleNamespace(objects=lambda **kwargs: group)
    client = SimpleNamespace(
        bucket=lambda *args, **kwargs: bucket,
        job=lambda _job_id: SimpleNamespace(wait=wait),
    )
    monkeypatch.setattr(storage, "_ais_client", lambda _cfg: client)
    return calls


def test_prefetch_requires_successful_job_completion(cfg, monkeypatch):
    calls = mock_prefetch(monkeypatch, WaitResult(job_id="prefetch-job", success=True))
    assert storage.prefetch(cfg, "raw/", timeout_s=12) == "prefetch-job"
    assert calls == [{"timeout": 12, "verbose": False}]


@pytest.mark.parametrize("error", ["backend unavailable", "aborted", None])
def test_failed_prefetch_is_an_error(cfg, monkeypatch, error):
    mock_prefetch(monkeypatch, WaitResult(job_id="prefetch-job", success=False, error=error))
    with pytest.raises(RuntimeError, match=rf"prefetch-job failed: {error or 'aborted'}"):
        storage.prefetch(cfg, "raw/")


def test_prefetch_timeout_identifies_job_and_deadline(cfg, monkeypatch):
    mock_prefetch(monkeypatch, AISTimeout("prefetch"))
    with pytest.raises(RuntimeError, match="prefetch-job timed out after 12s"):
        storage.prefetch(cfg, "raw/", timeout_s=12)


def test_evict_scopes_cache_listing_to_common_prefix(cfg, monkeypatch):
    prefixes, evicted = [], []
    monkeypatch.setattr(
        storage, "cached_keys", lambda _cfg, prefix: prefixes.append(prefix) or {"raw/a/1.mp4"}
    )

    def objects(obj_names):
        evicted.extend(obj_names)
        return SimpleNamespace(evict=lambda: "evict-job")

    bucket = SimpleNamespace(objects=objects)
    client = SimpleNamespace(
        bucket=lambda *args, **kwargs: bucket,
        job=lambda _job_id: SimpleNamespace(wait=lambda **_: WaitResult(job_id="j", success=True)),
    )
    monkeypatch.setattr(storage, "_ais_client", lambda _cfg: client)
    storage.evict(cfg, ["raw/a/1.mp4", "raw/a/2.mp4"])
    assert prefixes == ["raw/a/"]
    assert evicted == ["raw/a/1.mp4"]


def test_evict_nothing_skips_cluster_calls(cfg, monkeypatch):
    monkeypatch.setattr(storage, "_ais_client", lambda _cfg: pytest.fail("cluster call"))
    monkeypatch.setattr(storage, "cached_keys", lambda *a: pytest.fail("bucket listing"))
    storage.evict(cfg, [])


# Shape of Cluster.get_performance(): raw target stats, counters as bare cumulative ints.
PERF = {
    "t1": {"tracker": {"aws.get.n": 3, "aws.get.size": 3000, "get.n": 9}},
    "t2": {"tracker": {"aws.get.n": 2, "aws.get.size": 500, "aws.head.n": 7}},
}


def test_backend_get_stats_sums_b2_gets_across_targets():
    assert storage.backend_get_stats(PERF) == (5, 3500)


def test_backend_get_stats_counts_omitted_counters_as_zero():
    # AIS drops zero-valued counters from the payload, e.g. before the first cold GET.
    assert storage.backend_get_stats({"t1": {"tracker": {"get.n": 4}}}) == (0, 0)


@pytest.mark.parametrize("perf", [{}, {"t1": {"stats": {}}}])
def test_backend_get_stats_rejects_unexpected_payload_shape(perf):
    with pytest.raises(ValueError, match="AIS performance payload"):
        storage.backend_get_stats(perf)


def test_ais_backend_gets_reads_cluster_performance(cfg, monkeypatch):
    client = SimpleNamespace(cluster=lambda: SimpleNamespace(get_performance=lambda: PERF))
    monkeypatch.setattr(storage, "_ais_client", lambda _cfg: client)
    assert storage.ais_backend_gets(cfg) == (5, 3500)
