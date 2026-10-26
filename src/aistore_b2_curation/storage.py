# SPDX-FileCopyrightText: Copyright (c) 2026 Backblaze, Inc.
# SPDX-License-Identifier: Apache-2.0
"""S3 client wiring for the two data paths: B2 directly, or B2 through AIStore.

AIS runs without AuthN in this sample, so clients that go through it carry placeholder
credentials. AIS targets hold the backend key; direct B2 tools also require it.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config as BotoConfig

from aistore_b2_curation.artifacts import upload_tree
from aistore_b2_curation.config import Config

if TYPE_CHECKING:
    from aistore.sdk import Client
    from aistore.sdk.bucket import Bucket

# Placeholder credentials for AIS: botocore insists on signing, AIS (no AuthN) ignores it.
AIS_ANON_USER = "ais-anonymous"

# Settings botocore also reads from the environment, so child processes (s3fs inside
# NeMo Curator) get the same behavior as our own clients.
_CHECKSUM_MODE = "when_required"  # skip optional CRC trailers some S3-compatible stores reject

_BOTO_CONFIG = BotoConfig(
    max_pool_connections=64,
    s3={"addressing_style": "path"},
    retries={"mode": "standard", "max_attempts": 5},
    request_checksum_calculation=_CHECKSUM_MODE,
    response_checksum_validation=_CHECKSUM_MODE,
)

TRANSFER_CONFIG = TransferConfig(
    multipart_threshold=64 * 1024**2,
    multipart_chunksize=64 * 1024**2,
    max_concurrency=8,
)


def b2_client(cfg: Config) -> Any:
    """boto3 S3 client that talks to B2 (the system of record) directly."""
    if not cfg.has_b2_credentials:
        raise ValueError("B2_KEY_ID and B2_APP_KEY are required for direct B2 access")
    return boto3.client(
        "s3",
        endpoint_url=cfg.b2_endpoint,
        region_name=cfg.b2_region,
        aws_access_key_id=cfg.b2_key_id,
        aws_secret_access_key=cfg.b2_app_key,
        config=_BOTO_CONFIG,
    )


def ais_s3_client(cfg: Config) -> Any:
    """boto3 S3 client that reads and writes the B2 bucket through AIS's S3 API."""
    return boto3.client(
        "s3",
        endpoint_url=cfg.ais_s3_endpoint,
        region_name=cfg.b2_region,
        aws_access_key_id=AIS_ANON_USER,
        aws_secret_access_key=AIS_ANON_USER,
        config=_BOTO_CONFIG,
    )


def ais_s3_env(cfg: Config, base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Environment for a child process whose S3 clients must go through AIS.

    botocore (and therefore s3fs/fsspec) honors AWS_ENDPOINT_URL, so tools that take only an
    `s3://` path, like NeMo Curator's VideoReader, need no code changes. B2 secrets and any
    AWS profile settings are removed to route the supported S3 read path through AIS.
    """
    env = dict(os.environ if base is None else base)
    for name in (
        "B2_KEY_ID",
        "B2_APP_KEY",
        "AWS_PROFILE",
        "AWS_DEFAULT_PROFILE",
        "AWS_SESSION_TOKEN",
        "AWS_ENDPOINT_URL_S3",
        "AWS_IGNORE_CONFIGURED_ENDPOINT_URLS",
    ):
        env.pop(name, None)
    env.update(
        AWS_ENDPOINT_URL=cfg.ais_s3_endpoint,
        AWS_ENDPOINT_URL_S3=cfg.ais_s3_endpoint,
        AWS_IGNORE_CONFIGURED_ENDPOINT_URLS="false",
        AWS_ACCESS_KEY_ID=AIS_ANON_USER,
        AWS_SECRET_ACCESS_KEY=AIS_ANON_USER,
        AWS_REGION=cfg.b2_region,
        AWS_REQUEST_CHECKSUM_CALCULATION=_CHECKSUM_MODE,
        AWS_RESPONSE_CHECKSUM_VALIDATION=_CHECKSUM_MODE,
    )
    return env


def _ais_client(cfg: Config) -> Client:
    from aistore.sdk import Client  # lazy: keeps `--dry-run` and unit tests import-light

    return Client(cfg.ais_endpoint)


def ais_bucket(cfg: Config) -> Bucket:
    """AIS native-API handle for the B2 bucket (attached as provider `aws`, i.e. s3://)."""
    return _ais_client(cfg).bucket(cfg.b2_bucket, provider="aws")


def prefetch(cfg: Config, prefix: str, timeout_s: float = 3600) -> str:
    """Warm the AIS cache with every object under `prefix`; block until done. Returns job ID.

    AIS treats a template without a brace range as a plain prefix.
    """
    from aistore.sdk.errors import Timeout as AISTimeout

    client = _ais_client(cfg)
    job_id = client.bucket(cfg.b2_bucket, provider="aws").objects(obj_template=prefix).prefetch()
    try:
        result = client.job(job_id).wait(timeout=timeout_s, verbose=False)
    except AISTimeout as exc:
        raise RuntimeError(f"AIS prefetch {job_id} timed out after {timeout_s:g}s") from exc
    if not result.success:
        raise RuntimeError(f"AIS prefetch {job_id} failed: {result.error or 'aborted'}")
    return job_id


def evict(cfg: Config, keys: list[str]) -> None:
    """Drop only the selected cached objects from AIS. B2 is untouched."""
    if not keys:
        return
    client = _ais_client(cfg)
    # List only the keys' shared prefix, not the whole bucket.
    cached = cached_keys(cfg, os.path.commonprefix(keys)) & set(keys)
    if cached:
        job_id = (
            client.bucket(cfg.b2_bucket, provider="aws").objects(obj_names=sorted(cached)).evict()
        )
        result = client.job(job_id).wait(timeout=3600, verbose=False)
        if not result.success:
            raise RuntimeError(f"AIS eviction failed: {result.error or 'aborted'}")


def cached_keys(cfg: Config, prefix: str = "") -> set[str]:
    """Names of objects under `prefix` whose data is currently cached in AIS."""
    from aistore.sdk.list_object_flag import ListObjectFlag

    entries = ais_bucket(cfg).list_all_objects(prefix=prefix, flags=[ListObjectFlag.CACHED])
    return {e.name for e in entries}


# AIS target counters for GETs sent to the `aws` (B2) backend: cold GETs and prefetch.
# S3 CopyObject does not record backend GET stats.
BACKEND_GET_COUNT = "aws.get.n"
BACKEND_GET_SIZE = "aws.get.size"


def backend_get_stats(perf: Mapping[str, Any]) -> tuple[int, int]:
    """Total (GET count, bytes) AIS fetched from B2, summed over a get_performance() payload.

    Counters are cumulative per target and AIS omits zero values, so a missing counter is 0.
    No targets, or a target without a `tracker`, is payload drift, not "no reads", and raises.
    """
    if not perf or any(not isinstance(node.get("tracker"), Mapping) for node in perf.values()):
        raise ValueError("unexpected AIS performance payload: no targets or no 'tracker' stats")
    trackers = [node["tracker"] for node in perf.values()]
    return (
        sum(t.get(BACKEND_GET_COUNT, 0) for t in trackers),
        sum(t.get(BACKEND_GET_SIZE, 0) for t in trackers),
    )


def ais_backend_gets(cfg: Config) -> tuple[int, int]:
    """Current cluster-wide (GET count, bytes) read from B2 by AIS; diff two calls for a delta."""
    return backend_get_stats(_ais_client(cfg).cluster().get_performance())


def list_keys(s3: Any, bucket: str, prefix: str = "") -> Iterator[tuple[str, int]]:
    """Yield (key, size) for every object under `prefix`."""
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            yield obj["Key"], obj["Size"]


def upload_dir(
    s3: Any,
    local_dir: Path,
    bucket: str,
    prefix: str,
    exclude: tuple[str, ...] = (),
    *,
    publish: bool = False,
    provenance: dict[str, Any] | None = None,
) -> tuple[int, int]:
    """Upload by content hash, optionally publishing `_SUCCESS.json` last."""
    return upload_tree(
        s3,
        local_dir,
        bucket,
        prefix,
        exclude=exclude,
        transfer_config=TRANSFER_CONFIG,
        publish=publish,
        provenance=provenance,
    )
