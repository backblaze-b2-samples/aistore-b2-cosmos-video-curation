# SPDX-FileCopyrightText: Copyright (c) 2026 Backblaze, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Measure object read throughput and latency: B2 direct vs AIS cold vs AIS warm.

Each mode GETs the same objects with the same concurrency and reads every byte:
  b2-direct  boto3 -> B2 S3 endpoint (baseline, no cache)
  ais-cold   AIS cache evicted first, so every GET is a cold GET from B2 through AIS
  ais-warm   untimed fill of the same keys first, then served from AIS target disks

B2 GETs and bytes are measured: client GETs in b2-direct mode, AIS backend counter deltas
otherwise.

Writes <--out-dir>/read_throughput-<UTC timestamp>.{json,md} (default results/). No numbers
are published with this repo; run it on your own hardware and network.

    uv run python benchmarks/read_throughput.py --prefix raw/ --limit 200 --workers 16
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import socket
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from aistore_b2_curation.artifacts import MANIFEST
from aistore_b2_curation.config import Config
from aistore_b2_curation.storage import (
    ais_backend_gets,
    ais_s3_client,
    b2_client,
    cached_keys,
    evict,
    list_keys,
)

MODES = ("b2-direct", "ais-cold", "ais-warm")
PRICING_URL = "https://www.backblaze.com/cloud-storage/pricing"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--prefix", default="raw/", help="object prefix to read (default: %(default)s)")
    p.add_argument(
        "--limit", type=int, default=200, help="max objects per mode (default: %(default)s)"
    )
    p.add_argument(
        "--workers", type=int, default=16, help="concurrent GETs per mode (default: %(default)s)"
    )
    p.add_argument(
        "--modes",
        default=",".join(MODES),
        help="comma-separated subset of %(default)s (default: all)",
    )
    p.add_argument("--link-gbps", type=float, help="nominal client link bandwidth, recorded as-is")
    p.add_argument(
        "--drop-caches",
        action="store_true",
        help="Linux only: drop the OS page cache (sudo) before each mode",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results"),
        help="directory for the JSON and Markdown reports (default: %(default)s)",
    )
    args = p.parse_args(argv)
    if args.drop_caches and sys.platform != "linux":
        print("warning: --drop-caches is Linux-only; ignoring", file=sys.stderr)
        args.drop_caches = False
    args.modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    if unknown := set(args.modes) - set(MODES):
        p.error(f"unknown modes: {sorted(unknown)}")
    if args.limit < 1 or args.workers < 1:
        p.error("--limit and --workers must be positive")
    return args


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile (no interpolation); stable for small samples."""
    if not values:
        return math.nan
    ordered = sorted(values)
    return ordered[max(0, math.ceil(pct / 100 * len(ordered)) - 1)]


def summarize(latencies_s: list[float], total_bytes: int, wall_s: float) -> dict[str, Any]:
    return {
        "objects": len(latencies_s),
        "bytes": total_bytes,
        "wall_s": round(wall_s, 3),
        "throughput_MBps": round(total_bytes / wall_s / 1e6, 2) if wall_s > 0 else math.nan,
        "p50_ms": round(percentile(latencies_s, 50) * 1e3, 2),
        "p99_ms": round(percentile(latencies_s, 99) * 1e3, 2),
    }


def read_all(s3: Any, bucket: str, keys: list[str], workers: int) -> dict[str, Any]:
    def get(key: str) -> tuple[float, int]:
        t0 = time.perf_counter()
        body = s3.get_object(Bucket=bucket, Key=key)["Body"]
        try:
            n = sum(len(block) for block in iter(lambda: body.read(1024 * 1024), b""))
        finally:
            body.close()
        return time.perf_counter() - t0, n

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(get, keys))
    wall = time.perf_counter() - t0
    return summarize([r[0] for r in results], sum(r[1] for r in results), wall)


def tcp_rtt_ms(url: str, samples: int = 5) -> float | None:
    """Median TCP connect time to the endpoint, a rough RTT proxy that needs no ICMP."""
    parsed = urlparse(url)
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    times = []
    for _ in range(samples):
        t0 = time.perf_counter()
        try:
            with socket.create_connection((parsed.hostname, port), timeout=5):
                times.append((time.perf_counter() - t0) * 1e3)
        except OSError:
            return None
    return round(statistics.median(times), 2)


def drop_page_cache() -> None:
    """Flush the Linux page cache so AIS-warm reads hit disk, not RAM (needs sudo)."""
    subprocess.run(["sudo", "sh", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"], check=True)


def to_markdown(report: dict[str, Any]) -> str:
    env = report["environment"]
    lines = [
        f"# read_throughput {report['timestamp']}",
        "",
        f"- prefix: `{report['prefix']}`, workers: {report['workers']}",
        f"- platform: {env['platform']}, cpus: {env['cpus']}, "
        f"link: {env['link_gbps'] or 'n/a'} Gbps",
        f"- TCP RTT to B2 endpoint: {env['b2_rtt_ms']} ms; page cache dropped: "
        f"{report['page_cache_dropped']}",
        "",
        "| mode | objects | MB | MB/s | p50 ms | p99 ms | B2 GETs | B2 GET MB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, r in report["results"].items():
        lines.append(
            f"| {mode} | {r['objects']} | {r['bytes'] / 1e6:.1f} | {r['throughput_MBps']} "
            f"| {r['p50_ms']} | {r['p99_ms']} | {r['b2_gets']} | {r['b2_get_bytes'] / 1e6:.1f} |"
        )
    lines += [
        "",
        "AIS-mode B2 traffic is the delta of AIS `aws.get.n`/`aws.get.size` target counters "
        "and includes any concurrent cluster traffic; b2-direct counts client GETs.",
        "B2 cost per pass ~= `B2 GET GB x egress rate + GET calls x class-B rate`; "
        f"see {PRICING_URL} for current rates and free-egress terms.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = Config.from_env(require_b2_credentials="b2-direct" in args.modes)
    ais = ais_s3_client(cfg)
    # Stop listing once --limit keys are found; prefixes can hold millions of objects.
    listing = list_keys(ais, cfg.b2_bucket, args.prefix)
    keys = list(islice((k for k, _ in listing if not k.endswith("/" + MANIFEST)), args.limit))
    if not keys:
        print(
            f"no objects under s3://{cfg.b2_bucket}/{args.prefix}; run `make seed`", file=sys.stderr
        )
        return 1

    if not args.drop_caches:
        print("OS page cache not dropped; warm numbers may include RAM hits", file=sys.stderr)
    results: dict[str, dict[str, Any]] = {}
    for mode in args.modes:
        if mode == "ais-cold":
            evict(cfg, keys)
        if mode == "ais-warm":
            # Warm only the selected keys, even when this is the first requested mode.
            read_all(ais, cfg.b2_bucket, keys, args.workers)
        if args.drop_caches:
            drop_page_cache()
        cached = cached_keys(cfg, args.prefix) & set(keys) if mode != "b2-direct" else set()
        s3 = b2_client(cfg) if mode == "b2-direct" else ais
        print(f"{mode}: reading {len(keys)} objects ...", file=sys.stderr)
        before = ais_backend_gets(cfg) if mode != "b2-direct" else (0, 0)
        r = read_all(s3, cfg.b2_bucket, keys, args.workers)
        r["cached_before"] = len(cached) if mode != "b2-direct" else None
        if mode == "b2-direct":
            r["b2_gets"], r["b2_get_bytes"] = r["objects"], r["bytes"]  # every GET hits B2
        else:
            after = ais_backend_gets(cfg)
            r["b2_gets"], r["b2_get_bytes"] = after[0] - before[0], after[1] - before[1]
        results[mode] = r

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = {
        "timestamp": ts,
        "prefix": args.prefix,
        "workers": args.workers,
        "page_cache_dropped": args.drop_caches,
        "environment": {
            "platform": platform.platform(),
            "cpus": os.cpu_count(),
            "link_gbps": args.link_gbps,
            "b2_rtt_ms": tcp_rtt_ms(cfg.b2_endpoint),
        },
        "results": results,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"read_throughput-{ts}.json").write_text(json.dumps(report, indent=2))
    md = to_markdown(report)
    (args.out_dir / f"read_throughput-{ts}.md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
