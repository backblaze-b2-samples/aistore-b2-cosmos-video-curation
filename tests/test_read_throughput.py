# SPDX-FileCopyrightText: Copyright (c) 2026 Backblaze, Inc.
# SPDX-License-Identifier: Apache-2.0
import json
import math
from types import SimpleNamespace

import pytest
import read_throughput as rt


def test_percentile_nearest_rank():
    values = [float(v) for v in range(1, 101)]
    assert rt.percentile(values, 50) == 50
    assert rt.percentile(values, 99) == 99
    assert rt.percentile([3.0], 99) == 3.0
    assert math.isnan(rt.percentile([], 50))


def test_summarize_units():
    s = rt.summarize([0.010, 0.020, 0.030], total_bytes=6_000_000, wall_s=2.0)
    assert s == {
        "objects": 3,
        "bytes": 6_000_000,
        "wall_s": 2.0,
        "throughput_MBps": 3.0,
        "p50_ms": 20.0,
        "p99_ms": 30.0,
    }


def test_parse_args_rejects_unknown_mode():
    assert rt.parse_args(["--modes", "ais-cold, ais-warm"]).modes == ["ais-cold", "ais-warm"]
    with pytest.raises(SystemExit):
        rt.parse_args(["--modes", "b2,s3-express"])


def test_parse_args_workers_and_out_dir(tmp_path):
    args = rt.parse_args(["--workers", "4", "--out-dir", str(tmp_path)])
    assert (args.workers, args.out_dir) == (4, tmp_path)
    assert rt.parse_args([]).workers == 16
    with pytest.raises(SystemExit):
        rt.parse_args(["--workers", "0"])
    with pytest.raises(SystemExit):
        rt.parse_args(["--concurrency", "4"])


def test_markdown_has_no_hardcoded_prices():
    result = {
        "objects": 1,
        "bytes": 2_000_000,
        "throughput_MBps": 1.0,
        "p50_ms": 1.0,
        "p99_ms": 1.0,
        "b2_gets": 0,
        "b2_get_bytes": 0,
    }
    md = rt.to_markdown(
        {
            "timestamp": "t",
            "prefix": "raw/",
            "workers": 4,
            "page_cache_dropped": False,
            "environment": {"platform": "p", "cpus": 1, "link_gbps": None, "b2_rtt_ms": None},
            "results": {"ais-warm": result},
        }
    )
    assert "workers: 4" in md
    assert "| ais-warm | 1 | 2.0 |" in md
    assert "| 0 | 0.0 |" in md and "estimated" not in md
    assert rt.PRICING_URL in md and "$" not in md


def test_b2_reads_are_measured_per_mode_and_eviction_is_scoped(b2_env, monkeypatch, tmp_path):
    for name, value in b2_env.items():
        monkeypatch.setenv(name, value)
    sizes = {"raw/small.mp4": 1_000_000, "raw/large.mp4": 999_000_000}
    monkeypatch.setattr(rt, "ais_s3_client", lambda cfg: object())
    monkeypatch.setattr(rt, "b2_client", lambda cfg: object())
    monkeypatch.setattr(rt, "list_keys", lambda *a: sizes.items())
    monkeypatch.setattr(rt, "cached_keys", lambda *a: {"raw/small.mp4"})
    evicted = []
    monkeypatch.setattr(rt, "evict", lambda cfg, keys: evicted.extend(keys))
    monkeypatch.setattr(rt, "read_all", lambda *a: rt.summarize([1, 1], 1_000_000_000, 2))
    monkeypatch.setattr(rt, "tcp_rtt_ms", lambda *a: 1)
    # Cumulative AIS counters, sampled before/after each AIS mode's timed pass.
    counters = iter([(10, 100), (12, 1_000_000_100), (12, 1_000_000_100), (12, 1_000_000_100)])
    monkeypatch.setattr(rt, "ais_backend_gets", lambda cfg: next(counters))
    assert rt.main(["--out-dir", str(tmp_path)]) == 0
    report = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert report["workers"] == 16 and "concurrency" not in report
    results = report["results"]
    assert (results["b2-direct"]["b2_gets"], results["b2-direct"]["b2_get_bytes"]) == (
        2,
        1_000_000_000,
    )
    assert (results["ais-cold"]["b2_gets"], results["ais-cold"]["b2_get_bytes"]) == (
        2,
        1_000_000_000,
    )
    assert (results["ais-warm"]["b2_gets"], results["ais-warm"]["b2_get_bytes"]) == (0, 0)
    assert set(evicted) == set(sizes)


def test_warm_fill_happens_before_page_cache_drop(b2_env, monkeypatch, tmp_path):
    for name, value in b2_env.items():
        monkeypatch.setenv(name, value)
    args = SimpleNamespace(
        modes=["ais-warm"],
        prefix="raw/",
        limit=1,
        workers=1,
        drop_caches=True,
        link_gbps=None,
        out_dir=tmp_path,
    )
    monkeypatch.setattr(rt, "parse_args", lambda argv: args)
    monkeypatch.setattr(rt, "ais_s3_client", lambda cfg: object())
    monkeypatch.setattr(rt, "list_keys", lambda *a: [("raw/v.mp4", 1)])
    monkeypatch.setattr(rt, "cached_keys", lambda *a: {"raw/v.mp4"})
    monkeypatch.setattr(rt, "tcp_rtt_ms", lambda *a: 1)
    monkeypatch.setattr(rt, "ais_backend_gets", lambda cfg: (0, 0))
    events = []

    def read(*args):
        events.append("read")
        return rt.summarize([1], 1, 1)

    monkeypatch.setattr(rt, "read_all", read)
    monkeypatch.setattr(rt, "drop_page_cache", lambda: events.append("drop"))
    assert rt.main([]) == 0
    assert events == ["read", "drop", "read"]


def test_limit_stops_listing_early(b2_env, monkeypatch, tmp_path):
    for name, value in b2_env.items():
        monkeypatch.setenv(name, value)
    consumed = []

    def listing(*args):
        for i in range(10_000):
            consumed.append(i)
            yield f"raw/{i}.mp4", 1

    monkeypatch.setattr(rt, "ais_s3_client", lambda cfg: object())
    monkeypatch.setattr(rt, "list_keys", listing)
    monkeypatch.setattr(rt, "cached_keys", lambda *a: set())
    monkeypatch.setattr(rt, "read_all", lambda *a: rt.summarize([1], 1, 1))
    monkeypatch.setattr(rt, "tcp_rtt_ms", lambda *a: 1)
    monkeypatch.setattr(rt, "ais_backend_gets", lambda cfg: (0, 0))
    argv = ["--modes", "ais-warm", "--limit", "1", "--out-dir", str(tmp_path)]
    assert rt.main(argv) == 0
    assert len(consumed) == 1
