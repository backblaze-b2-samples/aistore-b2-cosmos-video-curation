# SPDX-FileCopyrightText: Copyright (c) 2026 Backblaze, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Curator interruption must stop its workers and prevent output publication."""

import os
import signal
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import run_pipeline

from aistore_b2_curation import curator_process

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX Curator process groups")


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT])
def test_signal_forwarded_and_reported_when_curator_exits_zero(signum, tmp_path):
    stopped = tmp_path / "stopped"
    script = """
import os, pathlib, signal, sys, time
stopped, signum = pathlib.Path(sys.argv[1]), int(sys.argv[2])
def stop(received, frame):
    stopped.write_text(str(received))
    raise SystemExit(0)
signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
os.kill(os.getppid(), signum)
while True:
    time.sleep(0.02)
"""
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    rc = curator_process.run_curator(
        [sys.executable, "-c", script, str(stopped), str(int(signum))], os.environ.copy()
    )
    assert rc == 128 + signum
    assert stopped.read_text() == str(int(signum))
    assert {sig: signal.getsignal(sig) for sig in previous} == previous


def test_stubborn_curator_and_worker_are_killed_within_shutdown_bound(tmp_path, monkeypatch):
    heartbeat = tmp_path / "heartbeat"
    worker = """
import pathlib, signal, sys, time
signal.signal(signal.SIGINT, signal.SIG_IGN)
signal.signal(signal.SIGTERM, signal.SIG_IGN)
heartbeat = pathlib.Path(sys.argv[1])
for step in range(10000):
    heartbeat.write_text(str(step))
    time.sleep(0.02)
"""
    driver = """
import os, pathlib, signal, subprocess, sys, time
signal.signal(signal.SIGINT, signal.SIG_IGN)
signal.signal(signal.SIGTERM, signal.SIG_IGN)
child = subprocess.Popen([sys.executable, '-c', sys.argv[1], sys.argv[2]])
while not pathlib.Path(sys.argv[2]).exists():
    time.sleep(0.01)
os.kill(os.getppid(), signal.SIGTERM)
child.wait()
"""
    monkeypatch.setattr(curator_process, "_SHUTDOWN_TIMEOUT_S", 0.2)
    started = time.monotonic()
    rc = curator_process.run_curator(
        [sys.executable, "-c", driver, worker, str(heartbeat)], os.environ.copy()
    )
    assert rc == 128 + signal.SIGTERM
    assert time.monotonic() - started < 5
    final = heartbeat.read_text()
    time.sleep(0.1)
    assert heartbeat.read_text() == final


def test_clean_launcher_with_live_worker_is_not_success(tmp_path, monkeypatch):
    heartbeat = tmp_path / "heartbeat"
    worker = """
import pathlib, sys, time
heartbeat = pathlib.Path(sys.argv[1])
for step in range(10000):
    heartbeat.write_text(str(step))
    time.sleep(0.02)
"""
    driver = """
import pathlib, subprocess, sys, time
subprocess.Popen([sys.executable, '-c', sys.argv[1], sys.argv[2]])
while not pathlib.Path(sys.argv[2]).exists():
    time.sleep(0.01)
"""
    monkeypatch.setattr(curator_process, "_EXIT_GRACE_S", 0.1)
    monkeypatch.setattr(curator_process, "_SHUTDOWN_TIMEOUT_S", 0.2)
    rc = curator_process.run_curator(
        [sys.executable, "-c", driver, worker, str(heartbeat)], os.environ.copy()
    )
    assert rc == 1
    final = heartbeat.read_text()
    time.sleep(0.1)
    assert heartbeat.read_text() == final


def test_helper_exiting_shortly_after_clean_launcher_is_success(tmp_path, monkeypatch):
    # e.g. a Ray or wandb helper that tears down a moment after the launcher returns 0.
    driver = """
import subprocess, sys
subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(0.3)'])
"""
    monkeypatch.setattr(curator_process, "_EXIT_GRACE_S", 5.0)
    monkeypatch.setattr(curator_process, "_SHUTDOWN_TIMEOUT_S", 0.2)
    assert curator_process.run_curator([sys.executable, "-c", driver], os.environ.copy()) == 0


def test_worker_outliving_launcher_gets_full_grace_period(tmp_path, monkeypatch):
    ready, stopped = tmp_path / "ready", tmp_path / "stopped"
    worker = """
import pathlib, signal, sys, time
ready, stopped = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
def stop(signum, frame):
    time.sleep(0.3)
    stopped.write_text("clean")
    raise SystemExit(0)
signal.signal(signal.SIGTERM, stop)
ready.write_text("")
while True:
    time.sleep(0.02)
"""
    driver = """
import pathlib, subprocess, sys, time
subprocess.Popen([sys.executable, '-c', *sys.argv[1:]])
while not pathlib.Path(sys.argv[2]).exists():
    time.sleep(0.01)
"""
    monkeypatch.setattr(curator_process, "_EXIT_GRACE_S", 0.1)
    monkeypatch.setattr(curator_process, "_SHUTDOWN_TIMEOUT_S", 5.0)
    rc = curator_process.run_curator(
        [sys.executable, "-c", driver, worker, str(ready), str(stopped)], os.environ.copy()
    )
    assert rc == 1
    assert stopped.read_text() == "clean"


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT])
def test_interrupted_curator_output_is_not_uploaded(signum, tmp_path, b2_env, monkeypatch):
    for name, value in b2_env.items():
        monkeypatch.setenv(name, value)
    script = tmp_path / "curator.py"
    script.write_text("")
    monkeypatch.setattr(run_pipeline, "CURATOR_PYTHON", Path(sys.executable))
    monkeypatch.setattr(run_pipeline, "CURATOR_SCRIPT", script)
    monkeypatch.setattr(run_pipeline, "check_media_tools", lambda _encoder: None)
    monkeypatch.setattr(run_pipeline, "check_input_publication", lambda *a: None)
    cmd = [
        sys.executable,
        "-c",
        f"import os, signal, time; os.kill(os.getppid(), {int(signum)}); time.sleep(20)",
    ]
    monkeypatch.setattr(run_pipeline, "build_command", lambda *args: cmd)
    monkeypatch.setattr(
        run_pipeline, "upload_dir", lambda *a, **k: pytest.fail("interrupted upload")
    )
    monkeypatch.setattr(run_pipeline, "ais_s3_client", lambda *a: pytest.fail("interrupted client"))
    monkeypatch.setattr(run_pipeline, "ais_backend_gets", lambda _cfg: (0, 0))
    rc = run_pipeline.main(
        ["--encoder", "libvpx-vp9", "--video-limit", "1", "--work-dir", str(tmp_path / "work")]
    )
    assert rc == 128 + signum
    assert len(list((tmp_path / "work").iterdir())) == 1


@pytest.mark.parametrize("result", [0, 3, "launch-failure"])
def test_original_signal_handlers_restore_for_all_exit_paths(result, tmp_path):
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    if result == "launch-failure":
        with pytest.raises(FileNotFoundError):
            curator_process.run_curator([str(tmp_path / "missing")], os.environ.copy())
    else:
        assert (
            curator_process.run_curator(
                [sys.executable, "-c", f"raise SystemExit({result})"], os.environ.copy()
            )
            == result
        )
    assert {sig: signal.getsignal(sig) for sig in previous} == previous


def test_group_signal_reaps_mac_os_zombie_leader_before_retry(monkeypatch):
    calls = []
    proc = SimpleNamespace(pid=123, poll=lambda: calls.append("poll"))

    def killpg(*args):
        calls.append("signal")
        if calls.count("signal") == 1:
            raise PermissionError("zombie leader")
        raise ProcessLookupError("group is gone")

    monkeypatch.setattr(curator_process.os, "killpg", killpg)
    assert curator_process._signal_group(proc, signal.SIGTERM) is False
    assert calls == ["poll", "signal", "poll", "signal"]


def test_permanent_shutdown_failure_is_reported_as_failure(monkeypatch, capsys):
    def denied(*args, **kwargs):
        raise PermissionError("remaining workers cannot be signaled")

    monkeypatch.setattr(curator_process, "_stop_group", denied)
    assert curator_process.run_curator([sys.executable, "-c", "pass"], os.environ.copy()) == 1
    assert "shutdown failed: remaining workers cannot be signaled" in capsys.readouterr().err
