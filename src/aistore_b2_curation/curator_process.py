# SPDX-FileCopyrightText: Copyright (c) 2026 Backblaze, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Own the local Curator process group until its workers have stopped."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from contextlib import suppress

_SHUTDOWN_TIMEOUT_S = 30.0
_EXIT_GRACE_S = 5.0


def _signal_group(proc: subprocess.Popen, signum: int) -> bool:
    proc.poll()
    try:
        os.killpg(proc.pid, signum)
    except PermissionError:
        # macOS can report EPERM for an exited, unreaped group leader.
        proc.poll()
        try:
            os.killpg(proc.pid, signum)
        except ProcessLookupError:
            return False
    except ProcessLookupError:
        return False
    return True


def _wait_group(proc: subprocess.Popen, timeout: float) -> bool:
    """Poll until the process group is empty; return whether it emptied within `timeout`."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            if not _signal_group(proc, 0):
                return True
        except PermissionError:
            # Reparented zombie workers can briefly keep a macOS group registered.
            pass
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)


def _stop_group(proc: subprocess.Popen, *, signal_sent: bool) -> bool:
    """Reap the leader and stop remaining workers; return whether the group exited cleanly."""
    # Short-lived helpers (wandb, Ray, dataloader workers) may trail a clean launcher exit;
    # a group that empties on its own within the grace period exited cleanly.
    if _wait_group(proc, 0 if signal_sent else _EXIT_GRACE_S):
        return True
    if not signal_sent:
        with suppress(PermissionError):
            _signal_group(proc, signal.SIGTERM)
    # Wait on the whole group, not the leader: workers may outlive a fast-exiting launcher.
    if not _wait_group(proc, _SHUTDOWN_TIMEOUT_S):
        _signal_group(proc, signal.SIGKILL)
    proc.wait(timeout=5)
    return False


def run_curator(cmd: list[str], env: dict[str, str]) -> int:
    """Forward interruptions to local Curator workers and reject unfinished group exits."""
    if os.name != "posix":
        raise RuntimeError("Curator processing requires a POSIX host")
    proc = None
    interrupted = None
    clean_group = False
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}

    def handle_signal(signum, _frame):
        nonlocal interrupted
        first = interrupted is None
        if first:
            interrupted = signum
        if proc is not None:
            try:
                _signal_group(proc, signum if first else signal.SIGKILL)
            except OSError as exc:
                print(f"[curator] signal forwarding failed: {exc}", file=sys.stderr)

    try:
        for sig in previous:
            signal.signal(sig, handle_signal)
        proc = subprocess.Popen(cmd, env=env, start_new_session=True)
        if interrupted is not None:
            _signal_group(proc, interrupted)
        while interrupted is None and proc.poll() is None:
            with suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=0.1)
    finally:
        try:
            if proc is not None:
                clean_group = _stop_group(proc, signal_sent=interrupted is not None)
        except (OSError, subprocess.SubprocessError) as exc:
            print(f"[curator] shutdown failed: {exc}", file=sys.stderr)
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
    if interrupted is not None:
        return 128 + interrupted
    if proc.returncode is None:
        return 1
    if proc.returncode == 0 and not clean_group:
        print(
            "[curator] launcher exited with active workers; output is incomplete", file=sys.stderr
        )
        return 1
    return proc.returncode
