"""Regression: a successful suite must not leave its own background processes alive.

Before the fix, once the group leader exited, ``ps -o lstart=`` returned nothing, so
``terminate_process_group`` saw a fingerprint mismatch and returned without signalling,
and the setsid-survivor sweep ran only on timeout/cancel/non-zero exit. A suite such as
``sh -c 'sleep 30 & exit 0'`` left ``sleep`` running after a PASS.

Ownership rule kept: only the suite's own process group (pgid == the pid TestBench spawned
and has not yet reaped) and processes holding the run's private sandbox directory are
signalled. Unrelated processes, including other TestBench runs, are never touched.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import time
import unittest

from hermes_testbench.isolation import default_scratch_root, run_isolated_command


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # A reaped-by-init zombie still answers kill(0); treat state Z as dead.
    out = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True)
    return bool(out.stdout.strip()) and not out.stdout.strip().startswith("Z")


def _wait_dead(pid: int, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _alive(pid):
            return True
        time.sleep(0.02)
    return not _alive(pid)


class LeaderExitCleanupTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(dir=str(default_scratch_root()), prefix="tb-orph-")
        self.root = Path(self.tmp.name)
        self.pid_file = self.root / "bg.pid"
        self.bystander: subprocess.Popen | None = None

    def tearDown(self) -> None:
        if self.bystander is not None and self.bystander.poll() is None:
            self.bystander.kill()
            self.bystander.wait()
        if self.pid_file.is_file():
            try:
                os.kill(int(self.pid_file.read_text().strip()), 9)
            except (OSError, ValueError):
                pass
        self.tmp.cleanup()

    def _run(self, script: str, **kw):
        return run_isolated_command(
            ["/bin/sh", "-c", script],
            cwd=self.root,
            evidence_dir=self.root / "ev",
            step_name="leader_exit",
            run_id="tb_leader_exit",
            use_seatbelt=False,
            scratch_root=self.root,
            **kw,
        )

    def test_successful_suite_background_child_in_same_group_is_reaped(self) -> None:
        res = self._run(f"sleep 30 & echo $! > {self.pid_file}; exit 0")
        self.assertEqual(res.exit_code, 0)
        bg = int(self.pid_file.read_text().strip())
        self.assertTrue(_wait_dead(bg), f"background child {bg} survived a successful suite")

    def test_successful_suite_child_ignoring_sigterm_is_killed(self) -> None:
        res = self._run(f"(trap '' TERM; sleep 30) & echo $! > {self.pid_file}; exit 0")
        self.assertEqual(res.exit_code, 0)
        bg = int(self.pid_file.read_text().strip())
        self.assertTrue(_wait_dead(bg, 3.0), f"TERM-ignoring child {bg} survived a successful suite")

    def test_failing_suite_background_child_is_reaped_and_counted(self) -> None:
        res = self._run(f"sleep 30 & echo $! > {self.pid_file}; exit 3")
        self.assertEqual(res.exit_code, 3)
        self.assertTrue(_wait_dead(int(self.pid_file.read_text().strip())))
        self.assertGreaterEqual(res.isolation_meta.get("reaped_survivors", 0), 1)

    def test_unrelated_process_with_same_command_is_never_signalled(self) -> None:
        # Outside the suite's group and not holding its sandbox: must survive.
        self.bystander = subprocess.Popen(["sleep", "30"], start_new_session=True)
        res = self._run(f"sleep 30 & echo $! > {self.pid_file}; exit 0")
        self.assertEqual(res.exit_code, 0)
        self.assertTrue(_wait_dead(int(self.pid_file.read_text().strip())))
        time.sleep(0.2)
        self.assertIsNone(self.bystander.poll(), "unrelated process was signalled")

    def test_plain_successful_suite_unchanged(self) -> None:
        res = self._run("echo ok")
        self.assertEqual(res.exit_code, 0)
        self.assertIn("ok", res.stdout_text)
        self.assertFalse(res.timed_out)
        self.assertEqual(res.isolation_meta.get("leaked_children"), 0)


if __name__ == "__main__":
    unittest.main()
