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
    """Liveness probe that treats an unreaped zombie as dead.

    ``kill(pid, 0)`` cannot distinguish a live process from a zombie: the pid is still
    allocated, so the call succeeds. On macOS a child of a short-lived intermediate can
    linger as ``Z`` until init reaps it, and a "did it die?" assertion would then wait
    out its whole budget on a process that has in fact exited. Hence the ``ps -o stat=``
    follow-up and the ``startswith("Z")`` filter; an empty ``ps`` output means the pid
    vanished between the two calls and is also dead.

    ``PermissionError`` is reported as alive: the pid exists but belongs to another
    user, and this module must never mistake "cannot inspect" for "gone".
    """
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
    """Poll until ``pid`` is dead, returning its state at the deadline.

    The final check is repeated after the loop so a process that died on the last
    iteration is not reported as a survivor. The default 2 s budget covers
    SIGTERM→SIGKILL escalation (grace 0.5 s) plus scheduling latency; the
    TERM-ignoring test asks for 3 s because it must outlast the same escalation.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _alive(pid):
            return True
        time.sleep(0.02)
    return not _alive(pid)


class LeaderExitCleanupTests(unittest.TestCase):
    """Detached-worker reaping after the suite's leader exits.

    Every suite is ``/bin/sh -c '<script>'`` under ``run_isolated_command`` with
    ``use_seatbelt=False`` and a private scratch root, so the subject is the process
    lifecycle, not confinement. ``$!`` is written to a pid file because the shell's
    background pid is the only handle the test has on a child it never spawned; the
    test then observes that pid, not the harness's counters, so a harness that
    undercounts cannot make a surviving process disappear.

    ``tearDown`` is a safety net: if an assertion fails before the background child is
    reaped, both the child (from the pid file) and the bystander are killed. Without it a
    red test would leak ``sleep 30`` processes and pollute the next run.
    """

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
        """Run one shell script through isolation; extra kwargs go to the harness.

        Kept as one helper so the tests differ only in the script and the timeout they
        pass, which is what makes them readable as a table of lifecycle cases.
        """
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
        """A background child in the suite's own group dies even though the leader exited 0.

        The historical bug: a ``ps -o lstart=`` fingerprint of a reaped leader is
        ``None``, so ``terminate_process_group`` refused to signal and the clean-exit
        path never ran the survivor sweep. Now the post-leader-exit member sweep
        (``_reap_group_after_leader_exit``) is member-based and fires for *any* exit
        status, so this needs no timeout or non-zero exit to work.

        ``exit 0`` is asserted first: if the shell had failed, a passing reap would
        say nothing about the clean-exit path the test exists to cover.
        """
        res = self._run(f"sleep 30 & echo $! > {self.pid_file}; exit 0")
        self.assertEqual(res.exit_code, 0)
        bg = int(self.pid_file.read_text().strip())
        self.assertTrue(_wait_dead(bg), f"background child {bg} survived a successful suite")

    def test_successful_suite_child_ignoring_sigterm_is_killed(self) -> None:
        """A child that traps TERM still dies on a successful suite (SIGKILL escalation).

        The group reaper sends SIGTERM, waits 0.5 s, then SIGKILL; a trap-only child
        therefore proves the escalation is real rather than nominal. The 3 s budget is
        longer than the other tests' 2 s because the survivor must survive the grace
        period before it can be killed, and the assertion is about the final state.
        """
        res = self._run(f"(trap '' TERM; sleep 30) & echo $! > {self.pid_file}; exit 0")
        self.assertEqual(res.exit_code, 0)
        bg = int(self.pid_file.read_text().strip())
        self.assertTrue(_wait_dead(bg, 3.0), f"TERM-ignoring child {bg} survived a successful suite")

    def test_failing_suite_background_child_is_reaped_and_counted(self) -> None:
        """A non-zero exit reaps the group and reports it in ``reaped_survivors``.

        Two observations, deliberately separate: the child's death (the lifecycle
        guarantee) and the counter (the provenance claim). The counter is asserted
        ``>= 1`` rather than ``== 1`` because the sweep counts whatever it signalled;
        pinning the exact number would couple this test to the detector's dedup order.
        ``leaked_children`` is not asserted here — the nonzero exit path reports
        survivors as reaped, and only an unkillable process would increment leaks.
        """
        res = self._run(f"sleep 30 & echo $! > {self.pid_file}; exit 3")
        self.assertEqual(res.exit_code, 3)
        self.assertTrue(_wait_dead(int(self.pid_file.read_text().strip())))
        self.assertGreaterEqual(res.isolation_meta.get("reaped_survivors", 0), 1)

    def test_unrelated_process_with_same_command_is_never_signalled(self) -> None:
        """Negative control: an unrelated ``sleep 30`` with identical argv must survive.

        The survivor sweep matches the per-run sandbox path twice — as an ``lsof``
        file holder and as a substring of the ``ps`` command line, which also
        catches an inherited cwd — plus process-group membership. It never matches
        a generic command name such as ``sleep``, which is why the identical-argv
        bystander survives; the test would fail if someone "simplified" the sweep
        to ``pkill sleep``. The bystander is started with
        ``start_new_session=True`` so it is outside the suite's group *and* outside
        the test runner's group, isolating the ownership rule from the R1 bug. The
        0.2 s pause gives a wrongly-sent signal time to take effect before
        ``poll()`` is read.
        """
        # Outside the suite's group and not holding its sandbox: must survive.
        self.bystander = subprocess.Popen(["sleep", "30"], start_new_session=True)
        res = self._run(f"sleep 30 & echo $! > {self.pid_file}; exit 0")
        self.assertEqual(res.exit_code, 0)
        self.assertTrue(_wait_dead(int(self.pid_file.read_text().strip())))
        time.sleep(0.2)
        self.assertIsNone(self.bystander.poll(), "unrelated process was signalled")

    def test_plain_successful_suite_unchanged(self) -> None:
        """Regression guard: an ordinary successful suite keeps zero leaked children.

        ``leaked_children == 0`` here means "the harness found nothing to kill", not
        "the suite cleaned up after itself" — but on a suite with no background child
        the two coincide, and the assertion pins the cheap path (no forced sweep on a
        clean exit) from regressing into reporting phantom leaks.
        """
        res = self._run("echo ok")
        self.assertEqual(res.exit_code, 0)
        self.assertIn("ok", res.stdout_text)
        self.assertFalse(res.timed_out)
        self.assertEqual(res.isolation_meta.get("leaked_children"), 0)


if __name__ == "__main__":
    unittest.main()
