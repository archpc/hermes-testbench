"""Regression tests for review findings A4, A5, C4 (opus-resumed-review, 1a10976).

Each finding is pinned by the *observable* it broke, not by an internal call:

A4  The discovery probe for ``scripts/check-wiring.py`` must accept an unwired
    home only when the script itself says so (nonzero exit **and** the
    ``advisor profile missing`` marker). A bare nonzero exit is not evidence:
    a script that crashes for an unrelated reason also exits nonzero, and the
    old probe scored that as "correctly rejected the unwired home".
A5  TUI overview rows must keep active runs in pre-1a10976 FIFO order — RUNNING
    before QUEUED, oldest first inside a class — even though ``state.runs`` is
    newest-first.
C4  Only ``hermes_agent_root is None`` records an empty command. A root with no
    ``venv/bin/python`` still records the full argv
    (``<root>/venv/bin/python -m pytest -q tests/tools/test_bot_mode_dm.py``)
    alongside the SKIP case, so the recorded command shows what would have run.

All probes run the real ``_CHECK_WIRING_PROBE_SCRIPT`` source in a subprocess
against synthetic homes under ``default_scratch_root()``; nothing touches the
operator's real ``~/.hermes``.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from hermes_testbench.discovery import _CHECK_WIRING_PROBE_SCRIPT
from hermes_testbench.isolation import default_scratch_root
from hermes_testbench.messaging_suite import PYTEST_SUITE, _pytest_suite
from hermes_testbench.schema import RunRecord, RunStatus
from hermes_testbench.tui_render import TuiState, _overview_runs

# A conforming check-wiring.py: rejects an unwired home with the marker the probe
# looks for. Kept deliberately minimal so the probe, not the fixture, is under test.
_WIRING_OK = (
    "import argparse, pathlib, sys\n"
    "ap = argparse.ArgumentParser()\n"
    "ap.add_argument('--hermes-home')\n"
    "home = pathlib.Path(ap.parse_args().hermes_home)\n"
    "if not (home / 'profiles' / 'opus' / 'config.yaml').is_file():\n"
    "    print('FAILED:\\n  1. advisor profile missing: x')\n"
    "    sys.exit(1)\n"
    "print('wiring OK')\n"
)

# Rejects the unwired home for an unrelated reason (crash), not the wiring check.
# Exits 1 like the fixture above, so return code alone cannot distinguish them.
_WIRING_WRONG_REASON = (
    "import argparse, pathlib, sys\n"
    "ap = argparse.ArgumentParser()\n"
    "ap.add_argument('--hermes-home')\n"
    "home = pathlib.Path(ap.parse_args().hermes_home)\n"
    "if not (home / 'profiles' / 'opus' / 'config.yaml').is_file():\n"
    "    raise RuntimeError('unrelated crash')\n"
    "print('wiring OK')\n"
)


class CheckWiringProbeTests(unittest.TestCase):
    """A4: discovery probe must require the same marker as messaging_suite w2."""

    def setUp(self) -> None:
        # Probe homes are created inside the probe itself; this directory only
        # holds the candidate script. Scratch root (not system TMPDIR) so a
        # Seatbelt-confined run can still read it.
        self.tmp = tempfile.TemporaryDirectory(dir=str(default_scratch_root()), prefix="tb-a4-")
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _probe(self, body: str) -> subprocess.CompletedProcess:
        """Run the shipped probe source against ``body`` as a candidate script.

        ``-c`` executes the probe with the script path as ``argv[1]``, matching
        how ``discovery._script_suites`` splices it into a suite command. A 60 s
        timeout bounds the two nested ``subprocess.run`` calls the probe makes.
        """
        script = self.root / "check-wiring.py"
        script.write_text(body, encoding="utf-8")
        return subprocess.run(
            [sys.executable, "-c", _CHECK_WIRING_PROBE_SCRIPT, str(script)],
            capture_output=True,
            text=True,
            timeout=60,
        )

    def test_real_wiring_rejection_passes(self) -> None:
        # Positive control: the probe must report both synthetic checks as PASS
        # and exit 0, otherwise A4 would be "fixed" by rejecting everything.
        proc = self._probe(_WIRING_OK)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("TB-PASS check-wiring:unwired_home_rejected", proc.stdout)

    def test_nonzero_exit_without_marker_is_not_accepted(self) -> None:
        # Negative control, and the actual regression: same exit status as the
        # positive case, no ``advisor profile missing`` text. The probe must call
        # this a FAIL — acceptance requires rc != 0 *and* the marker *and* (for
        # the wired home) rc == 0 with ``wiring OK``.
        proc = self._probe(_WIRING_WRONG_REASON)
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        self.assertIn("TB-FAIL check-wiring:unwired_home_rejected", proc.stdout)


def _run(run_id: str, status: str, created_at: str) -> RunRecord:
    # Minimal record: only the fields ordering reads (status, created_at, project).
    return RunRecord(
        run_id=run_id,
        project="p-" + run_id[-2:],
        project_root="/tmp/x",
        revision="r",
        profile="unit",
        status=status,
        created_at=created_at,
    )


class TuiActiveOrderingTests(unittest.TestCase):
    """A5: active runs keep their pre-1a10976 oldest-first (FIFO) order within each status."""

    def test_active_runs_oldest_first_running_before_queued(self) -> None:
        # state.runs is newest-first (store.list_runs order)
        # Status class outranks age: a queued job never sorts above a running one,
        # even when it is newer. Within a class, ascending timestamp = FIFO.
        runs = (
            _run("tb_q2", RunStatus.QUEUED, "2026-11-15T10:00:04.000000Z"),
            _run("tb_r2", RunStatus.RUNNING, "2026-11-15T10:00:03.000000Z"),
            _run("tb_q1", RunStatus.QUEUED, "2026-11-15T10:00:02.000000Z"),
            _run("tb_r1", RunStatus.RUNNING, "2026-11-15T10:00:01.000000Z"),
        )
        ids = [r.run_id for r in _overview_runs(TuiState(runs=runs))]
        self.assertEqual(ids, ["tb_r1", "tb_r2", "tb_q1", "tb_q2"])

    def test_active_tie_breaks_older_index_first(self) -> None:
        # Equal timestamps are realistic (whole-second stamps). ``-idx`` in the
        # sort key is what makes the *larger* index (older, since the list is
        # newest-first) come first — i.e. the tiebreak is the row position, not
        # the run id or the parsed timestamp.
        same = "2026-11-15T10:00:00Z"
        runs = (_run("tb_new", RunStatus.RUNNING, same), _run("tb_old", RunStatus.RUNNING, same))
        ids = [r.run_id for r in _overview_runs(TuiState(runs=runs))]
        self.assertEqual(ids, ["tb_old", "tb_new"])


class SkippedPytestCommandTests(unittest.TestCase):
    """C4: a pytest suite with no source tree records an empty command."""

    def test_no_hermes_agent_root_records_empty_command(self) -> None:
        # With no source tree there is no interpreter to name, so ``command``
        # stays [] rather than an argv that would fail if replayed. The skip
        # itself is still recorded as one skipped case (not an absence), so a
        # green run cannot quietly omit the pytest dimension.
        res = _pytest_suite(None, evidence_dir=None)
        self.assertEqual(res.name, PYTEST_SUITE)
        self.assertEqual(res.status, RunStatus.SKIP)
        self.assertEqual(res.command, [])


if __name__ == "__main__":
    unittest.main()
