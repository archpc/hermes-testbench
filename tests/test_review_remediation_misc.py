"""Regression tests for review findings A4, A5, C4 (opus-resumed-review, 1a10976)."""

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
        self.tmp = tempfile.TemporaryDirectory(dir=str(default_scratch_root()), prefix="tb-a4-")
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _probe(self, body: str) -> subprocess.CompletedProcess:
        script = self.root / "check-wiring.py"
        script.write_text(body, encoding="utf-8")
        return subprocess.run(
            [sys.executable, "-c", _CHECK_WIRING_PROBE_SCRIPT, str(script)],
            capture_output=True,
            text=True,
            timeout=60,
        )

    def test_real_wiring_rejection_passes(self) -> None:
        proc = self._probe(_WIRING_OK)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("TB-PASS check-wiring:unwired_home_rejected", proc.stdout)

    def test_nonzero_exit_without_marker_is_not_accepted(self) -> None:
        proc = self._probe(_WIRING_WRONG_REASON)
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        self.assertIn("TB-FAIL check-wiring:unwired_home_rejected", proc.stdout)


def _run(run_id: str, status: str, created_at: str) -> RunRecord:
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
        runs = (
            _run("tb_q2", RunStatus.QUEUED, "2026-11-15T10:00:04.000000Z"),
            _run("tb_r2", RunStatus.RUNNING, "2026-11-15T10:00:03.000000Z"),
            _run("tb_q1", RunStatus.QUEUED, "2026-11-15T10:00:02.000000Z"),
            _run("tb_r1", RunStatus.RUNNING, "2026-11-15T10:00:01.000000Z"),
        )
        ids = [r.run_id for r in _overview_runs(TuiState(runs=runs))]
        self.assertEqual(ids, ["tb_r1", "tb_r2", "tb_q1", "tb_q2"])

    def test_active_tie_breaks_older_index_first(self) -> None:
        same = "2026-11-15T10:00:00Z"
        runs = (_run("tb_new", RunStatus.RUNNING, same), _run("tb_old", RunStatus.RUNNING, same))
        ids = [r.run_id for r in _overview_runs(TuiState(runs=runs))]
        self.assertEqual(ids, ["tb_old", "tb_new"])


class SkippedPytestCommandTests(unittest.TestCase):
    """C4: a skipped pytest suite with no runnable command records no command."""

    def test_no_hermes_agent_root_records_empty_command(self) -> None:
        res = _pytest_suite(None, evidence_dir=None)
        self.assertEqual(res.name, PYTEST_SUITE)
        self.assertEqual(res.status, RunStatus.SKIP)
        self.assertEqual(res.command, [])


if __name__ == "__main__":
    unittest.main()
