"""Installed-artifact smoke tests: exercise a ``testbench`` installed from a wheel.

Source-checkout tests cannot catch packaging defects (missing modules, entry points,
worker spawn under a different interpreter or layout). These tests run only when
``TESTBENCH_INSTALLED_BIN`` names an installed console script, e.g. from CI:

    uv build && uv tool install --python 3.12 dist/*.whl
    TESTBENCH_INSTALLED_BIN=$(command -v testbench) python -m unittest tests.test_installed_artifact -v

They import nothing from the source tree: every check goes through the installed binary,
and the source tree is moved off the worker's import path by running from a temp cwd.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import unittest

BIN = os.environ.get("TESTBENCH_INSTALLED_BIN", "").strip()


def _interpreter_of(script: str) -> str:
    with open(os.path.realpath(script), "rb") as fh:
        first = fh.readline().decode("utf-8", "replace").strip()
    return first[2:].strip() if first.startswith("#!") else ""


@unittest.skipUnless(BIN, "TESTBENCH_INSTALLED_BIN not set (installed-artifact smoke only)")
class InstalledArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="tb-installed-")
        self.root = Path(self.tmp.name).resolve()
        self.home = self.root / "home"
        self.proj = self.root / "proj"
        (self.proj / "tests").mkdir(parents=True)
        (self.proj / "tests" / "test_ok.py").write_text(
            "import unittest\n\nclass T(unittest.TestCase):\n"
            "    def test_ok(self):\n        pass\n"
            "    def test_two(self):\n        pass\n",
            encoding="utf-8",
        )
        self.env = {k: v for k, v in os.environ.items() if not k.startswith(("TESTBENCH_", "PYTHONPATH"))}
        self.env["TESTBENCH_ALLOWED_ROOTS"] = str(self.root)
        self.env["TESTBENCH_SCRATCH_DIR"] = str(self.root / "scratch")
        (self.root / "scratch").mkdir()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _tb(self, *args: str, timeout: float = 120.0) -> subprocess.CompletedProcess:
        return subprocess.run(
            [BIN, "--home", str(self.home), *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=self.env,
            cwd=str(self.root),
        )

    def test_version_reports_installation_interpreter(self) -> None:
        proc = self._tb("--version")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("testbench ", proc.stdout)
        installed_py = _interpreter_of(BIN)
        if installed_py:
            self.assertIn(installed_py, proc.stdout)

    def test_discover_and_sync_run_counts_and_junit(self) -> None:
        disc = self._tb("discover", str(self.proj), "--json")
        self.assertEqual(disc.returncode, 0, disc.stderr)
        spec = json.loads(disc.stdout)
        self.assertEqual(len(spec["suites"]), 1)

        run = self._tb("run", str(self.proj), "--profile", "unit", "--json")
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        rec = json.loads(run.stdout)
        self.assertEqual(rec["status"], "PASS")
        self.assertEqual((rec["total_count"], rec["pass_count"], rec["fail_count"]), (2, 2, 0))
        junit = self._tb("results", rec["run_id"], "--format", "junit")
        self.assertEqual(junit.returncode, 0, junit.stderr)
        self.assertIn('tests="2" failures="0" errors="0" skipped="0"', junit.stdout)

    def test_async_worker_settles_under_the_installation_interpreter(self) -> None:
        run = self._tb("run", str(self.proj), "--profile", "unit", "--async", "--json")
        self.assertIn(run.returncode, (0, 1), run.stderr)
        rid = json.loads(run.stdout)["run_id"]
        deadline = time.monotonic() + 90
        rec = {}
        while time.monotonic() < deadline:
            st = self._tb("status", rid, "--json")
            rec = json.loads(st.stdout)
            rec = rec[0] if isinstance(rec, list) else rec
            if rec["status"] not in ("QUEUED", "RUNNING"):
                break
            time.sleep(0.25)
        self.assertEqual(rec["status"], "PASS", rec.get("first_failure"))
        env = rec.get("environment") or {}
        self.assertTrue(env.get("worker_python"), env)
        self.assertEqual(env.get("worker_python"), env.get("python_executable"))
        installed_py = _interpreter_of(BIN)
        if installed_py:
            self.assertEqual(os.path.realpath(env["worker_python"]), os.path.realpath(installed_py))
        self.assertNotIn(str(Path(__file__).resolve().parent.parent), env["worker_python"])

    def test_reproduce_matches_original(self) -> None:
        run = self._tb("run", str(self.proj), "--profile", "unit", "--json")
        rid = json.loads(run.stdout)["run_id"]
        rep = self._tb("reproduce", rid, "--json")
        self.assertEqual(rep.returncode, 0, rep.stderr)
        out = json.loads(rep.stdout)
        self.assertEqual((out["status"], out["pass_count"], out["fail_count"]), ("PASS", 2, 0))

    def test_mcp_handshake_lists_tools(self) -> None:
        msgs = [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize",
             "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "ci", "version": "0"}}},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        ]
        proc = subprocess.run(
            [BIN, "--home", str(self.home), "mcp"],
            input="".join(json.dumps(m) + "\n" for m in msgs),
            capture_output=True, text=True, timeout=60, env=self.env, cwd=str(self.root),
        )
        replies = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
        tools = next(r for r in replies if r.get("id") == 2)["result"]["tools"]
        self.assertIn("testbench_run", {t["name"] for t in tools})

    def test_no_leaked_processes_or_sandboxes(self) -> None:
        self._tb("run", str(self.proj), "--profile", "unit")
        leftovers = [p.name for p in (self.root / "scratch").iterdir() if p.name.startswith("tbw-")]
        self.assertEqual(leftovers, [])
        if shutil.which("pgrep"):
            ps = subprocess.run(["pgrep", "-f", f"_worker.*{self.home}"], capture_output=True, text=True)
            self.assertEqual(ps.stdout.strip(), "")


if __name__ == "__main__":
    unittest.main()
