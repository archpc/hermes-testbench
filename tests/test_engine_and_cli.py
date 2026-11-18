"""TDD tests for TestBenchEngine, scoped authorization, async detached workers, cancellation, and CLI."""

from __future__ import annotations

import json
import os
import selectors
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from hermes_testbench.isolation import real_user_home
from hermes_testbench.cli import main as cli_main
from hermes_testbench.engine import AuthorizationError, TestBenchEngine, check_scoped_access
from hermes_testbench.schema import ProjectSpec, RunStatus, SuiteSpec
from hermes_testbench.store import TestBenchStore

PY = sys.executable
REPO_ROOT = Path(__file__).resolve().parent.parent


class EngineAndCliTests(unittest.TestCase):
    def setUp(self) -> None:
        scratch = Path.home() / ".hermes/cache/scratch"
        scratch.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=str(scratch), prefix="tb-eng-")
        self.root = Path(self.tmp.name)
        self.tb_home = self.root / "tb_home"
        self.allowed_code = self.root / "Code"
        self.allowed_code.mkdir(mode=0o700)
        self.proj = self.allowed_code / "demo_proj"
        self.proj.mkdir(mode=0o700)
        (self.proj / "run_unit.py").write_text(
            "import sys\nsys.stderr.write('Ran 3 tests in 0.01s\\n\\nOK\\n')\nsys.exit(0)\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _fake_discover(self, root: Path, **_kw) -> ProjectSpec:
        return ProjectSpec(
            name=root.name,
            root=str(root),
            languages=["python"],
            revision="abc1234",
            branch="main",
            suites=[
                SuiteSpec(
                    name="unit-fast",
                    adapter="custom",
                    profile="unit",
                    command=[PY, str(root / "run_unit.py")],
                    cwd=str(root),
                    timeout_s=10.0,
                ),
                SuiteSpec(
                    name="slow-job",
                    adapter="custom",
                    profile="integration",
                    command=[PY, "-c", "import time; time.sleep(30)"],
                    cwd=str(root),
                    timeout_s=60.0,
                ),
            ],
        )

    def test_scoped_access_blocks_hermes_skills_and_outside_roots_unless_registered(self) -> None:
        with TestBenchStore(self.tb_home) as store:
            ok_path = check_scoped_access(self.proj, store, allowed_roots=[self.allowed_code])
            self.assertEqual(ok_path, self.proj.resolve())

            hermes_skills = real_user_home() / ".hermes" / "skills"
            if hermes_skills.exists():
                with self.assertRaises(AuthorizationError):
                    check_scoped_access(hermes_skills, store, allowed_roots=[self.allowed_code])

            outside = self.root / "unregistered_outside"
            outside.mkdir()
            with self.assertRaises(AuthorizationError):
                check_scoped_access(outside, store, allowed_roots=[self.allowed_code])

            store.register_project(outside)
            self.assertEqual(
                check_scoped_access(outside, store, allowed_roots=[self.allowed_code]),
                outside.resolve(),
            )

    def test_sync_run_persists_results_and_reproduce_replays_without_rediscovery(self) -> None:
        with TestBenchStore(self.tb_home) as store:
            eng = TestBenchEngine(store, discover_fn=self._fake_discover)
            run = eng.run_sync(self.proj, profile="unit")
            self.assertEqual(run.status, RunStatus.PASS)
            self.assertTrue(Path(run.run_json_path or "").is_file())

            repro_cmd = eng.reproduce(run.run_id, print_only=True)
            self.assertIn("run_unit.py", repro_cmd["repro_command"])

    def test_explicit_missing_suite_does_not_exit_success(self) -> None:
        proc = subprocess.run([PY, "-m", "hermes_testbench", "--home", str(self.tb_home),
                               "run", str(self.proj), "--suite", "not-a-suite", "--json"],
                              cwd=REPO_ROOT, capture_output=True, text=True, timeout=10)
        self.assertEqual(json.loads(proc.stdout)["status"], RunStatus.SKIP)
        self.assertNotEqual(proc.returncode, 0)

    def test_foreground_stream_shows_sanitized_output_before_suite_finishes(self) -> None:
        (self.proj / ".testbench.toml").write_text(
            '[[suite]]\nname = "stream"\nadapter = "shell"\nprofile = "unit"\n'
            f'command = [{json.dumps(PY)}, "run_unit.py"]\ntimeout_s = 15\n', encoding="utf-8")
        (self.proj / "run_unit.py").write_text(
            'import sys, time\n'
            'print("LIVE-BEFORE-EXIT " + "x" * 400, flush=True)\n'
            'time.sleep(2)\nprint("TB-PASS done", flush=True)\n', encoding="utf-8")
        argv = [PY, "-m", "hermes_testbench", "--home", str(self.tb_home),
                "run", str(self.proj), "--suite", "stream", "--stream"]
        with subprocess.Popen(argv, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as proc:
            try:
                assert proc.stdout is not None
                with selectors.DefaultSelector() as ready:
                    ready.register(proc.stdout, selectors.EVENT_READ)
                    self.assertTrue(ready.select(10), "no streaming output")
                    first = os.read(proc.stdout.fileno(), 8192)
                self.assertIn(b"LIVE-BEFORE-EXIT", first)
                self.assertIsNone(proc.poll(), "output appeared only after the suite finished")
                rest, _ = proc.communicate(timeout=15)
                self.assertIn(b"TB-PASS done", first + rest)
                self.assertEqual(proc.returncode, 0)
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    proc.wait(timeout=15)

    def test_async_worker_survives_caller_return_and_explicit_cancel_terminates_owned_job(self) -> None:
        # Write a real slow script so CLI detached _worker can run without custom python callbacks
        (self.proj / ".testbench.toml").write_text(
            '[project]\nname = "demo_proj"\n\n'
            '[[suites]]\nname = "slow"\nadapter = "shell"\nprofile = "integration"\n'
            f'command = [{PY!r}, "-c", "import time; time.sleep(30)"]\ntimeout_s = 60.0\n',
            encoding="utf-8",
        )
        with TestBenchStore(self.tb_home) as store:
            store.trust_project(self.proj)
            eng = TestBenchEngine(store)
            launched = eng.spawn_async(self.proj, profile="integration", wait_up_to_s=0.3)
            self.assertIn(launched.status, (RunStatus.QUEUED, RunStatus.RUNNING))
            rid = launched.run_id

            # Wait until worker acquires lock & marks RUNNING
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                cur = store.get_run(rid)
                if cur and cur.status == RunStatus.RUNNING and cur.worker_pid:
                    break
                time.sleep(0.05)
            cur = store.get_run(rid)
            self.assertIsNotNone(cur)
            assert cur is not None
            self.assertEqual(cur.status, RunStatus.RUNNING)
            wpid = cur.worker_pid
            self.assertIsNotNone(wpid)

            cancelled = eng.cancel(rid)
            self.assertEqual(cancelled.status, RunStatus.CANCELLED)

            # Worker process must exit promptly
            deadline2 = time.monotonic() + 3.0
            alive = True
            while time.monotonic() < deadline2:
                try:
                    os.kill(int(wpid or 0), 0)
                    time.sleep(0.05)
                except ProcessLookupError:
                    alive = False
                    break
            self.assertFalse(alive, f"Detached worker {wpid} survived cancel!")

    def test_cli_json_workflow_status_results_and_unknown_cancel_refusal(self) -> None:
        env = dict(os.environ)
        env["TESTBENCH_HOME"] = str(self.tb_home)
        env["PYTHONPATH"] = str(REPO_ROOT)

        (self.proj / ".testbench.toml").write_text(
            '[project]\nname = "demo_proj"\n\n'
            '[[suites]]\nname = "fast"\nadapter = "shell"\nprofile = "unit"\n'
            f'command = [{PY!r}, {str(self.proj / "run_unit.py")!r}]\ntimeout_s = 10.0\n',
            encoding="utf-8",
        )
        r_run = subprocess.run(
            [PY, "-m", "hermes_testbench", "run", str(self.proj), "--profile", "unit", "--json"],
            env=env,
            capture_output=True,
            text=True,
            timeout=15,
        )
        self.assertEqual(r_run.returncode, 0, r_run.stderr + r_run.stdout)
        run_obj = json.loads(r_run.stdout)
        self.assertEqual(run_obj["status"], "PASS")
        rid = run_obj["run_id"]

        r_res = subprocess.run(
            [PY, "-m", "hermes_testbench", "results", rid, "--format", "json"],
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(r_res.returncode, 0)
        self.assertEqual(json.loads(r_res.stdout)["run_id"], rid)

        # Cancelling a non-existent ID fails cleanly and never signals arbitrary processes
        rc_bad = cli_main(["--home", str(self.tb_home), "cancel", "tb_nonexistent_id"])
        self.assertNotEqual(rc_bad, 0)


if __name__ == "__main__":
    unittest.main()
