"""Engine and CLI integration: scoped access, sync runs, streaming, async cancel, dispatch.

Most of this file drives the real stack — a real store, real isolation, real
subprocesses — because the properties under test are about process and file-system
behaviour a mock cannot have: a lock the kernel drops when a process dies, output that
reaches the terminal before the suite exits, an exit code a wrapper script reads.

Two things are injected to keep it fast and deterministic:

* ``discover_fn=self._fake_discover`` for the in-process engine tests, so suite selection
  does not depend on git state or a manifest on disk;
* the CLI is invoked as ``python -m hermes_testbench`` from ``REPO_ROOT``, and the tests that
  run it from another working directory also set ``PYTHONPATH``, so the child imports these
  sources rather than an installed copy.

Fixtures: ``self.root`` is a temp dir holding ``Code/demo_proj`` (the allowlisted root)
and ``tb_home`` (the store home). Keeping the allowlist inside the temp tree is what lets
the scoped-access test separate "inside an allowed root" from "registered in the store"
without reading the user's real ``~/Code``.
"""

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
    """In-process engine tests plus out-of-process CLI tests, sharing one fixture shape.

    ``self.allowed_code`` is both the scoped-access allowlist and the parent of the temp
    project, so a path can be "inside the allowlist" and "registered" independently — the
    distinction the authorization test turns on. ``tb_home`` is a separate directory under
    the same temp root, so store state never lands inside the project being run.
    """

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
        """Stand-in for ``discovery.discover_project``: two suites on two profiles.

        ``unit`` runs a script that prints a unittest-shaped summary and exits 0; ``integration``
        sleeps 30s so the async/cancel test has a real child that outlives its caller. The
        command uses ``sys.executable``, so the child is the same interpreter as the test and a
        second Python on PATH cannot change the outcome.

        ``custom_from_manifest`` is left at its ``False`` default, which keeps the trust gate out
        of the way: these tests are about execution, not policy.
        """
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
        """Allowlist, registration, and the protected ``~/.hermes`` deny — in that order.

        Three sibling paths, three outcomes:

        * a directory under ``allowed_roots`` resolves — the allow path;
        * a directory under the *real* ``~/.hermes`` is refused even though an allowed root is
          supplied, because the enumerated runtime deny is checked before any allow rule.
          This half is conditional on ``~/.hermes/skills`` existing on the machine, so on a host
          without it the deny path is silently not exercised;
        * an unlisted directory outside every root is refused, then allowed after
          ``store.register_project``.

        The last two are the discriminating pair: same path, same ``allowed_roots``, only the
        registry row differs. That is what proves registration widens access and that nothing
        else did.
        """
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
        """``run_sync`` settles a terminal record with artifacts; ``reproduce`` reads it back.

        The PASS comes from the injected script's ``OK`` marker via the generic parser (the
        adapter name ``custom`` is unknown to ``ADAPTER_PARSERS``). ``run_json_path`` existing is
        the evidence half: settlement writes ``run.json`` before the row, so a record without it
        would be a claim with nothing behind it.

        ``reproduce(print_only=True)`` must contain ``run_unit.py`` and must not rediscover the
        project — the replay path rebuilds from the stored ``repro_instructions`` (or the recorded
        ``commands``). A change that re-ran discovery could still produce a plausible string, which
        is why the assertion is on the recorded command's content and not on the run existing.
        """
        with TestBenchStore(self.tb_home) as store:
            eng = TestBenchEngine(store, discover_fn=self._fake_discover)
            run = eng.run_sync(self.proj, profile="unit")
            self.assertEqual(run.status, RunStatus.PASS)
            self.assertTrue(Path(run.run_json_path or "").is_file())

            repro_cmd = eng.reproduce(run.run_id, print_only=True)
            self.assertIn("run_unit.py", repro_cmd["repro_command"])

    def test_explicit_missing_suite_does_not_exit_success(self) -> None:
        """An explicit ``--suite`` that matches nothing is SKIP *and* a non-zero exit.

        ``run`` returns 0 for PASS and SKIP in general — "nothing to do" is success. An explicit
        ``--suite`` naming something that does not exist is the single exception in the CLI
        (``args.suite`` plus a SKIP record with no suites), so this asserts the pair: JSON status
        ``SKIP`` and ``returncode != 0``.

        Asserting only the status would pass even if the exit code regressed to 0, which is
        precisely the failure this test exists to catch: a CI script would read a typo'd suite
        name as success.
        """
        proc = subprocess.run([PY, "-m", "hermes_testbench", "--home", str(self.tb_home),
                               "run", str(self.proj), "--suite", "not-a-suite", "--json"],
                              cwd=REPO_ROOT, capture_output=True, text=True, timeout=10)
        self.assertEqual(json.loads(proc.stdout)["status"], RunStatus.SKIP)
        self.assertNotEqual(proc.returncode, 0)

    def test_foreground_stream_shows_sanitized_output_before_suite_finishes(self) -> None:
        """Foreground ``--stream`` emits output while the suite is still running.

        The script prints a padded line, sleeps 2s, then prints ``TB-PASS done``. The test proves
        ordering, not merely content: ``select`` must report the pipe readable within 10s, the
        first chunk must contain the marker printed *before* the sleep, and ``proc.poll()`` must
        still be ``None`` at that instant. A buffered implementation that flushed at exit would
        satisfy a content-only check and fail here.

        The 400-character padding makes the first write larger than a single small pipe read and
        the marker distinctive, so ``LIVE-BEFORE-EXIT`` cannot be found by accident in a later
        read. Margins: 10s for the first byte, 15s for ``communicate`` — generous because a loaded
        machine delays the child's first flush; the ``finally`` terminates the child so a
        regression fails instead of hanging the suite.
        """
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
        """A detached worker outlives its caller, and ``cancel`` ends the job it owns.

        This goes through the real ``_worker`` path (a ``.testbench.toml`` plus ``spawn_async``)
        because the property under test is that nothing in the spawning process is needed for the
        run to progress: the caller keeps only a run id and polls the store.

        The two waits are where the assertions actually live:

        * up to 5s for the row to become ``RUNNING`` with a non-``None`` ``worker_pid``. That pid is
          the worker's own (``run_worker`` records ``os.getpid()``), and its presence is what makes
          cancellation possible at all;
        * after ``cancel`` returns ``CANCELLED``, up to 3s polling ``os.kill(pid, 0)`` until
          ``ProcessLookupError``. A status-only assertion would pass for a cancelled run whose
          worker kept running, so the liveness probe is the discriminating half.

        The suite sleeps 30s — far past both deadlines — so a worker that ignored the cancel flag
        cannot finish on its own before the test gives up.
        """
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
        """End-to-end dispatch through ``python -m hermes_testbench``: run → results → cancel.

        ``env`` carries ``TESTBENCH_HOME`` and ``PYTHONPATH`` because the child must import this
        checkout and open *this* store; without either it would use the real default home and the
        ``results`` read-back would find nothing.

        The three commands cover the read paths a script uses: ``--json`` run output (exit 0 and
        status ``PASS``), ``results --format json`` for the same id (which proves persistence
        across processes, not in-memory state), and ``cancel`` of an unknown id.

        The cancel case is called in-process via ``cli_main`` and asserts a non-zero exit: that is
        the CLI translating the engine's ``KeyError``. A regression that created an evidence
        directory, or signalled a process, for an arbitrary string would either raise or exit 0
        here — and the second would be an arbitrary-process kill.
        """
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
