"""Installed-artifact smoke tests: exercise a ``testbench`` installed from a wheel.

Source-checkout tests cannot catch packaging defects (missing modules, entry points,
worker spawn under a different interpreter or layout). These tests run only when
``TESTBENCH_INSTALLED_BIN`` names an installed console script, e.g. from CI:

    uv build && uv tool install --python 3.12 dist/*.whl
    TESTBENCH_INSTALLED_BIN=$(command -v testbench) python -m unittest tests.test_installed_artifact -v

They import nothing from the source tree: every check goes through the installed binary,
and the source tree is moved off the worker's import path by running from a temp cwd.

Contract with the environment, precisely:

* ``TESTBENCH_INSTALLED_BIN`` is read **at import time** into ``BIN`` and the skip is
  class-level, so unset means every test skips rather than errors. A non-empty but
  unusable path (typo, deleted venv) does *not* skip — it fails, which is the
  intended behaviour: CI that forgot to install must not look green.
* Each subprocess env is the caller's, minus every ``TESTBENCH_*`` variable and
  ``PYTHONPATH``. That matters twice: an inherited ``TESTBENCH_HOME``/
  ``TESTBENCH_SCRATCH_DIR``/``TESTBENCH_ALLOWED_ROOTS`` would silently point the run
  at the developer's real store, and an inherited ``PYTHONPATH`` would let the
  installed binary import the *checkout* instead of its own wheel — destroying the
  only thing these tests exist to prove. ``cwd`` is the temp root for the same
  reason: a cwd-relative import cannot find ``hermes_testbench`` there.
* ``TESTBENCH_ALLOWED_ROOTS`` is pinned to the temp root, so a path bug surfaces as
  a refusal inside the sandbox instead of being masked by a permissive default.

What this suite does *not* prove: that the wheel's metadata is complete for every
platform, or anything about a Windows launcher (``_interpreter_of`` reads a POSIX
shebang). Where the check would have to guess, it is skipped rather than asserted —
see ``_interpreter_of``.
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
    """Shebang interpreter of the installed console script, or ``""`` when there is none.

    ``realpath`` first: ``uv tool install`` and pip both leave a symlink in a bin
    directory, and the interesting shebang is the one on the target (the venv's own
    interpreter), not on the shim. Returns ``""`` for a launcher without a shebang
    (Windows ``.exe``, a shell shim) — callers then skip the interpreter comparison
    rather than compare against a guess.
    """
    with open(os.path.realpath(script), "rb") as fh:
        first = fh.readline().decode("utf-8", "replace").strip()
    return first[2:].strip() if first.startswith("#!") else ""


@unittest.skipUnless(BIN, "TESTBENCH_INSTALLED_BIN not set (installed-artifact smoke only)")
class InstalledArtifactTests(unittest.TestCase):
    """End-to-end checks against the installed console script only."""

    def setUp(self) -> None:
        """Build a self-contained project + store + scratch, all inside one temp root.

        The project is a two-test ``unittest`` module on purpose: it needs no
        third-party runner, so the counts asserted later cannot depend on whether
        pytest happened to be installed alongside the wheel. ``home`` and ``scratch``
        live under the same root that ``TESTBENCH_ALLOWED_ROOTS`` names, so an
        escaping path fails here rather than writing to the real home.
        """
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
        """Run the installed binary with this test's home, cwd, and scrubbed env.

        Never raises on a non-zero exit: every caller asserts the code it expects,
        so a failure message carries the command's own stderr instead of a generic
        ``CalledProcessError``. The timeout is per call and real — an installed
        worker that hangs (bad spawn, interpreter mismatch) must fail the CI job
        rather than stall it.
        """
        return subprocess.run(
            [BIN, "--home", str(self.home), *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=self.env,
            cwd=str(self.root),
        )

    def test_version_reports_installation_interpreter(self) -> None:
        """``--version`` must name the interpreter actually running the code.

        ``cli._version_string()`` embeds ``sys.executable``, so this reads back the
        interpreter the console script really started — not the one its shebang
        names. A shebang pointing at ``/usr/bin/python3`` while the package lives in a
        ``uv`` venv would show up as a mismatch here; a stale ``PYTHONPATH`` that made
        the shim import a checkout would still pass this and fail the worker test
        below, which is why both exist.

        The shebang comparison is conditional (no shebang: no comparison). The
        unconditional prefix check still fails if the entry point is missing from the
        wheel, because the process would not produce a banner at all.
        """
        proc = self._tb("--version")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("testbench ", proc.stdout)
        installed_py = _interpreter_of(BIN)
        if installed_py:
            self.assertIn(installed_py, proc.stdout)

    def test_discover_and_sync_run_counts_and_junit(self) -> None:
        """Discovery, a synchronous run, and the JUnit renderer all work from the wheel.

        ``len(spec["suites"]) == 1`` says the packaging did not lose an adapter module
        and that a plain ``unittest`` project still collapses to exactly one suite
        (extra suites would mean a doubled auto-detect; zero would mean a missing
        adapter). The ``(2, 2, 0)`` tuple is read from the *persisted record*, so it
        also proves the store schema round-trips through a fresh install.

        The JUnit assertion matches an attribute substring
        (``tests="2" failures="0" errors="0" skipped="0"``). That is a contract with
        the renderer's attribute order, not with JUnit itself — reordering attributes
        is a test change, deliberately, because the string is what consumers diff.
        """
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
        """The detached ``_worker`` runs under the installed interpreter, not a host pin.

        The exit code of ``run --async`` is accepted as 0 or 1 on purpose: the CLI
        returns 0 while the run is still ACTIVE and otherwise derives the code from the
        verdict, so which one appears depends on whether the worker finished inside the
        spawn window. The verdict is taken from ``status`` after polling instead — this
        test is about *which interpreter* settled the run, not about the exit code.

        ``status`` is called with an explicit run id; the list/object tolerance is
        defensive, since the no-id form prints a JSON array and the single-id form a
        JSON object.

        The three environment assertions are the actual regression: ``worker_python``
        equals ``python_executable`` (the worker was spawned by the same interpreter
        that created the run — a re-spawn under another python splits them), it
        realpath-matches the console script's shebang, and it does **not** live under
        the checkout. That last one is what catches an editable-install or
        ``PYTHONPATH`` leak, which every other test here would happily tolerate.

        Sequential polling with a 90 s deadline: the worker is a real process doing a
        real run, so the loop cannot assume any particular settle time. On timeout the
        assertion reports the last observed status and ``first_failure``, which is the
        only useful signal left at that point.
        """
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
        """``reproduce`` replays the recorded command and lands on the same counts.

        Reproduce creates a new run from the persisted spec, so equal
        ``(status, pass_count, fail_count)`` proves the record carried everything the
        engine needs (root, profile, suite commands) — a store that could merely *read*
        a run back would pass every other test and fail this one.
        """
        run = self._tb("run", str(self.proj), "--profile", "unit", "--json")
        rid = json.loads(run.stdout)["run_id"]
        rep = self._tb("reproduce", rid, "--json")
        self.assertEqual(rep.returncode, 0, rep.stderr)
        out = json.loads(rep.stdout)
        self.assertEqual((out["status"], out["pass_count"], out["fail_count"]), ("PASS", 2, 0))

    def test_mcp_handshake_lists_tools(self) -> None:
        """The stdio MCP server completes a handshake and advertises ``testbench_run``.

        Messages are piped as newline-delimited JSON with the ``initialize`` request,
        the ``notifications/initialized`` notification, then ``tools/list`` — the order
        a real client uses. The notification is included because a server that only
        accepts requests (never notifications) would still pass a request-only test and
        then hang against a conforming client.

        The reply is located by ``id`` rather than by position, so log lines or
        out-of-order responses do not produce a false negative. The process exit status
        is deliberately not asserted: the server terminates when stdin reaches EOF, and
        that shutdown path is not part of the contract under test — the reply stream is.
        """
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
        """A synchronous run leaves no sandbox directory and no live worker behind.

        Two independent leftovers, both checked here because they have different
        causes: ``tbw-*`` entries in ``TESTBENCH_SCRATCH_DIR`` mean teardown failed to
        delete a sandbox (a surviving process, an open file), while a live
        ``_worker`` process means the detached spawn outlived its parent. The run's own
        exit code is not asserted — this test is about what remains afterwards, and a
        failing-but-clean run still has to clean up.

        ``pgrep`` is optional (absent on minimal containers); when it is missing the
        process check is skipped rather than faked. What remains is a point-in-time
        observation, not a proof of absence: a worker that is still exiting would not
        show up. The scratch check is the deterministic half.
        """
        self._tb("run", str(self.proj), "--profile", "unit")
        leftovers = [p.name for p in (self.root / "scratch").iterdir() if p.name.startswith("tbw-")]
        self.assertEqual(leftovers, [])
        if shutil.which("pgrep"):
            ps = subprocess.run(["pgrep", "-f", f"_worker.*{self.home}"], capture_output=True, text=True)
            self.assertEqual(ps.stdout.strip(), "")


if __name__ == "__main__":
    unittest.main()
