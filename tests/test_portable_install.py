"""Regressions: the installed interpreter is the one that runs everything.

1. The detached ``_worker`` must run under the interpreter that launched TestBench
   (``sys.executable`` of the installation environment), never a host-pinned path.
2. Default interpreter selection and bin directories carry no machine-specific path.
3. Manifest ``{python}`` argv[0] resolves to the default interpreter.

The worker test spawns from a throwaway venv so the spawning interpreter path differs
from any host-pinned candidate; it also passes when the suite runs against an installed
wheel, because the package root is taken from the imported module.

How the venv makes the assertion bite: ``EnvBuilder(symlinks=True)`` leaves
``bin/python`` as a symlink to the base interpreter, and ``sys.executable`` inside that
venv reports the *venv* path (verified: the symlink is not resolved). So
``spawner == str(vpy)`` holds while any resolved host path differs — which is exactly
the distinction the worker must preserve. That is also why the comparison here is on
strings, while ``test_installed_artifact`` realpaths both sides of a shebang check: in
that case one side is a symlinked shim and the other is whatever the interpreter
printed, so string equality would be an accident of layout.

Isolation detail: every temp tree is created under ``default_scratch_root()``, so the
worker's own sandbox lands inside the test's scratch area instead of the real one.
``PYTHONPATH`` is set to the package parent (not the repo root) so a subprocess under
the throwaway interpreter can import the package under test without an install.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import venv

import hermes_testbench
from hermes_testbench import adapters
from hermes_testbench.discovery import default_interpreter, discover_project
from hermes_testbench.isolation import default_scratch_root, extra_bin_dirs, real_user_home

# The *package* parent, not the repository root: the child only needs
# ``hermes_testbench`` importable, and pointing it at the repo would also make the
# checkout's tests/ and scripts/ importable — i.e. it would hide import-path defects.
PKG_PARENT = Path(hermes_testbench.__file__).resolve().parent.parent

# Program run by the throwaway venv's interpreter. It goes through the public engine
# surface (store + ``spawn_async``) rather than the CLI so the environment it prints is
# the one the *engine* wrote, with no CLI-level rewriting in between. ``wait_up_to_s=0``
# forces the detached path — the whole point is that the worker is a separate process.
# Exit-code wise it always prints one JSON line, so the parent asserts on the payload
# instead of guessing from stderr.
SPAWN = r"""
import json, sys, time
from pathlib import Path
from hermes_testbench.engine import TestBenchEngine
from hermes_testbench.schema import RunStatus
from hermes_testbench.store import TestBenchStore
home, proj = Path(sys.argv[1]), Path(sys.argv[2])
with TestBenchStore(home) as store:
    eng = TestBenchEngine(store)
    rec = eng.spawn_async(proj, profile="unit", wait_up_to_s=0.0)
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        cur = store.get_run(rec.run_id)
        if cur is not None and cur.status in RunStatus.TERMINAL:
            break
        time.sleep(0.1)
    cur = store.get_run(rec.run_id)
    print(json.dumps({"status": cur.status, "env": cur.environment, "spawner": sys.executable}))
"""


class WorkerInterpreterTests(unittest.TestCase):
    """The detached worker's interpreter is inherited, not discovered."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(dir=str(default_scratch_root()), prefix="tb-interp-")
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_async_worker_runs_under_the_spawning_interpreter(self) -> None:
        """Three equalities, each ruling out a different wrong answer.

        ``spawner == str(vpy)`` confirms the harness itself is what we think it is, so a
        failure below is a product defect rather than a test that ran under the wrong
        interpreter. ``python_executable == spawner`` says run creation recorded the real
        launcher; ``worker_python == spawner`` says the spawn reused it. A hard-coded
        ``/usr/bin/python3`` or a resolved base-interpreter path breaks the second and
        third while leaving the first intact — that asymmetry is the diagnostic.

        ``TESTBENCH_*`` is stripped from the child env so an ambient
        ``TESTBENCH_PYTHON`` cannot substitute an interpreter and make this pass for the
        wrong reason. Note this test does not prove the worker *ran* the tests: the
        status check is what does that, and it is asserted first because a spawn that
        produced no terminal record would otherwise report an empty env and look like an
        interpreter mismatch.
        """
        env_dir = self.root / "venv"
        venv.EnvBuilder(with_pip=False, symlinks=True).create(env_dir)
        vpy = env_dir / "bin" / "python"
        proj = self.root / "proj"
        (proj / "tests").mkdir(parents=True)
        (proj / "tests" / "test_ok.py").write_text(
            "import unittest\n\nclass T(unittest.TestCase):\n    def test_ok(self):\n        pass\n",
            encoding="utf-8",
        )
        env = {k: v for k, v in os.environ.items() if not k.startswith("TESTBENCH_")}
        env["PYTHONPATH"] = str(PKG_PARENT)
        env["TESTBENCH_ALLOWED_ROOTS"] = str(self.root)
        proc = subprocess.run(
            [str(vpy), "-c", SPAWN, str(self.root / "home"), str(proj)],
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
            cwd=str(self.root),
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        out = json.loads(proc.stdout.strip().splitlines()[-1])
        self.assertEqual(out["status"], "PASS", out)
        self.assertEqual(out["env"].get("python_executable"), out["spawner"])
        self.assertEqual(out["env"].get("worker_python"), out["spawner"])
        self.assertEqual(out["spawner"], str(vpy))


class NoMachineSpecificPathTests(unittest.TestCase):
    """Default selection must be derivable on any host, with no path baked in."""

    def test_default_interpreter_is_running_interpreter_or_override(self) -> None:
        """``TESTBENCH_PYTHON`` is honoured only when it names a usable executable.

        The first assertion runs with the variable *popped*, so an ambient override
        cannot mask a broken fallback. The last one — a syntactically fine but
        nonexistent path — is the discriminating case: returning it would make
        discovery emit a command that cannot run, i.e. a broken suite instead of a
        fallback. Restoration in ``finally`` is required because this mutates process
        env shared by later tests.
        """
        old = os.environ.pop("TESTBENCH_PYTHON", None)
        try:
            self.assertEqual(default_interpreter(), Path(sys.executable))
            os.environ["TESTBENCH_PYTHON"] = sys.executable
            self.assertEqual(default_interpreter(), Path(sys.executable))
            os.environ["TESTBENCH_PYTHON"] = "/nonexistent/python-for-test"
            self.assertEqual(default_interpreter(), Path(sys.executable))
        finally:
            os.environ.pop("TESTBENCH_PYTHON", None)
            if old is not None:
                os.environ["TESTBENCH_PYTHON"] = old

    def test_extra_bin_dirs_derive_from_real_user_home(self) -> None:
        """``~/.local/bin`` is computed from the *real* home, and adapters agree on it.

        ``real_user_home()``, not ``$HOME``: isolation redirects ``HOME`` into the
        sandbox, so deriving from the environment would drop the user's tool directory
        the moment a suite runs. The second assertion pins adapters'
        ``_extra_bin_dirs`` to the same function — a private copy that drifted would
        make ``cargo``/``uv`` discoverable during a run but not during discovery,
        which reads as flaky SKIPs rather than a bug.
        """
        dirs = extra_bin_dirs()
        self.assertEqual(dirs[0], str(real_user_home() / ".local" / "bin"))
        self.assertEqual(adapters._extra_bin_dirs(), dirs)

    def test_package_source_has_no_user_specific_absolute_paths(self) -> None:
        """No ``/Users/`` or ``/home/`` literal survives in shipped package code.

        A baked-in home path makes the package non-portable in the quietest possible
        way: it works on the author's machine. The scan is textual and line-based, and
        it exempts only lines whose first non-space character is ``#`` — so a path in a
        docstring or an f-string is an offender too. That is intentional (a docstring
        path is still a stale path), but it means documentation added here must not
        quote a real home directory.

        ``rglob("*.py")`` reads every module in the installed package, so the check
        covers files this test file never imports.
        """
        offenders = []
        for path in Path(hermes_testbench.__file__).resolve().parent.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for marker in ("/Users/", "/home/"):
                for lineno, line in enumerate(text.splitlines(), 1):
                    if marker in line and not line.lstrip().startswith("#"):
                        offenders.append(f"{path.name}:{lineno}: {line.strip()[:100]}")
        self.assertEqual(offenders, [])


class ManifestPythonTokenTests(unittest.TestCase):
    """``{python}`` is a discovery-time substitution, not a run-time lookup."""

    def test_python_token_resolves_to_default_interpreter(self) -> None:
        """Only argv[0] is substituted, and only when it is exactly ``{python}``.

        The manifest is written into a temp project (created under the scratch root)
        because discovery reads a real ``.testbench.toml``; nothing is installed or
        executed. Substituting at discovery rather than at spawn is what keeps the
        recorded command and its repro instruction identical — the second assertion
        pins that the remaining argv is passed through untouched, so the substitution
        cannot silently rewrite flags that happen to contain the token text.
        """
        with tempfile.TemporaryDirectory(dir=str(default_scratch_root()), prefix="tb-tok-") as tmp:
            root = Path(tmp)
            (root / ".testbench.toml").write_text(
                '[project]\nname = "tok"\n\n[[suite]]\nname = "u"\nadapter = "python_unittest"\n'
                'command = ["{python}", "-m", "unittest", "-v"]\n',
                encoding="utf-8",
            )
            spec = discover_project(root)
            self.assertEqual(spec.suites[0].command[0], str(default_interpreter()))
            self.assertEqual(spec.suites[0].command[1:], ["-m", "unittest", "-v"])


if __name__ == "__main__":
    unittest.main()
