"""Regressions: the installed interpreter is the one that runs everything.

1. The detached ``_worker`` must run under the interpreter that launched TestBench
   (``sys.executable`` of the installation environment), never a host-pinned path.
2. Default interpreter selection and bin directories carry no machine-specific path.
3. Manifest ``{python}`` argv[0] resolves to the default interpreter.

The worker test spawns from a throwaway venv so the spawning interpreter path differs
from any host-pinned candidate; it also passes when the suite runs against an installed
wheel, because the package root is taken from the imported module.
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

PKG_PARENT = Path(hermes_testbench.__file__).resolve().parent.parent

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
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(dir=str(default_scratch_root()), prefix="tb-interp-")
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_async_worker_runs_under_the_spawning_interpreter(self) -> None:
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
    def test_default_interpreter_is_running_interpreter_or_override(self) -> None:
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
        dirs = extra_bin_dirs()
        self.assertEqual(dirs[0], str(real_user_home() / ".local" / "bin"))
        self.assertEqual(adapters._extra_bin_dirs(), dirs)

    def test_package_source_has_no_user_specific_absolute_paths(self) -> None:
        offenders = []
        for path in Path(hermes_testbench.__file__).resolve().parent.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for marker in ("/Users/", "/home/"):
                for lineno, line in enumerate(text.splitlines(), 1):
                    if marker in line and not line.lstrip().startswith("#"):
                        offenders.append(f"{path.name}:{lineno}: {line.strip()[:100]}")
        self.assertEqual(offenders, [])


class ManifestPythonTokenTests(unittest.TestCase):
    def test_python_token_resolves_to_default_interpreter(self) -> None:
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
