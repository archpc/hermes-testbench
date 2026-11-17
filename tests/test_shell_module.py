"""Shell module tests: isolation scrub, declared shell_test suites, completions.

Coverage map:
* shell startup/config environment is scrubbed before any suite runs
* ``shell_test`` executes exactly the declared files, under a pinned
  interpreter with rc files disabled, reusing the ``shell`` marker parser
* misconfiguration is rejected at discovery rather than silently ignored
* generated completions cover the real CLI contract and run on bash 3.2 / zsh 5.9
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from hermes_testbench.cli import build_parser
from hermes_testbench.discovery import discover_project
from hermes_testbench.engine import TestBenchEngine
from hermes_testbench.isolation import build_isolated_env
from hermes_testbench.schema import ProfileName, RunStatus
from hermes_testbench.shell_completion import command_tree, render, render_zsh
from hermes_testbench.store import TestBenchStore

PY = sys.executable
SHELL_LEAK_KEYS = (
    "ZDOTDIR",
    "BASH_ENV",
    "ENV",
    "SHELLOPTS",
    "BASHOPTS",
    "CDPATH",
    "GLOBIGNORE",
)


def _tmpdir(prefix: str = "tb-shell-") -> Path:
    return Path(tempfile.mkdtemp(prefix=prefix, dir=os.environ.get("TMPDIR")))


def _write(path: Path, text: str, *, executable: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    if executable:
        path.chmod(0o755)
    return path


class IsolationScrubTests(unittest.TestCase):
    def test_shell_startup_keys_are_scrubbed_from_the_suite_environment(self) -> None:
        base = {key: f"/leaked/{key}" for key in SHELL_LEAK_KEYS}
        base["KEEP_ME"] = "kept"
        base["OPENAI_API_KEY"] = "sk-should-not-leak"
        env = build_isolated_env(_tmpdir(), run_id="r1", nonce="n1", base_env=base)

        for key in SHELL_LEAK_KEYS:
            self.assertNotIn(key, env, f"{key} leaked into the isolated environment")
        self.assertNotIn("OPENAI_API_KEY", env)
        self.assertEqual(env["KEEP_ME"], "kept")

    def test_existing_isolation_contract_is_preserved(self) -> None:
        env = build_isolated_env(_tmpdir(), run_id="r1", nonce="n1", base_env={"PATH": "/usr/bin:/bin"})
        self.assertEqual(env["TESTBENCH_SANDBOX"], "1")
        self.assertEqual(env["TESTBENCH_RUN_ID"], "r1")
        self.assertNotEqual(env["HOME"], os.path.expanduser("~"))
        self.assertTrue(env["TMPDIR"].startswith("/"))
        self.assertIn("LC_ALL", env)


class ShellTestDiscoveryTests(unittest.TestCase):
    def _manifest(self, root: Path, suite_body: str) -> Path:
        return _write(root / ".testbench.toml", f'[project]\nname = "shellproj"\n\n{suite_body}')

    def test_one_suite_per_declared_file_with_pinned_interpreter(self) -> None:
        root = _tmpdir()
        _write(root / "tests/a_test.sh", "#!/bin/bash\nexit 0\n")
        _write(root / "tests/b_test.sh", "#!/bin/bash\nexit 0\n")
        self._manifest(
            root,
            '[[suite]]\nname = "shell-unit"\nadapter = "shell_test"\nprofile = "unit"\n'
            'shell = "bash"\nfiles = ["tests/a_test.sh", "tests/b_test.sh"]\ntimeout_s = 30.0\n',
        )
        spec = discover_project(root)
        self.assertEqual(
            [s.name for s in spec.suites],
            ["shell-unit:tests/a_test.sh", "shell-unit:tests/b_test.sh"],
        )
        resolved = shutil.which("bash") or "bash"
        for suite in spec.suites:
            self.assertEqual(suite.adapter, "shell_test")
            self.assertEqual(suite.profile, ProfileName.UNIT)
            self.assertEqual(suite.command[0], resolved)
            self.assertEqual(suite.command[1:3], ["--noprofile", "--norc"])
            self.assertEqual(suite.required_paths, [suite.command[-1]])
            self.assertIn(resolved, suite.required_binaries)

    def test_zsh_declaration_disables_rc_files(self) -> None:
        root = _tmpdir()
        _write(root / "z_test.zsh", "exit 0\n")
        self._manifest(
            root,
            '[[suite]]\nname = "z"\nadapter = "shell_test"\nfiles = ["z_test.zsh"]\nshell = "zsh"\n',
        )
        suite = discover_project(root).suites[0]
        self.assertEqual(suite.command[0], shutil.which("zsh") or "zsh")
        self.assertEqual(suite.command[1], "-f")
        # Effective value on the discovered spec, not the parser's default:
        # an omitted profile must land as UNIT where execution reads it.
        self.assertEqual(suite.profile, ProfileName.UNIT)

    def test_explicit_interpreter_must_be_absolute(self) -> None:
        root = _tmpdir()
        _write(root / "a_test.sh", "exit 0\n")
        self._manifest(
            root,
            '[[suite]]\nname = "s"\nadapter = "shell_test"\nfiles = ["a_test.sh"]\n'
            'interpreter = "bash"\n',
        )
        with self.assertRaises(ValueError) as caught:
            discover_project(root)
        self.assertIn("absolute path", str(caught.exception))

    def test_rejects_command_files_conflict_empty_files_and_bad_shell(self) -> None:
        cases = {
            "command conflict": 'adapter = "shell_test"\nfiles = ["a.sh"]\ncommand = ["/bin/true"]\n',
            "empty files": 'adapter = "shell_test"\nfiles = []\n',
            "bad shell": 'adapter = "shell_test"\nfiles = ["a.sh"]\nshell = "fish"\n',
            "absolute file": 'adapter = "shell_test"\nfiles = ["/etc/passwd"]\n',
            "escape": 'adapter = "shell_test"\nfiles = ["../a.sh"]\n',
        }
        for label, body in cases.items():
            with self.subTest(label):
                root = _tmpdir()
                _write(root / "a.sh", "exit 0\n")
                self._manifest(root, f'[[suite]]\nname = "s"\n{body}')
                with self.assertRaises(ValueError):
                    discover_project(root)

    def test_shell_test_keys_are_rejected_on_other_adapters(self) -> None:
        root = _tmpdir()
        self._manifest(
            root,
            '[[suite]]\nname = "s"\nadapter = "shell"\nfiles = ["a.sh"]\ncommand = ["/bin/true"]\n',
        )
        with self.assertRaises(ValueError) as caught:
            discover_project(root)
        self.assertIn("only valid with adapter", str(caught.exception))


class ShellTestExecutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.home = _tmpdir("tb-shell-home-")
        self.proj = _tmpdir("tb-shell-proj-")

    def _run(self, manifest_body: str):
        _write(self.proj / ".testbench.toml", f'[project]\nname = "shellproj"\n\n{manifest_body}')
        with TestBenchStore(self.home) as store:
            eng = TestBenchEngine(store)
            return eng.run_sync(self.proj, profile="unit")

    def test_markers_reach_existing_parser_and_exit_status_is_honoured(self) -> None:
        _write(
            self.proj / "tests/ok_test.sh",
            "printf 'TB-PASS ok:one\\nTB-PASS ok:two\\n'\nexit 0\n",
            executable=True,
        )
        _write(
            self.proj / "tests/bad_test.sh",
            "printf 'TB-PASS bad:one\\nTB-FAIL bad:two boom\\n'\nexit 1\n",
            executable=True,
        )
        run = self._run(
            '[[suite]]\nname = "unit"\nadapter = "shell_test"\nprofile = "unit"\n'
            'shell = "bash"\nfiles = ["tests/ok_test.sh", "tests/bad_test.sh"]\ntimeout_s = 30.0\n'
        )
        by_name = {s.name: s for s in run.suites}
        ok = by_name["unit:tests/ok_test.sh"]
        bad = by_name["unit:tests/bad_test.sh"]
        self.assertEqual(ok.status, RunStatus.PASS)
        self.assertEqual((ok.total_count, ok.pass_count, ok.fail_count), (2, 2, 0))
        self.assertEqual(bad.status, RunStatus.FAIL)
        self.assertEqual((bad.total_count, bad.pass_count, bad.fail_count), (2, 1, 1))
        self.assertEqual([c.name for c in bad.cases], ["bad:one", "bad:two boom"])
        self.assertEqual(run.status, RunStatus.FAIL)

    def test_exit_status_only_script_never_invents_case_counts(self) -> None:
        _write(self.proj / "quiet_test.sh", "echo nothing structured\nexit 0\n", executable=True)
        run = self._run(
            '[[suite]]\nname = "q"\nadapter = "shell_test"\nfiles = ["quiet_test.sh"]\ntimeout_s = 30.0\n'
        )
        suite = run.suites[0]
        self.assertEqual(suite.status, RunStatus.PASS)
        self.assertEqual((suite.total_count, suite.pass_count), (1, 1))
        self.assertEqual(suite.cases, [])
        self.assertNotIn("shell_test", (suite.summary_excerpt or "").lower())

    def test_parent_rc_environment_does_not_reach_the_script(self) -> None:
        poison = _write(self.proj / "poison.sh", "export TB_POISONED=1\n")
        probe = "\n".join(
            [
                "#!/bin/bash",
                "rc=0",
                # Plain variables: bash never re-creates these, so any non-empty
                # value here is inherited leakage.
                *(f'if [ -n "${{{key}:-}}" ]; then echo "TB-FAIL leaked:{key}"; rc=1; '
                  f'else echo "TB-PASS clean:{key}"; fi'
                  for key in ("ZDOTDIR", "BASH_ENV", "ENV", "CDPATH", "GLOBIGNORE")),
                # SHELLOPTS/BASHOPTS always exist inside bash and list its own
                # options; what matters is that the parent's option bundles did
                # not get imported.
                'case "$SHELLOPTS" in *xtrace*|*errexit*) echo "TB-FAIL leaked:SHELLOPTS=$SHELLOPTS"; rc=1 ;; '
                '*) echo "TB-PASS clean:SHELLOPTS" ;; esac',
                'case "$BASHOPTS" in *nocaseglob*|*globstar*) echo "TB-FAIL leaked:BASHOPTS=$BASHOPTS"; rc=1 ;; '
                '*) echo "TB-PASS clean:BASHOPTS" ;; esac',
                'if [ -n "${TB_POISONED:-}" ]; then echo "TB-FAIL poisoned:BASH_ENV"; rc=1; fi',
                "exit $rc",
            ]
        )
        _write(self.proj / "probe_test.sh", probe + "\n", executable=True)
        saved = {key: os.environ.get(key) for key in SHELL_LEAK_KEYS}
        try:
            os.environ["BASH_ENV"] = str(poison)
            os.environ["ENV"] = str(poison)
            os.environ["ZDOTDIR"] = str(self.proj)
            os.environ["CDPATH"] = str(self.proj)
            os.environ["GLOBIGNORE"] = "*.tmp"
            os.environ["SHELLOPTS"] = "xtrace"
            os.environ["BASHOPTS"] = "nocaseglob"
            run = self._run(
                '[[suite]]\nname = "probe"\nadapter = "shell_test"\nfiles = ["probe_test.sh"]\n'
                'timeout_s = 30.0\n'
            )
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        suite = run.suites[0]
        self.assertEqual(suite.status, RunStatus.PASS, suite.summary_excerpt)
        self.assertEqual(suite.fail_count, 0)

    def test_missing_declared_file_skips_instead_of_failing(self) -> None:
        run = self._run(
            '[[suite]]\nname = "gone"\nadapter = "shell_test"\nfiles = ["tests/absent_test.sh"]\n'
        )
        suite = run.suites[0]
        self.assertEqual(suite.status, RunStatus.SKIP)
        self.assertIn("required path missing", suite.summary_excerpt)

    def test_missing_interpreter_skips_with_named_reason(self) -> None:
        _write(self.proj / "a_test.sh", "exit 0\n", executable=True)
        run = self._run(
            '[[suite]]\nname = "x"\nadapter = "shell_test"\nfiles = ["a_test.sh"]\n'
            'interpreter = "/nonexistent/bash9"\n'
        )
        suite = run.suites[0]
        self.assertEqual(suite.status, RunStatus.SKIP)
        self.assertIn("/nonexistent/bash9", suite.summary_excerpt)


class CompletionTests(unittest.TestCase):
    def test_every_visible_command_and_option_appears_in_both_scripts(self) -> None:
        parser = build_parser()
        tree = command_tree(parser)
        self.assertIn("completion", tree)
        bash_script = render("bash")
        zsh_script = render("zsh")
        for name, spec in tree.items():
            self.assertIn(name, bash_script)
            self.assertIn(name, zsh_script)
            for opt in spec.options:
                self.assertIn(opt, bash_script, f"{name} {opt} missing from bash")
                self.assertIn(opt, zsh_script, f"{name} {opt} missing from zsh")
            for sub_name, sub in spec.subcommands.items():
                self.assertIn(sub_name, bash_script)
                self.assertIn(sub_name, zsh_script)
                for opt in sub.options:
                    self.assertIn(opt, bash_script, f"{name} {sub_name} {opt} missing from bash")
                    self.assertIn(opt, zsh_script, f"{name} {sub_name} {opt} missing from zsh")

    def test_hidden_worker_command_is_not_advertised(self) -> None:
        self.assertNotIn("_worker", render("bash"))
        self.assertNotIn("_worker", render("zsh"))

    def test_profile_candidates_come_from_the_schema_contract(self) -> None:
        script = render("bash")
        self.assertIn("--profile)", script)
        profile_line = next(line for line in script.splitlines() if "--profile)" in line)
        for profile in ProfileName.STANDARD:
            self.assertIn(profile, profile_line)

    def test_bash_32_syntax_and_candidate_behaviour(self) -> None:
        bash = "/bin/bash"
        self.assertTrue(Path(bash).is_file())
        script = _write(_tmpdir() / "testbench.bash", render("bash"))
        self.assertEqual(subprocess.run([bash, "-n", str(script)]).returncode, 0)

        def complete(words: list[str], cword: int) -> list[str]:
            snippet = (
                f"source {str(script)!r}; "
                f"COMP_WORDS=({words[0]!r} " + " ".join(repr(w) for w in words[1:]) + "); "
                f"COMP_CWORD={cword}; _testbench; printf '%s\\n' \"${{COMPREPLY[@]}}\""
            )
            proc = subprocess.run([bash, "--noprofile", "--norc", "-c", snippet],
                                  capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            return [line for line in proc.stdout.splitlines() if line]

        self.assertEqual(complete(["testbench", "ru"], 1)[:1], ["run"])
        self.assertIn("--profile", complete(["testbench", "run", "--pro"], 2))
        self.assertEqual(
            complete(["testbench", "run", "--profile", ""], 3),
            list(ProfileName.STANDARD),
        )
        self.assertIn("compare", complete(["testbench", "bench", "c"], 2))
        self.assertIn("--repeats", complete(["testbench", "bench", "run", "--rep"], 3))

    def test_zsh_syntax_load_and_candidates(self) -> None:
        zsh = shutil.which("zsh")
        if not zsh:
            self.skipTest("zsh not available")
        root = _tmpdir()
        script = _write(root / "_testbench", render_zsh(build_parser()))
        self.assertEqual(subprocess.run([zsh, "-n", str(script)]).returncode, 0)

        snippet = (
            f"fpath=({str(root)!r} $fpath); autoload -Uz _testbench; "
            "compadd() { shift; print -l -- \"$@\" }; "
            "_describe() { print -r -- \"DESC:$2\" }; "
            "words=(testbench run --profile ''); CURRENT=4; _testbench"
        )
        proc = subprocess.run([zsh, "-f", "-c", snippet], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(
            [line for line in proc.stdout.splitlines() if line], list(ProfileName.STANDARD)
        )

    def test_cli_subcommand_prints_a_script_without_touching_the_store(self) -> None:
        env = dict(os.environ)
        env["TESTBENCH_HOME"] = str(_tmpdir("tb-shell-nohome-"))
        for shell, needle in (("bash", "complete -F _testbench testbench"), ("zsh", "#compdef testbench")):
            proc = subprocess.run(
                [PY, "-m", "hermes_testbench", "completion", shell],
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn(needle, proc.stdout)
            self.assertEqual(proc.stderr, "")


if __name__ == "__main__":
    unittest.main()
