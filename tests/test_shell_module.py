"""Shell module tests: isolation scrub, declared shell_test suites, completions.

Coverage map:
* shell startup/config environment is scrubbed before any suite runs
* ``shell_test`` executes exactly the declared files, under a pinned
  interpreter with rc files disabled, reusing the ``shell`` marker parser
* misconfiguration is rejected at discovery rather than silently ignored
* generated completions cover the real CLI contract and run on bash 3.2 / zsh 5.9

Three properties are worth stating explicitly because the tests are easy to
misread:

* The scrub is by variable *name* (``_SCRUB_EXACT`` plus ``_SCRUB_KEY_RE``), not by
  provenance — a secret in an unrecognised name passes through. These tests assert
  the names that are known to leak (*rc file pointers, option bundles, ``*_API_KEY``),
  so a new leak class needs a new name here, not a stronger assertion.
* ``shell_test`` suites do not invent aggregate suites: one declared file becomes
  one suite named ``<suite>:<path>``, so per-file exit status, markers, evidence and
  repro commands survive. Discovery pins the interpreter to an absolute path, which
  is why a missing shell is a SKIP with a reason rather than a crash.
* Completion tests are output tests. ``shell_completion`` walks argparse internals
  (no public tree API exists), so generation is checked by *running the emitted
  script* on the real shells rather than by trusting the introspection. The bash
  half runs on ``/bin/bash`` (3.2 on macOS); the zsh half is skipped when zsh is
  absent, so a Linux CI box proves less here than a developer Mac.

Environment sensitivity: ``PY`` is ``sys.executable``, so the ``testbench
completion`` subprocess resolves ``hermes_testbench`` from the current working
directory (or an installed copy). Run these files from the repo root, as the
suite's own manifest does.
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
# Variables a caller's shell uses to point at rc files and to carry option bundles.
# ZDOTDIR/BASH_ENV/ENV are pointers into the *real* configuration (an overlay HOME does
# not stop them); SHELLOPTS/BASHOPTS are inherited option sets that can hand a script
# ``errexit``/``nocaseglob`` it never asked for; CDPATH/GLOBIGNORE change how pathnames
# resolve. Each one, if inherited, makes a shell suite behave differently on two hosts
# that claim the same inputs.
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
    """Create a scratch directory; ``mkdtemp`` falls back to the system temp when
    ``$TMPDIR`` is unset. Never cleaned up by the caller.

    Deliberately not a ``TemporaryDirectory``: several tests hand the path to discovery
    or to a subprocess and only some of them are torn down by the OS cleanup. Leftovers
    are visible on purpose — a leaked sandbox should be inspectable, and every test here
    works inside its own directory so a stale tree cannot make another test pass.
    """
    return Path(tempfile.mkdtemp(prefix=prefix, dir=os.environ.get("TMPDIR")))


def _write(path: Path, text: str, *, executable: bool = False) -> Path:
    """Write ``text`` (creating parents) and optionally mark it executable.

    The mode matters: ``shell_test`` runs the file through the pinned interpreter, but
    the manifest and discovery paths still resolve the path itself, and a test that
    forgot ``executable=True`` would exercise a different code path than a real suite.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    if executable:
        path.chmod(0o755)
    return path


class IsolationScrubTests(unittest.TestCase):
    """The environment handoff: what must disappear, what must stay, what must be set."""

    def test_shell_startup_keys_are_scrubbed_from_the_suite_environment(self) -> None:
        """Named rc/option variables and ``*_API_KEY`` are dropped; unrelated ones survive.

        ``KEEP_ME`` is the negative control: a scrubber that returned an empty dict (or
        dropped the whole parent env) would satisfy the first two assertions while
        breaking every tool that needs ``PATH``-adjacent settings. ``OPENAI_API_KEY``
        checks the key-pattern half of the scrub rather than the exact-name half — the
        two are separate mechanisms and a regression in one is invisible to the other.
        """
        base = {key: f"/leaked/{key}" for key in SHELL_LEAK_KEYS}
        base["KEEP_ME"] = "kept"
        base["OPENAI_API_KEY"] = "sk-should-not-leak"
        env = build_isolated_env(_tmpdir(), run_id="r1", nonce="n1", base_env=base)

        for key in SHELL_LEAK_KEYS:
            self.assertNotIn(key, env, f"{key} leaked into the isolated environment")
        self.assertNotIn("OPENAI_API_KEY", env)
        self.assertEqual(env["KEEP_ME"], "kept")

    def test_existing_isolation_contract_is_preserved(self) -> None:
        """The rest of the isolation contract survives widening the scrub.

        A narrow fix for the rc leak could plausibly have replaced the whole env rather
        than filtered it. These assertions pin what other modules depend on: the sandbox
        marker, the run id adapters read, a ``HOME`` that is *not* the real one, an
        absolute ``TMPDIR``, and a pinned locale. ``PATH`` is supplied explicitly
        because building from an empty base would say nothing about a real caller.
        """
        env = build_isolated_env(_tmpdir(), run_id="r1", nonce="n1", base_env={"PATH": "/usr/bin:/bin"})
        self.assertEqual(env["TESTBENCH_SANDBOX"], "1")
        self.assertEqual(env["TESTBENCH_RUN_ID"], "r1")
        self.assertNotEqual(env["HOME"], os.path.expanduser("~"))
        self.assertTrue(env["TMPDIR"].startswith("/"))
        self.assertIn("LC_ALL", env)


class ShellTestDiscoveryTests(unittest.TestCase):
    """``[[suite]] adapter = "shell_test"`` is expanded and validated at discovery."""

    def _manifest(self, root: Path, suite_body: str) -> Path:
        """Write a minimal ``[project]`` plus one suite body; returns the manifest path."""
        return _write(root / ".testbench.toml", f'[project]\nname = "shellproj"\n\n{suite_body}')

    def test_one_suite_per_declared_file_with_pinned_interpreter(self) -> None:
        """Two declared files become two suites, each with an absolute interpreter pin.

        The name shape ``shell-unit:<rel>`` is asserted exactly because it is the key
        for ``--suite`` selection, history rows and repro commands: a change here is a
        user-visible break, not cosmetic. ``command[1:3]`` pins rc disabling, while
        ``required_paths``/``required_binaries`` are what turn a missing file or shell
        into a named SKIP rather than a phantom PASS — those last two are the
        load-bearing assertions, not the naming.

        ``shutil.which("bash") or "bash"`` mirrors discovery's own fallback: without
        bash on PATH the pinned value is the bare name and the suite would SKIP at run
        time. Comparing against the same expression keeps the test from claiming more
        than discovery guarantees.
        """
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
        """zsh is pinned with ``-f``; an omitted ``profile`` still lands as UNIT.

        The ``profile`` assertion is about the *effective* spec, not the parser's
        default: execution reads ``suite.profile``, so a suite whose profile is applied
        only later in the pipeline would run under the wrong filter while looking
        correct in the manifest. ``-f`` is asserted rather than ``--no-rcs`` because the
        command line is exactly what a repro replays.
        """
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
        """A relative ``interpreter`` is refused, and the message names the requirement.

        Absolute-only is what makes the pinned path meaningful: a bare ``bash`` would be
        resolved by whichever PATH the run inherited, which is the defect the pin
        exists to prevent. The message check keeps the failure diagnosable — discovery
        errors surface as CLI errors, so "Got ValueError" without the reason would send
        a user to the wrong place.
        """
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
        """Five malformed declarations, each rejected at discovery, not at run time.

        The cases cover distinct failure modes rather than one shape five ways:
        ``command`` + ``files`` (two sources of truth), ``files = []`` (a suite that
        would run nothing), an unsupported ``shell`` (only bash/zsh/sh have rc flags),
        an absolute file (a manifest must not point outside the checkout), and a ``..``
        escape (same, reached by traversal). ``subTest`` keeps one failing case
        attributable instead of aborting the loop, so a partial regression still names
        itself.

        Rejection at discovery is the assertion's real content: a manifest error that
        survived to run time would surface as a failing suite, indistinguishable from a
        real test failure.
        """
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
        """``files``/``shell``/``interpreter`` mean nothing on a ``shell`` suite.

        The neighbouring adapter is the dangerous one: a ``shell`` suite already takes
        an explicit ``command``, so a leftover ``files`` would be silently ignored and
        look honoured. The message check pins that the error explains *why* (the keys
        belong to one adapter), which is the difference between a fixable manifest and
        a mysterious one.
        """
        root = _tmpdir()
        self._manifest(
            root,
            '[[suite]]\nname = "s"\nadapter = "shell"\nfiles = ["a.sh"]\ncommand = ["/bin/true"]\n',
        )
        with self.assertRaises(ValueError) as caught:
            discover_project(root)
        self.assertIn("only valid with adapter", str(caught.exception))


class ShellTestExecutionTests(unittest.TestCase):
    """Real runs of declared shell files through the engine, with markers as ground truth."""

    def setUp(self) -> None:
        self.home = _tmpdir("tb-shell-home-")
        self.proj = _tmpdir("tb-shell-proj-")

    def _run(self, manifest_body: str):
        """Write the manifest, then run the ``unit`` profile synchronously.

        Synchronous on purpose: these assertions are about counts and statuses, and a
        detached run would add a polling loop that can only fail in new ways. The store
        is a throwaway ``home`` per test, so no run history from another test can be
        picked up by the profile selection.
        """
        _write(self.proj / ".testbench.toml", f'[project]\nname = "shellproj"\n\n{manifest_body}')
        with TestBenchStore(self.home) as store:
            eng = TestBenchEngine(store)
            return eng.run_sync(self.proj, profile="unit")

    def test_markers_reach_existing_parser_and_exit_status_is_honoured(self) -> None:
        """``TB-PASS``/``TB-FAIL`` lines are attributed per file, and exit 1 still fails.

        The two scripts are deliberately asymmetric: the passing one prints two markers
        and exits 0; the failing one prints one marker of each kind and exits 1. That
        makes the counts discriminating in both directions — ``(2,2,0)`` cannot come
        from an exit-status fallback, and ``(2,1,1)`` is exactly the marker-only
        count, so the assertion pins marker attribution, not exit-status honouring
        (the FAIL status is where the exit cross-check shows). The case names are
        asserted verbatim because ``_summarize_markers`` keeps ``bad:two boom`` as
        the whole case name and message; a naive ``split(':')`` would truncate it.

        The run-level FAIL assertion confirms per-suite failures aggregate rather than
        staying local to the suite object.
        """
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
        """A script with no markers reports exactly one synthetic case, not zero or many.

        ``(total, pass) == (1, 1)`` with ``cases == []`` is the documented
        unstructured-fallback shape: the exit status decides, and the case list stays
        empty because nothing was actually named. The two failure modes this rules out
        are a phantom 0/0 PASS (a run that reports success while counting nothing) and a
        fabricated case named after the file.

        The ``summary_excerpt`` assertion pins that the excerpt comes from the script's
        own output, not from the adapter's own name — an excerpt containing
        ``shell_test`` would mean the diagnostic text was generated rather than observed,
        which is precisely what an operator would misread as evidence.
        """
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
        """A hostile parent env does not change what the script observes.

        The fixture poisons the *test process'* environment, then runs a suite that
        reports on its own variables. Two different assertion styles are needed because
        bash re-creates some of them: plain names (``ZDOTDIR``, ``BASH_ENV``, ``ENV``,
        ``CDPATH``, ``GLOBIGNORE``) must be empty inside the script, whereas
        ``SHELLOPTS``/``BASHOPTS`` always exist there and list the shell's own options —
        so the probe checks that the *parent's* bundles (``xtrace``, ``errexit``,
        ``nocaseglob``, ``globstar``) did not come along. Asserting emptiness on those
        two would be wrong, not stricter.

        ``TB_POISONED`` closes the loop: ``BASH_ENV``/``ENV`` point at a script that
        exports it, so if an rc file had been sourced the marker would appear even
        though the variable itself was scrubbed. That is a different failure mode from
        inheritance and the only case that would catch a "scrub the name, still source
        the file" bug.

        The saved/restored ``os.environ`` block is mandatory: a leaked ``BASH_ENV`` here
        would silently alter every later shell test in the process.
        """
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
        """A declared file that does not exist is a SKIP with a named reason.

        Without this, a manifest naming a renamed script would fail the suite and read
        as a product regression. The reason string is asserted because a bare SKIP is
        indistinguishable from a suite that skipped for an unrelated cause — the message
        is the only part an operator can act on.
        """
        run = self._run(
            '[[suite]]\nname = "gone"\nadapter = "shell_test"\nfiles = ["tests/absent_test.sh"]\n'
        )
        suite = run.suites[0]
        self.assertEqual(suite.status, RunStatus.SKIP)
        self.assertIn("required path missing", suite.summary_excerpt)

    def test_missing_interpreter_skips_with_named_reason(self) -> None:
        """A pinned interpreter that does not exist SKIPs, naming the path tried.

        This is the counterpart to the absolute-path requirement: pinning is only safe
        if a missing pin degrades instead of crashing or, worse, falling back to
        ``bash`` from PATH and reporting a pass for a suite that never ran where the
        manifest said. The path is checked in the excerpt for the same reason as above —
        "skipped" alone would not say which interpreter was missing.
        """
        _write(self.proj / "a_test.sh", "exit 0\n", executable=True)
        run = self._run(
            '[[suite]]\nname = "x"\nadapter = "shell_test"\nfiles = ["a_test.sh"]\n'
            'interpreter = "/nonexistent/bash9"\n'
        )
        suite = run.suites[0]
        self.assertEqual(suite.status, RunStatus.SKIP)
        self.assertIn("/nonexistent/bash9", suite.summary_excerpt)


class CompletionTests(unittest.TestCase):
    """Generated scripts cover the CLI contract and actually run on the target shells."""

    def test_every_visible_command_and_option_appears_in_both_scripts(self) -> None:
        """Everything ``build_parser`` exposes is present in both emitted scripts.

        The expected set is derived from the parser at test time, so adding a subcommand
        or flag cannot silently fall out of completion — the failure message names the
        command and option, which is the only way to find it among the full
        command/option set.

        The comparison is a substring test, so it is a *presence* check, not a wiring
        check: a name appearing in a comment would satisfy it. Wiring is what the two
        behavioural tests below assert; this one catches the far more common defect, a
        new command that generation never learned about.
        """
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
        """``_worker`` is a private entry point and must not be offered to users.

        It is registered with ``help=argparse.SUPPRESS``, and ``shell_completion``
        filters suppressed choices; advertising it would invite users to invoke a
        detached-worker protocol by hand. The check is over both scripts because the two
        generators collect hidden names independently.
        """
        self.assertNotIn("_worker", render("bash"))
        self.assertNotIn("_worker", render("zsh"))

    def test_profile_candidates_come_from_the_schema_contract(self) -> None:
        """``--profile`` values are read from ``ProfileName.STANDARD``, not a copy.

        The assertion is scoped to the single ``case`` line that answers ``--profile``,
        so a profile name appearing elsewhere in the script cannot satisfy it. This is
        the schema-to-shell link: a new profile must reach completion without anyone
        editing the template, and a removed one must disappear.
        """
        script = render("bash")
        self.assertIn("--profile)", script)
        profile_line = next(line for line in script.splitlines() if "--profile)" in line)
        for profile in ProfileName.STANDARD:
            self.assertIn(profile, profile_line)

    def test_bash_32_syntax_and_candidate_behaviour(self) -> None:
        """The bash script parses on 3.2 and produces the candidates it claims to.

        Two layers. ``bash -n`` catches syntax the system bash rejects — macOS ships
        3.2.57, which lacks ``mapfile``, associative arrays and ``${x,,}``, so a modern
        bash on the developer's PATH would hide the break. The completion harness then
        *sources* the script in a clean bash and drives ``_testbench`` directly with a
        hand-set ``COMP_WORDS``/``COMP_CWORD``, which is the only faithful way to test
        completion without an interactive TTY.

        The assertions discriminate different code paths, not variants of one: prefix
        matching of a command (``ru`` → ``run``), prefix matching of an option
        (``--pro``), value completion with an *empty* current word (must be exactly
        ``ProfileName.STANDARD``, in order — the equality, not a subset, is what proves
        the value list is complete), a second-level command (``bench c`` → ``compare``),
        and a second-level option (``bench run --rep``). A script that only handled
        top-level words would pass the first two and fail the last two.

        ``--noprofile --norc`` on the harness shell keeps the caller's rc files out, for
        the same reason the product scrubs them.
        """
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
        """The zsh script parses, autoloads from an ``fpath``, and offers the profiles.

        Conditional on zsh being present (``skipTest``), which is an honest gap: this
        half of the contract is unverified on a host without zsh rather than silently
        passing.

        ``compadd`` and ``_describe`` are stubbed before the body runs, so only the
        candidate list is asserted — zsh's own display, grouping and matching behaviour
        are *not* under test, and trying to assert them here would couple the test to a
        zsh release. The stub still exercises the real dispatch: ``words[2]`` is the
        subcommand and ``CURRENT=4`` points at the value being completed, so a script
        that mis-indexed ``words`` would emit the wrong list.

        ``-f`` skips rc files for the same reason as the product's own zsh pin.
        """
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
        """``testbench completion <shell>`` emits the script and nothing else.

        ``TESTBENCH_HOME`` points at a fresh empty directory, and the assertions are
        exit 0, a shell-specific marker line, and *empty stderr*. That establishes the
        generation path needs no initialised store; it does not directly prove nothing
        was written (the directory is empty before and after either way). The mechanism
        is in ``cli.main``, which returns from the ``completion`` branch before
        constructing ``TestBenchStore`` — so a future refactor that moved the branch
        inside the store block would still pass here and needs the code-level guard.

        ``PY`` is ``sys.executable`` with the parent env, so this also pins that the
        ``-m`` entry point works, i.e. the package is importable in the interpreter
        running the suite.
        """
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
