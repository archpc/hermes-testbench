"""Discovery and native-adapter tests (TDD: written before implementation)."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from hermes_testbench.discovery import DEFAULT_INTERPRETER, detect_git_metadata, discover_project


def _tmpdir(prefix: str = "tb-test-") -> Path:
    return Path(tempfile.mkdtemp(prefix=prefix, dir=os.environ.get("TMPDIR")))


def _run_git(cwd: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-c", "user.name=TB", "-c", "user.email=tb@example.invalid", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise AssertionError(f"git {' '.join(args)} failed: {proc.stderr}")
    return proc.stdout.strip()


def _init_repo(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    _run_git(root, "init", "-q", "-b", "main")
    (root / "tracked.txt").write_text("one\n", encoding="utf-8")
    _run_git(root, "add", "tracked.txt")
    _run_git(root, "commit", "-q", "-m", "initial")


class DetectGitMetadataTests(unittest.TestCase):
    def test_reports_short_sha_and_branch_for_clean_repo(self) -> None:
        root = _tmpdir()
        _init_repo(root)
        revision, branch = detect_git_metadata(root)
        expected = _run_git(root, "rev-parse", "--short", "HEAD")
        self.assertEqual(revision, expected)
        self.assertEqual(len(revision), len(expected))
        self.assertEqual(branch, "main")

    def test_marks_dirty_only_for_uncommitted_tracked_changes(self) -> None:
        root = _tmpdir()
        _init_repo(root)
        (root / "tracked.txt").write_text("changed\n", encoding="utf-8")
        revision, branch = detect_git_metadata(root)
        self.assertTrue(revision.endswith("+dirty"), revision)
        self.assertEqual(revision.count("+dirty"), 1)
        self.assertEqual(branch, "main")

    def test_ignores_untracked_files(self) -> None:
        root = _tmpdir()
        _init_repo(root)
        (root / "untracked.txt").write_text("new\n", encoding="utf-8")
        revision, _branch = detect_git_metadata(root)
        self.assertFalse(revision.endswith("+dirty"), revision)

    def test_non_repository_directory_is_unversioned(self) -> None:
        root = _tmpdir()
        self.assertEqual(detect_git_metadata(root), ("unversioned", ""))

    def test_broken_git_pointer_degrades_cleanly(self) -> None:
        root = _tmpdir()
        (root / ".git").write_text("gitdir: /nonexistent/definitely-missing\n", encoding="utf-8")
        self.assertEqual(detect_git_metadata(root), ("unversioned", ""))


MANIFEST_TOML = """
[project]
name = "demo-project"
languages = ["python"]

[[suite]]
name = "custom-unit"
adapter = "python_unittest"
profile = "unit"
command = ["python3", "-m", "unittest", "discover", "-s", "tests"]
timeout_s = 42
cwd = "."
needs_tmux = true
allow_network = false
allow_repo_write = false
billable = false
required_paths = ["tests"]
required_binaries = ["python3"]
description = "manifest-defined unit suite"
warmup_runs = 2
repeat_runs = 3

[suite.env]
FOO = "bar"

[[suite]]
name = "net-suite"
adapter = "shell"
profile = "integration"
command = ["/usr/bin/true"]
allow_network = true
"""


def _write_manifest(root: Path, text: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / ".testbench.toml"
    path.write_text(text, encoding="utf-8")
    return path


class DiscoverProjectManifestTests(unittest.TestCase):
    def test_parses_local_manifest_as_custom_and_untrusted(self) -> None:
        root = _tmpdir()
        manifest = _write_manifest(root, MANIFEST_TOML)
        spec = discover_project(root)

        self.assertEqual(spec.name, "demo-project")
        self.assertEqual(spec.root, str(root.resolve()))
        self.assertEqual(spec.manifest_path, str(manifest.resolve()))
        self.assertEqual(len(spec.manifest_sha256 or ""), 64)
        self.assertFalse(spec.trusted)
        self.assertEqual([s.name for s in spec.suites], ["custom-unit", "net-suite"])

        unit = spec.suites[0]
        self.assertTrue(unit.custom_from_manifest)
        self.assertEqual(unit.adapter, "python_unittest")
        self.assertEqual(unit.profile, "unit")
        self.assertEqual(unit.command[:3], ["python3", "-m", "unittest"])
        self.assertEqual(unit.timeout_s, 42)
        self.assertTrue(unit.needs_tmux)
        self.assertEqual(unit.env_overrides, {"FOO": "bar"})
        self.assertEqual(unit.required_paths, ["tests"])
        self.assertEqual(unit.required_binaries, ["python3"])
        self.assertEqual(unit.description, "manifest-defined unit suite")
        self.assertEqual(unit.warmup_runs, 2)
        self.assertEqual(unit.repeat_runs, 3)

        self.assertTrue(spec.suites[1].allow_network)
        self.assertTrue(spec.suites[1].custom_from_manifest)

    def test_manifest_is_trusted_only_after_store_trust(self) -> None:
        from hermes_testbench.store import TestBenchStore

        root = _tmpdir()
        _write_manifest(root, MANIFEST_TOML)
        with TestBenchStore(home=_tmpdir("tb-home-")) as store:
            self.assertFalse(discover_project(root, store=store).trusted)
            store.trust_project(root)
            self.assertTrue(discover_project(root, store=store).trusted)
            _write_manifest(root, MANIFEST_TOML + "\n# tampered\n")
            self.assertFalse(discover_project(root, store=store).trusted)

    def test_external_manifest_is_not_marked_custom(self) -> None:
        root = _tmpdir()
        ext_dir = _tmpdir("tb-ext-")
        ext = ext_dir / "suite.toml"
        ext.write_text(
            "\n".join(
                [
                    "[project]",
                    'name = "external-name"',
                    "",
                    "[[suite]]",
                    'name = "ext-shell"',
                    'adapter = "shell"',
                    'profile = "unit"',
                    'command = ["/usr/bin/true"]',
                ]
            ),
            encoding="utf-8",
        )
        spec = discover_project(root, external_manifest=ext)
        self.assertEqual(spec.name, "external-name")
        self.assertEqual(spec.manifest_path, str(ext.expanduser().resolve()))
        self.assertFalse(spec.trusted)
        self.assertEqual(len(spec.suites), 1)
        self.assertFalse(spec.suites[0].custom_from_manifest)

    def test_oversized_manifest_is_rejected(self) -> None:
        root = _tmpdir()
        filler = "#" * (256 * 1024 + 16)
        _write_manifest(root, MANIFEST_TOML + "\n" + filler)
        with self.assertRaises(ValueError):
            discover_project(root)

    def test_invalid_toml_is_rejected_with_value_error(self) -> None:
        root = _tmpdir()
        _write_manifest(root, "[[suite]\nname = oops\n")
        with self.assertRaises(ValueError):
            discover_project(root)

    def test_string_command_is_rejected(self) -> None:
        root = _tmpdir()
        _write_manifest(
            root,
            "\n".join(
                [
                    "[[suite]]",
                    'name = "bad"',
                    'adapter = "shell"',
                    'profile = "unit"',
                    'command = "echo hi"',
                ]
            ),
        )
        with self.assertRaises(ValueError):
            discover_project(root)

    def test_unknown_adapter_is_rejected(self) -> None:
        root = _tmpdir()
        _write_manifest(
            root,
            "\n".join(
                [
                    "[[suite]]",
                    'name = "bad"',
                    'adapter = "totally_made_up"',
                    'profile = "unit"',
                    'command = ["true"]',
                ]
            ),
        )
        with self.assertRaises(ValueError):
            discover_project(root)

    def test_invalid_profile_is_rejected(self) -> None:
        root = _tmpdir()
        _write_manifest(
            root,
            "\n".join(
                [
                    "[[suite]]",
                    'name = "bad"',
                    'adapter = "shell"',
                    'profile = "not-a-profile"',
                    'command = ["true"]',
                ]
            ),
        )
        with self.assertRaises(ValueError):
            discover_project(root)

    def test_unknown_empty_directory_yields_no_suites(self) -> None:
        root = _tmpdir()
        (root / "README.md").write_text("nothing to run\n", encoding="utf-8")
        spec = discover_project(root)
        self.assertEqual(spec.suites, [])
        self.assertEqual(spec.languages, [])
        self.assertIsNone(spec.manifest_path)


class DiscoverProjectZeroConfigTests(unittest.TestCase):
    def test_rust_project_uses_cargo_unit_profile(self) -> None:
        root = _tmpdir()
        (root / "Cargo.toml").write_text(
            '[package]\nname = "demo"\nversion = "0.1.0"\n', encoding="utf-8"
        )
        spec = discover_project(root)
        self.assertIn("rust", spec.languages)
        self.assertEqual(len(spec.suites), 1)
        suite = spec.suites[0]
        self.assertEqual(suite.adapter, "cargo")
        self.assertEqual(suite.profile, "unit")
        self.assertEqual(suite.command, ["cargo", "test", "--offline", "--workspace"])
        self.assertIn("cargo", suite.required_binaries)
        self.assertGreaterEqual(suite.timeout_s, 60)
        self.assertFalse(suite.allow_repo_write, "cargo uses disposable CARGO_TARGET_DIR in sandbox")
        self.assertFalse(suite.custom_from_manifest)

    def test_python_project_falls_back_to_unittest_without_pytest(self) -> None:
        root = _tmpdir()
        (root / "tests").mkdir()
        (root / "tests" / "test_alpha.py").write_text(
            "import unittest\n\n\nclass AlphaTest(unittest.TestCase):\n"
            "    def test_ok(self) -> None:\n        self.assertTrue(True)\n",
            encoding="utf-8",
        )
        spec = discover_project(root)
        self.assertIn("python", spec.languages)
        suite = spec.suites[0]
        self.assertEqual(suite.adapter, "python_unittest")
        self.assertEqual(
            suite.command[1:], ["-m", "unittest", "discover", "-s", "tests", "-v"]
        )
        self.assertEqual(suite.command[0], str(DEFAULT_INTERPRETER))
        self.assertFalse(suite.needs_tmux)

    def test_python_project_prefers_project_venv_interpreter(self) -> None:
        root = _tmpdir()
        (root / "tests").mkdir()
        (root / "tests" / "test_alpha.py").write_text("import unittest\n", encoding="utf-8")
        venv_bin = root / "venv" / "bin"
        venv_bin.mkdir(parents=True)
        (venv_bin / "python").symlink_to(DEFAULT_INTERPRETER)

        spec = discover_project(root)
        suite = spec.suites[0]
        self.assertEqual(suite.command[0], str(root.resolve() / "venv" / "bin" / "python"))
        self.assertEqual(suite.adapter, "python_unittest")

    def test_python_project_uses_pytest_when_venv_has_it(self) -> None:
        root = _tmpdir()
        (root / "tests").mkdir()
        (root / "tests" / "test_alpha.py").write_text("def test_ok():\n    pass\n", encoding="utf-8")
        venv_bin = root / "venv" / "bin"
        venv_bin.mkdir(parents=True)
        stub = venv_bin / "python"
        stub.write_text(
            "#!/bin/sh\n"
            'case "$1" in\n'
            '  -c) exit 0 ;;\n'
            "esac\n"
            "exit 0\n",
            encoding="utf-8",
        )
        stub.chmod(0o755)

        spec = discover_project(root)
        suite = spec.suites[0]
        self.assertEqual(suite.adapter, "python_pytest")
        self.assertEqual(suite.command[0], str(root.resolve() / "venv" / "bin" / "python"))
        self.assertIn("-m", suite.command)
        self.assertIn("pytest", suite.command)

    def test_tmux_integration_test_marks_suite_needs_tmux(self) -> None:
        root = _tmpdir()
        (root / "tests").mkdir()
        (root / "tests" / "test_alpha.py").write_text("import unittest\n", encoding="utf-8")
        (root / "tests" / "test_tmux_integration.py").write_text(
            "import unittest\n", encoding="utf-8"
        )
        spec = discover_project(root)
        self.assertTrue(any(s.needs_tmux for s in spec.suites))

    def test_hermes_tmux_marker_file_marks_suite_needs_tmux(self) -> None:
        root = _tmpdir()
        (root / "tests").mkdir()
        (root / "tests" / "test_alpha.py").write_text("import unittest\n", encoding="utf-8")
        (root / "hermes_tmux").write_text("#!/bin/sh\n", encoding="utf-8")
        spec = discover_project(root)
        self.assertTrue(any(s.needs_tmux for s in spec.suites))

    def test_pyproject_only_project_is_still_detected(self) -> None:
        root = _tmpdir()
        (root / "pyproject.toml").write_text(
            '[project]\nname = "demo"\nversion = "0.1.0"\n', encoding="utf-8"
        )
        spec = discover_project(root)
        self.assertIn("python", spec.languages)
        suite = spec.suites[0]
        self.assertEqual(suite.adapter, "python_unittest")
        self.assertEqual(suite.command[0], str(DEFAULT_INTERPRETER))
        self.assertEqual(suite.command[1:3], ["-m", "unittest"])

    def test_polyglot_project_reports_every_language(self) -> None:
        root = _tmpdir()
        (root / "Cargo.toml").write_text('[package]\nname = "demo"\n', encoding="utf-8")
        (root / "pyproject.toml").write_text('[project]\nname = "demo"\n', encoding="utf-8")
        spec = discover_project(root)
        self.assertEqual(spec.languages, ["python", "rust"])
        self.assertGreaterEqual(len(spec.suites), 2)
        adapters = {s.adapter for s in spec.suites}
        self.assertEqual(adapters, {"cargo", "python_unittest"})

    def test_git_revision_and_branch_are_recorded(self) -> None:
        root = _tmpdir()
        (root / "Cargo.toml").write_text('[package]\nname = "demo"\n', encoding="utf-8")
        _init_repo(root)
        spec = discover_project(root)
        self.assertEqual(spec.revision, _run_git(root, "rev-parse", "--short", "HEAD"))
        self.assertEqual(spec.branch, "main")
        self.assertTrue(spec.revision)


class DiscoverProjectMultiLanguageTests(unittest.TestCase):
    def test_cobol_sources_build_optimized_and_debug_into_tmpdir(self) -> None:
        root = _tmpdir()
        (root / "hello.cob").write_text("       IDENTIFICATION DIVISION.\n", encoding="utf-8")
        spec = discover_project(root)
        self.assertEqual(spec.languages, ["cobol"])
        suite = spec.suites[0]
        self.assertEqual(suite.adapter, "cobol")
        self.assertEqual(suite.profile, "unit")
        self.assertEqual(suite.name, "cobol-unit")
        joined = " ".join(suite.command)
        for needle in ("cobc", "-O2", "-debug", "$TMPDIR", "hello.cob"):
            self.assertIn(needle, joined)
        self.assertIn("cobc", suite.required_binaries)

    def test_cobol_tui_harness_uses_uv_with_pyte(self) -> None:
        root = _tmpdir()
        harness = root / "bench" / "tui" / "cobol" / "test_cobol.py"
        harness.parent.mkdir(parents=True)
        harness.write_text("print('harness')\n", encoding="utf-8")
        spec = discover_project(root)
        self.assertEqual(spec.languages, ["cobol"])
        suite = spec.suites[0]
        self.assertEqual(suite.adapter, "cobol")
        self.assertEqual(
            suite.command,
            [
                "uv",
                "run",
                "--offline",
                "--with",
                "pyte",
                "python",
                "-m",
                "unittest",
                "discover",
                "-s",
                "bench/tui/cobol",
                "-v",
            ],
        )
        self.assertIn("uv", suite.required_binaries)

    def test_c_selftest_gets_asan_ubsan_suite(self) -> None:
        root = _tmpdir()
        (root / "queue_selftest.c").write_text("int main(void){return 0;}\n", encoding="utf-8")
        spec = discover_project(root)
        self.assertEqual(spec.languages, ["c"])
        suite = spec.suites[0]
        self.assertEqual(suite.adapter, "c_sanitizer")
        self.assertEqual(suite.profile, "security")
        joined = " ".join(suite.command)
        for needle in ("clang", "-O1", "-g", "-fsanitize=address,undefined", "$TMPDIR", "queue_selftest.c"):
            self.assertIn(needle, joined)
        self.assertIn("clang", suite.required_binaries)

    def test_verifier_scripts_map_to_re_project_and_shell(self) -> None:
        root = _tmpdir()
        scripts = root / "scripts"
        scripts.mkdir()
        for name in ("verify_docs.py", "verify_claims.py", "regress_release.py", "check-wiring.py"):
            (scripts / name).write_text("print('PASS x')\n", encoding="utf-8")
        spec = discover_project(root)
        by_name = {s.name: s for s in spec.suites}
        self.assertEqual(
            set(by_name), {"verify-docs", "verify-claims", "regress-release", "check-wiring"}
        )
        self.assertEqual(by_name["verify-docs"].adapter, "re_project")
        self.assertEqual(by_name["verify-claims"].adapter, "re_project")
        self.assertEqual(by_name["regress-release"].adapter, "shell")
        self.assertEqual(by_name["check-wiring"].adapter, "shell")
        for suite in spec.suites:
            self.assertEqual(Path(suite.command[0]).name, DEFAULT_INTERPRETER.name)
            self.assertTrue(suite.command[1].startswith("scripts/"), suite.command)
            self.assertEqual(suite.profile, "regression")

    def test_sqlite_databases_get_integrity_suite_and_skip_noise_dirs(self) -> None:
        import sqlite3

        root = _tmpdir()
        con = sqlite3.connect(str(root / "app.db"))
        con.execute("CREATE TABLE t (x INTEGER)")
        con.commit()
        con.close()
        (root / "cache.sqlite").write_text("", encoding="utf-8")
        noise = root / "node_modules" / "pkg"
        noise.mkdir(parents=True)
        (noise / "vendor.db").write_text("", encoding="utf-8")

        spec = discover_project(root)
        self.assertEqual(spec.languages, ["sqlite"])
        suite = spec.suites[0]
        self.assertEqual(suite.adapter, "sqlite_probe")
        self.assertEqual(suite.profile, "integration")
        self.assertIn("integrity_check", " ".join(suite.command))
        db_paths = suite.command[3:]
        self.assertEqual(sorted(db_paths), ["app.db", "cache.sqlite"])

    def test_full_polyglot_project_reports_every_language_and_suite(self) -> None:
        root = _tmpdir()
        (root / "Cargo.toml").write_text('[package]\nname = "demo"\n', encoding="utf-8")
        (root / "pyproject.toml").write_text('[project]\nname = "demo"\n', encoding="utf-8")
        (root / "hello.cob").write_text("       IDENTIFICATION DIVISION.\n", encoding="utf-8")
        (root / "probe_selftest.c").write_text("int main(void){return 0;}\n", encoding="utf-8")
        (root / "scripts").mkdir()
        (root / "scripts" / "verify_docs.py").write_text("print('PASS x')\n", encoding="utf-8")
        (root / "app.db").write_bytes(b"")
        spec = discover_project(root)
        self.assertEqual(spec.languages, ["c", "cobol", "python", "rust", "sqlite"])
        adapters = {s.adapter for s in spec.suites}
        self.assertEqual(
            adapters,
            {"cargo", "python_unittest", "cobol", "c_sanitizer", "re_project", "sqlite_probe"},
        )
        names = [s.name for s in spec.suites]
        self.assertEqual(len(names), len(set(names)), names)

    def test_hidden_and_vendor_directories_are_ignored(self) -> None:
        root = _tmpdir()
        (root / ".git").mkdir()
        (root / ".git" / "hook.cob").write_text("x\n", encoding="utf-8")
        (root / "vendor").mkdir()
        (root / "vendor" / "lib_selftest.c").write_text("int main(void){return 0;}\n", encoding="utf-8")
        spec = discover_project(root)
        self.assertEqual(spec.suites, [])


def _isolated_result(
    command,
    *,
    exit_code: int | None = 0,
    stdout_text: str = "",
    stderr_text: str = "",
    timed_out: bool = False,
    cancelled: bool = False,
):
    from hermes_testbench.isolation import IsolatedExecResult

    return IsolatedExecResult(
        command=list(command),
        cwd=".",
        exit_code=exit_code,
        signal_num=None,
        duration_ms=1.5,
        timed_out=timed_out,
        cancelled=cancelled,
        stdout_text=stdout_text,
        stderr_text=stderr_text,
        stdout_path="/nonexistent/stdout.log",
        stderr_path="/nonexistent/stderr.log",
        sanitized_path="/nonexistent/sanitized.log",
        isolation_meta={"seatbelt": "active"},
    )


def _suite(**kwargs):
    from hermes_testbench.schema import SuiteSpec

    params = {
        "name": "unit",
        "adapter": "shell",
        "profile": "unit",
        "command": ["/usr/bin/true"],
    }
    params.update(kwargs)
    return SuiteSpec(**params)


class ExecuteSuitePolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        from unittest import mock

        import hermes_testbench.adapters as adapters_mod

        self.adapters = adapters_mod
        self.root = _tmpdir()
        self.evidence = _tmpdir("tb-ev-")
        self._patcher = mock.patch.object(
            adapters_mod, "run_isolated_command", side_effect=self._fake_run
        )
        self.run_mock = self._patcher.start()
        self.addCleanup(self._patcher.stop)
        self.fake_kwargs: dict = {}

    def _fake_run(self, command, **kwargs):
        self.fake_kwargs = dict(kwargs)
        return _isolated_result(command, stdout_text="TB-PASS shell:unit\n1 passed, 0 skipped, 0 failed\n")

    def _exec(self, suite, **kwargs):
        return self.adapters.execute_suite(
            suite,
            project_root=self.root,
            evidence_dir=self.evidence,
            run_id="tb_policy",
            **kwargs,
        )

    def test_billable_suite_without_approval_is_blocked_before_spawn(self) -> None:
        from hermes_testbench.schema import RunStatus

        result, usage = self._exec(_suite(billable=True))
        self.assertEqual(result.status, RunStatus.BLOCKED)
        self.assertEqual(result.blocked_count, 1)
        self.assertEqual(result.total_count, 1)
        self.assertIsNone(usage)
        self.assertEqual(self.run_mock.call_count, 0)
        self.assertIn("billable", result.summary_excerpt.lower())

    def test_billable_suite_with_approval_but_no_token_budget_is_blocked(self) -> None:
        from hermes_testbench.schema import RunStatus

        result, _usage = self._exec(_suite(billable=True), allow_billable=True, max_tokens=0)
        self.assertEqual(result.status, RunStatus.BLOCKED)
        self.assertEqual(self.run_mock.call_count, 0)

    def test_billable_suite_runs_when_approved_with_budget(self) -> None:
        from hermes_testbench.schema import RunStatus

        result, _usage = self._exec(
            _suite(billable=True), allow_billable=True, max_tokens=1000
        )
        self.assertNotEqual(result.status, RunStatus.BLOCKED)
        self.assertEqual(self.run_mock.call_count, 1)

    def test_untrusted_custom_manifest_network_request_is_blocked(self) -> None:
        from hermes_testbench.schema import FailureKind, RunStatus

        suite = _suite(custom_from_manifest=True, allow_network=True)
        result, _usage = self._exec(suite, trusted_manifest=False)
        self.assertEqual(result.status, RunStatus.BLOCKED)
        self.assertEqual(result.blocked_count, 1)
        self.assertEqual(len(result.failures), 1)
        self.assertEqual(result.failures[0].kind, FailureKind.UNTRUSTED_MANIFEST)
        self.assertEqual(self.run_mock.call_count, 0)

    def test_untrusted_custom_manifest_repo_write_is_blocked(self) -> None:
        from hermes_testbench.schema import RunStatus

        suite = _suite(custom_from_manifest=True, allow_repo_write=True)
        result, _usage = self._exec(suite, trusted_manifest=False)
        self.assertEqual(result.status, RunStatus.BLOCKED)
        self.assertEqual(self.run_mock.call_count, 0)

    def test_trusted_manifest_allows_network_and_repo_write(self) -> None:
        from hermes_testbench.schema import RunStatus

        suite = _suite(custom_from_manifest=True, allow_network=True, allow_repo_write=True)
        result, _usage = self._exec(suite, trusted_manifest=True)
        self.assertEqual(result.status, RunStatus.PASS)
        self.assertEqual(self.run_mock.call_count, 1)

    def test_non_manifest_suite_is_not_gated_on_network(self) -> None:
        from hermes_testbench.schema import RunStatus

        result, _usage = self._exec(_suite(allow_network=True), trusted_manifest=False)
        self.assertEqual(result.status, RunStatus.PASS)
        self.assertEqual(self.run_mock.call_count, 1)

    def test_missing_required_binary_skips_suite(self) -> None:
        from hermes_testbench.schema import RunStatus

        result, usage = self._exec(
            _suite(required_binaries=["definitely-not-a-real-binary-xyz"])
        )
        self.assertEqual(result.status, RunStatus.SKIP)
        self.assertEqual(result.skip_count, 1)
        self.assertEqual(result.total_count, 1)
        self.assertIsNone(usage)
        self.assertEqual(self.run_mock.call_count, 0)
        self.assertIn("definitely-not-a-real-binary-xyz", result.summary_excerpt)

    def test_missing_required_path_skips_suite(self) -> None:
        from hermes_testbench.schema import RunStatus

        result, _usage = self._exec(_suite(required_paths=["missing/dir"]))
        self.assertEqual(result.status, RunStatus.SKIP)
        self.assertEqual(self.run_mock.call_count, 0)

    def test_suite_without_command_skips(self) -> None:
        from hermes_testbench.schema import RunStatus

        result, _usage = self._exec(_suite(command=[]))
        self.assertEqual(result.status, RunStatus.SKIP)
        self.assertEqual(self.run_mock.call_count, 0)

    def test_policy_result_records_repro_command_and_timestamps(self) -> None:
        result, _usage = self._exec(_suite(billable=True))
        self.assertIn("true", result.repro_command)
        self.assertTrue(result.started_at)
        self.assertTrue(result.finished_at)
        self.assertEqual(result.adapter, "shell")


class AdapterParserContractTests(unittest.TestCase):
    """Captured-text parser behaviour for runners without local toolchains."""

    def test_unittest_ok_with_skips(self) -> None:
        from hermes_testbench.adapters.parsers import parse_python_unittest

        outcome = parse_python_unittest("Ran 4 tests in 0.010s\n\nOK (skipped=2)\n")
        self.assertEqual((outcome.total, outcome.passed, outcome.skipped), (4, 2, 2))
        self.assertTrue(outcome.structured)

    def test_unittest_failed_clause_and_error_headers(self) -> None:
        from hermes_testbench.adapters.parsers import parse_python_unittest

        text = (
            "test_a (mod.T.test_a) ... ok\n"
            "test_b (mod.T.test_b) ... FAIL\n"
            "test_c (mod.T.test_c) ... ERROR\n"
            "======================================================================\n"
            "FAIL: test_b (mod.T.test_b)\n"
            "----------------------------------------------------------------------\n"
            "Traceback (most recent call last):\n"
            'AssertionError: 1 != 2\n'
            "======================================================================\n"
            "ERROR: test_c (mod.T.test_c)\n"
            "----------------------------------------------------------------------\n"
            "RuntimeError: boom\n"
            "----------------------------------------------------------------------\n"
            "Ran 3 tests in 0.020s\n\n"
            "FAILED (failures=1, errors=1)\n"
        )
        outcome = parse_python_unittest(text)
        self.assertEqual(outcome.total, 3)
        self.assertEqual(outcome.passed, 1)
        self.assertEqual((outcome.failed, outcome.errors), (1, 1))
        self.assertEqual([f["case_name"] for f in outcome.failures], ["test_b (mod.T.test_b)", "test_c (mod.T.test_c)"])
        self.assertIn("1 != 2", outcome.failures[0]["message"])

    def test_pytest_summary_and_failed_lines(self) -> None:
        from hermes_testbench.adapters.parsers import parse_python_pytest

        text = (
            "tests/test_api.py::test_ok PASSED\n"
            "tests/test_api.py::test_bad FAILED\n"
            "========================= 1 failed, 1 passed, 2 skipped in 0.11s =========================\n"
            "FAILED tests/test_api.py::test_bad - AssertionError: 500 != 200\n"
        )
        outcome = parse_python_pytest(text)
        self.assertEqual((outcome.total, outcome.passed, outcome.failed, outcome.skipped), (4, 1, 1, 2))
        self.assertEqual(outcome.failures[0]["case_name"], "tests/test_api.py::test_bad")

    def test_sqlite_probe_and_pty_markers(self) -> None:
        from hermes_testbench.adapters.parsers import parse_pty_tmux, parse_sqlite_probe

        probe = parse_sqlite_probe("PASS: integrity_check\nFAIL: wal_sync\nSKIP: fts5\n")
        self.assertEqual((probe.total, probe.passed, probe.failed, probe.skipped), (3, 1, 1, 1))
        tty = parse_pty_tmux("\x1b[32mPASS: resize\x1b[0m\nPASS: detach\n")
        self.assertEqual((tty.total, tty.passed), (2, 2))

    def test_asan_report_is_a_single_failure(self) -> None:
        from hermes_testbench.adapters.parsers import parse_c_sanitizer

        text = (
            "=================================================================\n"
            "==49278==ERROR: AddressSanitizer: stack-buffer-overflow on address 0x00016eeee186\n"
            "WRITE of size 1 at 0x00016eeee186 thread T0\n"
            "SUMMARY: AddressSanitizer: stack-buffer-overflow probe.c:2 in main\n"
            "==49278==ABORTING\n"
        )
        outcome = parse_c_sanitizer(text)
        self.assertEqual(outcome.failed, 1)
        self.assertEqual(len(outcome.failures), 1)
        self.assertIn("stack-buffer-overflow", outcome.failures[0]["message"])

    def test_adapter_parser_map_covers_every_known_adapter(self) -> None:
        from hermes_testbench.adapters.parsers import ADAPTER_PARSERS
        from hermes_testbench.discovery import KNOWN_ADAPTERS

        self.assertEqual(set(ADAPTER_PARSERS), set(KNOWN_ADAPTERS))
        self.assertTrue(all(callable(fn) for fn in ADAPTER_PARSERS.values()))


class AdapterExecutionIntegrationTests(unittest.TestCase):
    """Real executions through execute_suite -> isolation, no mocked runner."""

    def setUp(self) -> None:
        import hermes_testbench.adapters as adapters_mod

        self.adapters = adapters_mod
        self.evidence = _tmpdir("tb-ev-int-")

    def _exec(self, suite, root: Path, **kwargs):
        return self.adapters.execute_suite(
            suite,
            project_root=root,
            evidence_dir=self.evidence,
            run_id="tb_integration",
            **kwargs,
        )

    def test_real_unittest_summary_on_stderr_is_parsed(self) -> None:
        from hermes_testbench.schema import RunStatus

        root = _tmpdir()
        (root / "test_calc.py").write_text(
            "import unittest\n\n"
            "class CalcTests(unittest.TestCase):\n"
            "    def test_add(self):\n"
            "        self.assertEqual(1 + 1, 2)\n\n"
            "    def test_broken(self):\n"
            "        self.assertEqual(2 + 2, 5)\n",
            encoding="utf-8",
        )
        suite = _suite(
            name="python-unittest",
            adapter="python_unittest",
            profile="unit",
            command=[sys.executable, "-m", "unittest", "test_calc", "-v"],
        )
        result, usage = self._exec(suite, root)
        self.assertIsNone(usage)
        self.assertEqual(result.status, RunStatus.FAIL)
        self.assertEqual(result.total_count, 2)
        self.assertEqual(result.pass_count, 1)
        self.assertEqual(result.fail_count, 1)
        self.assertEqual(result.exit_code, 1)
        self.assertEqual(len(result.failures), 1)
        self.assertIn("test_broken", result.failures[0].case_name)
        self.assertIn("Ran 2 tests", result.summary_excerpt)

    def test_real_cargo_workspace_sums_every_test_result_block(self) -> None:
        from hermes_testbench.schema import RunStatus

        cargo = shutil.which("cargo") or shutil.which("cargo", path="/opt/homebrew/bin")
        if not cargo:
            self.skipTest("cargo not installed")
        root = _tmpdir()
        (root / "Cargo.toml").write_text(
            '[workspace]\nmembers = ["alpha", "beta"]\nresolver = "2"\n', encoding="utf-8"
        )
        (root / "alpha" / "src").mkdir(parents=True)
        (root / "alpha" / "Cargo.toml").write_text(
            '[package]\nname = "alpha"\nversion = "0.1.0"\nedition = "2021"\n', encoding="utf-8"
        )
        (root / "alpha" / "src" / "lib.rs").write_text(
            "#[test]\nfn alpha_ok() { assert_eq!(1 + 1, 2); }\n", encoding="utf-8"
        )
        (root / "beta" / "src").mkdir(parents=True)
        (root / "beta" / "Cargo.toml").write_text(
            '[package]\nname = "beta"\nversion = "0.1.0"\nedition = "2021"\n', encoding="utf-8"
        )
        (root / "beta" / "src" / "lib.rs").write_text(
            "#[test]\nfn beta_ok() { assert_eq!(2 * 2, 4); }\n\n"
            "#[test]\nfn beta_bad() { assert_eq!(2 + 2, 5); }\n",
            encoding="utf-8",
        )
        suite = _suite(
            name="cargo-workspace",
            adapter="cargo",
            profile="unit",
            command=[cargo, "test", "--offline", "--color", "never"],
            allow_repo_write=True,
            timeout_s=240.0,
            env_overrides={"CARGO_TARGET_DIR": "target"},
        )
        result, _usage = self._exec(suite, root)
        self.assertEqual(result.status, RunStatus.FAIL)
        self.assertEqual(result.total_count, 3)
        self.assertEqual(result.pass_count, 2)
        self.assertEqual(result.fail_count, 1)
        self.assertTrue(any("beta_bad" in f.case_name for f in result.failures), result.failures)

    def test_cargo_parser_sums_multi_crate_blocks_and_names_failures(self) -> None:
        from hermes_testbench.adapters.parsers import parse_cargo

        text = (
            "   Compiling alpha v0.1.0\n"
            "    Finished test [unoptimized]\n"
            "     Running unittests src/lib.rs (target/debug/deps/alpha-1234)\n"
            "running 2 tests\n"
            "test alpha::one ... ok\n"
            "test alpha::two ... ok\n"
            "\n"
            "test result: ok. 2 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.01s\n"
            "\n"
            "     Running unittests src/lib.rs (target/debug/deps/beta-5678)\n"
            "running 3 tests\n"
            "test beta::one ... ok\n"
            "test beta::two ... FAILED\n"
            "test beta::three ... ok\n"
            "\n"
            "test result: FAILED. 2 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.02s\n"
        )
        outcome = parse_cargo(text)
        self.assertTrue(outcome.structured)
        self.assertEqual(outcome.total, 6)
        self.assertEqual(outcome.passed, 4)
        self.assertEqual(outcome.failed, 1)
        self.assertEqual(outcome.skipped, 1)
        self.assertEqual([f["case_name"] for f in outcome.failures], ["beta::two"])

    def test_real_asan_clean_selftest_passes(self) -> None:
        from hermes_testbench.schema import RunStatus

        clang = shutil.which("clang") or "/usr/bin/clang"
        if not Path(clang).exists():
            self.skipTest("clang not installed")
        root = _tmpdir()
        (root / "selftest.c").write_text(
            "#include <stdio.h>\n"
            "static int add(int a, int b) { return a + b; }\n"
            "int main(void) {\n"
            "    if (add(2, 3) != 5) { printf(\"FAIL: add\\n\"); return 1; }\n"
            "    printf(\"SELFTEST OK\\n\");\n"
            "    return 0;\n"
            "}\n",
            encoding="utf-8",
        )
        build = subprocess.run(
            [clang, "-fsanitize=address,undefined", "-g", "-O1", "-o", "selftest", "selftest.c"],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(build.returncode, 0, build.stderr)
        suite = _suite(name="c-selftest", adapter="c_sanitizer", profile="integration", command=["./selftest"])
        result, _usage = self._exec(suite, root)
        self.assertEqual(result.status, RunStatus.PASS)
        self.assertEqual(result.fail_count, 0)
        self.assertGreaterEqual(result.pass_count, 1)

    def test_real_ubsan_diagnostic_fails_run_even_with_zero_exit(self) -> None:
        from hermes_testbench.schema import RunStatus

        clang = shutil.which("clang") or "/usr/bin/clang"
        if not Path(clang).exists():
            self.skipTest("clang not installed")
        root = _tmpdir()
        (root / "ubsan_probe.c").write_text(
            "#include <stdio.h>\n"
            "int main(void) {\n"
            "    int big = 2147483647;\n"
            "    big += 1; /* recoverable UBSan signed overflow: exit stays 0 */\n"
            "    printf(\"value=%d\\n\", big);\n"
            "    return 0;\n"
            "}\n",
            encoding="utf-8",
        )
        build = subprocess.run(
            [clang, "-fsanitize=address,undefined", "-g", "-O1", "-o", "ubsan_probe", "ubsan_probe.c"],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(build.returncode, 0, build.stderr)
        suite = _suite(name="c-ubsan", adapter="c_sanitizer", profile="integration", command=["./ubsan_probe"])
        result, _usage = self._exec(suite, root)
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(result.status, RunStatus.FAIL)
        self.assertGreaterEqual(result.fail_count, 1)
        self.assertTrue(
            any("runtime error" in f.message for f in result.failures), result.failures
        )

    def test_real_cobol_selftest_compiles_and_runs(self) -> None:
        from hermes_testbench.schema import RunStatus

        cobc = shutil.which("cobc") or (shutil.which("cobc", path="/opt/homebrew/bin"))
        if not cobc:
            self.skipTest("cobc not installed")
        root = _tmpdir()
        (root / "selftest.cob").write_text(
            "       IDENTIFICATION DIVISION.\n"
            "       PROGRAM-ID. selftest.\n"
            "       DATA DIVISION.\n"
            "       WORKING-STORAGE SECTION.\n"
            "       01 WS-N PIC 9(2) VALUE 0.\n"
            "       PROCEDURE DIVISION.\n"
            "           ADD 1 TO WS-N\n"
            "           IF WS-N = 1\n"
            "               DISPLAY \"PASS: counter\"\n"
            "           ELSE\n"
            "               DISPLAY \"FAIL: counter\"\n"
            "           END-IF\n"
            "           DISPLAY \"TESTS PASSED: 1 / 1\"\n"
            "           STOP RUN.\n",
            encoding="utf-8",
        )
        build = subprocess.run(
            [cobc, "-x", "-free", "-o", "selftest", "selftest.cob"],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(build.returncode, 0, build.stderr)
        suite = _suite(name="cobol-selftest", adapter="cobol", profile="unit", command=["./selftest"])
        result, _usage = self._exec(suite, root)
        self.assertEqual(result.status, RunStatus.PASS)
        self.assertEqual(result.total_count, 1)
        self.assertEqual(result.pass_count, 1)
        self.assertEqual(result.fail_count, 0)

    def test_real_re_project_skip_line_is_counted_as_skip(self) -> None:
        from hermes_testbench.schema import RunStatus

        root = _tmpdir()
        script = root / "verify_docs.sh"
        script.write_text("#!/bin/sh\necho 'SKIP: browser toolchain unavailable'\nexit 0\n", encoding="utf-8")
        script.chmod(0o755)
        suite = _suite(
            name="re-verify-docs",
            adapter="re_project",
            profile="security",
            command=["/bin/sh", "verify_docs.sh"],
        )
        result, _usage = self._exec(suite, root)
        self.assertEqual(result.status, RunStatus.PASS)
        self.assertEqual(result.total_count, 1)
        self.assertEqual(result.skip_count, 1)
        self.assertEqual(result.fail_count, 0)

    def test_real_re_project_pass_and_fail_lines_are_attributed(self) -> None:
        from hermes_testbench.schema import RunStatus

        root = _tmpdir()
        (root / "verify.sh").write_text(
            "#!/bin/sh\n"
            "echo 'PASS: header anchored'\n"
            "echo 'PASS endpoint table present'\n"
            "echo 'FAIL: stale diagram reference'\n"
            "exit 1\n",
            encoding="utf-8",
        )
        suite = _suite(
            name="re-verify",
            adapter="re_project",
            profile="security",
            command=["/bin/sh", "verify.sh"],
        )
        result, _usage = self._exec(suite, root)
        self.assertEqual(result.status, RunStatus.FAIL)
        self.assertEqual(result.total_count, 3)
        self.assertEqual(result.pass_count, 2)
        self.assertEqual(result.fail_count, 1)
        self.assertEqual([f.case_name for f in result.failures], ["stale diagram reference"])

    def test_agent_eval_never_fabricates_usage_without_emitted_line(self) -> None:
        from hermes_testbench.schema import RunStatus

        root = _tmpdir()
        (root / "eval.sh").write_text("#!/bin/sh\necho '1 passed, 0 skipped, 0 failed'\nexit 0\n", encoding="utf-8")
        suite = _suite(
            name="agent-eval",
            adapter="agent_eval",
            profile="agent-eval",
            command=["/bin/sh", "eval.sh"],
        )
        result, usage = self._exec(suite, root)
        self.assertEqual(result.status, RunStatus.PASS)
        self.assertIsNone(usage)
        self.assertNotIn("usage", result.metrics)
        self.assertNotIn("tokens", result.summary_excerpt.lower())
        self.assertNotIn("cost", result.summary_excerpt.lower())

    def test_agent_eval_uses_only_verbatim_usage_json(self) -> None:
        root = _tmpdir()
        (root / "eval_usage.sh").write_text(
            "#!/bin/sh\n"
            "echo '2 passed, 0 skipped, 0 failed'\n"
            'echo \'TESTBENCH_MODEL_USAGE={"input_tokens": 120, "output_tokens": 30, "cost_usd": 0.0004}\'\n'
            "exit 0\n",
            encoding="utf-8",
        )
        suite = _suite(
            name="agent-eval-usage",
            adapter="agent_eval",
            profile="agent-eval",
            command=["/bin/sh", "eval_usage.sh"],
        )
        result, usage = self._exec(suite, root)
        self.assertEqual(result.total_count, 2)
        self.assertEqual(usage, {"input_tokens": 120, "output_tokens": 30, "cost_usd": 0.0004})
