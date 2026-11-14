"""Discovery and native-adapter tests (TDD: written before implementation)."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
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
        self.assertTrue(suite.allow_repo_write, "cargo needs target/ writes")
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
            suite.command[:6], ["uv", "run", "--offline", "--with", "pyte", "python"]
        )
        self.assertEqual(suite.command[6], "bench/tui/cobol/test_cobol.py")
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
