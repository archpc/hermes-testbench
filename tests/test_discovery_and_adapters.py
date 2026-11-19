"""Discovery, adapter policy, and adapter output parsing — four layers, one file.

The layers are separated by what each can prove, and the split is what tells you where a
failure lives:

* ``DetectGitMetadataTests`` — git enrichment. Every failure mode degrades to
  ``("unversioned", "")``; the tests pin which situations count as failure, because a
  wrong-but-plausible revision silently attributes a run to someone else's commit.
* ``DiscoverProject*Tests`` — manifest and zero-config discovery. The manifest tests are
  about *refusal* (strict validation is a security feature, not tidiness); the zero-config
  tests are about which argv a set of conventions produces.
* ``ExecuteSuitePolicyTests`` / ``AdapterParserContractTests`` — pure policy and pure text.
  The policy class patches ``run_isolated_command`` and asserts ``call_count == 0`` for
  every blocked or skipped case, which is the only way to show that a gate happens *before*
  a spawn rather than merely returning a blocked result.
* ``AdapterExecutionIntegrationTests`` — real toolchains, nothing patched. These call
  ``skipTest`` when ``cargo``/``clang``/``cobc`` is absent, so a green run on a bare machine
  says nothing about them; the skips are explicit and named.

Helpers: ``_tmpdir`` honours ``$TMPDIR``, ``_init_repo``/``_run_git`` build throwaway
repositories with a fixed identity and no global git config, ``_isolated_result`` fabricates
an ``IsolatedExecResult`` for the mocked-runner tests, and ``_suite`` builds a ``SuiteSpec``
whose defaults are the least-privileged ones.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from hermes_testbench.discovery import default_interpreter, detect_git_metadata, discover_project


def _tmpdir(prefix: str = "tb-test-") -> Path:
    """Create a throwaway project directory under the ambient temp dir.

    ``$TMPDIR`` is honoured rather than ``tempfile.gettempdir()`` so the caller's sandbox
    policy applies: these tests build real git repositories and compile C, and a long or
    TMPDIR-excluded path is a common cause of spurious failures. Nothing is cleaned up — the
    directory is disposable by construction.
    """
    return Path(tempfile.mkdtemp(prefix=prefix, dir=os.environ.get("TMPDIR")))


def _run_git(cwd: Path, *args: str) -> str:
    """Run git with an inline identity and no ambient configuration.

    ``-c user.name=TB -c user.email=tb@example.invalid`` makes ``commit`` work on a machine
    with no global git identity, and keeps the test from depending on (or writing to) the
    developer's config. A non-zero exit raises ``AssertionError`` with stderr: a fixture that
    silently produced a half-initialised repo would turn every downstream assertion into a
    confusing false failure.
    """
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
    """Initialise ``root`` as a repository on ``main`` with one committed file.

    The explicit ``-b main`` is used by the dirty/untracked tests, which assert the branch
    name; without it the default could be ``master`` depending on the host's git version and
    config. ``tracked.txt`` exists so a later test can modify a *tracked* file, which is the
    only way to distinguish "dirty" from "untracked".
    """
    root.mkdir(parents=True, exist_ok=True)
    _run_git(root, "init", "-q", "-b", "main")
    (root / "tracked.txt").write_text("one\n", encoding="utf-8")
    _run_git(root, "add", "tracked.txt")
    _run_git(root, "commit", "-q", "-m", "initial")


class DetectGitMetadataTests(unittest.TestCase):
    """Git enrichment, including every way it is allowed to fail.

    The one case not covered here is "directory merely inside another repository": that
    needs a parent repo, and the guard handling it (``--show-toplevel`` compared against
    the resolved root) exists so the answer is not the parent's commit. Everything below
    documents a degradation that keeps a run attributable instead of plausible-but-wrong.
    """

    def test_reports_short_sha_and_branch_for_clean_repo(self) -> None:
        """Clean repository: the reported revision is exactly git's own short SHA.

        Compared against a fresh ``git rev-parse --short HEAD`` rather than a literal so an
        abbreviation-length change does not create a false failure; the length equality that follows
        is the explicit statement of "short" (a full-SHA regression would otherwise show up only as a
        string mismatch). ``branch == "main"`` also pins the fixture's explicit default, which the
        dirty and untracked tests depend on.
        """
        root = _tmpdir()
        _init_repo(root)
        revision, branch = detect_git_metadata(root)
        expected = _run_git(root, "rev-parse", "--short", "HEAD")
        self.assertEqual(revision, expected)
        self.assertEqual(len(revision), len(expected))
        self.assertEqual(branch, "main")

    def test_marks_dirty_only_for_uncommitted_tracked_changes(self) -> None:
        """An edited tracked file yields exactly one ``+dirty`` suffix.

        The revision is compared against a freshly computed ``git rev-parse --short HEAD`` rather
        than a literal, so an abbreviation-length change does not create a false failure; the
        length equality that follows is the explicit statement of "short" (a full-SHA regression
        would otherwise only show up as a string mismatch).

        ``revision.count("+dirty") == 1`` is the discriminator that matters: it catches an
        implementation that composed two status sources and appended the suffix twice.
        """
        root = _tmpdir()
        _init_repo(root)
        (root / "tracked.txt").write_text("changed\n", encoding="utf-8")
        revision, branch = detect_git_metadata(root)
        self.assertTrue(revision.endswith("+dirty"), revision)
        self.assertEqual(revision.count("+dirty"), 1)
        self.assertEqual(branch, "main")

    def test_ignores_untracked_files(self) -> None:
        """A brand-new untracked file must not mark the tree dirty.

        Negative control for the previous test: same repository, same kind of write, only the
        file's tracked-ness differs. ``status --untracked-files=no`` is the reason, and a
        regression that dropped that flag would make every build artifact dirty the revision —
        which then makes every stored revision unreproducible.
        """
        root = _tmpdir()
        _init_repo(root)
        (root / "untracked.txt").write_text("new\n", encoding="utf-8")
        revision, _branch = detect_git_metadata(root)
        self.assertFalse(revision.endswith("+dirty"), revision)

    def test_non_repository_directory_is_unversioned(self) -> None:
        """A plain directory returns ``("unversioned", "")`` without invoking git.

        Pins both halves of the fallback pair: the string the store persists as ``revision`` and
        the empty branch. The function must not raise for a directory that simply is not a repo —
        ``discover`` runs on arbitrary paths.
        """
        root = _tmpdir()
        self.assertEqual(detect_git_metadata(root), ("unversioned", ""))

    def test_broken_git_pointer_degrades_cleanly(self) -> None:
        """A ``.git`` *file* pointing at a missing directory degrades, never raises.

        This is the worktree/submodule ``gitdir:`` form with a dangling target. ``rev-parse``
        fails, ``_git`` returns ``None``, and the pair becomes ``("unversioned", "")``. A
        regression letting the git error escape would make an unrelated broken checkout
        unrunnable, which is why enrichment is allowed to fail.
        """
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
    """Write a manifest into ``root`` and return its path.

    ``mkdir(parents=True, exist_ok=True)`` lets a test pass the directory it already built or a
    fresh one; the returned path is what discovery is expected to record, so tests assert on it
    directly instead of reconstructing it.
    """
    root.mkdir(parents=True, exist_ok=True)
    path = root / ".testbench.toml"
    path.write_text(text, encoding="utf-8")
    return path


class DiscoverProjectManifestTests(unittest.TestCase):
    """Manifest mode: strict parsing, provenance, and trust.

    The validation tests share one theme — a malformed declaration is a hard
    ``ValueError`` at discovery time, never a suite that quietly does not run. That is a
    security posture rather than tidiness: an ignored typo in ``allow_network`` would turn
    "my network suite never executed" into a green run.
    """

    def test_parses_local_manifest_as_custom_and_untrusted(self) -> None:
        """Every declared field reaches ``SuiteSpec``, with provenance ``custom`` and untrusted.

        The ``custom_from_manifest=True`` / ``trusted=False`` pair is the load-bearing part: the
        engine's veto keys on exactly that combination, so a parse that lost either flag would make an
        unreviewed manifest's ``allow_network`` effective. ``len(manifest_sha256) == 64`` pins the hex
        digest form, because the trust comparison is string equality with that same function's output.

        The field-by-field coverage is deliberate: ``timeout_s``, ``needs_tmux``, the ``env`` alias,
        ``required_*``, ``description`` and the warmup/repeat counts all fail silently if dropped, and
        none of them changes the suite count.
        """
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
        """``trusted`` comes from the store's pinned digest, and dies with the content.

        Untrusted → trusted → untrusted across a one-line append. The middle step proves the flag
        is read from the registry instead of being defaulted; the last step is the whole point of
        pinning and cannot be faked by a boolean that was set once and cached.
        """
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
        """An operator-supplied manifest is parsed but neither custom nor trusted.

        This pins the documented asymmetry: the veto only applies to suites the *repository*
        authored, so an external manifest's ``allow_network``/``allow_repo_write`` are not gated.
        Asserting the provenance flags (what the gate reads) rather than the engine's behaviour
        keeps the responsibility where it lives; ``manifest_path`` is also pinned to the resolved
        external file so the record says which document was used.
        """
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
        """A manifest past the 256 KiB cap is refused before parsing.

        The filler is comment lines appended to an otherwise valid manifest, so the file *is*
        valid TOML and size is the only possible rejection reason. A regression that checked the
        size via ``stat`` before reading, dropped the cap, or truncated instead of refusing would
        let this through.
        """
        root = _tmpdir()
        filler = "#" * (256 * 1024 + 16)
        _write_manifest(root, MANIFEST_TOML + "\n" + filler)
        with self.assertRaises(ValueError):
            discover_project(root)

    def test_invalid_toml_is_rejected_with_value_error(self) -> None:
        """Malformed TOML surfaces as ``ValueError``, not ``tomllib.TOMLDecodeError``.

        The translation is the contract callers depend on: the CLI maps ``ValueError`` to a
        one-line ``testbench error:`` message with exit 1, while a raw decode error would escape
        as a traceback.
        """
        root = _tmpdir()
        _write_manifest(root, "[[suite]\nname = oops\n")
        with self.assertRaises(ValueError):
            discover_project(root)

    def test_string_command_is_rejected(self) -> None:
        """``command`` must be an argv list; a shell string is refused.

        Accepting ``"echo hi"`` would require either a shell (which the isolation layer
        deliberately does not provide) or an exec of a file literally named ``echo hi``. Refusing
        is the only reading that cannot be misinterpreted, and the error is raised at discovery
        rather than at spawn time.
        """
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
        """An adapter outside ``KNOWN_ADAPTERS`` is a hard error.

        ``execute_suite`` tolerates an unknown name by falling back to ``parse_generic``, so this
        discovery-side check is the only place a typo is caught before a suite runs and reports
        counts from the wrong parser.
        """
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
        """``profile`` must be one of ``ProfileName.STANDARD``.

        Profile is the selection key: a suite filed under an unknown profile is invisible to
        ``--profile`` selection and would look like a missing suite (a SKIP) rather than a typo,
        so the error has to happen here.
        """
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
        """No manifest and nothing detectable: empty suites, no manifest path, no languages.

        Pins that detection invents nothing. An empty project is a legitimate, reportable state
        (the CLI prints a full listing with an empty suite list), not an error and not a silently
        added default suite.
        """
        root = _tmpdir()
        (root / "README.md").write_text("nothing to run\n", encoding="utf-8")
        spec = discover_project(root)
        self.assertEqual(spec.suites, [])
        self.assertEqual(spec.languages, [])
        self.assertIsNone(spec.manifest_path)


class DiscoverProjectZeroConfigTests(unittest.TestCase):
    """Zero-config detection: conventions in, ``SuiteSpec`` shapes out.

    Every suite built in this mode keeps the dataclass defaults for ``allow_network`` and
    ``allow_repo_write`` (both false), so detection can never ask the trust gate for a
    capability — the flags are absent, not vetoed. The assertions are on the argv, because
    that is what the adapters and the recorded repro command consume.
    """

    def test_rust_project_uses_cargo_unit_profile(self) -> None:
        """``Cargo.toml`` → one offline ``cargo test --workspace`` suite with repo-write off.

        ``--offline`` and ``assertFalse(allow_repo_write)`` are the sandbox contract: cargo builds into
        a disposable ``CARGO_TARGET_DIR``, so a zero-config suite never needs the checkout writable.
        The timeout assertion is a floor, not the exact value, because it encodes "minutes, not the
        120s default" rather than a tuning constant. ``cargo`` in ``required_binaries`` is what turns a
        missing toolchain into SKIP-with-reason instead of a spawn error.
        """
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
        """No venv and no pytest in the chosen interpreter → ``unittest discover -s tests -v``.

        The argv *tail* is asserted so the interpreter path (machine-dependent) is separated from
        the shape, while ``command[0] == default_interpreter()`` pins that the suite names an
        absolute interpreter rather than ``python3`` off PATH — the difference between a repro that
        works and one that picks up a different environment.

        Environment sensitivity: the branch is chosen by ``_has_module(python, "pytest")``, which
        executes the interpreter. This test asserts ``python_unittest`` because pytest is not
        importable there; on an interpreter that has it the test would legitimately fail, since it
        is testing the *fallback*, not the preference.
        """
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
        self.assertEqual(suite.command[0], str(default_interpreter()))
        self.assertFalse(suite.needs_tmux)

    def test_python_project_prefers_project_venv_interpreter(self) -> None:
        """``./venv/bin/python`` wins over the ambient interpreter.

        The symlink makes the venv interpreter executable without installing anything, which is
        exactly what ``_interpreter_for`` checks (a file, with the execute bit). The assertion is
        on the resolved absolute path, so an implementation returning a relative path fails.

        As in the fallback test, ``python_unittest`` is asserted only because the linked
        interpreter lacks pytest: the preference under test is the interpreter, not the runner.
        """
        root = _tmpdir()
        (root / "tests").mkdir()
        (root / "tests" / "test_alpha.py").write_text("import unittest\n", encoding="utf-8")
        venv_bin = root / "venv" / "bin"
        venv_bin.mkdir(parents=True)
        (venv_bin / "python").symlink_to(default_interpreter())

        spec = discover_project(root)
        suite = spec.suites[0]
        self.assertEqual(suite.command[0], str(root.resolve() / "venv" / "bin" / "python"))
        self.assertEqual(suite.adapter, "python_unittest")

    def test_python_project_uses_pytest_when_venv_has_it(self) -> None:
        """A ``python`` that answers ``-c`` with exit 0 is treated as having pytest.

        The stub is a shell script that exits 0 for every argv, so ``_has_module`` sees success
        without pytest existing on the machine. That is deliberate: the detector's specified
        contract is "ask the interpreter", and this test pins the observable consequence (adapter
        ``python_pytest``, ``-m pytest`` in the argv) instead of requiring an installation.

        The same stub also satisfies ``_interpreter_for``'s executable check, so one file proves
        both the interpreter preference and the runner detection.
        """
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
        """A ``tests/test_tmux_integration.py`` file flips ``needs_tmux`` on the Python suite.

        ``any(...)`` rather than an index because the flag is a property of the proposed suite and
        the test is about the inference, not about which position carries it. Without the flag the
        suite runs headless and fails for lack of a server; with it, isolation allocates tmux.
        """
        root = _tmpdir()
        (root / "tests").mkdir()
        (root / "tests" / "test_alpha.py").write_text("import unittest\n", encoding="utf-8")
        (root / "tests" / "test_tmux_integration.py").write_text(
            "import unittest\n", encoding="utf-8"
        )
        spec = discover_project(root)
        self.assertTrue(any(s.needs_tmux for s in spec.suites))

    def test_hermes_tmux_marker_file_marks_suite_needs_tmux(self) -> None:
        """A ``hermes_tmux`` path at the project root is the second marker for the same inference.

        Same assertion as the previous test through a different signal, so a regression that kept
        only one of the two heuristics is caught by whichever one it dropped.
        """
        root = _tmpdir()
        (root / "tests").mkdir()
        (root / "tests" / "test_alpha.py").write_text("import unittest\n", encoding="utf-8")
        (root / "hermes_tmux").write_text("#!/bin/sh\n", encoding="utf-8")
        spec = discover_project(root)
        self.assertTrue(any(s.needs_tmux for s in spec.suites))

    def test_pyproject_only_project_is_still_detected(self) -> None:
        """``pyproject.toml`` alone, with no ``tests/`` directory, still yields a Python suite.

        Detection is allowed to propose a runnable suite for a package with no test directory
        because the runner discovers internally. The assertion keeps ``-m unittest`` in the argv
        and the absolute interpreter first, which is what makes the proposal executable.
        """
        root = _tmpdir()
        (root / "pyproject.toml").write_text(
            '[project]\nname = "demo"\nversion = "0.1.0"\n', encoding="utf-8"
        )
        spec = discover_project(root)
        self.assertIn("python", spec.languages)
        suite = spec.suites[0]
        self.assertEqual(suite.adapter, "python_unittest")
        self.assertEqual(suite.command[0], str(default_interpreter()))
        self.assertEqual(suite.command[1:3], ["-m", "unittest"])

    def test_polyglot_project_reports_every_language(self) -> None:
        """Python plus Rust: both languages reported and both adapters present.

        ``languages == ["python", "rust"]`` pins the sorted, de-duplicated union, because that order
        is what ``testbench discover`` prints. The adapter *set* is the discriminating half: it
        fails if one detector's suite replaced the other's instead of being added alongside it.
        """
        root = _tmpdir()
        (root / "Cargo.toml").write_text('[package]\nname = "demo"\n', encoding="utf-8")
        (root / "pyproject.toml").write_text('[project]\nname = "demo"\n', encoding="utf-8")
        spec = discover_project(root)
        self.assertEqual(spec.languages, ["python", "rust"])
        self.assertGreaterEqual(len(spec.suites), 2)
        adapters = {s.adapter for s in spec.suites}
        self.assertEqual(adapters, {"cargo", "python_unittest"})

    def test_git_revision_and_branch_are_recorded(self) -> None:
        """Discovery carries git metadata into ``ProjectSpec``.

        ``Cargo.toml`` is written *before* the initial commit, so the tree is clean and the revision
        is that commit; comparing against a fresh ``rev-parse`` avoids hard-coding an abbreviation
        length. ``assertTrue(spec.revision)`` guards against the empty string that would silently
        make every run look unversioned.
        """
        root = _tmpdir()
        (root / "Cargo.toml").write_text('[package]\nname = "demo"\n', encoding="utf-8")
        _init_repo(root)
        spec = discover_project(root)
        self.assertEqual(spec.revision, _run_git(root, "rev-parse", "--short", "HEAD"))
        self.assertEqual(spec.branch, "main")
        self.assertTrue(spec.revision)


class DiscoverProjectMultiLanguageTests(unittest.TestCase):
    """Cobol/C/verifier/SQLite detectors, and the walk's exclusion rules.

    These detectors emit one suite per source, or one aggregate suite with the files as
    argv, so the assertions are about argv shape and *which* files were collected — file
    selection is the part that can regress silently while every count still looks right.
    """

    def test_cobol_sources_build_optimized_and_debug_into_tmpdir(self) -> None:
        """One ``*.cob`` → a ``cobol-unit`` suite compiling ``-O2 -debug`` into ``$TMPDIR``.

        ``$TMPDIR`` is asserted literally because it is the isolation-provided scratch directory — a
        hard-coded path would escape the sandbox or collide between concurrent runs. ``-debug`` is what
        makes a failing program's line numbers usable, so it is part of the contract rather than a
        detail. ``cobc`` in ``required_binaries`` converts "no compiler" into SKIP.
        """
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
        """The TUI harness file produces exactly one suite, with the full uv argv.

        Exact-argv equality rather than substring checks: this command is what a repro replays, and
        it must run offline with a pinned ``pyte`` rather than whatever the environment happens to
        have. ``suites[0]`` being the harness also pins a quieter invariant — when the harness is
        present it *suppresses* the per-source ``cobol-unit`` suite, because the harness already
        compiles what it needs and two suites would run the same sources twice.
        """
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
        """Only ``*selftest*.c`` becomes a ``c_sanitizer`` suite, and it is profile ``security``.

        The filename filter is the point: zero-config must not compile every ``.c`` file in a tree,
        only the ones the project marks as self-tests. ``-fsanitize=address,undefined`` is asserted
        as one flag string because both sanitizers must survive edits to the command, and
        ``$TMPDIR`` is the isolation-provided scratch dir (a hard-coded path would escape it).
        ``clang`` in ``required_binaries`` is what turns a missing compiler into SKIP.
        """
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
        """Four known verifier scripts map to fixed names, adapters, and runners.

        The set equality on names checks both directions: every script was found, and none was
        renamed or duplicated. ``verify_docs``/``verify_claims`` are ``re_project`` because they
        emit PASS/FAIL markers; ``regress_release``/``check-wiring`` are ``shell``. All four sit on
        profile ``regression``, which is what keeps an expensive sweep out of ``--profile unit``.

        ``command[1].startswith("scripts/")`` pins that the argv uses the repo-relative path, so
        the recorded repro works from the project root; ``command[0]`` is compared by *name* to
        keep the assertion portable, since the venv-preference logic is covered elsewhere.
        """
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
            self.assertEqual(Path(suite.command[0]).name, default_interpreter().name)
            self.assertTrue(suite.command[1].startswith("scripts/"), suite.command)
            self.assertEqual(suite.profile, "regression")

    def test_sqlite_databases_get_integrity_suite_and_skip_noise_dirs(self) -> None:
        """Root ``*.db``/``*.sqlite`` files are probed; a copy under ``node_modules`` is not.

        ``command[3:]`` are the database arguments (after interpreter, ``-c`` and the inline probe
        script), so comparing that sorted slice is a direct statement about which files were
        collected. ``vendor.db`` under ``node_modules`` is the negative control for noise pruning:
        same suffix, excluded purely by directory, so the test fails if the walk stops pruning.

        ``integrity_check`` appears in the embedded probe, which opens each file with
        ``mode=ro&immutable=1`` — the reason this suite cannot damage what it inspects.
        """
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
        """Five languages and six adapters from one tree, with unique suite names.

        The sorted language list pins the union across detectors; the adapter set pins that each
        detector contributed rather than one overwriting another. Name uniqueness protects
        ``--suite`` selection: two suites sharing a name make selection ambiguous and hide one of
        the two results, which is also why discovery rejects duplicate manifest names.
        """
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
        """A ``.cob`` inside ``.git`` and a selftest inside ``vendor`` produce no suites.

        Two negative controls on the walk: hidden directories are skipped outright, and ``vendor``
        is in the noise set. Without them a vendored dependency's sources would silently become the
        project's test suite — and its failures the project's failures.
        """
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
    """Build one ``IsolatedExecResult`` for the mocked runner.

    ``stdout_text`` defaults to empty so each policy test decides what the parser sees, and the
    three ``*_path`` fields deliberately point at a path that does not exist — a policy test must
    never read a log, so an implementation that tried would fail loudly instead of passing on a
    real file left by an earlier run.
    """
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
    """A least-privileged ``shell`` suite with overridable fields.

    Defaults are ``/usr/bin/true`` on profile ``unit`` with every capability flag false, so each
    test states exactly the one flag it is about; an accidental gate engagement then shows up as
    an unexpected BLOCKED instead of being masked by a permissive default.
    """
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
    """Pre-spawn policy gates, with the runner replaced by a recorder.

    ``run_isolated_command`` is patched on the adapters module because ``execute_suite``
    looks it up there. That yields two independent signals per case — the returned
    ``SuiteResult`` and ``self.run_mock.call_count`` — and the second is what proves
    ordering: a gate that produced a BLOCKED result *after* spawning would satisfy a
    status-only assertion but not ``call_count == 0``.

    ``self.fake_kwargs`` records what the runner was called with, for tests that care about
    the isolation flags rather than the verdict.
    """

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
        """Billable without approval: BLOCKED, one blocked case, runner never called.

        The accounting is asserted alongside the verdict — ``total_count == 1`` with
        ``blocked_count == 1`` — because a blocked suite reporting zero totals would disappear from
        the run summary's denominators. ``usage is None`` pins that a blocked suite cannot bill, and
        the summary containing "billable" is what the operator actually reads.
        """
        from hermes_testbench.schema import RunStatus

        result, usage = self._exec(_suite(billable=True))
        self.assertEqual(result.status, RunStatus.BLOCKED)
        self.assertEqual(result.blocked_count, 1)
        self.assertEqual(result.total_count, 1)
        self.assertIsNone(usage)
        self.assertEqual(self.run_mock.call_count, 0)
        self.assertIn("billable", result.summary_excerpt.lower())

    def test_billable_suite_with_approval_but_no_token_budget_is_blocked(self) -> None:
        """Approval without a token budget is not approval.

        ``allow_billable=True`` with ``max_tokens=0`` must still block: the budget is the second half
        of the gate, and an implementation that checked only the boolean would spend money. The
        discriminator is ``call_count == 0``, not the status.
        """
        from hermes_testbench.schema import RunStatus

        result, _usage = self._exec(_suite(billable=True), allow_billable=True, max_tokens=0)
        self.assertEqual(result.status, RunStatus.BLOCKED)
        self.assertEqual(self.run_mock.call_count, 0)

    def test_billable_suite_runs_when_approved_with_budget(self) -> None:
        """Positive control for the two billable tests: approval plus budget reaches the runner.

        ``assertNotEqual(BLOCKED)`` rather than ``== PASS`` keeps the test about the gate rather than
        the parser, and ``call_count == 1`` proves it spawned exactly once — a retry loop could
        otherwise hide a double execution behind the same verdict.
        """
        from hermes_testbench.schema import RunStatus

        result, _usage = self._exec(
            _suite(billable=True), allow_billable=True, max_tokens=1000
        )
        self.assertNotEqual(result.status, RunStatus.BLOCKED)
        self.assertEqual(self.run_mock.call_count, 1)

    def test_untrusted_custom_manifest_network_request_is_blocked(self) -> None:
        """Untrusted manifest requesting ``allow_network``: BLOCKED, kind ``untrusted_manifest``.

        The failure *kind* is asserted, not just the status, because it is the vocabulary the
        reporting layer maps differently from an assertion failure — reporting a policy denial as a
        test failure would misattribute it to the project. Exactly one failure record, and no spawn.
        """
        from hermes_testbench.schema import FailureKind, RunStatus

        suite = _suite(custom_from_manifest=True, allow_network=True)
        result, _usage = self._exec(suite, trusted_manifest=False)
        self.assertEqual(result.status, RunStatus.BLOCKED)
        self.assertEqual(result.blocked_count, 1)
        self.assertEqual(len(result.failures), 1)
        self.assertEqual(result.failures[0].kind, FailureKind.UNTRUSTED_MANIFEST)
        self.assertEqual(self.run_mock.call_count, 0)

    def test_untrusted_custom_manifest_repo_write_is_blocked(self) -> None:
        """``allow_repo_write`` is vetoed by the same gate, not only network.

        A separate test because the two capabilities are separate fields: a gate written as
        ``if allow_network`` alone would pass the network case and fail here.
        """
        from hermes_testbench.schema import RunStatus

        suite = _suite(custom_from_manifest=True, allow_repo_write=True)
        result, _usage = self._exec(suite, trusted_manifest=False)
        self.assertEqual(result.status, RunStatus.BLOCKED)
        self.assertEqual(self.run_mock.call_count, 0)

    def test_trusted_manifest_allows_network_and_repo_write(self) -> None:
        """Trust flips the identical suite to PASS: the gate keys on provenance, not on the flags.

        This is the control for the two untrusted cases — same suite, only ``trusted_manifest``
        differs — so a gate that blocked unconditionally, or one that ignored trust, is caught here
        rather than by the untrusted tests.
        """
        from hermes_testbench.schema import RunStatus

        suite = _suite(custom_from_manifest=True, allow_network=True, allow_repo_write=True)
        result, _usage = self._exec(suite, trusted_manifest=True)
        self.assertEqual(result.status, RunStatus.PASS)
        self.assertEqual(self.run_mock.call_count, 1)

    def test_non_manifest_suite_is_not_gated_on_network(self) -> None:
        """A suite that did not come from the project manifest may request the network freely.

        Pins the documented boundary: the veto is scoped to ``custom_from_manifest``, so detected
        suites (and external-manifest suites) are not gated. Without this test, an implementation
        that gated every ``allow_network`` would look correct on the untrusted cases while breaking
        legitimate zero-config suites.
        """
        from hermes_testbench.schema import RunStatus

        result, _usage = self._exec(_suite(allow_network=True), trusted_manifest=False)
        self.assertEqual(result.status, RunStatus.PASS)
        self.assertEqual(self.run_mock.call_count, 1)

    def test_missing_required_binary_skips_suite(self) -> None:
        """A missing ``required_binaries`` entry is SKIP, with the binary named in the reason.

        SKIP rather than FAIL: a machine without the toolchain has not tested anything, so calling
        it a failure would blame the project. The reason must name the binary or the operator cannot
        tell which tool to install. No spawn, and ``usage is None``.
        """
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
        """A missing ``required_paths`` entry also SKIPs, before any spawn.

        Separate from the binary case because the two are resolved differently — paths against the
        project root, binaries against PATH plus the extra bin dirs — so they are two code paths
        with two different messages.
        """
        from hermes_testbench.schema import RunStatus

        result, _usage = self._exec(_suite(required_paths=["missing/dir"]))
        self.assertEqual(result.status, RunStatus.SKIP)
        self.assertEqual(self.run_mock.call_count, 0)

    def test_suite_without_command_skips(self) -> None:
        """An empty argv is SKIP, never an exec of ``[]``.

        The discriminator is that no runner was called: exec'ing an empty list raises
        ``ValueError``, which ``execute_suite`` would report as ERROR/spawn_error. Refusing earlier
        keeps "this suite has nothing to run" distinct from "this suite failed to start".
        """
        from hermes_testbench.schema import RunStatus

        result, _usage = self._exec(_suite(command=[]))
        self.assertEqual(result.status, RunStatus.SKIP)
        self.assertEqual(self.run_mock.call_count, 0)

    def test_policy_result_records_repro_command_and_timestamps(self) -> None:
        """Even a blocked suite records a replayable command and both timestamps.

        ``started_at``/``finished_at``/``repro_command`` and ``adapter`` are what ``run.json`` and
        ``repro.sh`` are rendered from, so a gate returning a bare ``BLOCKED`` would produce an
        artifact with empty fields. The repro contains the argv shlex-joined behind the resolved
        cwd, so the operator can rerun the command by hand even though it was never executed.
        """
        result, _usage = self._exec(_suite(billable=True))
        self.assertIn("true", result.repro_command)
        self.assertTrue(result.started_at)
        self.assertTrue(result.finished_at)
        self.assertEqual(result.adapter, "shell")


class AdapterParserContractTests(unittest.TestCase):
    """Captured-text parsers, with no toolchain and no process.

    The inputs are literal runner output copied from real logs, which is why these tests run
    on any machine: the subject is the mapping from text to counts, including the shapes that
    must *not* produce counts. Duplicated markers and summary lines are included on purpose,
    because double counting is the characteristic parser defect.
    """

    def test_unittest_ok_with_skips(self) -> None:
        """``Ran 4`` plus ``OK (skipped=2)`` → 4 total, 2 passed, 2 skipped.

        ``passed`` is derived (total − skipped − failures − errors), so this asserts the derivation
        rather than a counted number. ``structured`` is the flag ``execute_suite`` uses to decide
        whether to trust the counts or fall back to the exit status; without it, a green run with
        skips would collapse to "exit 0 = one anonymous pass" and the skip would vanish from the
        run's denominators.
        """
        from hermes_testbench.adapters.parsers import parse_python_unittest

        outcome = parse_python_unittest("Ran 4 tests in 0.010s\n\nOK (skipped=2)\n")
        self.assertEqual((outcome.total, outcome.passed, outcome.skipped), (4, 2, 2))
        self.assertTrue(outcome.structured)

    def test_unittest_failed_clause_and_error_headers(self) -> None:
        """One FAIL and one ERROR: exact counts, ordered failure records, message extraction.

        Three discriminators:

        * ``passed == 1`` with ``total == 3`` — ``passed`` is derived (total − skipped − failures −
          errors), so this asserts the derivation rather than a counted number.
        * the failure list is ``[test_b, test_c]`` in header order, named ``name (loc)``. That form is
          the key the verbose lines use too, so a different display form would orphan the cases from
          their headers.
        * ``"1 != 2"`` in the first message proves the traceback block was mined for the exception
          line instead of the header line being used as the message.

        The ``... ok|FAIL|ERROR`` verbose lines are present so header-wins-over-verbose is exercised:
        the same test must not be recorded twice.
        """
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
        """pytest's summary line supplies totals; the ``FAILED`` line supplies identity.

        The four-tuple ``(4, 1, 1, 2)`` is asserted because ``total`` is the sum of the summary's
        buckets, not the count of case lines — an implementation that counted printed lines would
        report 2 and pass a status-only check. ``case_name`` is the full ``path::test`` node id
        because that is what pytest's short summary prints; stripping it to the function name would
        lose the file that identifies the failure.
        """
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
        """Marker grammar: ``PASS:``/``FAIL:``/``SKIP:`` counts, and ANSI stripped before scanning.

        The PTY input carries a colour escape before ``PASS: resize`` and still counts, which proves
        normalisation happens before the marker match. Without it a coloured TUI dump parses as zero
        cases and the run would fall through to exit-status logic — reported as one anonymous pass.
        """
        from hermes_testbench.adapters.parsers import parse_pty_tmux, parse_sqlite_probe

        probe = parse_sqlite_probe("PASS: integrity_check\nFAIL: wal_sync\nSKIP: fts5\n")
        self.assertEqual((probe.total, probe.passed, probe.failed, probe.skipped), (3, 1, 1, 1))
        tty = parse_pty_tmux("\x1b[32mPASS: resize\x1b[0m\nPASS: detach\n")
        self.assertEqual((tty.total, tty.passed), (2, 2))

    def test_asan_report_is_a_single_failure(self) -> None:
        """One ASan report is one failure, with the diagnostic in the message.

        ``failed == 1`` with exactly one failure record: the ``SUMMARY:`` and ``WRITE of size`` lines
        belong to the same report, so a line-oriented implementation that counted every match would
        report three. ``stack-buffer-overflow`` in the message pins that the
        ``ERROR: AddressSanitizer:`` line is the record rather than the first stack frame.
        """
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
        """``ADAPTER_PARSERS`` and ``KNOWN_ADAPTERS`` must be the same set.

        A structural test, not a behavioural one: an adapter added to discovery but not to the parser
        map would silently fall back to ``parse_generic`` at runtime — a downgrade that presents as a
        parser bug rather than a missing registration. Asserting both directions also catches a
        parser registered for an adapter discovery would refuse, and the ``callable`` check rejects a
        name mapped to ``None`` or to a value that only looks like a function.
        """
        from hermes_testbench.adapters.parsers import ADAPTER_PARSERS
        from hermes_testbench.discovery import KNOWN_ADAPTERS

        self.assertEqual(set(ADAPTER_PARSERS), set(KNOWN_ADAPTERS))
        self.assertTrue(all(callable(fn) for fn in ADAPTER_PARSERS.values()))


class AdapterExecutionIntegrationTests(unittest.TestCase):
    """Real executions through ``execute_suite`` and the isolation layer.

    Nothing here is patched, so the tests depend on the host. Each one that needs a toolchain
    checks for it first and calls ``skipTest`` with a named reason. A skip is not a pass: on a
    machine without ``cargo``/``clang``/``cobc`` the corresponding assertions never run, and a
    green suite result says nothing about them.

    Suites are built with ``_suite`` and executed against a temp project root, so the real
    argv runs and the real parser produces the counts — the difference from the parser tests
    above is that the text was not written by hand.
    """

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
        """unittest writes its summary to stderr; the parser must read the combined stream.

        The script fails one of two tests, so the run is FAIL with ``exit_code == 1``, exactly one
        failure naming ``test_broken``, and ``Ran 2 tests`` in the excerpt. Reading only stdout is
        the natural bug — unittest is silent there — and this test fails on it because the counts
        would be missing and the suite would collapse to one synthetic exit-status failure.
        """
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
        """Two crates, three tests: counts are summed across cargo's per-binary result blocks.

        ``3 total / 2 passed / 1 failed`` comes from *two* ``test result:`` blocks; taking only the
        last block would report ``2/1/1``, which looks entirely plausible — which is why the totals
        are asserted and not just the failure. The failing case name must contain ``beta_bad``,
        proving attribution survives aggregation.

        ``skipTest`` when cargo is absent; ``allow_repo_write=True`` because cargo writes into the
        temp project's ``target``; the 240s suite timeout covers a cold build.
        """
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
        """The same aggregation as a pure text test, so it runs where cargo is not installed.

        ``skipped == 1`` from cargo's ``1 ignored`` pins the ``ignored`` → skip mapping (cargo has no
        "skipped" vocabulary), and the single failure is named ``beta::two`` — the module-qualified
        path cargo prints. ``structured`` is asserted because unstructured output is exactly what
        would silently downgrade the run to exit-status counting.
        """
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
        """A clean ASan/UBSan binary passes, with no invented failure.

        The binary is built by the test rather than by the suite, so a compiler or flag problem fails
        here instead of being misread as a test failure. ``pass_count >= 1`` rather than ``== 1``:
        either the ``SELFTEST OK`` success marker or a marker count is a legitimate source, and the
        assertion is that *something* was counted as a pass.

        ``skipTest`` when clang is absent, including when only the ``/usr/bin/clang`` fallback is
        missing.
        """
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
        """A recoverable UBSan diagnostic fails the run even though the process exits 0.

        This is the test that justifies parsing instead of trusting exit statuses: signed overflow is
        recoverable, so the program prints a value and returns 0, and the only evidence is the
        ``runtime error:`` line. ``exit_code == 0`` and ``status == FAIL`` are asserted together so
        neither can be satisfied by coincidence, and the failure message must carry ``runtime error``
        so the operator sees the diagnostic rather than a generic verdict.
        """
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
        """A GnuCOBOL self-test prints ``PASS: counter`` and is counted as exactly one pass.

        The program also prints ``TESTS PASSED: 1 / 1``, so the counts must come out 1 total / 1
        passed / 0 failed without double counting the marker and the summary. ``skipTest`` when
        ``cobc`` is absent; the ``-x -free`` build is done by the test so a compiler problem is not
        reported as a COBOL failure.
        """
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
        """A verifier that only prints ``SKIP: ...`` exits 0 → suite PASS with one skipped check.

        The suite status describes the *process*, ``skip_count`` describes the *checks*; a regression
        that mapped any skip to a suite-level SKIP would hide the fact that the verifier ran
        successfully and would turn a deliberate "browser toolchain unavailable" into a non-zero exit
        for callers scripting on the run status.
        """
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
        """Two PASS lines and one FAIL: 3 total, 2 passed, 1 failed, named by the marker text.

        The failure's ``case_name`` is the text after ``FAIL:`` — that is what makes the record
        readable in the report. The script exits 1, so an implementation that ignored markers would
        still call the run FAIL; the counts and the attribution are what this test adds on top of the
        verdict.
        """
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
        """No ``TESTBENCH_MODEL_USAGE`` line → ``usage is None`` and no cost anywhere.

        Three assertions close the same hole at three levels: the returned usage, ``result.metrics``,
        and the summary text. An implementation that synthesised ``{"tokens": 0}`` would pass the
        first check and fail on ``"tokens" not in summary``. A fabricated zero cost is worse than no
        number at all, because it reports a billable run as free.
        """
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
        """The emitted JSON is taken verbatim, including its dollar cost.

        Exact dict equality is the assertion: no unit conversion, no rounding, no key renaming. The
        report must quote the number the runner produced, because that number is what gets billed —
        and the suite's own counts (``2 total``) come from the summary line beside it, so the two
        sources are proven independent.
        """
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
