"""Phase 3 mutation-testing contract: external tools, disposable staging, explicit gating.

These tests run the *real* ``mutmut`` and ``cargo-mutants`` binaries when they are
available (``TESTBENCH_MUTMUT`` / ``TESTBENCH_CARGO_MUTANTS`` or ``PATH``), and otherwise
cover classification, gating and isolation with pure/unit-level checks.

Where the boundary sits, so a green local run is not read as more than it is:

* ``RealToolEndToEndTests`` needs the binary *and* ``can_apply_seatbelt()``, i.e.
  macOS with ``sandbox-exec``. Every such test is skipped on Linux, which is why the
  unit classes below carry the contract for CI.
* The fixtures are committed to a git repo at build time. That is not decoration:
  the pristine checks compare ``git status --porcelain`` and scan for the sentinels a
  mutation tool drops, so a tool escaping its stage is visible rather than silently
  absorbed into the working tree.
* Gating is tested by mocking only the *platform* question (``can_apply_seatbelt``)
  and the *resolution* question (``resolve_tool``), never the staging or verdict
  code; the assertions are on BLOCKED/SKIP/ERROR plus ``on_spawn`` counts, so "did
  not run" is proven separately from "reported badly".
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from hermes_testbench.adapters import execute_suite
from hermes_testbench.adapters import mutation as mutation_mod
from hermes_testbench.adapters import parsers
from hermes_testbench.discovery import KNOWN_ADAPTERS, discover_project
from hermes_testbench.isolation import can_apply_seatbelt, run_isolated_command
from hermes_testbench.schema import (
    FailureKind,
    ProfileName,
    RunStatus,
    SuiteSpec,
)

PY = sys.executable

def _tool_dir() -> Path:
    """Directory scanned for locally staged tool builds when no env override names one."""
    override = os.environ.get("TESTBENCH_MUT_TOOLS_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    return Path.home() / ".hermes" / "cache" / "scratch" / "tb-mut-tools" / "bin"


MUTMUT_BIN = Path(os.environ.get("TESTBENCH_MUTMUT") or (_tool_dir() / "mutmut"))
CARGO_MUTANTS_BIN = Path(os.environ.get("TESTBENCH_CARGO_MUTANTS") or (_tool_dir() / "cargo-mutants"))


def _tool_env() -> dict:
    """Env overrides for the tools that actually exist on this host.

    Only present paths are exported: an empty override would shadow a tool the
    test deliberately removes from ``PATH`` in other cases.
    """
    env = {}
    if MUTMUT_BIN.is_file():
        env["TESTBENCH_MUTMUT"] = str(MUTMUT_BIN)
    if CARGO_MUTANTS_BIN.is_file():
        env["TESTBENCH_CARGO_MUTANTS"] = str(CARGO_MUTANTS_BIN)
    return env


def _git_init(root: Path) -> None:
    """Commit the fixture so ``git status --porcelain`` has a clean baseline.

    The identity flags are passed per-invocation rather than configured, so the
    fixtures do not depend on or modify the developer's git config.
    """
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.email=tb@example.invalid",
            "-c",
            "user.name=testbench",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )


def _git_status(root: Path) -> str:
    """Porcelain status of a fixture checkout; ``check=False`` so a failure reads as output."""
    proc = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout


def _write_mutmut_project(root: Path) -> None:
    """Python fixture with one fully-covered function and one known survivor.

    ``add`` is pinned by one assertion, ``add(2, 3) == 5``, which catches every
    mutant of ``a + b``. ``classify`` is only exercised at 11 and 1, which leaves ``>`` → ``>=``
    alive: ``survived == 1`` is therefore a fixture property, and the tests can
    assert it without depending on mutmut's mutation generator.
    """
    (root / "src").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "pyproject.toml").write_text(
        "[project]\n"
        'name = "tb-mutation-demo"\n'
        'version = "0.1.0"\n'
        "\n"
        "[tool.mutmut]\n"
        'source_paths = ["src"]\n'
        'pytest_add_cli_args_test_selection = ["tests"]\n'
        "\n"
        "[tool.pytest.ini_options]\n"
        'pythonpath = ["src"]\n',
        encoding="utf-8",
    )
    # ``add`` is fully covered (its mutants are caught). ``classify`` only exercises the
    # boundary from both sides, so the ``>`` -> ``>=`` mutant survives: one known survivor.
    (root / "src" / "demo.py").write_text(
        "def add(a, b):\n"
        "    return a + b\n"
        "\n"
        "\n"
        "def classify(n):\n"
        "    if n > 10:\n"
        '        return "big"\n'
        '    return "small"\n',
        encoding="utf-8",
    )
    (root / "tests" / "test_demo.py").write_text(
        "from demo import add, classify\n"
        "\n"
        "\n"
        "def test_add():\n"
        "    assert add(2, 3) == 5\n"
        "\n"
        "\n"
        "def test_classify():\n"
        '    assert classify(11) == "big"\n'
        '    assert classify(1) == "small"\n',
        encoding="utf-8",
    )
    _git_init(root)


def _write_cargo_project(root: Path) -> None:
    """Rust mirror of the Python fixture: covered ``add``, boundary-only ``classify``."""
    (root / "src").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "Cargo.toml").write_text(
        "[package]\n"
        'name = "tbmutationdemo"\n'
        'version = "0.1.0"\n'
        'edition = "2021"\n'
        "\n"
        "[lib]\n"
        'path = "src/lib.rs"\n',
        encoding="utf-8",
    )
    (root / "src" / "lib.rs").write_text(
        "pub fn add(a: i32, b: i32) -> i32 {\n"
        "    a + b\n"
        "}\n"
        "\n"
        "pub fn classify(n: i32) -> &'static str {\n"
        "    if n > 10 {\n"
        '        "big"\n'
        "    } else {\n"
        '        "small"\n'
        "    }\n"
        "}\n",
        encoding="utf-8",
    )
    (root / "tests" / "basic.rs").write_text(
        "use tbmutationdemo::{add, classify};\n"
        "\n"
        "#[test]\n"
        "fn add_works() {\n"
        "    assert_eq!(add(2, 3), 5);\n"
        "}\n"
        "\n"
        "#[test]\n"
        "fn classify_works() {\n"
        '    assert_eq!(classify(11), "big");\n'
        '    assert_eq!(classify(1), "small");\n'
        "}\n",
        encoding="utf-8",
    )
    _git_init(root)


class _MutationTestCase(unittest.TestCase):
    """Shared scratch layout: ``evidence/`` for artifacts, ``scratch/`` for staged copies.

    Both live under one temp root so ``tearDown`` removes a stage that a failed
    test left behind and the next test cannot read it as its own.
    """

    def setUp(self) -> None:
        scratch = Path.home() / ".hermes/cache/scratch"
        scratch.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=str(scratch), prefix="tb-mut-")
        self.root = Path(self.tmp.name)
        self.evidence = self.root / "evidence"
        self.evidence.mkdir()
        self.scratch = self.root / "scratch"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _run(self, suite: SuiteSpec, project_root: Path, **kwargs):
        return execute_suite(
            suite,
            project_root=project_root,
            evidence_dir=self.evidence,
            run_id="tb_mut_test",
            scratch_root=self.scratch,
            **kwargs,
        )

    def _assert_checkout_pristine(self, project_root: Path) -> None:
        """Fail if the tool wrote into the real checkout.

        Two independent signals: the sentinel names a mutation tool drops in the
        project root, and a non-empty ``git status --porcelain``. Neither is
        exhaustive — ignored paths and a write-then-restore are invisible — but
        together they catch the ordinary escape, and the sandbox is the real
        guarantee.
        """
        for name in ("mutants", "mutants.out", "mutants.out.old", ".mutmut-cache"):
            self.assertFalse(
                (project_root / name).exists(), f"{name} leaked into the real checkout"
            )
        self.assertEqual(_git_status(project_root), "", "real checkout was modified")


class RealToolEndToEndTests(_MutationTestCase):
    """Real binaries, real staging, real Seatbelt.

    Every case here is skipped without the binary and without
    ``sandbox-exec``; the 600–900 s suite timeouts are the outer bound, since a
    hung tool is torn down by the isolation layer rather than by the test.
    """

    @unittest.skipUnless(MUTMUT_BIN.is_file(), "mutmut binary unavailable")
    @unittest.skipUnless(can_apply_seatbelt(), "macOS sandbox-exec unavailable")
    def test_mutmut_catches_and_survives_on_a_disposable_copy(self) -> None:
        """One mutant caught and one survived, both by fixture design.

        The five documented ``metrics["mutation"]`` keys are asserted present
        (``set(metrics) & {...} == {...}``), with extra keys tolerated;
        ``pass_count``/``fail_count``
        must equal caught/survived, which is the projection ``_bucket_counts``
        performs (a mutation suite counts mutant verdicts, not tests). The
        evidence assertions cover the other half of the contract: owner-only
        ``0600`` and the 8 MiB bound, with the stored metrics matching the ones
        returned.
        """
        project = self.root / "pymut"
        _write_mutmut_project(project)
        suite = SuiteSpec(
            name="mutmut",
            adapter="mutmut",
            profile=ProfileName.MUTATION,
            command=["mutmut", "run"],
            timeout_s=600.0,
        )
        with mock.patch.dict(os.environ, _tool_env(), clear=False):
            result, _ = self._run(suite, project)

        self.assertEqual(result.status, RunStatus.PASS, result.summary_excerpt)
        metrics = result.metrics["mutation"]
        self.assertGreaterEqual(metrics["caught"], 1, metrics)
        self.assertGreaterEqual(metrics["survived"], 1, metrics)
        self.assertEqual(
            set(metrics) & {"caught", "survived", "invalid", "timeout", "inconclusive"},
            {"caught", "survived", "invalid", "timeout", "inconclusive"},
        )
        self.assertEqual(result.pass_count, metrics["caught"])
        self.assertEqual(result.fail_count, metrics["survived"])
        survived_cases = [c for c in result.cases if c.status == RunStatus.FAIL]
        self.assertTrue(survived_cases, "surviving mutants must appear as case entries")
        self._assert_checkout_pristine(project)

        # Bounded, owner-only evidence bundle.
        evidence = self.evidence / "mutation" / "mutmut.json"
        self.assertTrue(evidence.is_file())
        self.assertEqual(stat.S_IMODE(evidence.stat().st_mode), 0o600)
        self.assertLessEqual(evidence.stat().st_size, 8 * 1024 * 1024)
        payload = json.loads(evidence.read_text(encoding="utf-8"))
        self.assertEqual(payload["metrics"]["caught"], metrics["caught"])

    @unittest.skipUnless(CARGO_MUTANTS_BIN.is_file(), "cargo-mutants binary unavailable")
    @unittest.skipUnless(can_apply_seatbelt(), "macOS sandbox-exec unavailable")
    @unittest.skipUnless(shutil.which("cargo"), "cargo unavailable")
    def test_cargo_mutants_catches_and_survives_on_a_disposable_copy(self) -> None:
        """The Rust path: same buckets, plus ``score < 1.0``.

        The score assertion is the one that fails if a survivor is silently
        reclassified as caught; ``cargo`` is additionally required because
        ``cargo-mutants`` is a cargo subcommand and fails without it.
        """
        project = self.root / "rsmut"
        _write_cargo_project(project)
        suite = SuiteSpec(
            name="cargo-mutants",
            adapter="cargo_mutants",
            profile=ProfileName.MUTATION,
            command=["cargo-mutants", "mutants"],
            timeout_s=900.0,
        )
        with mock.patch.dict(os.environ, _tool_env(), clear=False):
            result, _ = self._run(suite, project)

        self.assertEqual(result.status, RunStatus.PASS, result.summary_excerpt)
        metrics = result.metrics["mutation"]
        self.assertGreaterEqual(metrics["caught"], 1, metrics)
        self.assertGreaterEqual(metrics["survived"], 1, metrics)
        self.assertLess(metrics["score"], 1.0)
        self._assert_checkout_pristine(project)

    @unittest.skipUnless(MUTMUT_BIN.is_file(), "mutmut binary unavailable")
    @unittest.skipUnless(can_apply_seatbelt(), "macOS sandbox-exec unavailable")
    def test_reproduce_of_mutation_run_succeeds_and_keeps_checkout_clean(self) -> None:
        """``testbench reproduce`` must replay a mutation suite through the engine.

        A ``RunRecord`` stores ``SuiteResult.cwd`` as an absolute path (the project root).
        Replaying that value verbatim used to feed it into the staged-copy containment
        check, where an absolute path silently escapes ``stage_dir`` and the reproduce run
        errored out with "suite cwd escapes the staged copy".

        The absolute value is asserted first, because that is exactly the input
        ``_relativize_suite_cwd`` has to rewrite; the repro run must then get a
        new id, the ``repro:mutation`` profile and the same bucket counts as the
        original. Without the cwd assertion the test would still pass if the
        recording format changed and the regression stopped being reachable.
        """
        from hermes_testbench.engine import TestBenchEngine
        from hermes_testbench.store import TestBenchStore

        project = self.root / "pymutrepro"
        _write_mutmut_project(project)
        home = self.root / "home"

        with mock.patch.dict(os.environ, _tool_env(), clear=False):
            store = TestBenchStore(home)
            try:
                engine = TestBenchEngine(store)
                run = engine.run_sync(project, profile="mutation")
                self.assertEqual(run.status, RunStatus.PASS, run.first_failure)
                self.assertEqual([s.adapter for s in run.suites], ["mutmut"])
                first = run.suites[0].metrics["mutation"]
                self.assertGreaterEqual(first["caught"], 1, first)
                self.assertGreaterEqual(first["survived"], 1, first)
                self.assertEqual(
                    run.suites[0].cwd, str(project.resolve()), "recorded cwd is absolute"
                )

                reproduced = engine.reproduce(run.run_id)
            finally:
                store.close()

        self.assertNotEqual(reproduced["repro_run_id"], run.run_id)
        self.assertEqual(
            reproduced["status"],
            RunStatus.PASS,
            reproduced.get("first_failure") or reproduced,
        )
        with TestBenchStore(home) as store:
            repro_run = store.get_run(reproduced["repro_run_id"])
        assert repro_run is not None
        self.assertEqual(repro_run.profile, "repro:mutation")
        self.assertEqual([s.adapter for s in repro_run.suites], ["mutmut"])
        metrics = repro_run.suites[0].metrics["mutation"]
        self.assertGreaterEqual(metrics["caught"], 1, metrics)
        self.assertGreaterEqual(metrics["survived"], 1, metrics)
        self.assertEqual(repro_run.pass_count, metrics["caught"])
        self.assertEqual(repro_run.fail_count, metrics["survived"])
        self._assert_checkout_pristine(project)

    @unittest.skipUnless(MUTMUT_BIN.is_file(), "mutmut binary unavailable")
    @unittest.skipUnless(can_apply_seatbelt(), "macOS sandbox-exec unavailable")
    def test_cli_mutation_gate_fails_when_a_survivor_lowers_the_score(self) -> None:
        """A gate above the achieved score fails; a gate below it passes.

        The 1.0 gate is the strict case and the 0.1 gate is its control: without
        the second, a suite that always FAILs whenever a gate is set would still
        pass this test. The failure kind and the "gate" wording pin the reason,
        not merely the status — an ERROR would otherwise satisfy a status check.
        Both gates are stamped the way the CLI does it (``mutation_gate_from_cli``),
        so Gate A's manifest-trust check is not what is being exercised.
        """
        project = self.root / "pygate"
        _write_mutmut_project(project)
        suite = SuiteSpec(
            name="mutmut",
            adapter="mutmut",
            profile=ProfileName.MUTATION,
            command=["mutmut", "run"],
            timeout_s=600.0,
        )
        suite.mutation_score_gate = 1.0
        suite.mutation_gate_from_cli = True
        with mock.patch.dict(os.environ, _tool_env(), clear=False):
            strict, _ = self._run(suite, project)
        self.assertEqual(strict.status, RunStatus.FAIL, strict.summary_excerpt)
        self.assertTrue(
            any(f.kind == FailureKind.ASSERTION for f in strict.failures), strict.failures
        )
        self.assertIn("gate", strict.summary_excerpt)
        self._assert_checkout_pristine(project)

        relaxed = SuiteSpec(
            name="mutmut",
            adapter="mutmut",
            profile=ProfileName.MUTATION,
            command=["mutmut", "run"],
            timeout_s=600.0,
        )
        relaxed.mutation_score_gate = 0.1
        relaxed.mutation_gate_from_cli = True
        with mock.patch.dict(os.environ, _tool_env(), clear=False):
            soft, _ = self._run(relaxed, project)
        self.assertEqual(soft.status, RunStatus.PASS, soft.summary_excerpt)
        self._assert_checkout_pristine(project)


class GatingAndAvailabilityTests(_MutationTestCase):
    """Policy gates must never turn missing tools or missing confinement into PASS."""

    def _suite(self, name: str = "mutmut", **kwargs) -> SuiteSpec:
        """Mutation suite with overridable adapter/command/timeout plus any ``SuiteSpec`` field."""
        return SuiteSpec(
            name=name,
            adapter=kwargs.pop("adapter", "mutmut"),
            profile=ProfileName.MUTATION,
            command=kwargs.pop("command", ["mutmut", "run"]),
            timeout_s=kwargs.pop("timeout_s", 60.0),
            **kwargs,
        )

    def test_untrusted_manifest_gate_is_blocked_not_evaluated(self) -> None:
        """Gate A: a manifest-declared gate is refused without manifest trust.

        ``custom_from_manifest`` without ``mutation_gate_from_cli`` is exactly
        the shape discovery produces for ``mutation_score_gate`` in
        ``.testbench.toml``. BLOCKED with ``UNTRUSTED_MANIFEST`` is asserted
        rather than "not PASS", so an ERROR that happens to be red does not
        satisfy the test.
        """
        project = self.root / "untrusted"
        _write_mutmut_project(project)
        suite = self._suite(custom_from_manifest=True)
        suite.mutation_score_gate = 0.9
        with mock.patch.dict(os.environ, _tool_env(), clear=False):
            result, _ = self._run(suite, project, trusted_manifest=False)
        self.assertEqual(result.status, RunStatus.BLOCKED)
        self.assertEqual(result.blocked_count, 1)
        self.assertEqual(result.failures[0].kind, FailureKind.UNTRUSTED_MANIFEST)
        self.assertIn("untrusted manifest", result.summary_excerpt)
        self._assert_checkout_pristine(project)

    @unittest.skipUnless(MUTMUT_BIN.is_file(), "mutmut binary unavailable")
    @unittest.skipUnless(can_apply_seatbelt(), "macOS sandbox-exec unavailable")
    def test_trusted_manifest_gate_is_evaluated(self) -> None:
        """Positive control for Gate A: with trust, the same manifest gate is scored.

        The gate is 0.1 so it passes; the assertion on a non-None score is what
        proves the run actually measured something rather than short-circuiting
        to PASS.
        """
        project = self.root / "trusted"
        _write_mutmut_project(project)
        suite = self._suite(custom_from_manifest=True)
        suite.mutation_score_gate = 0.1
        with mock.patch.dict(os.environ, _tool_env(), clear=False):
            result, _ = self._run(suite, project, trusted_manifest=True)
        self.assertEqual(result.status, RunStatus.PASS, result.summary_excerpt)
        self.assertIsNotNone(result.metrics["mutation"]["score"])

    def test_missing_binary_returns_skip_never_pass(self) -> None:
        """Gate C alone: confinement is mocked available so the binary decides.

        On Linux ``can_apply_seatbelt()`` is False, so without the mock Gate B
        would win and the test would pass for the wrong reason. Only the two env
        overrides are removed and ``PATH`` narrowed; ``resolve_tool`` still
        searches ``extra_bin_dirs()`` (``~/.local/bin``, ``/opt/homebrew/bin``,
        ``/usr/local/bin``), so a tool installed there resolves anyway. Narrowing
        ``PATH`` therefore does not guarantee a missing tool: a known
        environmental hermeticity defect, fixed separately by mocking
        ``extra_bin_dirs`` (c43254f); the test logic here is unchanged
        (``mutation.py:124-129``, ``isolation.py:141-149``).
        """
        project = self.root / "notool"
        _write_mutmut_project(project)
        # Narrow ``PATH`` and drop both overrides. Confinement is mocked as available
        # so this exercises Gate C (missing binary) alone: on Linux CI there is no
        # ``sandbox-exec`` and the confinement gate would otherwise win.
        env = {
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "TESTBENCH_MUTMUT": "",
            "TESTBENCH_CARGO_MUTANTS": "",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            os.environ.pop("TESTBENCH_MUTMUT", None)
            os.environ.pop("TESTBENCH_CARGO_MUTANTS", None)
            with mock.patch.object(mutation_mod, "can_apply_seatbelt", return_value=True):
                result, _ = self._run(self._suite(), project)
        self.assertEqual(result.status, RunStatus.SKIP)
        self.assertNotEqual(result.status, RunStatus.PASS)
        self.assertEqual(result.skip_count, 1)
        self.assertIn("required binary not found", result.summary_excerpt)
        self._assert_checkout_pristine(project)

    def test_unconfined_platform_refuses_to_spawn_the_tool(self) -> None:
        """Gate B: no sandbox-exec means BLOCKED with ``SANDBOX_DENIED``.

        ``pass_count == 0`` is asserted as well as the status, because "blocked"
        and "did not pass anything" are separate claims and only the second one
        rules out a partially-executed suite.
        """
        project = self.root / "unconfined"
        _write_mutmut_project(project)
        env = dict(_tool_env())
        env["TESTBENCH_SEATBELT_ACTIVE"] = ""
        with mock.patch.dict(os.environ, env, clear=False):
            os.environ.pop("TESTBENCH_SEATBELT_ACTIVE", None)
            with mock.patch.object(mutation_mod, "can_apply_seatbelt", return_value=False):
                result, _ = self._run(self._suite(), project)
        self.assertEqual(result.status, RunStatus.BLOCKED)
        self.assertEqual(result.failures[0].kind, FailureKind.SANDBOX_DENIED)
        self.assertIn("confinement unavailable", result.summary_excerpt)
        self.assertEqual(result.pass_count, 0)
        self._assert_checkout_pristine(project)

    def test_spoofed_seatbelt_env_var_still_blocks_when_seatbelt_unavailable(self) -> None:
        """``TESTBENCH_SEATBELT_ACTIVE`` is a marker, not an escape hatch.

        The tool is mocked resolvable and the env marker is set, so the only
        thing that can prevent a spawn is the platform capability check.
        ``spawns == []`` (via ``on_spawn``) is the assertion that matters: a
        BLOCKED status alone would not prove that nothing ran.
        """
        project = self.root / "spoofed"
        _write_mutmut_project(project)
        spawns = []
        env = dict(_tool_env())
        env["TESTBENCH_SEATBELT_ACTIVE"] = "1"
        with mock.patch.dict(os.environ, env, clear=False):
            with mock.patch.object(mutation_mod, "can_apply_seatbelt", return_value=False):
                with mock.patch.object(
                    mutation_mod, "resolve_tool", return_value=["/bin/true"]
                ):
                    self.assertFalse(mutation_mod.can_apply_seatbelt())
                    result, _ = self._run(
                        self._suite(), project, on_spawn=lambda *a: spawns.append(a)
                    )
        # ``TESTBENCH_SEATBELT_ACTIVE`` is only an environment variable, so it is not
        # evidence that this process can install a per-stage Seatbelt profile. On a host
        # without ``sandbox-exec`` (every Linux box) the tool must never be spawned.
        self.assertEqual(result.status, RunStatus.BLOCKED)
        self.assertEqual(result.blocked_count, 1)
        self.assertEqual(result.failures[0].kind, FailureKind.SANDBOX_DENIED)
        self.assertIn("confinement unavailable", result.summary_excerpt)
        self.assertEqual(spawns, [], "a mutation tool was spawned without confinement")
        self.assertEqual(result.pass_count, 0)
        self._assert_checkout_pristine(project)

    def test_suite_cwd_cannot_escape_the_staged_copy(self) -> None:
        """Lexical containment: absolute, ``..`` and nested traversal are ERROR.

        Each bad cwd is a separate run, and ``spawns`` is asserted empty only
        after the loop — a case that reports ERROR *after* spawning would
        otherwise slip through. ``HARNESS_ERROR`` (not a spawn error) keeps the
        blame on the harness's own containment check.
        """
        project = self.root / "escapedcwd"
        _write_mutmut_project(project)
        spawns = []
        for bad in ("../outside", "/tmp", "src/../../outside"):
            suite = self._suite(cwd=bad)
            with mock.patch.object(
                mutation_mod, "can_apply_seatbelt", return_value=True
            ), mock.patch.object(mutation_mod, "resolve_tool", return_value=["/bin/true"]):
                result, _ = self._run(suite, project, on_spawn=lambda *a: spawns.append(a))
            self.assertEqual(
                result.status, RunStatus.ERROR, f"{bad}: {result.summary_excerpt}"
            )
            self.assertEqual(result.failures[0].kind, FailureKind.HARNESS_ERROR)
            self.assertIn("escapes the staged copy", result.summary_excerpt)
        self.assertEqual(
            spawns, [], "a mutation tool ran from a working directory outside the stage"
        )
        self._assert_checkout_pristine(project)

    def test_symlinked_staged_cwd_is_rejected_after_staging(self) -> None:
        """The pre-staging lexical check cannot see links the copy itself creates.

        ``stage_project_tree`` copies symlinks verbatim, so a symlinked suite cwd can pass the
        ``resolve()``-based containment check (the link does not exist yet in the empty stage)
        and only become an escape once the copy lands on disk. Re-verifying after staging must
        reject it, spawn nothing, and remove the disposable stage.

        "symlink escape" discriminates the post-copy check from the lexical one
        (which says "escapes the staged copy"), and the empty scratch glob proves
        the rejected path still runs the ``finally`` teardown.
        """
        project = self.root / "symlinkcwd"
        _write_mutmut_project(project)
        outside = self.root / "outside-target"
        outside.mkdir()
        (project / "linked").symlink_to(outside, target_is_directory=True)
        subprocess.run(["git", "-C", str(project), "add", "-A"], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(project),
                "-c",
                "user.email=tb@example.invalid",
                "-c",
                "user.name=testbench",
                "commit",
                "-qm",
                "fixture: symlinked cwd",
            ],
            check=True,
        )
        spawns = []
        suite = self._suite(cwd="linked")
        with mock.patch.object(
            mutation_mod, "can_apply_seatbelt", return_value=True
        ), mock.patch.object(mutation_mod, "resolve_tool", return_value=["/bin/true"]):
            result, _ = self._run(suite, project, on_spawn=lambda *a: spawns.append(a))
        self.assertEqual(result.status, RunStatus.ERROR, result.summary_excerpt)
        self.assertEqual(result.failures[0].kind, FailureKind.HARNESS_ERROR)
        self.assertIn("symlink escape", result.summary_excerpt)
        self.assertEqual(spawns, [], "a mutation tool ran through a symlinked staged cwd")
        self.assertEqual(
            list(self.scratch.glob("tbmut-*")),
            [],
            "the disposable stage survived a rejected cwd",
        )
        self._assert_checkout_pristine(project)


class MutationGateValidationTests(unittest.TestCase):
    """A gate is a fraction of caught mutants; out-of-range values fail loudly."""

    def _suite(self) -> SuiteSpec:
        """An ungated mutation suite; the gate value itself is the variable under test."""
        return SuiteSpec(
            name="mutmut",
            adapter="mutmut",
            profile=ProfileName.MUTATION,
            command=["mutmut", "run"],
            timeout_s=60.0,
        )

    def test_engine_rejects_gates_outside_the_unit_interval(self) -> None:
        """Out-of-range and NaN gates raise, and stamp nothing.

        The post-loop ``mutation_score_gate is None`` / ``mutation_gate_from_cli
        is False`` assertions are the important half: validation that raises
        *after* mutating the first suite would still leave it corrupted.
        """
        from hermes_testbench.engine import _apply_mutation_gate

        suite = self._suite()
        for bad in (-1.0, 8.0, 1.5, float("nan")):
            with self.assertRaises(ValueError, msg=f"gate {bad!r} was accepted"):
                _apply_mutation_gate([suite], bad)
        self.assertIsNone(suite.mutation_score_gate)
        self.assertFalse(suite.mutation_gate_from_cli)

    def test_cli_exits_nonzero_on_an_out_of_range_gate(self) -> None:
        """The CLI turns the ``ValueError`` into exit 1, including for ``nan``.

        Strings are what argparse yields, so this covers the conversion path the
        engine-level test does not: ``nan`` must be rejected after parsing, not
        compared as a float.
        """
        from hermes_testbench import cli

        for bad in ("-1", "8", "nan"):
            code = cli.main(
                ["run", ".", "--profile", "mutation", f"--mutation-gate={bad}"]
            )
            self.assertEqual(code, 1, f"--mutation-gate={bad} exited {code}")


class SeatbeltStagingRegressionTests(_MutationTestCase):
    """Seatbelt must allow writes inside the staged copy and deny them in the checkout."""

    @unittest.skipUnless(can_apply_seatbelt(), "macOS sandbox-exec unavailable")
    def test_stage_dir_is_writable_while_real_project_root_is_read_only(self) -> None:
        """The two halves in one probe: a write in the stage succeeds, the same write
        in the real checkout raises.

        Asserting the child's exit status *and* the printed verdict keeps the two
        outcomes distinguishable: an exception before the first write would also
        leave the real root untouched, but would print nothing and exit non-zero.
        """
        project = self.root / "realrepo"
        project.mkdir()
        (project / "source.py").write_text("VALUE = 1\n", encoding="utf-8")
        stage = self.root / "stage"
        shutil.copytree(project, stage)

        script = (
            "import pathlib, sys\n"
            f"stage = pathlib.Path({str(stage)!r})\n"
            f"real = pathlib.Path({str(project)!r})\n"
            "(stage / 'staged-write.txt').write_text('ok')\n"
            "denied = False\n"
            "try:\n"
            "    (real / 'staged-write.txt').write_text('bad')\n"
            "except OSError:\n"
            "    denied = True\n"
            "print(f'STAGE=ok REAL={\"denied\" if denied else \"allowed\"}')\n"
            "sys.exit(0 if denied else 1)\n"
        )
        res = run_isolated_command(
            [PY, "-c", script],
            cwd=stage,
            evidence_dir=self.evidence,
            step_name="stage_write_probe",
            run_id="tb_mut_sb",
            allow_repo_write=False,
            extra_writable_dirs=(stage,),
            scratch_root=self.scratch,
        )
        self.assertEqual(res.isolation_meta.get("seatbelt"), "active")
        self.assertEqual(res.exit_code, 0, res.stdout_text + res.stderr_text)
        self.assertIn("STAGE=ok REAL=denied", res.stdout_text)
        self.assertTrue((stage / "staged-write.txt").is_file())
        self.assertFalse((project / "staged-write.txt").exists())

    @unittest.skipUnless(can_apply_seatbelt(), "macOS sandbox-exec unavailable")
    def test_without_extra_writable_dirs_the_stage_is_read_only(self) -> None:
        """The relaxation exists only for the internal staging call, never by default.

        Same probe as the previous test with ``extra_writable_dirs`` omitted: if
        the stage were writable here, the staging helper's permission would be
        indistinguishable from the sandbox's default, and every other suite would
        inherit it.
        """
        stage = self.root / "stage-default"
        stage.mkdir()
        script = (
            "import pathlib, sys\n"
            f"p = pathlib.Path({str(stage)!r}) / 'nope.txt'\n"
            "try:\n"
            "    p.write_text('bad')\n"
            "    print('WRITE=allowed')\n"
            "except OSError:\n"
            "    print('WRITE=denied')\n"
        )
        res = run_isolated_command(
            [PY, "-c", script],
            cwd=stage,
            evidence_dir=self.evidence,
            step_name="stage_default_probe",
            run_id="tb_mut_sb2",
            scratch_root=self.scratch,
        )
        self.assertIn("WRITE=denied", res.stdout_text)
        self.assertFalse((stage / "nope.txt").exists())


class MutationParserTests(unittest.TestCase):
    """Every canonical bucket must be reachable from captured tool text."""

    def test_cargo_mutants_outcomes_json_covers_all_buckets(self) -> None:
        """The structured ``outcomes.json`` path: one entry per bucket, baseline excluded.

        ``total`` counts mutant entries only (the Baseline scenario is not a
        mutant), and ``passed``/``failed`` are the caught/survived projections;
        the case statuses pin the mapping used by the suite's per-mutant rows.
        """
        payload = {
            "total_mutants": 5,
            "outcomes": [
                {"scenario": "Baseline", "summary": "Success"},
                {"scenario": {"Mutant": {"name": "src/lib.rs:1:5: c1"}}, "summary": "CaughtMutant"},
                {"scenario": {"Mutant": {"name": "src/lib.rs:2:5: m1"}}, "summary": "MissedMutant"},
                {"scenario": {"Mutant": {"name": "src/lib.rs:3:5: u1"}}, "summary": "Unviable"},
                {"scenario": {"Mutant": {"name": "src/lib.rs:4:5: t1"}}, "summary": "Timeout"},
                {"scenario": {"Mutant": {"name": "src/lib.rs:5:5: f1"}}, "summary": "Failure"},
            ],
        }
        outcome = parsers.parse_cargo_mutants("", outcomes_json=json.dumps(payload))
        self.assertEqual(
            outcome.buckets,
            {"caught": 1, "survived": 1, "invalid": 1, "timeout": 1, "inconclusive": 1},
        )
        self.assertTrue(outcome.structured)
        self.assertFalse(outcome.baseline_failed)
        self.assertEqual(outcome.total, 5)
        self.assertEqual(outcome.passed, 1)
        self.assertEqual(outcome.failed, 1)
        statuses = {c["name"]: c["status"] for c in outcome.cases}
        self.assertEqual(statuses["src/lib.rs:1:5: c1"], "PASS")
        self.assertEqual(statuses["src/lib.rs:2:5: m1"], "FAIL")

    def test_cargo_mutants_console_lines_carry_every_bucket(self) -> None:
        """Console classification, used when ``outcomes.json`` is absent.

        Each status word appears exactly once, so a parser that assigned every
        UNVIABLE/FAILED line to the same bucket would fail one of the five
        assertions rather than passing by coincidence.
        """
        text = (
            "Found 6 mutants to test\n"
            "ok       Unmutated baseline in 0s build + 0s test\n"
            "CAUGHT   src/lib.rs:2:5: replace add -> i32 with 0 in 0s build + 0s test\n"
            "MISSED   src/lib.rs:6:10: replace > with >= in classify in 0s build + 0s test\n"
            "TIMEOUT  src/lib.rs:7:5: replace loop with () in spin in 0s build + 20s test\n"
            "UNVIABLE src/lib.rs:9:5: replace body with todo!() in broken\n"
            "FAILED   src/lib.rs:11:5: replace x with y in weird in 0s build + 0s test\n"
            "6 mutants tested in 5s: 1 caught, 1 missed, 1 timeout, 1 unviable, 1 failed\n"
        )
        outcome = parsers.parse_cargo_mutants(text)
        self.assertEqual(outcome.buckets["caught"], 1)
        self.assertEqual(outcome.buckets["survived"], 1)
        self.assertEqual(outcome.buckets["timeout"], 1)
        self.assertEqual(outcome.buckets["invalid"], 1)
        self.assertEqual(outcome.buckets["inconclusive"], 1)
        self.assertFalse(outcome.baseline_failed)

    def test_cargo_mutants_tally_only_log_is_classified(self) -> None:
        """A tally-only log still yields buckets: the summary lines are a fallback.

        Two summaries are summed (6 missed + 3 caught, then 1 timeout), so a
        parser that read only the last line would report 0/0/1 instead; 3/6/1 is
        the first-line shape.
        """
        outcome = parsers.parse_cargo_mutants(
            "9 mutants tested in 5s: 6 missed, 3 caught\n1 mutants tested in 1s: 1 timeout\n"
        )
        self.assertEqual(outcome.buckets["caught"], 3)
        self.assertEqual(outcome.buckets["survived"], 6)
        self.assertEqual(outcome.buckets["timeout"], 1)

    def test_cargo_mutants_baseline_failure_is_inconclusive_not_pass(self) -> None:
        """A broken unmutated baseline is inconclusive, never a pass.

        Without the ``baseline_failed`` flag an empty bucket set reads as "no
        survivors", which is the most dangerous possible misreading; one
        inconclusive bucket plus ``passed == 0`` is the honest result.
        """
        outcome = parsers.parse_cargo_mutants(
            "error: test failed, to rerun pass `-p demo --test basic`\n"
            "*** result: Failure(101)\n"
            "ERROR cargo test failed in an unmutated tree, so no mutants were tested\n"
        )
        self.assertTrue(outcome.baseline_failed)
        self.assertEqual(outcome.buckets["inconclusive"], 1)
        self.assertEqual(outcome.passed, 0)

    def test_mutmut_results_text_covers_all_buckets(self) -> None:
        """``mutmut results`` text: killed and "caught by" are both caught.

        Unknown status words (``suspicious``, ``skipped``) and ``no tests`` land
        in ``inconclusive``, not ``invalid``; the 3/0 split is asserted
        explicitly so a parser that widened ``invalid`` cannot pass unseen.
        """
        results_text = (
            "    demo.x_add__mutmut_1: killed\n"
            "    demo.x_add__mutmut_2: caught by type check\n"
            "    demo.x_classify__mutmut_1: survived\n"
            "    demo.x_timeout__mutmut_1: timeout\n"
            "    demo.x_untested__mutmut_1: no tests\n"
            "    demo.x_odd__mutmut_1: suspicious\n"
            "    demo.x_odd__mutmut_2: skipped\n"
        )
        outcome = parsers.parse_mutmut("", results_text=results_text)
        self.assertEqual(outcome.buckets["caught"], 2)
        self.assertEqual(outcome.buckets["survived"], 1)
        self.assertEqual(outcome.buckets["timeout"], 1)
        self.assertEqual(outcome.buckets["inconclusive"], 3)
        self.assertEqual(outcome.buckets["invalid"], 0)
        self.assertEqual(outcome.total, 7)

    def test_mutmut_progress_tally_is_the_fallback(self) -> None:
        """The emoji progress line is the fallback when no per-mutant line is recognised.

        The two carriage-return updates carry the same numbers, so the parser
        must be idempotent per line rather than summing every redraw. The tally
        is positional: ``🎉`` is caught, ``🙁`` survived, ``🫥`` inconclusive —
        asserted separately so a shifted column fails instead of passing.
        """
        run_text = (
            "Running mutation testing\n"
            "\r⠼ 12/12  🎉 6 🫥 5  ⏰ 0  🤔 0  🙁 1  🔇 0  🧙 0\r"
            "\r⠴ 12/12  🎉 6 🫥 5  ⏰ 0  🤔 0  🙁 1  🔇 0  🧙 0\n"
        )
        outcome = parsers.parse_mutmut(run_text)
        self.assertEqual(outcome.buckets["caught"], 6)
        self.assertEqual(outcome.buckets["survived"], 1)
        self.assertEqual(outcome.buckets["inconclusive"], 5)
        self.assertEqual(outcome.total, 12)

    def test_mutmut_baseline_failure_marker(self) -> None:
        """A baseline marker from the run log sets ``baseline_failed``.

        The marker ("runner returned 1") is only trusted with no per-mutant
        verdict present; this asserts the no-verdict case, where the run would
        otherwise look like zero surviving mutants.
        """
        outcome = parsers.parse_mutmut(
            "Running clean tests\n"
            "1 failed, 2 passed in 0.01s\n"
            "failed to collect stats. runner returned 1\n"
        )
        self.assertTrue(outcome.baseline_failed)
        self.assertEqual(outcome.buckets["inconclusive"], 1)
        self.assertEqual(outcome.passed, 0)

    def test_mutmut_statuses_map_to_canonical_case_statuses(self) -> None:
        """killed→PASS, survived→FAIL, timeout→SKIP is the per-case projection.

        The exact dict comparison catches an added or renamed case as well as a
        changed status: a skip that became a pass would silently improve a
        stored run's ``pass_count``.
        """
        outcome = parsers.parse_mutmut(
            "", results_text="    m1: killed\n    m2: survived\n    m3: timeout\n"
        )
        by_name = {c["name"]: c["status"] for c in outcome.cases}
        self.assertEqual(by_name, {"m1": "PASS", "m2": "FAIL", "m3": "SKIP"})


class MutationDiscoveryTests(_MutationTestCase):
    """Discovery must be opt-in and must describe the staged-copy contract, not guess."""

    def test_adapters_and_profile_are_registered(self) -> None:
        """Registration is the wiring three other layers assume: adapter names, profile, parsers."""
        self.assertIn("cargo_mutants", KNOWN_ADAPTERS)
        self.assertIn("mutmut", KNOWN_ADAPTERS)
        self.assertIn(ProfileName.MUTATION, ProfileName.STANDARD)
        self.assertIn("cargo_mutants", parsers.ADAPTER_PARSERS)
        self.assertIn("mutmut", parsers.ADAPTER_PARSERS)

    def test_rust_project_discovers_a_mutation_suite(self) -> None:
        """Opt-in: a plain ``Cargo.toml`` yields nothing; ``.cargo/mutants.toml`` adds it.

        The first assertion is the negative control — discovery must not infer
        mutation from a language marker, because the suite costs minutes to
        hours. ``allow_repo_write=False`` pins that the discovered suite is
        described as staged-copy work, which is what the adapter depends on.
        """
        project = self.root / "rustdiscover"
        (project / "src").mkdir(parents=True)
        (project / "Cargo.toml").write_text(
            '[package]\nname = "demo"\nversion = "0.1.0"\n', encoding="utf-8"
        )
        (project / "src" / "lib.rs").write_text("pub fn one() -> i32 { 1 }\n", encoding="utf-8")
        # A plain Cargo.toml is never enough: mutation must be opted into via the tool's
        # own configuration, because the suite costs minutes to hours.
        plain = [s for s in discover_project(project).suites if s.profile == ProfileName.MUTATION]
        self.assertEqual(plain, [])

        (project / ".cargo").mkdir()
        (project / ".cargo" / "mutants.toml").write_text("timeout_multiplier = 2.0\n", encoding="utf-8")
        spec = discover_project(project)
        mutation_suites = [s for s in spec.suites if s.profile == ProfileName.MUTATION]
        self.assertEqual([s.adapter for s in mutation_suites], ["cargo_mutants"])
        self.assertEqual(mutation_suites[0].name, "cargo-mutants")
        self.assertFalse(mutation_suites[0].allow_repo_write)

    def test_mutmut_config_discovers_a_mutation_suite(self) -> None:
        """``[tool.mutmut]`` is the Python opt-in marker, mirroring the Rust check."""
        project = self.root / "pydiscover"
        (project / "src").mkdir(parents=True)
        (project / "tests").mkdir()
        (project / "pyproject.toml").write_text(
            '[project]\nname = "demo"\nversion = "0.1.0"\n\n[tool.mutmut]\nsource_paths = ["src"]\n',
            encoding="utf-8",
        )
        (project / "tests" / "test_x.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
        spec = discover_project(project)
        mutation_suites = [s for s in spec.suites if s.profile == ProfileName.MUTATION]
        self.assertEqual([s.adapter for s in mutation_suites], ["mutmut"])

    def test_python_project_without_mutmut_config_has_no_mutation_suite(self) -> None:
        """No opt-in, no suite: a Python project with tests is not a mutation target."""
        project = self.root / "plainpy"
        (project / "tests").mkdir(parents=True)
        (project / "pyproject.toml").write_text(
            '[project]\nname = "demo"\nversion = "0.1.0"\n', encoding="utf-8"
        )
        (project / "tests" / "test_x.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
        spec = discover_project(project)
        self.assertFalse([s for s in spec.suites if s.profile == ProfileName.MUTATION])

    def test_manifest_gate_is_parsed_and_validated(self) -> None:
        """A manifest gate is parsed and attributed; out of range fails at discovery.

        ``mutation_gate_from_cli`` must stay False — that flag is what tells the
        adapter the gate came from an operator rather than from the untrusted
        manifest. The second half rewrites the value to ``8`` and expects
        ``ValueError``, so an invalid gate cannot reach a run at all.
        """
        project = self.root / "manifestgate"
        project.mkdir()
        (project / ".testbench.toml").write_text(
            "[project]\n"
            'name = "gated"\n'
            "\n"
            "[[suite]]\n"
            'name = "rust-mutation"\n'
            'adapter = "cargo_mutants"\n'
            'profile = "mutation"\n'
            'command = ["cargo-mutants", "mutants"]\n'
            "mutation_score_gate = 0.8\n",
            encoding="utf-8",
        )
        spec = discover_project(project)
        self.assertEqual(len(spec.suites), 1)
        suite = spec.suites[0]
        self.assertEqual(suite.mutation_score_gate, 0.8)
        self.assertFalse(suite.mutation_gate_from_cli)
        self.assertTrue(suite.custom_from_manifest)

        (project / ".testbench.toml").write_text(
            (project / ".testbench.toml")
            .read_text(encoding="utf-8")
            .replace("mutation_score_gate = 0.8", "mutation_score_gate = 8"),
            encoding="utf-8",
        )
        with self.assertRaises(ValueError):
            discover_project(project)

    def test_staging_skips_build_and_mutation_artifacts(self) -> None:
        """Staging copies sources and drops caches/build output/prior mutant state.

        The prior-run directories matter most: if ``mutants.out`` were copied,
        a stale verdict could be read as this run's. ``keep.py`` is the positive
        control proving the copy is not simply empty.
        """
        project = self.root / "stageskip"
        (project / "src").mkdir(parents=True)
        (project / "src" / "keep.py").write_text("x = 1\n", encoding="utf-8")
        for name in ("target", "mutants", "mutants.out", ".mutmut-cache", "node_modules"):
            (project / name).mkdir()
            (project / name / "junk.txt").write_text("junk\n", encoding="utf-8")
        stage = self.root / "stage-copy"
        mutation_mod.stage_project_tree(project, stage)
        self.assertTrue((stage / "src" / "keep.py").is_file())
        for name in ("target", "mutants", "mutants.out", ".mutmut-cache", "node_modules"):
            self.assertFalse((stage / name).exists(), f"{name} should not be staged")


if __name__ == "__main__":
    unittest.main()
