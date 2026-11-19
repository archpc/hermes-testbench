"""End-to-end TDD tests for Phase 2A flake hunting.

Every attempt is a full independent ``run_sync`` execution, so these tests use
real fixture suites from ``.testbench.toml`` manifests, run through the real
CLI subprocess, and then assert against the persisted store and evidence
directories. The flaky fixture is a stateful counter: attempt 1 fails and every
later attempt passes, which proves a passing retry never erases the failed
attempt or its artifacts.

Two layers, and only the second one pays for subprocesses:

* ``FlakeClassificationUnitTests`` builds ``RunRecord`` objects in memory and
  covers ``flake.py`` alone — verdict edges, target discovery, and the
  participation gate. No store, no fixture, no timing.
* ``FlakeHunterEndToEndTests`` shells out to ``python -m hermes_testbench``
  with its own ``--home``, writes the manifest into a temp project, and reads
  the results back from the store. Only the timeout test passes ``--timeout``
  (0.5 s); the rest rely on a fast fixture and per-subprocess timeouts, so a
  hung child fails the test rather than stalling the suite.

Nothing here is skipped: a missing configured tool would be a fixture bug, not
an environment difference, and the timeout/skip cases construct those states
deliberately.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from hermes_testbench.flake import (
    FlakeVerdict,
    analyze_flake_history,
    build_report,
    classify_statuses,
    flake_metadata,
)
from hermes_testbench.schema import (
    SCHEMA_VERSION,
    CaseResult,
    RunRecord,
    RunStatus,
    SuiteResult,
)
from hermes_testbench.store import TestBenchStore

PY = sys.executable
REPO_ROOT = Path(__file__).resolve().parent.parent

PASS_SCRIPT = "import sys\nprint('all good')\nsys.exit(0)\n"
FAIL_SCRIPT = "import sys\nprint('boom', file=sys.stderr)\nsys.exit(7)\n"
SLEEP_SCRIPT = "import time\ntime.sleep(30)\n"
FLAKY_SCRIPT = """
from pathlib import Path
import sys

state = Path(__file__).with_name("attempt_counter.txt")
n = int(state.read_text().strip() or 0) if state.exists() else 0
n += 1
state.write_text(str(n))
print(f"attempt {n}")
sys.exit(0 if n >= 2 else 1)
"""

CASE_SCRIPT = """
import unittest


class SampleCases(unittest.TestCase):
    def test_always_ok(self):
        self.assertTrue(True)

    def test_always_bad(self):
        self.assertEqual(1, 2)
"""


def _toml_value(value) -> str:
    """Serialize one manifest value the way the discovery parser expects to read it."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    raise TypeError(f"unsupported manifest value: {value!r}")


def _manifest(name: str, suites: list) -> str:
    chunks = ["[project]", f"name = {json.dumps(name)}", ""]
    for suite in suites:
        chunks.append("[[suites]]")
        for key in (
            "name",
            "adapter",
            "profile",
            "command",
            "cwd",
            "timeout_s",
            "repeat_runs",
            "allow_repo_write",
            "required_binaries",
            "required_paths",
            "description",
        ):
            if key in suite:
                chunks.append(f"{key} = {_toml_value(suite[key])}")
        chunks.append("")
    return "\n".join(chunks)


def _suite(name: str, script: str, **extra) -> dict:
    base = {
        "name": name,
        "adapter": "shell",
        "profile": "unit",
        "command": [PY, script],
        "timeout_s": 20.0,
    }
    base.update(extra)
    return base


def _make_run(
    run_id: str,
    status: str,
    *,
    attempt: int = 1,
    repeats: int = 2,
    group: str = "fg_unit",
    suite_status: str = "",
    cases: tuple = (),
) -> RunRecord:
    suite = SuiteResult(
        name="unit-suite",
        adapter="shell",
        profile="unit",
        command=["true"],
        cwd="/",
        status=suite_status or status,
        cases=[CaseResult(name=n, suite="unit-suite", status=s) for n, s in cases],
    )
    return RunRecord(
        run_id=run_id,
        project="flake_demo",
        project_root="/",
        revision="rev1",
        profile="unit",
        status=status,
        created_at=f"2026-11-15T00:00:0{attempt}.000000Z",
        environment={"flake": {"group_id": group, "attempt": attempt, "repeats": repeats}},
        evidence_dir=f"/runs/{run_id}",
        suites=[suite],
    )


class FlakeClassificationUnitTests(unittest.TestCase):
    """Pure classification contract: no store, no subprocess, no fixture."""

    def test_status_matrix(self) -> None:
        P, F = RunStatus.PASS, RunStatus.FAIL
        self.assertEqual(classify_statuses([P]), FlakeVerdict.INCONCLUSIVE)
        self.assertEqual(classify_statuses([]), FlakeVerdict.INCONCLUSIVE)
        self.assertEqual(classify_statuses([P, P, P]), FlakeVerdict.ALWAYS_PASS)
        self.assertEqual(classify_statuses([F, F]), FlakeVerdict.ALWAYS_FAIL)
        self.assertEqual(classify_statuses([P, F]), FlakeVerdict.MIXED)
        self.assertEqual(classify_statuses([F, P, P]), FlakeVerdict.MIXED)
        self.assertEqual(classify_statuses([P, RunStatus.TIMEOUT]), FlakeVerdict.INCONCLUSIVE)
        self.assertEqual(classify_statuses([P, RunStatus.SKIP]), FlakeVerdict.INCONCLUSIVE)
        self.assertEqual(classify_statuses([RunStatus.SKIP, F]), FlakeVerdict.INCONCLUSIVE)
        self.assertEqual(classify_statuses([P, F, RunStatus.ERROR]), FlakeVerdict.MIXED)

    def test_mixed_report_preserves_both_run_id_sets(self) -> None:
        runs = [
            _make_run("tb_a", RunStatus.FAIL, attempt=1),
            _make_run("tb_b", RunStatus.PASS, attempt=2),
        ]
        report = build_report(runs, expected_repeats=2)
        self.assertEqual(report.verdict, FlakeVerdict.MIXED)
        suite = report.suite_targets[0]
        self.assertEqual(suite.verdict, FlakeVerdict.MIXED)
        self.assertEqual(suite.passing_run_ids, ["tb_b"])
        self.assertEqual(suite.failing_run_ids, ["tb_a"])
        self.assertEqual(suite.passing_evidence_dirs, ["/runs/tb_b"])
        self.assertEqual(suite.failing_evidence_dirs, ["/runs/tb_a"])
        self.assertEqual(suite.statuses, [RunStatus.FAIL, RunStatus.PASS])
        self.assertTrue(suite.is_flaky)

    def test_case_level_targets_are_classified_independently(self) -> None:
        runs = [
            _make_run(
                "tb_a",
                RunStatus.FAIL,
                attempt=1,
                suite_status=RunStatus.FAIL,
                cases=(("test_ok", RunStatus.PASS), ("test_flaky", RunStatus.FAIL)),
            ),
            _make_run(
                "tb_b",
                RunStatus.PASS,
                attempt=2,
                cases=(("test_ok", RunStatus.PASS), ("test_flaky", RunStatus.PASS)),
            ),
        ]
        report = build_report(runs, expected_repeats=2)
        self.assertEqual(report.verdict, FlakeVerdict.MIXED)
        by_name = {t.case: t for t in report.case_targets}
        self.assertEqual(by_name["test_ok"].verdict, FlakeVerdict.ALWAYS_PASS)
        self.assertEqual(by_name["test_flaky"].verdict, FlakeVerdict.MIXED)
        self.assertEqual(by_name["test_flaky"].passing_run_ids, ["tb_b"])
        self.assertEqual(by_name["test_flaky"].failing_run_ids, ["tb_a"])

    def test_missing_case_in_an_attempt_makes_that_case_inconclusive_alone(self) -> None:
        runs = [
            _make_run("tb_a", RunStatus.TIMEOUT, attempt=1, suite_status=RunStatus.TIMEOUT),
            _make_run("tb_b", RunStatus.PASS, attempt=2, cases=(("test_ok", RunStatus.PASS),)),
        ]
        report = build_report(runs, expected_repeats=2)
        self.assertEqual(report.verdict, FlakeVerdict.INCONCLUSIVE)
        case = report.case_targets[0]
        self.assertEqual(case.verdict, FlakeVerdict.INCONCLUSIVE)
        self.assertEqual(case.missing_attempts, [1])

    def test_plain_runs_without_flake_metadata_are_not_attempts(self) -> None:
        plain = RunRecord(
            run_id="tb_plain",
            project="flake_demo",
            project_root="/",
            revision="rev1",
            profile="unit",
            status=RunStatus.PASS,
        )
        self.assertIsNone(flake_metadata(plain))
        history = analyze_flake_history([plain])
        self.assertEqual(history.groups, [])

    def test_undecisive_single_attempt_is_inconclusive(self) -> None:
        report = build_report([_make_run("tb_a", RunStatus.PASS, repeats=1)], expected_repeats=1)
        self.assertEqual(report.verdict, FlakeVerdict.INCONCLUSIVE)

    def test_schema_version_is_untouched(self) -> None:
        self.assertEqual(SCHEMA_VERSION, 1)


class FlakeHunterEndToEndTests(unittest.TestCase):
    def setUp(self) -> None:
        scratch = Path.home() / ".hermes" / "cache" / "scratch"
        scratch.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=str(scratch), prefix="tb-flake-")
        self.root = Path(self.tmp.name)
        self.tb_home = self.root / "tb_home"
        self.scratch = self.root / "scratch"
        self.scratch.mkdir()
        self.proj = self.root / "flake_demo"
        self.proj.mkdir()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    # --- fixture helpers -------------------------------------------------

    def _script(self, name: str, body: str) -> str:
        path = self.proj / name
        path.write_text(body, encoding="utf-8")
        return str(path)

    def _write_manifest(self, suites: list, name: str = "flake_demo") -> None:
        (self.proj / ".testbench.toml").write_text(_manifest(name, suites), encoding="utf-8")

    def _cli(self, args: list, timeout: float = 120.0) -> subprocess.CompletedProcess:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO_ROOT)
        env["TMPDIR"] = str(self.scratch)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        return subprocess.run(
            [PY, "-m", "hermes_testbench", "--home", str(self.tb_home), *args],
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def _flake_run(self, *extra: str, timeout: float = 120.0) -> subprocess.CompletedProcess:
        return self._cli(["flake", "run", str(self.proj), *extra], timeout=timeout)

    def _store_runs(self, project: str = "flake_demo") -> list:
        with TestBenchStore(self.tb_home) as store:
            return store.list_runs(project=project, limit=100)

    @staticmethod
    def _payload(proc: subprocess.CompletedProcess) -> dict:
        try:
            return json.loads(proc.stdout)
        except json.JSONDecodeError as exc:  # pragma: no cover - diagnostics aid
            raise AssertionError(f"stdout was not JSON:\n{proc.stdout}\nstderr:\n{proc.stderr}") from exc

    # --- deterministic pass ---------------------------------------------

    def test_deterministic_pass_is_always_pass_exit_zero_with_distinct_runs(self) -> None:
        script = self._script("pass_case.py", PASS_SCRIPT)
        self._write_manifest([_suite("stable", script)])
        proc = self._flake_run("--repeats", "3", "--json")
        self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout)
        payload = self._payload(proc)
        self.assertEqual(payload["verdict"], FlakeVerdict.ALWAYS_PASS)
        self.assertEqual(payload["attempts"], 3)
        self.assertEqual(payload["expected_repeats"], 3)
        self.assertTrue(payload["group_id"])
        suite = payload["targets"][0]
        self.assertEqual(suite["verdict"], FlakeVerdict.ALWAYS_PASS)
        self.assertEqual(suite["pass_count"], 3)
        self.assertEqual(suite["fail_count"], 0)
        self.assertEqual(len(payload["run_ids"]), 3)
        self.assertEqual(len(set(payload["run_ids"])), 3)

        runs = self._store_runs()
        self.assertEqual(len(runs), 3)
        self.assertEqual({r.status for r in runs}, {RunStatus.PASS})
        for run in runs:
            meta = flake_metadata(run)
            assert meta is not None
            self.assertEqual(meta["group_id"], payload["group_id"])
            self.assertEqual(meta["repeats"], 3)
            self.assertIn(meta["attempt"], (1, 2, 3))
            self.assertTrue((Path(run.evidence_dir) / "run.json").is_file())
        attempts = sorted(int((flake_metadata(r) or {}).get("attempt", 0)) for r in runs)
        self.assertEqual(attempts, [1, 2, 3])

    # --- deterministic failure ------------------------------------------

    def test_deterministic_failure_is_always_fail_exit_one_and_all_runs_preserved(self) -> None:
        script = self._script("fail_case.py", FAIL_SCRIPT)
        self._write_manifest([_suite("broken", script)])
        proc = self._flake_run("--repeats", "3", "--json")
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        payload = self._payload(proc)
        self.assertEqual(payload["verdict"], FlakeVerdict.ALWAYS_FAIL)
        suite = payload["targets"][0]
        self.assertEqual(suite["verdict"], FlakeVerdict.ALWAYS_FAIL)
        self.assertEqual(suite["fail_count"], 3)
        self.assertEqual(suite["pass_count"], 0)

        runs = self._store_runs()
        self.assertEqual(len(runs), 3)
        self.assertEqual(len({r.run_id for r in runs}), 3)
        self.assertEqual({r.status for r in runs}, {RunStatus.FAIL})
        for run in runs:
            self.assertTrue((Path(run.evidence_dir) / "run.json").is_file())

    # --- stateful flake: fail attempt 1, pass 2..N ----------------------

    def test_flaky_counter_is_mixed_and_failed_attempt_survives_passing_retries(self) -> None:
        script = self._script("flaky_counter.py", FLAKY_SCRIPT)
        self._write_manifest([_suite("counter", script, allow_repo_write=True)])
        with TestBenchStore(self.tb_home) as store:
            store.trust_project(self.proj)

        proc = self._flake_run("--repeats", "3", "--json")
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        payload = self._payload(proc)
        self.assertEqual(payload["verdict"], FlakeVerdict.MIXED)
        suite = payload["targets"][0]
        self.assertEqual(suite["verdict"], FlakeVerdict.MIXED)
        self.assertEqual(suite["statuses"], [RunStatus.FAIL, RunStatus.PASS, RunStatus.PASS])
        self.assertEqual(suite["pass_count"], 2)
        self.assertEqual(suite["fail_count"], 1)
        self.assertEqual(len(suite["failing_run_ids"]), 1)
        self.assertEqual(len(suite["passing_run_ids"]), 2)
        failed_id = suite["failing_run_ids"][0]
        passing_ids = suite["passing_run_ids"]
        self.assertNotIn(failed_id, passing_ids)
        self.assertEqual(len(set(payload["run_ids"])), 3)

        # The failed attempt and its evidence must remain inspectable AFTER
        # the passing retries finished: no last-writer-wins overwrite.
        runs = {r.run_id: r for r in self._store_runs()}
        self.assertEqual(len(runs), 3)
        self.assertEqual(runs[failed_id].status, RunStatus.FAIL)
        self.assertEqual(runs[failed_id].environment["flake"]["attempt"], 1)
        failed_json = Path(runs[failed_id].evidence_dir) / "run.json"
        self.assertTrue(failed_json.is_file())
        failed_payload = json.loads(failed_json.read_text(encoding="utf-8"))
        self.assertEqual(failed_payload["status"], RunStatus.FAIL)
        for rid in passing_ids:
            self.assertEqual(runs[rid].status, RunStatus.PASS)
            self.assertTrue((Path(runs[rid].evidence_dir) / "run.json").is_file())

        # Historical analysis sees the same metadata with no re-execution.
        analyze = self._cli(
            ["flake", "analyze", "--group", payload["group_id"], "--json"],
        )
        self.assertEqual(analyze.returncode, 0, analyze.stderr + analyze.stdout)
        history = self._payload(analyze)
        self.assertEqual(len(history["groups"]), 1)
        self.assertEqual(history["groups"][0]["verdict"], FlakeVerdict.MIXED)
        self.assertEqual(history["groups"][0]["targets"][0]["failing_run_ids"], [failed_id])

    # --- timeout / skip are inconclusive --------------------------------

    def test_timeout_attempts_are_inconclusive_and_exit_one(self) -> None:
        script = self._script("sleep_case.py", SLEEP_SCRIPT)
        self._write_manifest([_suite("slow", script, timeout_s=20.0)])
        proc = self._flake_run("--repeats", "2", "--timeout", "0.5", "--json", timeout=60.0)
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        payload = self._payload(proc)
        self.assertEqual(payload["verdict"], FlakeVerdict.INCONCLUSIVE)
        suite = payload["targets"][0]
        self.assertEqual(suite["verdict"], FlakeVerdict.INCONCLUSIVE)
        self.assertEqual(suite["other_count"], 2)
        self.assertEqual(suite["statuses"], [RunStatus.TIMEOUT, RunStatus.TIMEOUT])
        runs = self._store_runs()
        self.assertEqual({r.status for r in runs}, {RunStatus.TIMEOUT})

    def test_skipped_suite_is_inconclusive_and_exit_one(self) -> None:
        script = self._script("never_runs.py", PASS_SCRIPT)
        self._write_manifest(
            [
                _suite(
                    "needs_missing_tool",
                    script,
                    required_binaries=["tb-definitely-missing-binary-xyz"],
                )
            ]
        )
        proc = self._flake_run("--repeats", "2", "--json")
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        payload = self._payload(proc)
        self.assertEqual(payload["verdict"], FlakeVerdict.INCONCLUSIVE)
        self.assertEqual(payload["targets"][0]["statuses"], [RunStatus.SKIP, RunStatus.SKIP])
        self.assertEqual({r.status for r in self._store_runs()}, {RunStatus.SKIP})

    # --- SuiteSpec.repeat_runs is honoured ------------------------------

    def test_manifest_repeat_runs_sets_attempt_count_without_flag(self) -> None:
        script = self._script("pass_case.py", PASS_SCRIPT)
        self._write_manifest([_suite("stable", script, repeat_runs=3)])
        proc = self._flake_run("--json")
        self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout)
        payload = self._payload(proc)
        self.assertEqual(payload["attempts"], 3)
        self.assertEqual(payload["expected_repeats"], 3)
        self.assertEqual(len(self._store_runs()), 3)

    def test_explicit_repeats_flag_overrides_manifest_repeat_runs(self) -> None:
        script = self._script("pass_case.py", PASS_SCRIPT)
        self._write_manifest([_suite("stable", script, repeat_runs=5)])
        proc = self._flake_run("--repeats", "2", "--json")
        self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout)
        self.assertEqual(self._payload(proc)["attempts"], 2)
        self.assertEqual(len(self._store_runs()), 2)

    # --- case-level classification from real structured output ----------

    def test_structured_unittest_cases_are_classified_per_case(self) -> None:
        script = self._script("sample_tests.py", CASE_SCRIPT)
        self._write_manifest(
            [
                {
                    "name": "sample-cases",
                    "adapter": "python_unittest",
                    "profile": "unit",
                    "command": [PY, "-m", "unittest", "-v", "sample_tests"],
                    "timeout_s": 30.0,
                }
            ]
        )
        proc = self._flake_run("--repeats", "2", "--json")
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        payload = self._payload(proc)
        self.assertEqual(payload["verdict"], FlakeVerdict.ALWAYS_FAIL)
        suite = payload["targets"][0]
        self.assertEqual(suite["level"], "suite")
        self.assertEqual(suite["verdict"], FlakeVerdict.ALWAYS_FAIL)

        cases = {t["case"]: t for t in payload["targets"] if t["level"] == "case"}
        self.assertEqual(len(cases), 2, payload["targets"])
        ok_case = next(t for name, t in cases.items() if "test_always_ok" in name)
        bad_case = next(t for name, t in cases.items() if "test_always_bad" in name)
        self.assertEqual(ok_case["verdict"], FlakeVerdict.ALWAYS_PASS)
        self.assertEqual(ok_case["pass_count"], 2)
        self.assertEqual(bad_case["verdict"], FlakeVerdict.ALWAYS_FAIL)
        self.assertEqual(bad_case["fail_count"], 2)
        self.assertEqual(len(bad_case["failing_run_ids"]), 2)
        self.assertEqual(len(set(payload["run_ids"])), 2)

    # --- historical analyze ---------------------------------------------

    def test_analyze_filters_by_project_revision_and_group(self) -> None:
        pass_script = self._script("pass_case.py", PASS_SCRIPT)
        flaky_script = self._script("flaky_counter.py", FLAKY_SCRIPT)
        self._write_manifest(
            [
                _suite("stable", pass_script),
                _suite("counter", flaky_script, allow_repo_write=True),
            ]
        )
        with TestBenchStore(self.tb_home) as store:
            store.trust_project(self.proj)

        # An ordinary run must never be treated as an attempt.
        plain = self._cli(["run", str(self.proj), "--suite", "stable", "--json"])
        self.assertEqual(plain.returncode, 0, plain.stderr + plain.stdout)

        stable = self._flake_run("--suite", "stable", "--repeats", "2", "--json")
        self.assertEqual(stable.returncode, 0, stable.stderr + stable.stdout)
        stable_group = self._payload(stable)["group_id"]

        flaky = self._flake_run("--suite", "counter", "--repeats", "2", "--json")
        self.assertEqual(flaky.returncode, 1, flaky.stderr + flaky.stdout)
        flaky_group = self._payload(flaky)["group_id"]
        self.assertNotEqual(stable_group, flaky_group)

        by_project = self._cli(["flake", "analyze", "--project", "flake_demo", "--json"])
        self.assertEqual(by_project.returncode, 0, by_project.stderr + by_project.stdout)
        history = self._payload(by_project)
        self.assertEqual(len(history["groups"]), 2)
        verdicts = {g["group_id"]: g["verdict"] for g in history["groups"]}
        self.assertEqual(verdicts[stable_group], FlakeVerdict.ALWAYS_PASS)
        self.assertEqual(verdicts[flaky_group], FlakeVerdict.MIXED)
        self.assertEqual(history["verdict_counts"][FlakeVerdict.MIXED], 1)
        self.assertEqual(history["verdict_counts"][FlakeVerdict.ALWAYS_PASS], 1)
        # Newest group sorts first.
        self.assertEqual(history["groups"][0]["group_id"], flaky_group)

        by_revision = self._cli(
            ["flake", "analyze", "--revision", "unversioned", "--group", flaky_group, "--json"]
        )
        self.assertEqual(by_revision.returncode, 0, by_revision.stderr + by_revision.stdout)
        narrowed = self._payload(by_revision)
        self.assertEqual(len(narrowed["groups"]), 1)
        self.assertEqual(narrowed["groups"][0]["group_id"], flaky_group)
        self.assertEqual(narrowed["groups"][0]["attempts"], 2)
        self.assertEqual(
            [t["target"] for t in narrowed["groups"][0]["targets"]], ["counter"]
        )

        no_match = self._cli(["flake", "analyze", "--revision", "deadbeef", "--json"])
        self.assertEqual(no_match.returncode, 1)
        self.assertEqual(self._payload(no_match)["groups"], [])

        no_groups = self._cli(["flake", "analyze", "--project", "other_project", "--json"])
        self.assertEqual(no_groups.returncode, 1)

    def test_analyze_never_reruns_suites(self) -> None:
        script = self._script("pass_case.py", PASS_SCRIPT)
        self._write_manifest([_suite("stable", script)])
        run_proc = self._flake_run("--repeats", "2", "--json")
        self.assertEqual(run_proc.returncode, 0, run_proc.stderr + run_proc.stdout)
        before = len(self._store_runs())

        analyze = self._cli(["flake", "analyze", "--json"])
        self.assertEqual(analyze.returncode, 0, analyze.stderr + analyze.stdout)
        history = self._payload(analyze)
        self.assertEqual(len(history["groups"]), 1)
        self.assertEqual(history["groups"][0]["attempts"], 2)
        self.assertEqual(len(self._store_runs()), before)

    def test_repeats_out_of_range_fails_cleanly(self) -> None:
        script = self._script("pass_case.py", PASS_SCRIPT)
        self._write_manifest([_suite("stable", script)])
        proc = self._flake_run("--repeats", "0", "--json")
        self.assertEqual(proc.returncode, 1)
        self.assertIn("repeats", proc.stderr)
        self.assertEqual(len(self._store_runs()), 0)


if __name__ == "__main__":
    unittest.main()
