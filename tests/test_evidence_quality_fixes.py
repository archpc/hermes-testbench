"""Vertical TDD regression tests for the five evidence-quality defects.

Each test is named after the finding it closes (``d1``..``d5``) and reproduces
the *observed* symptom from the review rather than an internal call, so the
assertions stay meaningful if the implementation is reorganised:

D1  Two runs created in the same UTC second listed in the wrong order, and the
    TUI/`last_verified_per_project` views agreeing with the wrong order.
D2  A bare multi-item ``FAILED:`` header collapsed into a synthetic ``case_1``,
    and a real ``check-wiring.py`` that accepts ``--hermes-home`` could not be
    exercised hermetically during discovery.
D3  ``messaging-verify`` reported the literal string ``verified`` as its
    revision, wrote no sanitized evidence logs, and benchmark runs stored a
    prose summary instead of a runnable repro command.
D4  JUnit XML counters disagreed with the recorded counts and with the number
    of ``<testcase>`` elements.
D5  An uncaught exception in a test was reported as a ``fail`` (not an error),
    an untrusted manifest was reported as generic ``blocked_policy``, and a
    blocked suite double-counted its policy failure plus a synthetic skip.

Every test works inside a temporary directory under ``default_scratch_root()``
(the disposable HOME/TMPDIR), never the operator's real state, except D3's
first half, which deliberately drives the real ``messaging-verify`` path and is
therefore host-sensitive — see that test's docstring.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from hermes_testbench.adapters.parsers import parse_cargo, parse_python_pytest, parse_python_unittest, parse_shell
from hermes_testbench.benchmark import run_workload
from hermes_testbench.cli import main as cli_main
from hermes_testbench.discovery import discover_project
from hermes_testbench.engine import TestBenchEngine
from hermes_testbench.isolation import default_scratch_root
from hermes_testbench.reporting import parse_junit_xml_bounded, render_junit_xml
from hermes_testbench.schema import CaseResult, FailureKind, FailureRecord, RunRecord, RunStatus, SuiteResult, new_run_id, utc_now_iso
from hermes_testbench.store import TestBenchStore
from hermes_testbench.tui_render import TuiState, _history_runs, _overview_runs


class EvidenceQualityRegressionTests(unittest.TestCase):
    def setUp(self) -> None:
        # One scratch-root temp dir per test; tb_home keeps store state local.
        self.tmp = tempfile.TemporaryDirectory(dir=str(default_scratch_root()), prefix="tb-eq-")
        self.root = Path(self.tmp.name)
        self.tb_home = self.root / "tb_home"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_d1_same_second_runs_and_benchmarks_preserve_strict_creation_order(self) -> None:
        """Same-second runs must list newest-first, deterministically.

        Two independent monotonicity claims, then the storage/view consequences:

        1. ``new_run_id`` and ``utc_now_iso`` are per-process monotonic, so IDs
           sort lexicographically in creation order and timestamps in call order.
           The 32-ID uniqueness assert is only a within-process claim; cross-
           process uniqueness is probabilistic (seq bucket + random bytes).
        2. Two runs with an *identical* whole-second ``created_at`` must still
           order deterministically. ``list_runs`` orders by normalised timestamp
           then ``rowid DESC`` then ``run_id DESC``. Here ``run_id DESC`` alone
           would put ``eca06163`` first (``e`` > ``2``), so the expected order
           is evidence that ``rowid DESC`` — insertion order — wins the tie.
           ``last_verified_per_project`` repeats the same key expression, and the
           TUI terminal views break the tie with ``-idx`` (newest-first input),
           which is why both are asserted alongside the store query.
        """
        # 1. Rapid new_run_id() and utc_now_iso() calls within the same millisecond must stay monotonic.
        rids = [new_run_id() for _ in range(32)]
        stamps = [utc_now_iso() for _ in range(32)]
        self.assertEqual(rids, sorted(rids), "new_run_id() must sort monotonically within one second")
        self.assertEqual(len(set(rids)), 32)
        self.assertEqual(stamps, sorted(stamps), "utc_now_iso() must sort monotonically")

        # 2. Exact demo collision: older run 'eca06163' followed in same UTC second by newer '2e1baddc'.
        # The stamps are hand-written whole seconds on purpose — the normaliser
        # pads them so they compare equal, exercising the rowid tiebreak rather
        # than the timestamp.
        with TestBenchStore(self.tb_home) as store:
            older = RunRecord(
                run_id="tb_20261115_013630_eca06163",
                project="demo",
                project_root="/tmp/demo",
                revision="4f62632",
                profile="regression",
                status=RunStatus.FAIL,
                created_at="2026-11-15T01:36:30Z",
                finished_at="2026-11-15T01:36:30Z",
            )
            newer = RunRecord(
                run_id="tb_20261115_013630_2e1baddc",
                project="demo",
                project_root="/tmp/demo",
                revision="8234029",
                profile="security",
                status=RunStatus.PASS,
                created_at="2026-11-15T01:36:30Z",
                finished_at="2026-11-15T01:36:31Z",
            )
            store.create_run(older)
            store.finalize_run(older)
            store.create_run(newer)
            store.finalize_run(newer)

            listed = store.list_runs(limit=10)
            self.assertEqual(
                [r.run_id for r in listed],
                ["tb_20261115_013630_2e1baddc", "tb_20261115_013630_eca06163"],
            )
            latest_map = store.last_verified_per_project()
            self.assertEqual(latest_map["demo"].run_id, "tb_20261115_013630_2e1baddc")

            # TUI views consuming listed (newest-first) must also pick newer when created_at ties.
            state = TuiState(runs=tuple(listed))
            self.assertEqual(_overview_runs(state)[0].run_id, "tb_20261115_013630_2e1baddc")
            self.assertEqual(_history_runs(state)[0].run_id, "tb_20261115_013630_2e1baddc")

    def test_d4_junit_xml_preserves_exact_total_pass_fail_error_skip_counts_and_case_elements(self) -> None:
        """JUnit counters and element counts must both equal the record's numbers.

        ``render_junit_xml`` emits adapter-reported cases/failures first and then
        *back-fills* synthetic elements from the counters, because report
        consumers expect ``tests == sum(testcase)``. The suites here cover the
        three shapes that used to disagree:

        * unittest: 3 real cases, 2 pass + 1 fail;
        * quiet pytest: only a summary line plus one FAILED case name, so 7
          total (1 failed, 4 passed, 2 skipped) with 7 elements, most synthetic;
        * an all-passing suite with no case records at all, 3 back-filled passes.

        The root counters (13/2/0/2) are summed from the suite attributes, and
        the per-suite ``tests == len(findall('testcase'))`` assertion is the
        actual invariant: it would fail on any back-fill gap or double-emit.
        """
        import xml.etree.ElementTree as ET

        # 1. Verbose unittest with 2 ok + 1 FAIL (exact reproduction of demo Section 2)
        ut_fail_out = parse_python_unittest(
            "test_lower_bound (test_quota.QuotaClampTests.test_lower_bound) ... ok\n"
            "test_upper_bound (test_quota.QuotaClampTests.test_upper_bound) ... FAIL\n"
            "test_within_bounds (test_quota.QuotaClampTests.test_within_bounds) ... ok\n\n"
            "======================================================================\n"
            "FAIL: test_upper_bound (test_quota.QuotaClampTests.test_upper_bound)\n"
            "----------------------------------------------------------------------\n"
            "AssertionError: 500 != 100\n\n"
            "----------------------------------------------------------------------\n"
            "Ran 3 tests in 0.001s\n\n"
            "FAILED (failures=1)\n"
        )
        self.assertEqual(len(ut_fail_out.cases), 3)
        suite_fail = SuiteResult(
            name="python-unittest",
            adapter="python_unittest",
            profile="unit",
            command=["python3", "-m", "unittest"],
            cwd=".",
            status=RunStatus.FAIL,
            total_count=ut_fail_out.total,
            pass_count=ut_fail_out.passed,
            fail_count=ut_fail_out.failed,
            skip_count=ut_fail_out.skipped,
            error_count=ut_fail_out.errors,
            failures=[FailureRecord(suite="python-unittest", **f) for f in ut_fail_out.failures],
            cases=[CaseResult(suite="python-unittest", **c) for c in ut_fail_out.cases],
        )

        # 2. Quiet pytest summary where only the 1 failed case name was printed alongside 4 passed + 2 skipped
        # Parser totals come from the summary line; only the FAILED case is named.
        pt_out = parse_python_pytest(
            "....Fss\n"
            "================ 1 failed, 4 passed, 2 skipped in 0.05s ================\n"
            "FAILED tests/test_mod.py::test_boom - AssertionError: bad\n"
        )
        suite_pt = SuiteResult(
            name="python-pytest",
            adapter="python_pytest",
            profile="unit",
            command=["pytest", "-q"],
            cwd=".",
            status=RunStatus.FAIL,
            total_count=pt_out.total,
            pass_count=pt_out.passed,
            fail_count=pt_out.failed,
            skip_count=pt_out.skipped,
            error_count=pt_out.errors,
            failures=[FailureRecord(suite="python-pytest", **f) for f in pt_out.failures],
            cases=[CaseResult(suite="python-pytest", **c) for c in pt_out.cases],
        )

        # 3. All-passing 3-test run with zero individual case records (quiet dot mode)
        suite_quiet_ok = SuiteResult(
            name="quiet-unit",
            adapter="python_unittest",
            profile="unit",
            command=["python3", "-m", "unittest"],
            cwd=".",
            status=RunStatus.PASS,
            total_count=3,
            pass_count=3,
        )

        # Run-level counters are given explicitly: the renderer trusts suite
        # attributes, not run totals, so 13 is only reproduced if the three
        # suites sum to it.
        rec = RunRecord(
            run_id="tb_junit_counts",
            project="demo",
            project_root="/tmp/demo",
            revision="abc1234",
            profile="unit",
            status=RunStatus.FAIL,
            total_count=13,
            pass_count=9,
            fail_count=2,
            skip_count=2,
            suites=[suite_fail, suite_pt, suite_quiet_ok],
        )
        xml_text = render_junit_xml(rec)
        # Round-trip through the bounded parser: its counters come from the
        # attributes, which is exactly the contract report consumers read.
        parsed = parse_junit_xml_bounded(xml_text.encode("utf-8"))
        self.assertEqual(parsed["tests"], 13)
        self.assertEqual(parsed["failures"], 2)
        self.assertEqual(parsed["errors"], 0)
        self.assertEqual(parsed["skipped"], 2)

        root = ET.fromstring(xml_text)
        ts_list = root.findall("testsuite")
        self.assertEqual([int(ts.attrib["tests"]) for ts in ts_list], [3, 7, 3])
        for ts in ts_list:
            self.assertEqual(int(ts.attrib["tests"]), len(ts.findall("testcase")))

    def test_d5_failure_classification_and_error_counts_are_reported_accurately(self) -> None:
        """Harness errors, untrusted manifests and sandbox denials must be named.

        Three projects, each isolating one classification:

        * an uncaught ``RuntimeError`` inside a test: unittest reports it as
          ``ERROR:``, so the suite has ``error_count == 1`` and ``fail_count ==
          0``; the run kind is HARNESS_ERROR because ``_failed_suite_kind`` infers
          it only when errors exist and failures do not. The summary must say
          ``1 errors`` (the run summary string), not fold it into failures.
        * a manifest suite requesting ``allow_network`` without trust: Gate 2
          blocks it with a ``<policy>`` failure record whose kind is
          ``untrusted_manifest``, and ``_blocked_failure_kind`` surfaces that
          specific kind instead of ``blocked_policy``. The JUnit assertions pin
          the double-count fix: the ``<policy>`` failure is emitted as *skipped*
          and the blocked counter must not add a second synthetic skip.
        * a write outside the sandbox: only runs when ``sandbox-exec`` exists
          (macOS). The target is chosen by ``can_apply_seatbelt()``: under a fresh
          profile it is a path beside the project (outside the suite's writable
          dir); under an inherited outer Seatbelt the project root is already
          writable, so the probe targets a path in the real home that the outer
          profile denies too. ``assertFalse(outside.exists())`` is the actual
          confinement evidence — the status alone could be a plain assertion.
        """
        import shutil

        # 1. Uncaught RuntimeError in unittest -> harness_error + error_count=1 in run.summary
        proj_err = self.root / "proj_err"
        (proj_err / "tests").mkdir(parents=True)
        (proj_err / "tests" / "test_err.py").write_text(
            "import unittest\n"
            "class ErrTests(unittest.TestCase):\n"
            "    def test_boom(self):\n"
            "        raise RuntimeError('kaboom')\n",
            encoding="utf-8",
        )
        # 2. Untrusted manifest requesting network -> untrusted_manifest (not generic blocked_policy)
        proj_untrusted = self.root / "proj_untrusted"
        proj_untrusted.mkdir(parents=True)
        (proj_untrusted / ".testbench.toml").write_text(
            '[project]\nname = "untrusted"\n\n'
            '[[suite]]\nname = "net"\nadapter = "shell"\nprofile = "unit"\n'
            'command = ["/usr/bin/true"]\nallow_network = true\n',
            encoding="utf-8",
        )

        with TestBenchStore(self.tb_home) as store:
            eng = TestBenchEngine(store)
            rec_err = eng.run_sync(proj_err, profile="unit")
            self.assertEqual(rec_err.status, RunStatus.FAIL)
            self.assertEqual(rec_err.error_count, 1)
            self.assertEqual(rec_err.fail_count, 0)
            self.assertEqual(rec_err.failure_kind, FailureKind.HARNESS_ERROR)
            self.assertIn("1 errors", rec_err.summary)
            # "0.0" is the unittest "Ran 1 test in 0.0NNs" timing line: this
            # assertion is timing-sensitive (a run slower than 100 ms prints
            # "0.1NNs" and would fail) — it is pinning that the adapter kept the
            # runner's own excerpt, not a stable value.
            self.assertIn("0.0", rec_err.suites[0].summary_excerpt)

            rec_untrusted = eng.run_sync(proj_untrusted, profile="unit")
            self.assertEqual(rec_untrusted.status, RunStatus.BLOCKED)
            self.assertEqual(rec_untrusted.failure_kind, FailureKind.UNTRUSTED_MANIFEST)
            untrusted_xml = parse_junit_xml_bounded(render_junit_xml(rec_untrusted).encode("utf-8"))
            self.assertEqual(untrusted_xml["tests"], 1, "blocked suite must not double-count <policy> + skip")
            self.assertEqual(untrusted_xml["skipped"], 1)
            self.assertEqual(untrusted_xml["failures"], 0)

            if shutil.which("sandbox-exec"):
                from hermes_testbench.isolation import can_apply_seatbelt, real_user_home
                proj_sb = self.root / "proj_sb"
                (proj_sb / "tests").mkdir(parents=True)
                # When running inside an outer Seatbelt, self.root is already inside the outer writable TMPDIR;
                # ~/.hermes/_tb_d5_probe is denied by both fresh and inherited_outer Seatbelt profiles.
                outside = (
                    (self.root / "forbidden_canary.txt")
                    if can_apply_seatbelt()
                    else (real_user_home() / ".hermes" / "_tb_d5_forbidden_probe")
                )
                (proj_sb / "tests" / "test_sb.py").write_text(
                    "import pathlib, unittest\n"
                    "class SbTests(unittest.TestCase):\n"
                    "    def test_write_outside(self):\n"
                    f"        pathlib.Path({str(outside)!r}).write_text('nope')\n",
                    encoding="utf-8",
                )
                rec_sb = eng.run_sync(proj_sb, profile="unit")
                self.assertEqual(rec_sb.status, RunStatus.FAIL)
                self.assertEqual(rec_sb.failure_kind, FailureKind.SANDBOX_DENIED)
                self.assertFalse(outside.exists())

    def test_d2_multiline_failed_header_and_check_wiring_hermetic_discovery(self) -> None:
        """A FAILED: list must yield real items, and check-wiring must run hermetically.

        Part 1 is the parser defect: a bare ``FAILED:`` header followed by a
        numbered list used to produce a single synthetic ``case_1`` whose message
        was the header, losing both items. Asserting the *text* (not just the
        count) is what makes that distinguishable.

        Part 2 is the discovery defect: ``scripts/check-wiring.py`` is a known
        verifier script, but running it with no arguments reads the operator's
        real ``~/.hermes``. Discovery therefore wraps it in
        ``_CHECK_WIRING_PROBE_SCRIPT`` when — and only when — its source mentions
        both ``--hermes-home`` and ``hermes-bots``; the probe builds synthetic
        wired/unwired homes and prints two TB-PASS markers. ``pass_count == 2``
        is the observable that both synthetic checks ran and passed, and
        ``status == PASS`` proves no suite touched the real home (a failure there
        would surface as a FAIL).
        """
        # 1. Multiline FAILED: header must extract actual bulleted failure items, never 'case_1'
        raw_out = (
            "install      : /tmp/fake/.hermes\n"
            "bot-mode flag: none\n\n"
            "FAILED:\n"
            "  1. advisor profile missing: /tmp/fake/.hermes/profiles/opus\n"
            "  2. no profile carries ui_meta.hermes-bots: message_agent stays unavailable\n"
        )
        outcome = parse_shell(raw_out)
        self.assertEqual(outcome.failed, 2)
        self.assertNotIn("case_1", outcome.failures[0]["message"])
        self.assertIn("advisor profile missing", outcome.failures[0]["message"])
        self.assertIn("ui_meta.hermes-bots", outcome.failures[1]["message"])

        # 2. A check-wiring.py accepting --hermes-home and checking hermes-bots runs hermetically under Seatbelt
        # The fixture mirrors the real script's contract: it exits 1 with a
        # numbered FAILED marker for each missing condition, so the probe sees
        # both the exit status and the marker text it requires.
        proj_wiring = self.root / "agent-messaging-sim"
        (proj_wiring / "scripts").mkdir(parents=True)
        (proj_wiring / "scripts" / "check-wiring.py").write_text(
            "import argparse, pathlib, sqlite3, sys\n"
            "ap = argparse.ArgumentParser()\n"
            "ap.add_argument('--hermes-home', default=str(pathlib.Path.home() / '.hermes'))\n"
            "home = pathlib.Path(ap.parse_args().hermes_home)\n"
            "opus_cfg = home / 'profiles' / 'opus' / 'config.yaml'\n"
            "marker = home / 'profile.yaml'\n"
            "if not opus_cfg.is_file() or 'opus' not in opus_cfg.read_text().lower():\n"
            "    print('FAILED:\\n  1. advisor profile missing')\n"
            "    sys.exit(1)\n"
            "if not marker.is_file() or 'hermes-bots' not in marker.read_text():\n"
            "    print('FAILED:\\n  2. ui_meta.hermes-bots missing')\n"
            "    sys.exit(1)\n"
            "print('wiring OK')\n",
            encoding="utf-8",
        )
        spec = discover_project(proj_wiring)
        # Exactly one suite: the wrapping must replace the plain invocation, not
        # add a second one alongside it.
        self.assertEqual([s.name for s in spec.suites], ["check-wiring"])
        with TestBenchStore(self.tb_home) as store:
            rec = TestBenchEngine(store).run_sync(proj_wiring, profile="regression")
            self.assertEqual(rec.status, RunStatus.PASS, rec.first_failure)
            self.assertEqual(rec.pass_count, 2)
            self.assertEqual(rec.fail_count, 0)

    def test_d3_messaging_verify_records_git_revision_env_repro_and_sanitized_logs(self) -> None:
        """``messaging-verify`` must record observed facts, not labels.

        Host sensitivity, stated plainly: this half runs the *real* verification
        path with no mock, so its preconditions come from the operator's home.
        ``messaging_paths()`` only returns both roots; the selection
        (artifact root if it exists, else the hermes-agent checkout, else the
        current directory) lives in ``cli.main`` (``cli.py:688-690``).
        ``revision`` is therefore a short SHA, ``<sha>+dirty``, or ``unversioned``.
        The assertion is only that it is truthy and is not the literal
        ``"verified"`` — the defect being pinned was a fabricated label, not a
        particular revision.
        ``rc == 0`` requires the run to settle PASS or SKIP, which holds when the
        source tree is absent (all cases skipped) or conforming.

        The sanitized-log assertion relies on naming: ordinary suites write
        ``<suite-name-slug>.sanitized.log`` listing every case's status, while the
        pytest suite writes the runner's own output (as
        ``agent-messaging_pytest.sanitized.log``) when pytest produced any
        (``messaging_suite.py:795-810``). The contract suite's slug sorts first, so
        ``sanitized[0]`` is that log. Case names appear even for SKIP, which is
        why the known case name is a stable needle.

        The benchmark half pins that a benchmark run stores an exact CLI line
        (``testbench bench run ... --name noop-bench``) rather than prose, so the
        repro is executable. ``warmup=0`` keeps the run cheap; two repeats give
        the sample set the stats need.
        """
        import io
        import json
        from contextlib import redirect_stdout
        import sys

        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cli_main(["--home", str(self.tb_home), "messaging-verify", "--json"])
        self.assertEqual(rc, 0)
        payload = json.loads(buf.getvalue())
        self.assertNotEqual(payload["revision"], "verified")
        self.assertTrue(payload["revision"])
        self.assertTrue(payload["environment"].get("os"))
        self.assertTrue(payload["commands"])
        self.assertIn("testbench messaging-verify", payload["repro_instructions"])

        ev_dir = Path(payload["evidence_dir"])
        san_logs = sorted(ev_dir.glob("*.sanitized.log"))
        self.assertGreaterEqual(len(san_logs), 1, "messaging-verify must write sanitized suite logs")
        self.assertIn("a1_ack_is_queued_handoff_with_delivery_id", san_logs[0].read_text(encoding="utf-8"))

        # Also verify benchmark runs persist an executable repro command instead of prose summary
        with TestBenchStore(self.tb_home) as store:
            bench_rec, _bpayload = run_workload(
                "noop-bench",
                [sys.executable, "-c", "pass"],
                cwd=self.root,
                store=store,
                project="demo",
                revision="1234abc",
                warmup=0,
                repeats=2,
            )
            # The exact CLI shape, including the workload name — a prose summary
            # would satisfy neither assertion.
            self.assertIn("testbench bench run", bench_rec.repro_instructions)
            self.assertIn("--name noop-bench", bench_rec.repro_instructions)


if __name__ == "__main__":
    unittest.main()
