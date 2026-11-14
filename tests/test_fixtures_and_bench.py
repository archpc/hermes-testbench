"""TDD tests for deterministic fixtures, fault probes, and reproducible benchmarking."""

from __future__ import annotations

import json
import os
import shutil
import signal
import sqlite3
import stat
import sys
import tempfile
import textwrap
import time
import unittest
import unicodedata
from contextlib import closing
from pathlib import Path

from hermes_testbench.benchmark import (
    collect_environment_metadata,
    compare_benchmarks,
    compute_stats,
    format_benchmark_comparison,
    run_workload,
)
from hermes_testbench.fixtures import (
    create_cross_lang_mail_db,
    create_sqlite_contention_fixture,
    malformed_input_corpus,
    run_pty_resize_probe,
    run_repeated_init_probe,
    run_signal_crash_probe,
    run_sqlite_contention_probe,
    run_tmux_lifecycle_probe,
    terminal_escape_corpus,
    unicode_boundary_corpus,
)
from hermes_testbench.sanitize import sanitize_terminal_text, strip_terminal_escapes
from hermes_testbench.schema import CaseResult, ProfileName, RunStatus
from hermes_testbench.store import TestBenchStore

PY = sys.executable


def _scratch() -> Path:
    root = Path.home() / ".hermes/cache/scratch"
    root.mkdir(parents=True, exist_ok=True)
    return root


class _TempDirTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(dir=str(_scratch()), prefix="tb-fix-")
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)


class UnicodeCorpusTests(unittest.TestCase):
    def test_corpus_covers_every_required_unicode_category(self) -> None:
        entries = unicode_boundary_corpus()
        kinds = {e.kind for e in entries}
        required = {
            "utf8_2byte",
            "utf8_3byte",
            "utf8_4byte",
            "utf8_boundary",
            "cjk_wide",
            "zwj_emoji",
            "combining_marks",
            "rtl_bidi",
            "plane14_tags",
        }
        self.assertLessEqual(required, kinds, f"missing kinds: {required - kinds}")

        by_kind = {}
        for entry in entries:
            self.assertIsInstance(entry.text, str)
            self.assertTrue(entry.text)
            self.assertTrue(entry.payload())
            by_kind.setdefault(entry.kind, []).append(entry)

        self.assertEqual(len(by_kind["utf8_2byte"][0].text.encode("utf-8")), 2)
        self.assertEqual(len(by_kind["utf8_3byte"][0].text.encode("utf-8")), 3)
        self.assertEqual(len(by_kind["utf8_4byte"][0].text.encode("utf-8")), 4)

        boundary = by_kind["utf8_boundary"][0].text
        self.assertIn("\x7f", boundary)
        self.assertIn("\x80", boundary)

        cjk_widths = [unicodedata.east_asian_width(ch) for ch in by_kind["cjk_wide"][0].text]
        self.assertIn("W", cjk_widths)
        self.assertRegex(by_kind["cjk_wide"][0].text, "[\u4e00-\u9fff\u3040-\u30ff]")

        self.assertIn("\u200d", by_kind["zwj_emoji"][0].text)
        self.assertTrue(
            any(unicodedata.combining(ch) for ch in by_kind["combining_marks"][0].text)
        )
        self.assertTrue(
            any(0x202A <= ord(ch) <= 0x202E for ch in by_kind["rtl_bidi"][0].text)
        )
        self.assertTrue(
            any(0xE0000 <= ord(ch) <= 0xE007F for ch in by_kind["plane14_tags"][0].text)
        )

    def test_corpus_is_stable_and_names_are_unique(self) -> None:
        first = unicode_boundary_corpus()
        second = unicode_boundary_corpus()
        self.assertEqual([e.name for e in first], [e.name for e in second])
        self.assertEqual([e.text for e in first], [e.text for e in second])
        self.assertEqual(len({e.name for e in first}), len(first))

    def test_corpus_entries_round_trip_through_json_dict(self) -> None:
        entry = unicode_boundary_corpus()[0]
        dumped = json.dumps(entry.to_dict(), ensure_ascii=False)
        self.assertIn(entry.name, dumped)


class MalformedInputCorpusTests(unittest.TestCase):
    def test_corpus_covers_every_required_malformed_category(self) -> None:
        entries = malformed_input_corpus()
        kinds = {e.kind for e in entries}
        required = {
            "truncated_json",
            "truncated_toml",
            "nul_byte",
            "negative_timestamp",
            "rid_over_2_32",
            "huge_string",
        }
        self.assertLessEqual(required, kinds, f"missing kinds: {required - kinds}")

        by_kind = {e.kind: e for e in entries}

        with self.assertRaises(json.JSONDecodeError):
            json.loads(by_kind["truncated_json"].payload().decode("utf-8"))
        self.assertIn("[", by_kind["truncated_toml"].payload().decode("utf-8"))
        self.assertIn(b"\x00", by_kind["nul_byte"].payload())

        negative = json.loads(by_kind["negative_timestamp"].payload().decode("utf-8"))
        self.assertLess(negative["received_at"], 0)
        self.assertLess(negative["received_at"], 0)  # pre-1970 epoch

        big = json.loads(by_kind["rid_over_2_32"].payload().decode("utf-8"))
        self.assertGreater(big["rid"], 2**32)

        huge = by_kind["huge_string"]
        self.assertEqual(len(huge.payload().decode("utf-8")), 10240)

    def test_corpus_is_deterministic(self) -> None:
        self.assertEqual(
            [e.payload() for e in malformed_input_corpus()],
            [e.payload() for e in malformed_input_corpus()],
        )


class TerminalEscapeCorpusTests(unittest.TestCase):
    def test_corpus_covers_every_required_escape_family(self) -> None:
        entries = terminal_escape_corpus()
        kinds = {e.kind for e in entries}
        required = {"osc52", "osc8", "dcs", "csi_2j", "c1_8bit", "nul", "bel"}
        self.assertLessEqual(required, kinds, f"missing kinds: {required - kinds}")
        by_kind = {e.kind: e for e in entries}
        self.assertIn("\x1b]52;", by_kind["osc52"].text)
        self.assertIn("\x1b]8;;", by_kind["osc8"].text)
        self.assertIn("\x1bP", by_kind["dcs"].text)
        self.assertIn("\x1b[2J", by_kind["csi_2j"].text)
        self.assertIn("\x9b", by_kind["c1_8bit"].text)
        self.assertIn("\x00", by_kind["nul"].text)
        self.assertIn("\x07", by_kind["bel"].text)

    def test_every_escape_sequence_is_stripped_by_the_sanitizer(self) -> None:
        for entry in terminal_escape_corpus():
            stripped = strip_terminal_escapes(entry.text)
            self.assertNotIn("\x1b", stripped, entry.name)
            self.assertFalse(
                any(0x80 <= ord(ch) <= 0x9F for ch in stripped), entry.name
            )
            cleaned = sanitize_terminal_text(entry.text)
            self.assertNotIn("\x00", cleaned, entry.name)
            self.assertNotIn("\x07", cleaned, entry.name)
            self.assertNotIn("\x1b", cleaned, entry.name)


class SqliteContentionTests(_TempDirTest):
    def test_fixture_creates_wal_database_with_busy_timeout(self) -> None:
        db_path = self.root / "contention.db"
        info = create_sqlite_contention_fixture(db_path)
        self.assertTrue(db_path.is_file())
        self.assertEqual(info["journal_mode"], "wal")
        self.assertEqual(info["busy_timeout_ms"], 5000)
        self.assertEqual(info["tables"], ["contention"])
        self.assertEqual(info["rows"], 0)
        with closing(sqlite3.connect(str(db_path))) as conn:
            self.assertEqual(
                conn.execute("PRAGMA journal_mode").fetchone()[0].lower(), "wal"
            )

    def test_multi_process_wal_contention_and_busy_timeout_zero_handling(self) -> None:
        db_path = self.root / "contention.db"
        create_sqlite_contention_fixture(db_path)
        result = run_sqlite_contention_probe(
            db_path, writers=4, ops_per_writer=20, busy_timeout_ms=5000
        )
        expected = 4 * 20
        self.assertEqual(result["writers"], 4)
        self.assertEqual(result["ops_per_writer"], 20)
        self.assertEqual(result["expected_rows"], expected)
        self.assertTrue(result["wal_mode"])
        self.assertTrue(result["integrity_ok"])
        self.assertTrue(result["begin_immediate"])
        self.assertTrue(result["measurable_contention"])
        self.assertGreaterEqual(
            result["no_timeout_phase"]["busy_retries"],
            1,
            "busy_timeout=0 must surface at least one measured SQLITE_BUSY retry",
        )
        self.assertEqual(result["no_timeout_phase"]["rows_written"], expected)
        self.assertEqual(result["timeout_phase"]["rows_written"], expected)
        self.assertEqual(result["no_timeout_phase"]["writer_errors"], 0)
        self.assertEqual(result["timeout_phase"]["writer_errors"], 0)
        self.assertEqual(result["rows_in_db"], expected)
        self.assertEqual(result["leaked_children"], 0)
        with closing(sqlite3.connect(str(db_path))) as conn:
            rows = conn.execute("SELECT COUNT(*) FROM contention").fetchone()[0]
        self.assertEqual(rows, expected)


class PtyResizeTests(unittest.TestCase):
    def test_child_observes_exact_geometry_transition_via_sigwinch(self) -> None:
        result = run_pty_resize_probe((80, 24), (120, 36))
        self.assertTrue(result["pty_available"])
        self.assertEqual(list(result["initial_size"]), [80, 24])
        self.assertEqual(list(result["resized"]), [120, 36])
        self.assertEqual(list(result["initial_reported"]), [80, 24])
        self.assertEqual(list(result["final_reported"]), [120, 36])
        self.assertGreaterEqual(result["sigwinch_count"], 1)
        self.assertTrue(result["geometry_transition_ok"])
        self.assertEqual(result["child_exit_code"], 0)
        self.assertEqual(result["leaked_children"], 0)

    def test_probe_honours_custom_geometry(self) -> None:
        result = run_pty_resize_probe((100, 40), (90, 30))
        self.assertEqual(list(result["initial_reported"]), [100, 40])
        self.assertEqual(list(result["final_reported"]), [90, 30])


class SignalCrashTests(unittest.TestCase):
    def test_sigabrt_is_classified_as_a_signal_with_no_leaked_children(self) -> None:
        result = run_signal_crash_probe(signal.SIGABRT)
        self.assertEqual(result["signal_num"], int(signal.SIGABRT))
        self.assertEqual(result["exit_code"], 128 + int(signal.SIGABRT))
        self.assertEqual(result["expected_exit_code"], 128 + int(signal.SIGABRT))
        self.assertEqual(result["classification"], "signalled")
        self.assertEqual(result["leaked_children"], 0)
        self.assertTrue(result["grandchild_reaped"])

    def test_classification_generalises_to_other_signals(self) -> None:
        result = run_signal_crash_probe(signal.SIGTERM)
        self.assertEqual(result["signal_num"], int(signal.SIGTERM))
        self.assertEqual(result["exit_code"], 128 + int(signal.SIGTERM))


class RepeatedInitTests(unittest.TestCase):
    def test_repeated_store_and_fixture_initialization_is_idempotent(self) -> None:
        result = run_repeated_init_probe(times=5)
        self.assertEqual(result["times"], 5)
        self.assertTrue(result["idempotent"])
        self.assertEqual(result["distinct_fingerprints"], 1)
        self.assertEqual(result["errors"], [])
        self.assertGreaterEqual(result["schema_version"], 1)
        self.assertTrue(result["mail_db_deterministic"])
        self.assertEqual(len(result["mail_db_hashes"]), 2)


class TmuxLifecycleTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("tmux"), "tmux unavailable")
    def test_isolated_tmux_lifecycle_with_hx_options_crash_recovery_and_teardown(self) -> None:
        result = run_tmux_lifecycle_probe()
        self.assertTrue(result["available"], result.get("reason"))
        self.assertEqual(result["session"], "tb-lifecycle")
        self.assertGreaterEqual(result["windows"], 2)
        self.assertGreaterEqual(result["panes"], 2)
        self.assertEqual(result["hx_options"]["session"], "main")
        self.assertEqual(result["hx_options"]["window"], "work")
        self.assertEqual(result["hx_options"]["pane"], "pane-1")
        self.assertTrue(result["crash_recovered"])
        self.assertTrue(result["server_gone"])
        self.assertTrue(result["clean"])

    @unittest.skipUnless(shutil.which("tmux"), "tmux unavailable")
    def test_probe_honours_scratch_root_and_leaves_it_clean(self) -> None:
        scratch = Path(tempfile.mkdtemp(dir=str(_scratch()), prefix="tb-tmux-"))
        self.addCleanup(shutil.rmtree, scratch, True)
        result = run_tmux_lifecycle_probe(scratch)
        self.assertTrue(result["available"], result.get("reason"))
        self.assertTrue(result["server_gone"])
        self.assertEqual(list(scratch.glob("tb-*")), [])


class CrossLangMailDbTests(_TempDirTest):
    EXPECTED_COLUMNS = {
        "mailbox": ["id", "name", "role", "parent_id", "sort_order"],
        "email": [
            "rid",
            "id",
            "blob_id",
            "thread_id",
            "received_at",
            "size",
            "flags",
            "has_att",
            "from_name",
            "from_addr",
            "to_json",
            "subject",
            "preview",
            "message_id",
            "in_reply_to",
            "other_kw",
        ],
        "body": ["rid", "text"],
        "email_mailbox": ["mailbox_id", "received_at", "rid", "thread_id", "unread"],
        "email_fts": ["subject", "sender", "recipients", "body"],
    }

    def test_schema_matches_hxmail_and_listbench_exactly(self) -> None:
        db_path = self.root / "mail.db"
        info = create_cross_lang_mail_db(db_path)
        self.assertTrue(db_path.is_file())
        self.assertEqual(
            sorted(info["tables"]), sorted(self.EXPECTED_COLUMNS)
        )
        with closing(sqlite3.connect(str(db_path))) as conn:
            for table, columns in self.EXPECTED_COLUMNS.items():
                got = [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]
                self.assertEqual(got, columns, table)
            self.assertEqual(
                conn.execute("PRAGMA journal_mode").fetchone()[0].lower(), "wal"
            )

    def test_contents_cover_pre_epoch_rid_overflow_utf8_and_hostile_escapes(self) -> None:
        db_path = self.root / "mail.db"
        info = create_cross_lang_mail_db(db_path)
        self.assertGreaterEqual(info["pre_epoch_count"], 1)
        self.assertTrue(info["rid_over_2_32"])
        self.assertGreaterEqual(info["hostile_escape_rows"], 1)
        self.assertTrue(info["utf8_ok"])
        self.assertGreaterEqual(info["fts_query_hits"], 1)
        with closing(sqlite3.connect(str(db_path))) as conn:
            conn.text_factory = bytes
            negatives = conn.execute(
                "SELECT COUNT(*) FROM email WHERE received_at < 0"
            ).fetchone()[0]
            self.assertGreaterEqual(negatives, 1)
            max_rid = conn.execute("SELECT MAX(rid) FROM email").fetchone()[0]
            self.assertGreater(max_rid, 2**32)
            subject = conn.execute(
                "SELECT subject FROM email WHERE rid = ?", (max_rid,)
            ).fetchone()[0]
            self.assertIn("日本語", subject.decode("utf-8"))
            body = conn.execute(
                "SELECT text FROM body WHERE rid = ?", (max_rid,)
            ).fetchone()[0]
            self.assertIn(b"\x1b", body)
            self.assertIn(b"\x07", body)

    def test_database_bytes_are_deterministic_across_paths(self) -> None:
        first = self.root / "a.db"
        second = self.root / "b.db"
        info_a = create_cross_lang_mail_db(first)
        info_b = create_cross_lang_mail_db(second)
        self.assertEqual(info_a["sha256"], info_b["sha256"])
        self.assertEqual(first.read_bytes(), second.read_bytes())
        self.assertEqual(info_a["emails"], info_b["emails"])


class EnvironmentMetadataTests(_TempDirTest):
    def test_metadata_reports_versions_load_without_leaking_identity(self) -> None:
        env = collect_environment_metadata(self.root)
        for key in (
            "os_name",
            "os_release",
            "arch",
            "cpu_count",
            "loadavg",
            "python_version",
            "sqlite_version",
            "git_revision",
            "git_branch",
        ):
            self.assertIn(key, env)
        self.assertEqual(len(env["loadavg"]), 3)
        self.assertIsInstance(env["cpu_count"], int)
        self.assertGreater(env["cpu_count"], 0)
        self.assertIn("3.14", env["python_version"])
        self.assertTrue(env["sqlite_version"])
        for tool in ("rustc", "cargo", "cobc", "clang"):
            self.assertIn(f"{tool}_version", env)

        blob = json.dumps(env, sort_keys=True)
        self.assertNotIn(str(Path.home()), blob)
        self.assertNotIn("Users/", blob)
        self.assertNotIn(os.uname().nodename, blob)

    def test_git_revision_is_read_from_the_worktree(self) -> None:
        repo_root = Path(__file__).resolve().parent.parent
        env = collect_environment_metadata(repo_root)
        self.assertRegex(env["git_revision"], r"^([0-9a-f]{7,40}|)$")
        self.assertTrue(env["git_revision"])


class ComputeStatsTests(unittest.TestCase):
    def test_known_sample_statistics(self) -> None:
        stats = compute_stats([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(stats["runs"], 5)
        self.assertEqual(stats["min_ms"], 1.0)
        self.assertEqual(stats["max_ms"], 5.0)
        self.assertEqual(stats["p50_ms"], 3.0)
        self.assertEqual(stats["mean_ms"], 3.0)
        self.assertAlmostEqual(stats["p95_ms"], 4.8, places=6)
        self.assertAlmostEqual(stats["iqr_ms"], 2.0, places=6)
        self.assertAlmostEqual(stats["mad_ms"], 1.0, places=6)
        self.assertAlmostEqual(stats["stdev_ms"], 1.5811388300841898, places=6)

    def test_single_sample_has_zero_dispersion(self) -> None:
        stats = compute_stats([7.5])
        self.assertEqual(stats["runs"], 1)
        for key in ("p50_ms", "min_ms", "max_ms", "mean_ms", "iqr_ms", "mad_ms", "stdev_ms"):
            self.assertEqual(stats[key], 7.5 if key not in ("iqr_ms", "mad_ms", "stdev_ms") else 0.0)

    def test_empty_sample_set_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            compute_stats([])


class RunWorkloadTests(_TempDirTest):
    def setUp(self) -> None:
        super().setUp()
        self.store = TestBenchStore(home=self.root / "home")
        self.addCleanup(self.store.close)
        self.work = self.root / "work"
        self.work.mkdir(mode=0o700)

    def test_runs_warmup_plus_repeats_and_records_benchmark_separately(self) -> None:
        cmd = [PY, "-c", "import time; time.sleep(0.01)"]
        run, payload = run_workload(
            "sleep-10ms",
            cmd,
            cwd=self.work,
            store=self.store,
            project="fixture-project",
            revision="abc1234",
            warmup=1,
            repeats=3,
            timeout_s=30.0,
            scratch_root=self.root,
        )
        self.assertEqual(run.profile, ProfileName.BENCHMARK)
        self.assertEqual(run.status, RunStatus.PASS)
        self.assertEqual(run.revision, "abc1234")
        self.assertEqual(len(run.commands), 1)
        self.assertEqual(payload["stats"]["runs"], 3)
        self.assertEqual(len(payload["samples_ms"]), 3)
        self.assertEqual(len(payload["warmup_ms"]), 1)
        self.assertEqual(payload["workload"], "sleep-10ms")
        self.assertEqual(payload["environment"]["cpu_count"], collect_environment_metadata(self.work)["cpu_count"])
        self.assertNotIn("threshold", json.dumps(payload).lower())
        self.assertNotIn("passed", json.dumps({k: v for k, v in payload.items() if k != "stats"}).lower())

        stored = self.store.list_benchmarks(project="fixture-project", workload="sleep-10ms")
        self.assertEqual(len(stored), 1)
        self.assertEqual(stored[0]["median_ms"], payload["stats"]["p50_ms"])
        self.assertEqual(stored[0]["payload"]["stats"], payload["stats"])

        persisted = self.store.get_run(run.run_id)
        self.assertIsNotNone(persisted)
        assert persisted is not None
        self.assertEqual(persisted.profile, ProfileName.BENCHMARK)

    def test_nonzero_exit_is_recorded_as_failure_without_perf_thresholds(self) -> None:
        cmd = [PY, "-c", "import sys; sys.exit(3)"]
        run, payload = run_workload(
            "failing-workload",
            cmd,
            cwd=self.work,
            store=self.store,
            project="fixture-project",
            warmup=0,
            repeats=2,
            timeout_s=30.0,
            scratch_root=self.root,
        )
        self.assertEqual(run.status, RunStatus.FAIL)
        self.assertEqual(payload["exit_codes"], [3, 3])
        self.assertTrue(all(v >= 0 for v in payload["samples_ms"]))


class BenchmarkComparisonTests(unittest.TestCase):
    @staticmethod
    def _payload(revision: str, samples: list[float], cpu_count: int = 8) -> dict:
        return {
            "workload": "parse",
            "project": "proj",
            "revision": revision,
            "stats": compute_stats(samples),
            "environment": {
                "os_name": "Darwin",
                "os_release": "27.2.1",
                "arch": "arm64",
                "cpu_count": cpu_count,
                "python_version": "3.14.7",
                "sqlite_version": "3.50.0",
                "loadavg": [1.0, 1.0, 1.0],
            },
        }

    def test_deltas_are_ratios_without_any_pass_fail_verdict(self) -> None:
        baseline = self._payload("aaa1111", [10.0, 10.0, 10.0, 10.0, 10.0])
        candidate = self._payload("aaa1111", [10.5, 10.5, 10.5, 10.5, 10.5])
        comp = compare_benchmarks(baseline, candidate)
        self.assertEqual(comp["workload"], "parse")
        p50 = comp["deltas"]["p50_ms"]
        self.assertAlmostEqual(p50["baseline"], 10.0)
        self.assertAlmostEqual(p50["candidate"], 10.5)
        self.assertAlmostEqual(p50["delta_ms"], 0.5)
        self.assertAlmostEqual(p50["delta_pct"], 5.0)
        self.assertAlmostEqual(p50["ratio"], 1.05)
        self.assertIn("p95_ms", comp["deltas"])
        self.assertIn("iqr_ms", comp["deltas"])
        self.assertEqual(comp["drift_notes"], [])
        blob = json.dumps(comp).lower()
        for banned in ("passed", "failed", "threshold", "regression"):
            self.assertNotIn(banned, blob)

    def test_environment_and_revision_drift_are_reported_as_notes(self) -> None:
        baseline = self._payload("aaa1111", [10.0] * 5, cpu_count=8)
        candidate = self._payload("bbb2222", [12.0] * 5, cpu_count=4)
        comp = compare_benchmarks(baseline, candidate)
        self.assertTrue(any("revision" in n for n in comp["drift_notes"]))
        self.assertTrue(any("cpu_count" in n for n in comp["drift_notes"]))
        self.assertFalse(comp["comparable"])

    def test_formatted_report_is_human_readable_and_threshold_free(self) -> None:
        comp = compare_benchmarks(
            self._payload("aaa1111", [10.0] * 5),
            self._payload("bbb2222", [12.0] * 5),
        )
        text = format_benchmark_comparison(comp)
        self.assertIn("parse", text)
        self.assertIn("p50_ms", text)
        self.assertIn("ratio", text)
        self.assertIn("aaa1111", text)
        self.assertIn("bbb2222", text)
        self.assertIn("drift", text.lower())
        for banned in ("passed", "failed", "threshold"):
            self.assertNotIn(banned, text.lower())


if __name__ == "__main__":
    unittest.main()
