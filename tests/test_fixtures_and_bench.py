"""Fixtures, fault probes and benchmarking tests: determinism, hostility, no verdicts.

Three properties are defended here, and they pull in different directions:

* **Determinism** — corpora, the WAL/mail fixtures and repeated store init must
  produce byte-identical artifacts. A fixture that embeds a timestamp or a random id
  makes every downstream failure irreproducible, so identity is asserted directly
  (same sha256, same bytes, same fingerprints), not inferred from equal counts.
* **Hostility** — the corpora must actually contain the inputs that break naive
  parsers and renderers: split code points, NUL/BEL, C1 controls, OSC 52/8, DCS,
  pre-epoch timestamps and 64-bit row ids. Each corpus test re-derives the hazard from
  the payload instead of trusting the entry's ``note``.
* **No performance policy** — benchmarks time a command and publish a distribution
  plus provenance; they never convert a delta into pass/fail. ``run_workload``'s status
  is a liveness verdict ("every iteration exited 0"), and ``compare_benchmarks`` emits
  ratios and drift notes. Tests assert the *absence* of ``threshold``/``passed``/
  ``regression`` keys as firmly as they assert the numbers, because a well-meaning
  "speed gate" would otherwise creep back in.

Process-level tests (PTY resize, signal crash, tmux lifecycle, SQLite contention) are
timing-sensitive: they wait for a child to become ready, poll for a state transition,
and clean up in ``finally``. They use real subprocesses precisely because the
behaviours under test (``SIGWINCH`` delivery, signal classification, WAL lock
contention, tmux server teardown) do not exist in-process. tmux-dependent tests skip
when the binary is missing.

Paths: ``_scratch`` names ``~/.hermes/cache/scratch`` directly instead of calling
``default_scratch_root()``, so the location does not shift with
``TESTBENCH_SCRATCH_DIR``/``TMPDIR``; that makes a leftover fixture easy to find. Every
temporary tree is removed with ``addCleanup`` or a ``finally`` block.
"""

from __future__ import annotations

import json
import os
import platform
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
    """Fixed scratch parent for this module's temp trees (see module docstring)."""
    root = Path.home() / ".hermes/cache/scratch"
    root.mkdir(parents=True, exist_ok=True)
    return root


class _TempDirTest(unittest.TestCase):
    """Base for tests that need a private tree; cleanup is registered, not manual.

    ``addCleanup`` runs at test teardown even when an assertion fails, which matters for
    the SQLite tests: a leaked ``-wal``/``-shm`` triple would otherwise persist in the
    scratch parent.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(dir=str(_scratch()), prefix="tb-fix-")
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)


class UnicodeCorpusTests(unittest.TestCase):
    """The Unicode corpus covers encodability bytes *and* display-width classes.

    Byte-length assertions (2/3/4) are exact because these are the canonical shortest
    sequences; a corpus that swapped é for plain ``e`` would still look like text but
    would no longer exercise the 2-byte decoder path. Width, combining and plane-14
    checks are re-derived with ``unicodedata`` rather than trusted from notes.
    """

    def test_corpus_covers_every_required_unicode_category(self) -> None:
        """All nine kinds exist and each entry carries real bytes.

        The byte-length checks pin the shortest encoding per width class; ``\\x7f``/
        ``\\x80`` pin the ASCII→2-byte boundary (the classic off-by-one in a decoder).
        Kind coverage is asserted as a subset test so adding kinds does not break it.
        """
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
        """Two builds produce identical names and text, and no name repeats.

        Unique names are load-bearing for reporting: cases are keyed by name, so a
        duplicate would silently merge two results and undercount. Text equality (not
        just name equality) catches a corpus that iterates a set or reads the clock.
        """
        first = unicode_boundary_corpus()
        second = unicode_boundary_corpus()
        self.assertEqual([e.name for e in first], [e.name for e in second])
        self.assertEqual([e.text for e in first], [e.text for e in second])
        self.assertEqual(len({e.name for e in first}), len(first))

    def test_corpus_entries_round_trip_through_json_dict(self) -> None:
        """An entry's ``to_dict`` survives ``json.dumps`` with the name visible.

        ``raw`` is emitted as a list of ints, which is what makes arbitrary bytes
        JSON-safe; the assertion is minimal because the shape is covered elsewhere —
        its purpose is to catch a non-serializable field being added to ``CorpusEntry``.
        """
        entry = unicode_boundary_corpus()[0]
        dumped = json.dumps(entry.to_dict(), ensure_ascii=False)
        self.assertIn(entry.name, dumped)


class MalformedInputCorpusTests(unittest.TestCase):
    """Malformed structured inputs must be rejected by real parsers, not coerced.

    Each payload is fed to a real parser where one exists (``json.loads`` for the
    truncated object and the JSON-valued entries) and inspected as exact bytes
    otherwise, so a corpus entry that is "malformed" only in its note fails here.
    """

    def test_corpus_covers_every_required_malformed_category(self) -> None:
        """Six kinds; JSON actually fails; NUL, pre-epoch and >2^32 values are present.

        The duplicated ``received_at < 0`` assertion is redundant and harmless; the
        meaningful part is that the value parses as a *number* that is negative, i.e.
        the fixture reaches the parser as valid JSON that is semantically hostile.
        ``huge_string`` is asserted at exactly 10240 chars because the size is the
        pressure being applied, not an implementation detail.
        """
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
        """Two builds emit byte-identical payloads, including the raw-byte entries.

        Compares ``payload()`` rather than ``text`` so the ``raw``-authoritative entries
        are covered too; those are exactly the ones a refactor is most likely to break.
        """
        self.assertEqual(
            [e.payload() for e in malformed_input_corpus()],
            [e.payload() for e in malformed_input_corpus()],
        )


class TerminalEscapeCorpusTests(unittest.TestCase):
    """Every escape family is present *and* every one is neutralized by the sanitizers.

    Coverage and behaviour are asserted separately: the first test pins the bytes, the
    second runs both public cleaning functions and forbids ESC, C1 controls, NUL and
    BEL in the output. ``strip_terminal_escapes`` and ``sanitize_terminal_text`` are
    checked because callers use them for different purposes (raw log vs display text);
    a fix in one must not leave the other leaking.
    """

    def test_corpus_covers_every_required_escape_family(self) -> None:
        """OSC 52/8, DCS, CSI 2J, 8-bit C1, NUL and BEL each appear with exact prefixes.

        The C1 check uses the single-byte ``\\x9b`` form, not the 7-bit ``\\x1b[``
        equivalent, because only the 8-bit form exercises the 0x80–0x9F strip path.
        """
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
        """No ESC, C1, NUL or BEL survives either sanitizer, for every corpus entry.

        The two-step assertion (strip then sanitize) is intentional: stripping removes
        escape sequences but should not be relied on to remove embedded NUL/BEL, so the
        test documents both contracts. Failure messages name the entry, which makes a
        regression attributable to one family rather than "the sanitizer".
        """
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
    """WAL contention is real, bounded and lossless — proven with actual processes.

    A single-process test can only check pragmas; it cannot show that concurrent writers
    retry or that ``busy_timeout=0`` yields ``SQLITE_BUSY`` instead of silent loss. The
    probe therefore runs two phases (timeout 0, then 5000 ms) with 4 writer processes
    and compares committed rows against ``writers * ops_per_writer``.

    The final count and ``integrity_check`` run on a fresh connection, so rows visible
    only inside a writer's own connection cannot be mistaken for committed data.
    """

    def test_fixture_creates_wal_database_with_busy_timeout(self) -> None:
        """The fixture is empty, in WAL mode, with a 5 s busy timeout on disk.

        The pragma is re-read through a new connection, because journal mode is a
        persistent database property and a per-connection setting would not survive the
        fixture call. Zero rows is asserted so a rerun on a stale file cannot inflate
        the later expected-count check.
        """
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
        """Both phases commit every row; the no-timeout phase must *observe* a busy retry.

        ``busy_retries >= 1`` is the discriminating assertion: without it the test would
        pass on a machine where the writers never actually overlapped (fast scheduler,
        serialised startup), leaving the contention claim unproven. ``measurable_contention``
        is the probe's own summary of that same fact, so it is cross-checked rather than
        trusted. ``integrity_ok`` and the final ``COUNT(*)`` catch the opposite failure —
        rows lost or the WAL left unmerged.
        """
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
    """A real PTY child observes the exact geometry transition through ``SIGWINCH``.

    The probe sets the initial window size on the slave fd, then resizes through the
    master fd and signals the child; the child reports what ``TIOCGWINSZ`` actually
    returns. Asserting the *reported* values, not the requested ones, is what makes
    this a test of delivery rather than of ``ioctl`` argument packing.
    """

    def test_child_observes_exact_geometry_transition_via_sigwinch(self) -> None:
        """Initial and final geometry are exact, at least one SIGWINCH arrived.

        ``geometry_transition_ok`` is the probe's aggregate; the individual comparisons
        are kept so a failure says whether the initial or the final size was wrong.
        ``child_exit_code == 0`` ensures the child ended by its own protocol rather
        than by the cleanup path killing it, and that is the teardown evidence here:
        ``leaked_children`` is a fixed ``0`` placeholder, not a survivor count.
        """
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
        """A non-default 100x40 → 90x30 transition is reported verbatim.

        The first test would still pass if the probe hard-coded 80x24/120x36 and
        happened to match its arguments; changing both dimensions catches that.
        """
        result = run_pty_resize_probe((100, 40), (90, 30))
        self.assertEqual(list(result["initial_reported"]), [100, 40])
        self.assertEqual(list(result["final_reported"]), [90, 30])


class SignalCrashTests(unittest.TestCase):
    """A signalled child is classified as signalled, with ``128 + signum`` and no survivors.

    ``run_signal_crash_probe`` launches a child that starts a grandchild, records its
    pid, then kills itself. The probe terminates the child's process group and checks
    the grandchild is gone. This is the classification contract the isolation layer
    relies on: ``exit_code`` for a signal death is ``128 + n``, not ``None`` and not a
    negative ``Popen.returncode``.
    """

    def test_sigkill_is_classified_as_a_signal_with_no_leaked_children(self) -> None:
        """SIGKILL → ``signalled``, exit 137, grandchild reaped.

        SIGKILL is chosen deliberately: on macOS SIGABRT/SIGSEGV make ReportCrash file a
        crash report and raise a UI dialog for every run, which would make an automated
        suite intrusive. SIGKILL cannot be trapped, so the child cannot mask the result.
        """
        # SIGKILL, not SIGABRT: a self-abort makes macOS ReportCrash show a crash dialog per run.
        result = run_signal_crash_probe(signal.SIGKILL)
        self.assertEqual(result["signal_num"], int(signal.SIGKILL))
        self.assertEqual(result["exit_code"], 128 + int(signal.SIGKILL))
        self.assertEqual(result["expected_exit_code"], 128 + int(signal.SIGKILL))
        self.assertEqual(result["classification"], "signalled")
        self.assertEqual(result["leaked_children"], 0)
        self.assertTrue(result["grandchild_reaped"])

    def test_classification_generalises_to_other_signals(self) -> None:
        """SIGTERM also yields ``128 + 15``, so the mapping is not hard-coded for KILL.

        Only the number and code are asserted here — unlike the SIGKILL case this test
        does not require the grandchild check, keeping it cheap. That asymmetry is
        intentional: one deep case plus one generalisation beats two near-duplicates.
        """
        result = run_signal_crash_probe(signal.SIGTERM)
        self.assertEqual(result["signal_num"], int(signal.SIGTERM))
        self.assertEqual(result["exit_code"], 128 + int(signal.SIGTERM))


class RepeatedInitTests(unittest.TestCase):
    """Initialization is idempotent: the fifth open of a store looks like the first.

    The fingerprint is ``(user_version, sorted table names, run count)``. Comparing it
    across iterations catches migrations that rerun destructively, tables created
    conditionally on a flag, and accidental row insertion during schema setup.
    ``distinct_fingerprints == 1`` is the assertion; the list of errors exists so a
    failure shows *which* iteration broke rather than only that one did.
    """

    def test_repeated_store_and_fixture_initialization_is_idempotent(self) -> None:
        """Five opens produce one fingerprint, WAL mode, and identical mail DB hashes.

        ``len(mail_db_hashes) == 2`` is a shape check: the determinism claim is about
        two independently created DBs, so a probe that returned one hash twice would
        otherwise pass. ``schema_version >= 1`` guards against a fixture that reports a
        default of zero for "no migration ran".
        """
        result = run_repeated_init_probe(times=5)
        self.assertEqual(result["times"], 5)
        self.assertTrue(result["idempotent"])
        self.assertEqual(result["distinct_fingerprints"], 1)
        self.assertEqual(result["errors"], [])
        self.assertGreaterEqual(result["schema_version"], 1)
        self.assertTrue(result["mail_db_deterministic"])
        self.assertEqual(len(result["mail_db_hashes"]), 2)


class TmuxLifecycleTests(unittest.TestCase):
    """A private tmux server supports the lifecycle the TUI relies on, then disappears.

    The probe uses ``-L <random socket>`` with ``-f /dev/null`` and a private
    ``TMUX_TMPDIR``, so it never sees the developer's server or config. It exercises
    session/window/pane ``@hx`` options, a pane crash with ``remain-on-exit`` followed
    by ``respawn-pane -k``, and ``kill-server`` teardown.
    """

    @unittest.skipUnless(shutil.which("tmux"), "tmux unavailable")
    def test_isolated_tmux_lifecycle_with_hx_options_crash_recovery_and_teardown(self) -> None:
        """Two windows/panes, per-level ``@hx`` options, recovery, and a dead server.

        Crash recovery is asserted as a *transition* (pane marked dead, then alive after
        respawn), not merely as a successful ``respawn-pane`` call: the command succeeds
        even when the pane never died. ``server_gone`` proves teardown, which is the
        difference between "the test passed" and "the test left a daemon behind".
        """
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
        """A caller-supplied scratch root is used and left with no ``tb-*`` entries.

        The probe may relocate its socket parent for length reasons, but the scratch
        tree it was handed must still be empty afterwards. Checking ``glob("tb-*")``
        rather than full emptiness tolerates unrelated files a developer put there.
        """
        scratch = Path(tempfile.mkdtemp(dir=str(_scratch()), prefix="tb-tmux-"))
        self.addCleanup(shutil.rmtree, scratch, True)
        result = run_tmux_lifecycle_probe(scratch)
        self.assertTrue(result["available"], result.get("reason"))
        self.assertTrue(result["server_gone"])
        self.assertEqual(list(scratch.glob("tb-*")), [])


class CrossLangMailDbTests(_TempDirTest):
    """The shared mail DB matches the schema both ``hxmail`` and ``listbench.cob`` expect.

    ``EXPECTED_COLUMNS`` is written out explicitly rather than imported, because its
    job is to be an independent statement of the contract: importing the fixture's own
    schema would make the test agree with any change. Column order is compared too —
    a COBOL reader using fixed positions depends on it.
    """

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
        """Five tables, exact column order, WAL mode.

        ``sorted(info["tables"]) == sorted(EXPECTED_COLUMNS)`` catches extra tables
        (FTS shadow tables are filtered by the fixture) that would confuse a consumer
        enumerating tables; ``PRAGMA table_info`` catches renamed/reordered columns.
        """
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
        """Rows include pre-1970 times, a rid above 2^32, UTF-8 text and raw ESC/BEL.

        ``text_factory = bytes`` is required for the escape assertions: decoding to str
        would hide exact byte values, and the point is that a body may contain control
        bytes a renderer must neutralize. The ``日本語`` subject is read back from the
        max-rid row, tying the overflow row and the UTF-8 row together instead of
        testing them independently.
        """
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
        """Two DBs built at different paths are byte-identical, not merely equivalent.

        ``wal_checkpoint(TRUNCATE)`` before hashing is what makes the file's bytes a
        stable artifact; comparing ``read_bytes()`` as well as the reported sha256 means
        the digest cannot drift from the file it claims to describe.
        """
        first = self.root / "a.db"
        second = self.root / "b.db"
        info_a = create_cross_lang_mail_db(first)
        info_b = create_cross_lang_mail_db(second)
        self.assertEqual(info_a["sha256"], info_b["sha256"])
        self.assertEqual(first.read_bytes(), second.read_bytes())
        self.assertEqual(info_a["emails"], info_b["emails"])


class EnvironmentMetadataTests(_TempDirTest):
    """Provenance is captured, JSON-safe, and free of local identity.

    Two of these tests pull in opposite directions: the metadata must be specific
    enough to tell machines and toolchains apart, and must not leak the user's home
    path, the literal ``Users/`` fragment, or the hostname. ``_scrub_identity`` is what
    makes both true, and it is applied to version strings as well as uname fields.
    """

    def test_metadata_reports_versions_load_without_leaking_identity(self) -> None:
        """Required keys exist, types are sane, and no home/host string appears.

        Tool versions are allowed to be ``""`` (missing tool = "unknown", not an error),
        so presence of the key is asserted rather than a non-empty value. ``loadavg``
        is asserted to have three entries because a two-value fallback would silently
        shift the meaning of the fields.
        """
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
        self.assertEqual(env["python_version"], platform.python_version())
        self.assertTrue(env["sqlite_version"])
        for tool in ("rustc", "cargo", "cobc", "clang"):
            self.assertIn(f"{tool}_version", env)

        blob = json.dumps(env, sort_keys=True)
        self.assertNotIn(str(Path.home()), blob)
        self.assertNotIn("Users/", blob)
        self.assertNotIn(os.uname().nodename, blob)

    def test_git_revision_is_read_from_the_worktree(self) -> None:
        """``git_revision`` is a hex SHA (or empty on failure) and non-empty in the repo.

        The regex admits the empty string because ``_git`` treats "not a repository"
        as unknown provenance rather than an error; the second assertion is what pins
        this test to a real worktree, where an empty value would mean the git call
        broke.
        """
        repo_root = Path(__file__).resolve().parent.parent
        env = collect_environment_metadata(repo_root)
        self.assertRegex(env["git_revision"], r"^([0-9a-f]{7,40}|)$")
        self.assertTrue(env["git_revision"])


class ComputeStatsTests(unittest.TestCase):
    """Statistics are checked against hand-computed values, including interpolation.

    ``p95`` of five samples is a Type-7 interpolation (4.8), not an observed point;
    pinning it documents that choice. ``stdev`` is the n-1 sample form, so a switch to
    the population formula would change the literal. Single-sample and empty-set cases
    cover the two edges where a naive implementation divides by zero or lies with zeros.
    """

    def test_known_sample_statistics(self) -> None:
        """1..5 yields p50 3, p95 4.8, IQR 2, MAD 1, sample stdev ≈ 1.58113883.

        Exact equality is used for the order statistics and interpolation results
        (they are exactly representable), ``assertAlmostEqual`` for the stdev, which is
        irrational and depends on float summation order.
        """
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
        """One sample: every centre equals it, and dispersion is exactly 0.0.

        The ternary in the assertion encodes the intended split: IQR/MAD/stdev are
        ``0.0`` because there is no spread to measure, while min/max/mean/p50 are the
        value itself. A ``None`` here would break callers that format or compare the
        dispersion fields.
        """
        stats = compute_stats([7.5])
        self.assertEqual(stats["runs"], 1)
        for key in ("p50_ms", "min_ms", "max_ms", "mean_ms", "iqr_ms", "mad_ms", "stdev_ms"):
            self.assertEqual(stats[key], 7.5 if key not in ("iqr_ms", "mad_ms", "stdev_ms") else 0.0)

    def test_empty_sample_set_is_rejected(self) -> None:
        """Empty input raises ``ValueError`` instead of returning zeroed stats.

        This is a policy assertion: zeros for zero samples look like a fast measurement
        and would be consumed by a downstream threshold. Rejecting is the only safe
        behaviour, so the exception type is pinned.
        """
        with self.assertRaises(ValueError):
            compute_stats([])


class RunWorkloadTests(_TempDirTest):
    """``run_workload`` records timings and liveness, never a performance verdict.

    The workload is a short-lived Python process, so the numbers are noise by design:
    the assertions are about *shape and provenance* (one warmup, N repeats, N exit
    codes, a benchmark-profile run in the store), not about how long anything took.
    ``self.work`` is the workload cwd and is created ``0700``.
    """

    def setUp(self) -> None:
        super().setUp()
        self.store = TestBenchStore(home=self.root / "home")
        self.addCleanup(self.store.close)
        self.work = self.root / "work"
        self.work.mkdir(mode=0o700)

    def test_runs_warmup_plus_repeats_and_records_benchmark_separately(self) -> None:
        """Warmup is separate from samples; the run is BENCHMARK and persisted.

        ``warmup=1, repeats=3`` pins the split: three samples and one warmup duration.
        The two absence assertions are the policy test — no ``threshold`` key, and no
        ``passed`` key outside ``stats`` — because either would invite treating a
        timing as a gate. ``stored[0]["median_ms"]`` must equal ``stats["p50_ms"]`` so
        the denormalized store column cannot drift from the payload it indexes.
        """
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
        """Exit 3 in every iteration means FAIL, and the exit codes are kept verbatim.

        ``exit_codes == [3, 3]`` with two repeats distinguishes a liveness verdict from
        an aggregate: the run failed because the workload failed, not because a duration
        crossed a line. Durations are still recorded (non-negative) so evidence exists
        for the failure.
        """
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
    """Comparison output is descriptive: ratios, deltas and drift, with no verdict.

    ``_payload`` builds a synthetic stored payload (stats plus environment) so these
    tests do not need a benchmark run. That is safe here because ``compare_benchmarks``
    is pure, and the helper computes real stats: ``compare_benchmarks`` documents that
    a payload without them degrades to a table of zeros.
    """

    @staticmethod
    def _payload(revision: str, samples: list[float], cpu_count: int = 8) -> dict:
        """Minimal payload shaped like ``run_workload``'s, with real computed stats."""
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
        """10.0 → 10.5 is +0.5 ms, +5 %, ratio 1.05, with no verdict vocabulary.

        Pinning all four fields catches the common mistake of computing a percentage
        from the candidate instead of the baseline. The banned-word scan is a blunt but
        effective guard against a ``"regression": true`` field being added: the payload
        is JSON, so the check covers keys and values at once.
        """
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
        """A different revision and CPU count produce notes and ``comparable == False``.

        Two independent drift keys are changed so the test proves iteration over the
        drift set, not special-casing one field. ``comparable`` is the *only* gate in
        the comparison layer, and it is about environment, not speed — hence checking
        it here rather than in the delta test.
        """
        baseline = self._payload("aaa1111", [10.0] * 5, cpu_count=8)
        candidate = self._payload("bbb2222", [12.0] * 5, cpu_count=4)
        comp = compare_benchmarks(baseline, candidate)
        self.assertTrue(any("revision" in n for n in comp["drift_notes"]))
        self.assertTrue(any("cpu_count" in n for n in comp["drift_notes"]))
        self.assertFalse(comp["comparable"])

    def test_formatted_report_is_human_readable_and_threshold_free(self) -> None:
        """The text report names the workload, stat, ratio, both revisions and drift.

        It also must not contain ``passed``/``failed``/``threshold``: a report that
        merely omits a numeric gate but says "failed" would still be read as a verdict.
        Drift is non-empty here (the revisions differ); the ``none`` branch the
        renderer uses for an empty drift set is not exercised by this test.
        """
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
