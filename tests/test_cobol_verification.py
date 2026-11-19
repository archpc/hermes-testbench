"""Phase 1B: GnuCOBOL verification against a deterministic Decimal oracle.

The oracle layer (fixed-width records, signed decimals, ``ROUNDED``, truncation,
PICTURE overflow, malformed input) is exercised unconditionally. The compiler
layer is skipped only when ``cobc`` is genuinely absent; on a host with GnuCOBOL
installed every case below compiles and runs for real.

The central negative control is ``test_exit_zero_with_wrong_result_fails_verification``:
a program that compiles cleanly and exits 0 while adding a wrong total must FAIL,
because exit status alone is not verification.

Layout of the file
------------------
Four oracle classes (``FixedWidthRecordTests``, ``SignedDecimalOracleTests``,
``RoundingOracleTests``, ``PictureOverflowOracleTests``) plus
``CobolDiagnosticParsingTests`` run on any host: they exercise the reference oracle
and the diagnostic parser directly, no ``cobc`` involved. Everything from
``CobolVerificationCaseTests`` down is gated on ``COBC`` being resolvable at *import*
time, so the skip decision is made once per process from ``find_cobc()`` — an
installed-but-broken ``cobc`` is not a skip, it is a failing compile case.

What a test here is allowed to assert
-------------------------------------
Expected output is always derived from the oracle (``_ledger_expected``,
``format_edited``, ``cobol_round``) and then pinned to one literal string, so the
literal catches an oracle that drifts while the derivation keeps the comparison
independent of hand-typed arithmetic. A test that only compared the program to the
program would pass while both were wrong.

Negative controls are structural, not decorative: the central "must fail" cases first
establish the failure cause in isolation (compiles OK, exits 0, prints the wrong
number) and only then assert that verification rejected it. Without that
pre-condition a FAIL could be produced by a crash and the test would still be green.

Sources are written into a per-test ``_tmpdir`` under the isolation scratch root, with
``rmtree`` registered on the case; the CLI allocates its own build tree under the same
scratch root and ``verify_case`` gives every case a subdirectory named after it, so two
cases never share a binary even when they name the same source.
"""

from __future__ import annotations

import json
import os
from decimal import Decimal
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

from hermes_testbench.adapters import cobol_check, execute_suite
from hermes_testbench.isolation import default_scratch_root
from hermes_testbench.adapters.cobol_check import (
    CobolCase,
    Field,
    FixedRecordLayout,
    MalformedRecordError,
    PictureOverflowError,
    ReferenceLedger,
    cobol_round,
    cobol_truncate,
    compile_source,
    enforce_picture,
    fits_picture,
    format_edited,
    format_ledger_summary,
    parse_diagnostics,
    parse_fixed_records,
    run_program,
    verify_cases,
)
from hermes_testbench.adapters.parsers import parse_cobol
from hermes_testbench.discovery import discover_project
from hermes_testbench.schema import RunStatus, SuiteSpec

REPO_ROOT = Path(__file__).resolve().parents[1]
COBC = cobol_check.find_cobc()
requires_cobc = unittest.skipUnless(COBC, "cobc (GnuCOBOL) not installed")


def _tmpdir(prefix: str = "tb-cobol-", case: unittest.TestCase | None = None) -> Path:
    """Scratch directory under the isolation root, cleaned up with the test case.

    The dir comes from ``default_scratch_root()`` rather than ``/tmp`` so the suite
    obeys the same isolation policy as the runner; passing ``case`` registers the
    removal so a failing assertion still tears the tree down.
    """
    path = Path(tempfile.mkdtemp(prefix=prefix, dir=str(default_scratch_root())))
    if case is not None:
        case.addCleanup(shutil.rmtree, path, True)
    return path


def _suite(**kwargs) -> SuiteSpec:
    """A minimal ``SuiteSpec`` whose command is replaced by the caller.

    The default ``/usr/bin/true`` is only a placeholder required by the dataclass;
    ``_run_spec`` always overrides ``command`` with the module CLI.
    """
    params = {
        "name": "cobol-verify",
        "adapter": "cobol",
        "profile": "unit",
        "command": ["/usr/bin/true"],
    }
    params.update(kwargs)
    return SuiteSpec(**params)


# --------------------------------------------------------------------------
# Fixtures: real GnuCOBOL programs (free form unless the name says FIXED)
# --------------------------------------------------------------------------

LEDGER_LAYOUT = FixedRecordLayout(
    fields=(
        Field("account", 4, "text"),
        Field("sign", 1, "sign"),
        Field("amount", 6, "decimal", scale=2),
    )
)

LEDGER_INPUT = "AAAA+001050\nBBBB-000325\nCCCC+000005\n"

LEDGER_OK = """IDENTIFICATION DIVISION.
PROGRAM-ID. ledgerok.
DATA DIVISION.
WORKING-STORAGE SECTION.
01 WS-REC PIC X(11).
01 WS-SIGN PIC X.
01 WS-AMT-BYTES PIC X(6).
01 WS-AMT REDEFINES WS-AMT-BYTES PIC 9(4)V99.
01 WS-SIGNED PIC S9(6)V99.
01 WS-TOTAL PIC S9(7)V99 VALUE 0.
01 WS-COUNT PIC 9(3) VALUE 0.
01 WS-OUT PIC S9(7)V99 SIGN IS LEADING SEPARATE.
01 WS-EOF PIC X VALUE 'N'.
PROCEDURE DIVISION.
    PERFORM UNTIL WS-EOF = 'Y'
        MOVE SPACES TO WS-REC
        ACCEPT WS-REC
            ON EXCEPTION MOVE 'Y' TO WS-EOF
        END-ACCEPT
        IF WS-EOF = 'Y' OR WS-REC = SPACES
            MOVE 'Y' TO WS-EOF
        ELSE
            MOVE WS-REC(5:1) TO WS-SIGN
            MOVE WS-REC(6:6) TO WS-AMT-BYTES
            MOVE WS-AMT TO WS-SIGNED
            IF WS-SIGN = '-'
                MULTIPLY -1 BY WS-SIGNED
            END-IF
            ADD WS-SIGNED TO WS-TOTAL
            ADD 1 TO WS-COUNT
        END-IF
    END-PERFORM
    MOVE WS-TOTAL TO WS-OUT
    DISPLAY "COUNT=" WS-COUNT
    DISPLAY "TOTAL=" WS-OUT
    STOP RUN.
"""

# Same program, one bug: the sign is ignored. Compiles cleanly, exits 0.
LEDGER_WRONG = LEDGER_OK.replace(
    "            MOVE WS-AMT TO WS-SIGNED\n"
    "            IF WS-SIGN = '-'\n"
    "                MULTIPLY -1 BY WS-SIGNED\n"
    "            END-IF\n",
    "            MOVE WS-AMT TO WS-SIGNED\n",
).replace("PROGRAM-ID. ledgerok.", "PROGRAM-ID. ledgerwrong.")

# Same arithmetic plus explicit input validation: a malformed record aborts.
LEDGER_GUARD = """IDENTIFICATION DIVISION.
PROGRAM-ID. ledgerguard.
DATA DIVISION.
WORKING-STORAGE SECTION.
01 WS-REC PIC X(11).
01 WS-SIGN PIC X.
01 WS-AMT-BYTES PIC X(6).
01 WS-AMT REDEFINES WS-AMT-BYTES PIC 9(4)V99.
01 WS-SIGNED PIC S9(6)V99.
01 WS-TOTAL PIC S9(7)V99 VALUE 0.
01 WS-COUNT PIC 9(3) VALUE 0.
01 WS-OUT PIC S9(7)V99 SIGN IS LEADING SEPARATE.
01 WS-EOF PIC X VALUE 'N'.
PROCEDURE DIVISION.
    PERFORM UNTIL WS-EOF = 'Y'
        MOVE SPACES TO WS-REC
        ACCEPT WS-REC
            ON EXCEPTION MOVE 'Y' TO WS-EOF
        END-ACCEPT
        IF WS-EOF = 'Y' OR WS-REC = SPACES
            MOVE 'Y' TO WS-EOF
        ELSE
            MOVE WS-REC(5:1) TO WS-SIGN
            IF WS-SIGN NOT = '+' AND WS-SIGN NOT = '-'
                DISPLAY "MALFORMED: bad sign"
                STOP RUN RETURNING 2
            END-IF
            MOVE WS-REC(6:6) TO WS-AMT-BYTES
            IF WS-AMT-BYTES IS NOT NUMERIC
                DISPLAY "MALFORMED: non-numeric amount"
                STOP RUN RETURNING 2
            END-IF
            MOVE WS-AMT TO WS-SIGNED
            IF WS-SIGN = '-'
                MULTIPLY -1 BY WS-SIGNED
            END-IF
            ADD WS-SIGNED TO WS-TOTAL
            ADD 1 TO WS-COUNT
        END-IF
    END-PERFORM
    MOVE WS-TOTAL TO WS-OUT
    DISPLAY "COUNT=" WS-COUNT
    DISPLAY "TOTAL=" WS-OUT
    STOP RUN.
"""

ROUNDING_BODY = """IDENTIFICATION DIVISION.
PROGRAM-ID. {program}.
DATA DIVISION.
WORKING-STORAGE SECTION.
01 POS PIC S9(3)V99 VALUE 000.01.
01 NEG PIC S9(3)V99 VALUE -000.01.
01 TWO PIC 9 VALUE 2.
01 T PIC S9(3)V99 VALUE 0.
01 OUT-T PIC S9(3)V99 SIGN IS LEADING SEPARATE.
PROCEDURE DIVISION.
    COMPUTE T {rounded}= POS / TWO
    MOVE T TO OUT-T
    DISPLAY "POS=" OUT-T
    COMPUTE T {rounded}= NEG / TWO
    MOVE T TO OUT-T
    DISPLAY "NEG=" OUT-T
    STOP RUN.
"""

ROUNDING_OK = ROUNDING_BODY.format(program="roundok", rounded="ROUNDED ")
ROUNDING_TRUNC = ROUNDING_BODY.format(program="roundtrunc", rounded="")

OVERFLOW_WRAP = """IDENTIFICATION DIVISION.
PROGRAM-ID. overflowwrap.
DATA DIVISION.
WORKING-STORAGE SECTION.
01 BIG PIC 9(6) VALUE 99999.
01 ONE PIC 9 VALUE 1.
01 SMALL PIC 9(5) VALUE 0.
PROCEDURE DIVISION.
    COMPUTE SMALL = BIG + ONE
    DISPLAY "SMALL=" SMALL
    STOP RUN.
"""

OVERFLOW_GUARD = """IDENTIFICATION DIVISION.
PROGRAM-ID. overflowguard.
DATA DIVISION.
WORKING-STORAGE SECTION.
01 BIG PIC 9(6) VALUE 99999.
01 ONE PIC 9 VALUE 1.
01 SMALL PIC 9(5) VALUE 0.
PROCEDURE DIVISION.
    COMPUTE SMALL = BIG + ONE
        ON SIZE ERROR
            DISPLAY "OVERFLOW"
            STOP RUN RETURNING 3
    END-COMPUTE
    DISPLAY "SMALL=" SMALL
    STOP RUN.
"""

COMPILE_ERROR = """IDENTIFICATION DIVISION.
PROGRAM-ID. broken.
DATA DIVISION.
WORKING-STORAGE SECTION.
01 WS-N PIC 9(2) VALUE 0.
01 WS-T PIC 9(4)V99 VALUE 0.
PROCEDURE DIVISION.
    COMPUTE WS-T = WS-N / / 2
    DISPLAY WS-T
    STOP RUN.
"""

# Fixed-format (columns 1-6 blank) so discovery's default cobc invocation applies.
FIXED_FORMAT_SUM = """       IDENTIFICATION DIVISION.
       PROGRAM-ID. FIXEDSUM.
       DATA DIVISION.
       WORKING-STORAGE SECTION.
       01 WS-TOTAL PIC S9(5)V99 VALUE 0.
       01 WS-OUT PIC S9(5)V99 SIGN IS LEADING SEPARATE.
       PROCEDURE DIVISION.
           ADD 1200.50 TO WS-TOTAL
           SUBTRACT 300.25 FROM WS-TOTAL
           MOVE WS-TOTAL TO WS-OUT
           DISPLAY "TOTAL=" WS-OUT
           STOP RUN.
"""


def _ledger_expected(text: str, *, int_digits: int = 7, scale: int = 2) -> str:
    """Oracle-rendered ``COUNT=``/``TOTAL=`` block for ``text``, as COBOL would print it."""
    ledger = ReferenceLedger(LEDGER_LAYOUT, total_int_digits=int_digits, scale=scale)
    return format_ledger_summary(ledger.summarize(text), int_digits=int_digits, scale=scale)


def _write(root: Path, name: str, text: str) -> Path:
    """Write COBOL source (or a sidecar) and return its path."""
    path = root / name
    path.write_text(text, encoding="utf-8")
    return path


def _spec(root: Path, cases: list, name: str = "cases.json") -> Path:
    """Write a ``{"cases": [...]}`` spec next to the sources it names.

    Relative ``source`` paths resolve against the spec file, not the cwd, which is
    what lets the subprocess run from ``REPO_ROOT`` and still find the fixtures.
    """
    payload = {"cases": cases}
    path = root / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _exec_suite(suite: SuiteSpec, project_root: Path, evidence: Path):
    """Run one suite through the real ``execute_suite`` path with a fixed run id."""
    return execute_suite(
        suite,
        project_root=project_root,
        evidence_dir=evidence,
        run_id="tb_cobol_verify",
    )


# --------------------------------------------------------------------------
# Reference oracle (no compiler required)
# --------------------------------------------------------------------------


class FixedWidthRecordTests(unittest.TestCase):
    """The layout parser is the oracle's input gate: whatever it accepts becomes evidence."""

    def test_layout_splits_contiguous_fixed_width_fields(self) -> None:
        """Fields are contiguous slices summing to ``width``; the amount is magnitude only.

        The sign has its own one-character field, so ``amount`` must come back as the
        absolute value. ``str(...) == "3.25"`` additionally pins the scale: the Decimal
        keeps two places instead of collapsing to ``3``, which is what makes the later
        edited-format comparison byte-exact.
        """
        self.assertEqual(LEDGER_LAYOUT.width, 11)
        record = LEDGER_LAYOUT.parse_record("AAAA-000325")
        self.assertEqual(record["account"], "AAAA")
        self.assertEqual(record["sign"], "-")
        self.assertEqual(record["amount"], Decimal("-3.25").copy_abs())
        self.assertEqual(str(record["amount"]), "3.25")

    def test_short_and_long_records_are_malformed(self) -> None:
        """Both truncation directions are rejected, and the message states the expected width.

        ``subTest`` keeps the two variants independently reported rather than letting the
        first one hide the second. The width in the message is the fix hint an operator
        needs; a bare "malformed record" would not distinguish an off-by-one column shift
        from a corrupt file.
        """
        for bad in ("AAAA+00105", "AAAA+0010500"):
            with self.subTest(record=bad):
                with self.assertRaises(MalformedRecordError) as ctx:
                    parse_fixed_records(bad + "\n", LEDGER_LAYOUT)
                self.assertIn("layout expects 11", str(ctx.exception))

    def test_bad_sign_and_non_numeric_amounts_are_malformed(self) -> None:
        """Field-level guards fire independently and name the offending field.

        Same-length records means the width check cannot catch these; the message
        distinguishes a bad sign column from a non-numeric amount column, so a layout
        whose offsets were swapped fails differently from one whose values are wrong.
        """
        with self.assertRaises(MalformedRecordError) as ctx:
            parse_fixed_records("AAAAX001050\n", LEDGER_LAYOUT)
        self.assertIn("sign must be '+' or '-'", str(ctx.exception))

        with self.assertRaises(MalformedRecordError) as ctx:
            parse_fixed_records("AAAA+0010A0\n", LEDGER_LAYOUT)
        self.assertIn("must be 6 digits", str(ctx.exception))

    def test_trailing_newline_is_not_a_record_but_interior_blank_is(self) -> None:
        """Exactly one trailing newline is a terminator; a blank line in the middle is data.

        ``_record_lines`` drops a single trailing empty string after splitting on
        ``"\n"``. A parser that stripped all empty lines would silently swallow the
        interior blank instead of reporting it, so both halves of the asymmetry are
        asserted: two records on the left, a raise on the right.
        """
        records = parse_fixed_records("AAAA+001050\nBBBB-000325\n", LEDGER_LAYOUT)
        self.assertEqual(len(records), 2)
        with self.assertRaises(MalformedRecordError):
            parse_fixed_records("AAAA+001050\n\nBBBB-000325\n", LEDGER_LAYOUT)


class SignedDecimalOracleTests(unittest.TestCase):
    """Exact ``Decimal`` accumulation -- no tolerance anywhere in this class."""

    def test_signed_total_is_exact_decimal_arithmetic(self) -> None:
        """Count, total, and an independently written sum all agree.

        The last assertion recomputes the total from literals rather than through the
        ledger, so an oracle that mis-adds in a self-consistent way still gets caught.
        Binary floats would need a tolerance here; the oracle deliberately offers none,
        which is the property the compiled-program comparisons depend on.
        """
        summary = ReferenceLedger(LEDGER_LAYOUT).summarize(LEDGER_INPUT)
        self.assertEqual(summary.count, 3)
        self.assertEqual(summary.total, Decimal("7.30"))
        # Binary floats would need a tolerance here; the oracle deliberately does not.
        self.assertEqual(Decimal("10.50") - Decimal("3.25") + Decimal("0.05"), summary.total)

    def test_negative_totals_keep_their_sign(self) -> None:
        """A negative total stays negative and is edited into 7 integer digits, not 6.

        ``-13.75`` renders as ``-0000013.75``: the sign is separate from the digit
        width, so the magnitude is padded to ``int_digits`` regardless of sign. A
        renderer that padded the signed string would emit one digit too many and
        misalign every comparison against real COBOL output.
        """
        summary = ReferenceLedger(LEDGER_LAYOUT).summarize("AAAA-001050\nBBBB-000325\n")
        self.assertEqual(summary.total, Decimal("-13.75"))
        self.assertEqual(_ledger_expected("AAAA-001050\nBBBB-000325\n"), "COUNT=002\nTOTAL=-0000013.75\n")

    def test_total_overflow_is_detected_rather_than_wrapped(self) -> None:
        """Eleven 9999.99 records exceed ``S9(5)V99`` and must raise, not wrap.

        Overflow is enforced per accumulation step, so the error names the running
        total that broke the PICTURE rather than the final formatted string. The
        message quotes both the value and the limit, which is what tells a reader
        whether the layout is wrong or the input is.
        """
        text = "AAAA+999999\n" * 11  # 11 x 9999.99 = 109,999.89 > S9(5)V99 limit
        with self.assertRaises(PictureOverflowError) as ctx:
            ReferenceLedger(LEDGER_LAYOUT, total_int_digits=5).summarize(text)
        self.assertIn("109999.89 does not fit PIC S9(5)V9(2)", str(ctx.exception))

    def test_edited_format_matches_cobol_sign_leading_separate(self) -> None:
        """``SIGN IS LEADING SEPARATE`` always emits a sign, including ``+`` and zero.

        Zero renders ``+00000.00``, not ``00000.00``: COBOL prints a positive sign, so
        an oracle that omitted it would differ on every passing case and force the
        tests to compare against a shape the compiler never produces.
        """
        self.assertEqual(format_edited(Decimal("7.30"), int_digits=7, scale=2), "+0000007.30")
        self.assertEqual(format_edited(Decimal("-13.75"), int_digits=7, scale=2), "-0000013.75")
        self.assertEqual(format_edited(Decimal("0"), int_digits=5, scale=2), "+00000.00")
        self.assertEqual(format_edited(Decimal("900.25"), int_digits=5, scale=2), "+00900.25")


class RoundingOracleTests(unittest.TestCase):
    """``ROUNDED`` vs. implicit truncation, the two modes COBOL arithmetic can take."""

    def test_rounding_is_half_away_from_zero_for_both_signs(self) -> None:
        """Ties round away from zero symmetrically; ``2.675`` is the float trap.

        ``Decimal("2.675")`` has no binary representation error, so it must go to
        ``2.68``. The same literal as a float would round to ``2.67`` and quietly prove
        the wrong rounding mode.
        """
        self.assertEqual(cobol_round(Decimal("0.005"), 2), Decimal("0.01"))
        self.assertEqual(cobol_round(Decimal("-0.005"), 2), Decimal("-0.01"))
        self.assertEqual(cobol_round(Decimal("2.675"), 2), Decimal("2.68"))

    def test_truncation_rounds_toward_zero(self) -> None:
        """Truncation drops the remainder toward zero and keeps a signed zero.

        ``-0.005`` truncates to ``Decimal("-0.00")``, whose ``str`` is ``"-0.00"``.
        Asserting on the string pins that the negative zero survives quantisation: a
        formatter that normalised it to ``0.00`` would flip the sign character.
        """
        self.assertEqual(cobol_truncate(Decimal("0.005"), 2), Decimal("0.00"))
        self.assertEqual(cobol_truncate(Decimal("-0.005"), 2), Decimal("-0.00"))
        self.assertEqual(str(cobol_truncate(Decimal("-2.679"), 2)), "-2.67")

    def test_rounding_and_truncation_differ_exactly_at_the_half(self) -> None:
        """At a half the two modes diverge; away from it they agree.

        This is the discriminating case for the pair: a stub that implemented
        truncation for both functions would pass every other assertion in the class
        and fail only here, which is why the ``assertNotEqual`` is explicit.
        """
        value = Decimal("0.01") / Decimal(2)
        self.assertEqual(cobol_round(value, 2), Decimal("0.01"))
        self.assertEqual(cobol_truncate(value, 2), Decimal("0.00"))
        self.assertNotEqual(cobol_round(value, 2), cobol_truncate(value, 2))


class PictureOverflowOracleTests(unittest.TestCase):
    """``fits_picture`` must agree with ``enforce_picture`` in both directions."""

    def test_enforce_picture_rejects_the_silent_wrap_case(self) -> None:
        """100000 needs six digits and does not fit five; the bounds are inclusive.

        ``99999`` and ``-9999999.99`` are accepted (the limit itself is representable)
        while ``10000000.00`` is not, so the predicate is tested on the exact edges.
        The unsigned case rejects ``-0.01`` although its magnitude fits: sign validity
        is a separate rule from range, and only a negative near-zero can show it.
        """
        with self.assertRaises(PictureOverflowError):
            enforce_picture(Decimal(100000), int_digits=5, scale=0)
        self.assertFalse(fits_picture(Decimal(100000), int_digits=5, scale=0))
        self.assertTrue(fits_picture(Decimal(99999), int_digits=5, scale=0))
        self.assertTrue(fits_picture(Decimal("-9999999.99"), int_digits=7, scale=2))
        self.assertFalse(fits_picture(Decimal("10000000.00"), int_digits=7, scale=2))
        self.assertFalse(fits_picture(Decimal("-0.01"), int_digits=7, scale=2, signed=False))


class CobolDiagnosticParsingTests(unittest.TestCase):
    """Text-level parsing of cobc output; runs without a compiler installed."""

    RAW_COBOL = (
        "broken.cob:8: error: syntax error, unexpected /\n"
        "    6 | 01 WS-T PIC 9(4)V99 VALUE 0.\n"
        "    7 | PROCEDURE DIVISION.\n"
        "    8 >     COMPUTE WS-T = WS-N / / 2\n"
        "broken.cob:8: note: unknown statement 'FROBNICATE'\n"
    )

    def test_diagnostics_carry_file_line_and_severity(self) -> None:
        """Severity, path, and line survive parsing; echoed source lines do not.

        ``cobc`` interleaves ``6 |``/``8 >`` context lines with the diagnostic. Only
        lines that start with a ``.cob/.cbl/.cpy`` path are kept, so the note following
        the error is retained (order preserved) while the context is dropped.
        ``render()`` is asserted to round-trip exactly, because the reporter pastes it
        verbatim into the failure excerpt.
        """
        diagnostics = parse_diagnostics(self.RAW_COBOL)
        self.assertEqual([d.severity for d in diagnostics], ["error", "note"])
        self.assertEqual(diagnostics[0].file, "broken.cob")
        self.assertEqual(diagnostics[0].line, 8)
        self.assertIn("syntax error", diagnostics[0].message)
        self.assertEqual(
            diagnostics[0].render(), "broken.cob:8: error: syntax error, unexpected /"
        )

    def test_parse_cobol_attributes_compiler_errors_without_harness_markers(self) -> None:
        """With no ``TB-`` markers at all, one error line still yields exactly one failure.

        This is the fallback the adapter relies on when ``cobc`` dies before a suite can
        print anything: the failure is synthesised from the diagnostic, names the file
        and line, and the summary line says "cobc compile failed". Counts are pinned to
        1/1 so the fallback cannot double-count the error and the echoed context.
        """
        outcome = parse_cobol(self.RAW_COBOL)
        self.assertTrue(outcome.structured)
        self.assertEqual(outcome.failed, 1)
        self.assertEqual(outcome.total, 1)
        self.assertEqual(len(outcome.failures), 1)
        self.assertIn("syntax error", outcome.failures[0]["message"])
        self.assertIn("broken.cob:8", outcome.failures[0]["case_name"])
        self.assertIn("cobc compile failed", outcome.summary_line)

    def test_harness_markers_stay_authoritative_over_compiler_text(self) -> None:
        """Markers win over a stray compiler-shaped line in the same stream.

        The stream has one PASS marker, one FAIL marker, and an ``error:`` line nested
        under the FAIL block. Totals must be 2 (one case each), not 3: the error line is
        part of the failing case's excerpt, and treating it as a third case would let a
        compiler message inflate the failing count.
        """
        text = "TB-PASS cobol:ok.cob\nTB-FAIL cobol:bad.cob compile\n  bad.cob:3: error: syntax error\n"
        outcome = parse_cobol(text)
        self.assertEqual((outcome.total, outcome.passed, outcome.failed), (2, 1, 1))


# --------------------------------------------------------------------------
# Real compilation and execution
# --------------------------------------------------------------------------


class CobolVerificationCaseTests(unittest.TestCase):
    """Compile-and-run checks against the oracle, driven through ``verify_cases`` directly.

    Each case compiles in its own subdirectory of the test's scratch root, so the same
    source text used by several cases never shares a binary.
    """

    @requires_cobc
    def test_correct_ledger_program_passes_with_exact_output(self) -> None:
        """Positive control: a correct program passes and the expected text is pinned.

        The oracle output is compared to a literal before it is handed to
        ``verify_cases``, so a drift in ``format_ledger_summary`` fails here rather than
        showing up later as a mysterious program failure. ``TB-PASS cobol:ledger-ok``
        is asserted in the report text because that marker, not the counter, is what the
        parser reads back.
        """
        root = _tmpdir(case=self)
        _write(root, "ledger_ok.cob", LEDGER_OK)
        expected = _ledger_expected(LEDGER_INPUT)
        self.assertEqual(expected, "COUNT=003\nTOTAL=+0000007.30\n")

        report = verify_cases(
            [
                CobolCase(
                    name="ledger-ok",
                    source=root / "ledger_ok.cob",
                    expected_stdout=expected,
                    stdin_text=LEDGER_INPUT,
                )
            ],
            root / "build",
        )
        self.assertTrue(report.ok, report.text())
        self.assertEqual((report.passed, report.failed), (1, 0))
        self.assertIn("TB-PASS cobol:ledger-ok", report.text())

    @requires_cobc
    def test_exit_zero_with_wrong_result_fails_verification(self) -> None:
        """Compiles, exits 0, wrong total: must FAIL with an actionable diff.

        The failure cause is established first and separately: the wrong program is
        compiled, run directly, asserted to exit 0 and to print ``+0000013.80``. Only
        then is verification applied, so the FAIL cannot be an artifact of a crash or a
        build error. The diff assertions pin the *shape* of the report -- the differing
        line number, both values quoted, and both stdout blocks -- because "output
        differed" without the two sides would not be actionable.
        """
        root = _tmpdir(case=self)
        _write(root, "ledger_wrong.cob", LEDGER_WRONG)
        expected = _ledger_expected(LEDGER_INPUT)
        build = root / "build"

        compiled = compile_source(root / "ledger_wrong.cob", build)
        self.assertTrue(compiled.ok, compiled.output)
        direct = run_program(compiled.binary, stdin_text=LEDGER_INPUT)
        self.assertEqual(direct.exit_code, 0, direct.stderr)
        self.assertIn("TOTAL=+0000013.80", direct.stdout)

        report = verify_cases(
            [
                CobolCase(
                    name="ledger-wrong",
                    source=root / "ledger_wrong.cob",
                    expected_stdout=expected,
                    stdin_text=LEDGER_INPUT,
                )
            ],
            build / "verify",
        )
        self.assertFalse(report.ok)
        outcome = report.cases[0]
        self.assertFalse(outcome.ok)
        self.assertIn("TB-FAIL cobol:ledger-wrong output", outcome.lines)
        self.assertIn("first difference at output line 2", outcome.reason)
        self.assertIn("'TOTAL=+0000007.30'", outcome.reason)
        self.assertIn("'TOTAL=+0000013.80'", outcome.reason)
        text = "\n".join(outcome.lines)
        self.assertIn("expected stdout", text)
        self.assertIn("actual stdout", text)
        self.assertIn("0 passed, 0 skipped, 1 failed", report.text())

    @requires_cobc
    def test_compile_failure_reports_file_line_and_message(self) -> None:
        """A build failure is reported as ``compile``, never as an output mismatch.

        The aspect token in the marker is the contract the parser folds on, so
        ``TB-FAIL cobol:broken compile`` is asserted literally. The diagnostic is checked
        in both ``reason`` and the rendered lines: a truncation that keeps only the
        marker block must still leave a greppable ``broken.cob:8: error:``.
        """
        root = _tmpdir(case=self)
        _write(root, "broken.cob", COMPILE_ERROR)

        report = verify_cases(
            [CobolCase(name="broken", source=root / "broken.cob")], root / "build"
        )
        self.assertFalse(report.ok)
        outcome = report.cases[0]
        self.assertIn("TB-FAIL cobol:broken compile", outcome.lines)
        self.assertIn("broken.cob:8: error:", outcome.reason)
        self.assertIn("syntax error", outcome.reason)
        self.assertIn("broken.cob:8: error:", "\n".join(outcome.lines))

    @requires_cobc
    def test_nonzero_runtime_exit_fails_verification(self) -> None:
        """A nonzero exit fails even when no ``expected_stdout`` was supplied.

        With ``expected_stdout`` unset there is no output check to fall back on, so this
        isolates the exit-status rule. The aspect is ``rc=3`` (the observed code, not the
        expected one) and the reason reads ``exit 3, expected 0`` -- both are asserted
        because they are two different consumers of the same field.
        """
        root = _tmpdir(case=self)
        _write(
            root,
            "refuse.cob",
            "IDENTIFICATION DIVISION.\n"
            "PROGRAM-ID. refuse.\n"
            "PROCEDURE DIVISION.\n"
            '    DISPLAY "REFUSED"\n'
            "    STOP RUN RETURNING 3.\n",
        )
        report = verify_cases(
            [CobolCase(name="refuse", source=root / "refuse.cob")], root / "build"
        )
        self.assertFalse(report.ok)
        self.assertIn("TB-FAIL cobol:refuse rc=3", report.cases[0].lines)
        self.assertEqual(report.cases[0].reason, "exit 3, expected 0")
        self.assertIn("expected exit 0, got 3", "\n".join(report.cases[0].lines))

    @requires_cobc
    def test_rounding_program_passes_and_truncating_variant_fails(self) -> None:
        """One source, one keyword apart: ``ROUNDED`` passes, truncation fails.

        Both cases are handed the *same* expected text, so the only variable is the
        compiled semantics. If both passed, the oracle would not be discriminating
        rounding at all; if both failed, the expected text would be wrong. The split
        1/1 pins the middle. ``first difference at output line 1`` matters because the
        divergence is on ``POS``, not on the later ``NEG`` line.
        """
        root = _tmpdir(case=self)
        _write(root, "round_ok.cob", ROUNDING_OK)
        _write(root, "round_trunc.cob", ROUNDING_TRUNC)
        expected = (
            f"POS={format_edited(cobol_round(Decimal('0.005'), 2), int_digits=3, scale=2)}\n"
            f"NEG={format_edited(cobol_round(Decimal('-0.005'), 2), int_digits=3, scale=2)}\n"
        )
        self.assertEqual(expected, "POS=+000.01\nNEG=-000.01\n")

        report = verify_cases(
            [
                CobolCase(name="round-ok", source=root / "round_ok.cob", expected_stdout=expected),
                CobolCase(
                    name="round-trunc", source=root / "round_trunc.cob", expected_stdout=expected
                ),
            ],
            root / "build",
        )
        self.assertEqual((report.passed, report.failed), (1, 1))
        by_name = {c.name: c for c in report.cases}
        self.assertTrue(by_name["round-ok"].ok, by_name["round-ok"].lines)
        self.assertFalse(by_name["round-trunc"].ok)
        self.assertIn("first difference at output line 1", by_name["round-trunc"].reason)
        self.assertIn("POS=+000.01", by_name["round-trunc"].reason)
        self.assertIn("POS=+000.00", by_name["round-trunc"].reason)

    @requires_cobc
    def test_overflow_is_caught_when_wrapped_and_accepted_when_guarded(self) -> None:
        """Silent wrap is a failure; an ``ON SIZE ERROR`` handler is the accepted answer.

        Both programs are given ``expected_error="OVERFLOW"``, so the check is "nonzero
        exit *and* the rejection text". The wrap variant exits 0 and prints
        ``SMALL=00000`` -- the classic COBOL hazard -- and the evidence text is asserted
        to say so explicitly ("exited 0 and accepted it"). The guard variant exits 3
        with the message and passes, which proves the expected_error path is not simply
        rejecting everything.
        """
        root = _tmpdir(case=self)
        _write(root, "overflow_wrap.cob", OVERFLOW_WRAP)
        _write(root, "overflow_guard.cob", OVERFLOW_GUARD)
        with self.assertRaises(PictureOverflowError):
            enforce_picture(Decimal(99999) + 1, int_digits=5, scale=0)

        report = verify_cases(
            [
                CobolCase(name="overflow-wrap", source=root / "overflow_wrap.cob", expected_error="OVERFLOW"),
                CobolCase(
                    name="overflow-guard",
                    source=root / "overflow_guard.cob",
                    expected_error="OVERFLOW",
                ),
            ],
            root / "build",
        )
        by_name = {c.name: c for c in report.cases}
        self.assertTrue(by_name["overflow-guard"].ok, by_name["overflow-guard"].lines)
        self.assertFalse(by_name["overflow-wrap"].ok)
        wrap_text = "\n".join(by_name["overflow-wrap"].lines)
        self.assertIn("but the program exited 0 and accepted it", wrap_text)
        self.assertIn("expected a nonzero exit reporting 'OVERFLOW', got exit 0", wrap_text)
        self.assertIn("SMALL=00000", wrap_text)

    @requires_cobc
    def test_malformed_record_is_rejected_by_both_oracle_and_program(self) -> None:
        """Oracle and guard program agree that a bad sign is invalid; leniency is caught.

        The same source and the same malformed input are used twice. With
        ``expected_error`` the guard's rejection passes; with ``expected_stdout`` the
        guard's ``rc=2`` must surface as a failure, so a program that accepted the record
        could not pass both. Asserting ``rc=2`` in the evidence pins that the exit code
        is what got reported, not a coincidental output diff.
        """
        root = _tmpdir(case=self)
        _write(root, "ledger_guard.cob", LEDGER_GUARD)
        malformed = "AAAAX001050\n"
        with self.assertRaises(MalformedRecordError):
            ReferenceLedger(LEDGER_LAYOUT).summarize(malformed)

        report = verify_cases(
            [
                CobolCase(
                    name="malformed-guarded",
                    source=root / "ledger_guard.cob",
                    stdin_text=malformed,
                    expected_error="MALFORMED: bad sign",
                ),
                CobolCase(
                    name="malformed-unexpected",
                    source=root / "ledger_guard.cob",
                    stdin_text=malformed,
                    expected_stdout="COUNT=000\nTOTAL=+0000000.00\n",
                ),
            ],
            root / "build",
        )
        by_name = {c.name: c for c in report.cases}
        self.assertTrue(by_name["malformed-guarded"].ok, by_name["malformed-guarded"].lines)
        self.assertFalse(by_name["malformed-unexpected"].ok)
        self.assertIn("rc=2", "\n".join(by_name["malformed-unexpected"].lines))

    @requires_cobc
    def test_aggregate_report_totals_are_exact(self) -> None:
        """3 pass / 2 fail, and the printed text parses back to the same counts.

        The round-trip through ``parse_cobol`` is the point: the summary line and the
        markers are what the runner consumes, so a report whose properties were right
        while its text said otherwise would be invisible to the reporter. Three correct
        and two incorrect cases share one source file each, which also checks that
        outcomes are keyed by case, not by source path.
        """
        root = _tmpdir(case=self)
        _write(root, "ledger_ok.cob", LEDGER_OK)
        _write(root, "ledger_wrong.cob", LEDGER_WRONG)
        expected = _ledger_expected(LEDGER_INPUT)
        cases = [
            CobolCase(name="ok-1", source=root / "ledger_ok.cob", expected_stdout=expected, stdin_text=LEDGER_INPUT),
            CobolCase(name="ok-2", source=root / "ledger_ok.cob", expected_stdout=expected, stdin_text=LEDGER_INPUT),
            CobolCase(name="ok-3", source=root / "ledger_ok.cob", expected_stdout=expected, stdin_text=LEDGER_INPUT),
            CobolCase(name="bad-1", source=root / "ledger_wrong.cob", expected_stdout=expected, stdin_text=LEDGER_INPUT),
            CobolCase(name="bad-2", source=root / "ledger_wrong.cob", expected_stdout=expected, stdin_text=LEDGER_INPUT),
        ]
        report = verify_cases(cases, root / "build")
        self.assertEqual((report.passed, report.failed), (3, 2))
        self.assertIn("3 passed, 0 skipped, 2 failed", report.text())

        parsed = parse_cobol(report.text())
        self.assertEqual((parsed.total, parsed.passed, parsed.failed), (5, 3, 2))
        self.assertEqual(parsed.summary_line, "3 passed, 0 skipped, 2 failed")


class CobolReportingIntegrationTests(unittest.TestCase):
    """The verification helper reports through the existing cobol adapter.

    Unlike ``CobolVerificationCaseTests`` this goes through ``execute_suite`` with a real
    ``SuiteSpec``: the module is spawned as ``python -m
    hermes_testbench.adapters.cobol_check --spec ...``, so what is under test is the
    CLI contract (exit code + printed markers) and the adapter's parsing of it, not the
    in-process API. ``required_binaries=["cobc"]`` means the runner's own preflight --
    not the test's skip decorator -- would also refuse the suite on a host without the
    compiler.
    """

    def setUp(self) -> None:
        self.evidence = _tmpdir("tb-cobol-ev-", case=self)

    def _run_spec(self, spec: Path, project_root: Path):
        """Run the spec file through the module CLI; returns ``(SuiteResult, usage)``.

        ``project_root`` is ``REPO_ROOT`` because the subprocess imports the package from
        the checkout, while the spec's relative ``source`` paths resolve against the
        spec file itself.
        """
        suite = _suite(
            command=[
                sys.executable,
                "-m",
                "hermes_testbench.adapters.cobol_check",
                "--spec",
                str(spec),
            ],
            required_binaries=["cobc"],
        )
        return _exec_suite(suite, project_root, self.evidence)

    @requires_cobc
    def test_passing_case_reports_one_pass_through_execute_suite(self) -> None:
        """End-to-end pass: one case, no failures, and no usage record.

        ``usage`` is asserted ``None`` -- the cobol adapter reports no token accounting,
        and a fabricated usage object here would be indistinguishable from a real one
        downstream. The exact 1/1/0 triple catches a parser that counts the summary line
        as an extra case.
        """
        root = _tmpdir(case=self)
        _write(root, "ledger_ok.cob", LEDGER_OK)
        spec = _spec(
            root,
            [
                {
                    "name": "ledger-ok",
                    "source": "ledger_ok.cob",
                    "stdin": LEDGER_INPUT,
                    "expected_stdout": _ledger_expected(LEDGER_INPUT),
                }
            ],
        )
        result, usage = self._run_spec(spec, REPO_ROOT)
        self.assertIsNone(usage)
        self.assertEqual(result.status, RunStatus.PASS)
        self.assertEqual((result.total_count, result.pass_count, result.fail_count), (1, 1, 0))
        self.assertEqual(result.failures, [])

    @requires_cobc
    def test_wrong_result_with_zero_exit_fails_through_execute_suite(self) -> None:
        """The zero-exit/wrong-answer control, observed through the adapter's fields.

        ``failure.message`` carries the first-difference line and ``failure.excerpt``
        carries the two stdout blocks; the assertions are split that way on purpose,
        because a change that folded the excerpt into the message (or dropped it) would
        still leave a single green check if only one field were inspected.
        """
        root = _tmpdir(case=self)
        _write(root, "ledger_wrong.cob", LEDGER_WRONG)
        spec = _spec(
            root,
            [
                {
                    "name": "ledger-wrong",
                    "source": "ledger_wrong.cob",
                    "stdin": LEDGER_INPUT,
                    "expected_stdout": _ledger_expected(LEDGER_INPUT),
                }
            ],
        )
        result, _usage = self._run_spec(spec, REPO_ROOT)
        self.assertEqual(result.status, RunStatus.FAIL)
        self.assertEqual(result.fail_count, 1)
        failure = result.failures[0]
        self.assertIn("first difference at output line 2", failure.message)
        self.assertIn("TOTAL=+0000007.30", failure.excerpt)
        self.assertIn("TOTAL=+0000013.80", failure.excerpt)

    @requires_cobc
    def test_compile_failure_reaches_failure_excerpt_with_file_and_line(self) -> None:
        """Compiler diagnostics survive the CLI -> parser -> ``FailureRecord`` path.

        The failure is a build error, so the case name ends in ``compile`` and the
        diagnostic must appear in both ``message`` and ``excerpt``: the excerpt is the
        only part a human reads when the one-line message is not enough.
        """
        root = _tmpdir(case=self)
        _write(root, "broken.cob", COMPILE_ERROR)
        spec = _spec(root, [{"name": "broken", "source": "broken.cob"}])
        result, _usage = self._run_spec(spec, REPO_ROOT)
        self.assertEqual(result.status, RunStatus.FAIL)
        self.assertEqual(result.fail_count, 1)
        failure = result.failures[0]
        self.assertIn("broken.cob:8: error:", failure.message)
        self.assertIn("syntax error", failure.message)
        self.assertIn("broken.cob:8: error:", failure.excerpt)

    @requires_cobc
    def test_program_stdout_with_fail_marker_does_not_create_phantom_cases(self) -> None:
        """Program output cannot invent cases: only ``TB-`` markers are counted.

        The program prints ``FAIL: customer 42 rejected`` and a fake
        ``99 passed, 0 skipped, 0 failed`` summary. A parser that scanned output text
        for ``FAIL`` or trusted the last summary line would report 99 cases or a
        different failure count. Totals must stay 1/0/1 with one failure, and the
        offending line must appear only as that failure's excerpt.
        """
        root = _tmpdir(case=self)
        _write(
            root,
            "marker_stdout.cob",
            "IDENTIFICATION DIVISION.\n"
            "PROGRAM-ID. markerstdout.\n"
            "PROCEDURE DIVISION.\n"
            '    DISPLAY "FAIL: customer 42 rejected"\n'
            '    DISPLAY "99 passed, 0 skipped, 0 failed"\n'
            "    STOP RUN.\n",
        )
        spec = _spec(
            root,
            [
                {
                    "name": "marker-stdout",
                    "source": "marker_stdout.cob",
                    "expected_stdout": "OK: customer 42 accepted\n",
                }
            ],
        )
        result, _usage = self._run_spec(spec, REPO_ROOT)
        self.assertEqual(result.status, RunStatus.FAIL)
        self.assertEqual((result.total_count, result.pass_count, result.fail_count), (1, 0, 1))
        self.assertEqual(len(result.failures), 1)
        self.assertIn("first difference at output line 1", result.failures[0].message)
        self.assertIn("FAIL: customer 42 rejected", result.failures[0].excerpt)


class CobolDiscoverySuiteTests(unittest.TestCase):
    """Discovery's cobol-unit suite: expected-output sidecar and diagnostics.

    These cases do not use ``cobol_check``'s CLI at all. ``discover_project`` generates
    a ``/bin/sh`` suite from the ``.cob`` files it finds, which compiles with
    ``cobc -O2 -debug -x`` (no ``-free``), runs with stdin at ``/dev/null``, and compares
    stdout byte-for-byte against ``<source>.expected`` when that sidecar exists. So the
    fixtures here are deliberately written in column-6 fixed format, and the expected
    text is a plain file rather than a spec entry.
    """

    def setUp(self) -> None:
        self.evidence = _tmpdir("tb-cobol-disc-", case=self)

    def _run_discovery(self, root: Path):
        """Discover ``root``, assert exactly one ``cobol-unit`` suite, and run it.

        The ``len(suites) == 1`` check is itself an assertion: discovery emits the
        cobol-tui-harness suite instead when ``bench/tui/cobol/test_cobol.py`` is
        present, and a silent fallback to the wrong suite would make the rest of the
        test meaningless.
        """
        spec = discover_project(root)
        suites = [s for s in spec.suites if s.name == "cobol-unit"]
        self.assertEqual(len(suites), 1, [s.name for s in spec.suites])
        return _exec_suite(suites[0], root, self.evidence)

    @requires_cobc
    def test_matching_expected_output_sidecar_passes(self) -> None:
        """A correct fixed-format program passes against its ``.expected`` sidecar.

        ``FIXED_FORMAT_SUM`` uses ``PIC S9(5)V99`` and prints ``+00900.25``, so this also
        pins the fixed-format assumption: the generated ``cobc`` invocation has no
        ``-free``, and a free-form fixture would fail to compile here rather than pass.
        """
        root = _tmpdir(case=self)
        _write(root, "fixed_sum.cob", FIXED_FORMAT_SUM)
        _write(root, "fixed_sum.cob.expected", "TOTAL=+00900.25\n")
        result, _usage = self._run_discovery(root)
        self.assertEqual(result.status, RunStatus.PASS)
        self.assertEqual((result.total_count, result.pass_count, result.fail_count), (1, 1, 0))

    @requires_cobc
    def test_disagreeing_sidecar_fails_with_diff_evidence(self) -> None:
        """A one-cent disagreement fails and the evidence is a unified diff, not a summary.

        The sidecar says ``+00900.50``; the program prints ``+00900.25``. ``@@`` is
        asserted in the excerpt because the shell script emits ``diff -u`` hunks, and
        both values are checked (actual in the excerpt, expected in the message) so a
        truncated hunk still carries enough to act on. The case name must carry
        ``output`` to distinguish it from a build failure.
        """
        root = _tmpdir(case=self)
        _write(root, "fixed_sum.cob", FIXED_FORMAT_SUM)
        _write(root, "fixed_sum.cob.expected", "TOTAL=+00900.50\n")
        result, _usage = self._run_discovery(root)
        self.assertEqual(result.status, RunStatus.FAIL)
        self.assertEqual(result.fail_count, 1)
        failure = result.failures[0]
        self.assertIn("output", failure.case_name)
        self.assertIn("TOTAL=+00900.25", failure.excerpt)
        self.assertIn("@@", failure.excerpt)
        self.assertIn("output differs from", failure.message)
        self.assertIn("TOTAL=+00900.50", failure.message)

    @requires_cobc
    def test_compile_failure_diagnostic_reaches_the_failure_excerpt(self) -> None:
        """``FROBNICATE`` is not a COBOL verb; the resulting error must reach the excerpt.

        This is the discovery path's build-failure control: the source is syntactically
        fixed-format but has no valid statement, so the only possible outcome is a
        compile error. Asserting ``compile`` in the case name keeps it distinct from an
        output mismatch produced by an unexpected binary.
        """
        root = _tmpdir(case=self)
        _write(root, "fixed_broken.cob", "       IDENTIFICATION DIVISION.\n       FROBNICATE.\n")
        result, _usage = self._run_discovery(root)
        self.assertEqual(result.status, RunStatus.FAIL)
        self.assertEqual(result.fail_count, 1)
        self.assertIn("compile", result.failures[0].case_name)
        self.assertIn("error:", result.failures[0].message)
        self.assertIn("error:", result.failures[0].excerpt)


if __name__ == "__main__":
    unittest.main()
