# ADR-0006: External LLVM FileCheck steps and GnuCOBOL Decimal oracle over custom output-matching DSLs

- Status: accepted
- Date: 2026-11-16
- Deciders: Sein, Hermes Agent
- Supersedes: none
- Superseded by: none

## Context and problem

Two Phase 1 verification gaps existed in the polyglot adapters: nothing could assert on a program's *text output* (only exit status was checked), and the `cobol` adapter trusted `cobc` to exit 0 without ever checking whether the program computed the right arithmetic. Both gaps invite the same bad fix — write a small in-repo pattern matcher and a hand-rolled COBOL output comparison — which would be a second, unreviewed implementation of two things that already have authoritative external definitions (LLVM `FileCheck` directives and COBOL fixed-point decimal semantics).

## Constraints and requirements

- Python 3.11+ stdlib only; no new runtime dependency may be added to `hermes_testbench` (`pyproject.toml` `dependencies = []`).
- Every step must run through the existing isolated execution path (`execute_suite` / `run_isolated_command`) so Seatbelt confinement, timeout, evidence capture, in-stream redaction, and exit-status accounting are not re-implemented.
- Optional host binaries must not turn a missing tool into a false PASS or FAIL: absence becomes the standard Gate 3 `RunStatus.SKIP` (`adapters/__init__.py:256-274`, `suite.required_binaries`).
- Failure records must be attributable: source file, line number, and an actionable excerpt, not "output differed".

## Alternatives considered

| Alternative | Why it was weighed | Why it lost |
|---|---|---|
| Ship a custom literal/substring matcher with a `.check`-suffixed sidecar heuristic | Zero external binary, trivially portable, no `FileCheck` discovery | It is a private assertion language with no specification, and the `.check` filename heuristic silently changed which suites were verified depending on a file suffix; the review commit `ce57afb` removed that heuristic entirely. Ordering, `CHECK-DAG` grouping, count bounds, and implicit-NOT regions would all have had to be reinvented and separately trusted. |
| One `FileCheck` invocation carrying every intent as a distinct `--check-prefix` group | One process, one artifact, fewer spawns | `FileCheck` evaluates multiple check-prefixes as a *single interleaved sequence*, and a trailing `CHECK-DAG` group also suppresses implicit-NOT regions. A forbidden-string assertion could therefore pass only because of where an unrelated group ended. Each intent (required / ordered / forbidden / counts) is now its own invocation. |
| Compare COBOL output with a normalized string equality helper | Simple, no oracle model | It cannot distinguish "the program printed the wrong number" from "the program printed it with different padding", and it inherits whatever `cobc` happens to emit. Worse, an exit-code-only check accepts a program whose arithmetic is wrong but which terminates cleanly — exactly the class of COBOL defect this phase targets. |
| Use `cobc` itself as the expected-value oracle (compile twice, diff) | No decimal model to maintain | A compiler is not an oracle for its own output: a COBOL semantic bug that reproduces consistently compiles to itself twice. The oracle must be independent of the tool under test. |

## Decision

1. **Output contracts are expressed as data and executed by LLVM `FileCheck`.** `hermes_testbench/adapters/filecheck.py` defines an immutable `OutputContract` (`required`, `ordered`, `forbidden`, `counts`) which `build_check_steps` lowers to `CHECK-DAG`, successive `CHECK`, `--implicit-check-not`, and `CHECK-COUNT-N` + `CHECK-NOT` respectively. Each intent is a separate `FileCheck` invocation inside one generated `/bin/sh` script, executed through the existing `shell` adapter. `parse_filecheck_diagnostics` is a pure parser (no I/O, no spawning) wired into `parsers.parse_shell`, so failures carry kind `assertion`, source, line, and excerpt. No custom matcher exists.
2. **COBOL expectations come from a stdlib `decimal` reference oracle.** `hermes_testbench/adapters/cobol_check.py` models fixed-width signed-decimal arithmetic — exact sums, `ROUNDED` (half away from zero), implicit-decimal truncation, and PICTURE overflow — computes expected stdout independently, then compiles each case with `cobc`, runs it, and compares exact stdout / exit status / expected reference rejection. It emits the `TB-PASS`/`TB-FAIL` marker lines that `parsers.parse_cobol` already attributes, so no second runner and no second reporter path exist.
3. **Absent tools are SKIP, not PASS.** `FileCheck` and `cobc` are optional host binaries declared as `required_binaries`; Gate 3 returns `RunStatus.SKIP` with a recorded reason when either is missing, so a machine without LLVM or GnuCOBOL reports unverified rather than green.

## Rationale

Both decisions push correctness onto an external, independently specified implementation and keep TestBench in the role it already occupies: isolation, capture, attribution, and reporting. The diff is bounded to two adapter modules plus parser wiring instead of a new assertion language and a second arithmetic model.

## Tradeoffs and consequences

- Suites are SKIP on hosts lacking `FileCheck` or `cobc`; Gate 3 SKIPs are visible in reports but do not prove the program. Provisioning LLVM/GnuCOBOL is a host responsibility.
- Four intents mean up to four `FileCheck` processes per contract, and the whole contract currently runs through a single generated `sh` step, so a mid-contract failure reports the first failing step rather than an aggregate.
- The COBOL oracle only models the arithmetic subset the suite declares; constructs outside it (indexed tables, `COMP-1` floats, `EVALUATE` control flow) are not verified by this layer.
- `FileCheck` discovery searches `PATH`, `$HOMEBREW_PREFIX`, and Homebrew prefixes; a user-installed binary outside those locations needs an explicit path.

## Supporting evidence

- `hermes_testbench/adapters/filecheck.py` (module docstring states the single-invocation interleaving limit and the no-custom-matcher rule; `build_check_steps` line ~224).
- `hermes_testbench/adapters/cobol_check.py` (oracle layers, "exit status alone proves nothing about arithmetic").
- `hermes_testbench/adapters/__init__.py:256-274` (Gate 3 SKIP on missing `required_binaries`).
- PR #4 merge commit `170c1546f949a0a0b33740916a13c3a0d3287a37`; review fixes `ce57afb` (drop `.check` heuristic, quote cobol detail lines) and `061fe21` (Gate-3 FileCheck skip).
- Tests: `tests/test_output_contract.py`, `tests/test_cobol_verification.py`, `tests/test_review_remediation_parsers.py`.

## Verification status

verified (TestBench `unit` profile on `feat/phase2-flake-and-tape`, 2026-11-16: 188/188 passed including `tests.test_output_contract` and `tests.test_cobol_verification`; Gate-3 SKIP path exercised by `tests/test_terminal_tape.py` seatbelt skip and `tui-pty-and-tmux` skipped=1)

## Related decisions

- Conforms to ADR-0001 (single stdlib core), ADR-0004 (in-stream redaction and strict schema/reporting separation), and ADR-0005 (deny-by-default Seatbelt isolation reused by every step).
