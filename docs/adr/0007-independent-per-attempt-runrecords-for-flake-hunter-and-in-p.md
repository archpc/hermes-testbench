# ADR-0007: Independent per-attempt RunRecords for Flake Hunter and in-process VTScreen Terminal Tape deferring VHS

- Status: accepted
- Date: 2026-11-16
- Deciders: Sein, Hermes Agent
- Supersedes: none
- Superseded by: none

## Context and problem

Phase 2 needed two capabilities that both tempt a shortcut through existing rows and existing tooling:

1. **Flake detection.** A flaky target is only visible across repeated executions, so something must run the same target N times and keep every outcome. The cheapest implementation is to reuse one `run_id` and update it per attempt; `store._upsert_run_row` is `ON CONFLICT(run_id) DO UPDATE SET status=…, summary=…, payload_json=…`, so that shortcut silently erases the failing attempt from `runs`, `run.json`, and `junit.xml` — the exact evidence the feature exists to preserve. It would also force a schema change to express attempt identity.
2. **Terminal regression.** The obvious backend is VHS/Chromium; a screenshot or video of a TUI. That pulls GUI binaries, a host keychain, and screen-capture access into a suite that is supposed to run offline under Seatbelt.

## Constraints and requirements

- `SCHEMA_VERSION` must stay `1` and the `store.py` DDL must not change (ADR-0001/0004 storage contract). No migration.
- Every attempt must remain independently inspectable: its own `run_id`, its own `runs/<run_id>/` evidence, its own JUnit record. A later passing retry may never touch an earlier failed attempt.
- Classification must be pure and reusable over already-persisted history (`flake analyze`), not only over a live run.
- Terminal assertions must be deterministic and offline: no browser, no recorder, no ambient tmux, no writes outside the scratch directory.
- A golden screen is a human-approved artifact; suites and background agents must be structurally unable to approve their own baselines.

## Alternatives considered

| Alternative | Why it was weighed | Why it lost |
|---|---|---|
| Reuse one `run_id` across attempts and update it (or keep a `flake_attempts` child table) | Cheapest live-run path; one row per group in the TUI | `_upsert_run_row`'s `ON CONFLICT … DO UPDATE` erases the earlier attempt's status, counts, `payload_json`, and (through `finalize_run`) its `run.json`/`junit.xml`. A child table means DDL plus a `SCHEMA_VERSION` bump, breaking the v1 store contract for a pure analysis feature. |
| Store flake verdicts in a new column or table | Makes `flake analyze` cheap | Same DDL/schema cost, and the verdict is derivable from attempt rows plus `environment['flake']` metadata — storing it would let store and classifier disagree. |
| VHS (`vhs` + `ttyd`/`ffmpeg`) or headless Chromium for terminal capture | Real video/GIF, handles any TUI, familiar tool | Under the current Seatbelt profile a host binary that needs Keychain/OSCrypt still reaches `secd`/`SecurityAgent` over Mach IPC even with `~/Library/Keychains` read-denied and `HOME` redirected: headless Chrome exits `rc=21` with no screenshot (`vhs_probe2.py`/`vhs_probe3.py`), and that investigation recorded a host-level "Keychain Not Found / Reset To Defaults" `SecurityAgent` modal. It also needs write access outside the sandbox and depends on optional host binaries. |
| Screenshot diffing via `screencapture`/GUI automation | Already on macOS | Captures the user's real desktop (privacy), is non-deterministic, and needs a window server the sandbox denies. |

## Decision

1. **One attempt = one independent `RunRecord`.** `TestBenchEngine.flake_hunt` (`engine.py`) runs N complete `run_sync` executions, each allocating a fresh `new_run_id()` and tagging `environment['flake'] = {group_id, attempt, repeats}`. No attempt is updated, reused, or overwritten. `hermes_testbench/flake.py` is a pure classifier over those records — `always_pass` / `always_fail` / `mixed` / `inconclusive`, at suite or structured-case granularity — and drives both the live `flake run` subcommand and `flake analyze` over history. `SCHEMA_VERSION` remains `1`; the `store.py` DDL is untouched.
2. **Terminal capture is an in-process VT interpreter over a real PTY.** `hermes_testbench/pty_session.py` owns one `pty.openpty()` pair with an isolated env and optional Seatbelt, and `VTScreen` reproduces the character grid (cursor motion, erase, alt-screen save/restore, wide cells, escape consumption) so assertions run against cells, not escape soup. `hermes_testbench/tape.py` parses declarative `.tape` scripts (`Size`, `Command`, `Type`, `Key`, `Resize`, `WaitFor`, `WaitQuiet`, `ExpectScreen`, `ExpectRaw`, `Golden`) and replays them through that session.
3. **VHS/Chromium backends are refused, not silently unsupported.** Any tape naming `Backend vhs`, `Vhs`, `Browser`, `Screenshot`, or `chrome|chromium|playwright` fails at parse time with `FailureKind.BLOCKED_POLICY` and the fixed message `VHS backend deferred: host-side-effect isolation not established` — nothing is spawned.
4. **Goldens are read-only and approval is a separate human gate.** `tape run` compares against `GOLDEN_DIR/<name>.screen.txt` plus a manifest (`screen_sha256`, `approved_sha256`, `tape_sha256`, cols/rows, `approved_at`); missing, unapproved, or stale goldens end `BLOCKED`, and a mismatch ends `FAIL`/`assertion` with `NAME.actual.screen.txt` + `NAME.golden.diff` written into evidence. `approve_golden` refuses when `TESTBENCH_SANDBOX=1` or stdin is not a TTY.

## Rationale

Keeping attempt identity at the run-row level gets exact preservation, the existing artifact set, and the existing TUI history for free, with no schema movement. Replacing VHS with an in-process VT screen keeps the terminal suite inside the same isolation boundary as every other suite and removes a class of host-side effects the sandbox demonstrably cannot contain on macOS.

## Tradeoffs and consequences

- Flake hunting multiplies runtime by N and creates N evidence directories; `flake run` is therefore explicit, never on by default, and history pruning must be aware of attempt groups.
- Attempts are linked only by `environment['flake'].group_id` — there is no foreign key, so a hand-deleted run row silently shrinks a group and its verdict degrades to `inconclusive` rather than erroring.
- `VTScreen` models the cell grid, not a terminal's full feature set: sixel/kitty graphics, hyperlink rendering, and scrollback beyond current geometry are out of scope, and golden comparisons are sensitive to normalization rules (`normalized_screen`).
- Terminal tapes produce no video artifact for PRs — exactly what VHS would have provided. That stays unavailable until a sandbox-compatible capture path exists.
- Golden approval remains a manual, TTY-only step; CI cannot self-approve a changed screen.

## Supporting evidence

- `hermes_testbench/flake.py` (verdict definitions, `flake_metadata`, `classify_statuses`, `analyze_flake_history`); `engine.py::flake_hunt` (per-attempt `run_sync` plus `environment['flake']` tagging).
- `hermes_testbench/store.py:323-358` `_upsert_run_row` — `ON CONFLICT(run_id) DO UPDATE SET status=…, payload_json=…` (the erasure hazard); `schema.py:14` `SCHEMA_VERSION = 1`.
- `hermes_testbench/pty_session.py`, `hermes_testbench/tape.py` (`VHS_DEFERRAL_MESSAGE`, `_assert_interactive_approval`, golden manifest checks), `docs/TERMINAL-TAPES.md`.
- VHS refusal: `tests/test_terminal_tape.py::TapeCliTests.test_cli_refuses_vhs_without_spawning_anything`.
- macOS sandbox evidence (recorded 2026-11-15, outside this repository): probe scripts `/Users/sein/.hermes/cache/scratch/vhs_probe.py` … `vhs_probe5.py`; subagent transcript `/Users/sein/.hermes/cache/delegation/live/deleg_e0c17295/task-1.log` (Chrome `--headless=new` under `build_seatbelt_profile()` with redirected `HOME`: `rc=21`, `png=False`, Crashpad `Operation not permitted`); the `SecurityAgent` "Keychain Not Found" modal was observed at 19:18:46 in that investigation's unified log.
- Phase 2 commits: `a7061fddcffad97265cd19bc268013f5bcc9e0e8` (flake), `123399ab840e611cc69059a3fc67b6acdf7e2e8f` (tape), merged onto `170c154`.

## Verification status

partially verified (flake + tape behavior: TestBench `unit` 188/188 and `terminal` 150/151 with 1 Seatbelt SKIP on 2026-11-16, including `tests.test_flake_hunter` and `tests.test_terminal_tape`; the Seatbelt/Chromium `SecurityAgent` modal is cited from the 2026-11-15 investigation record and was not re-executed here — no GUI binary was launched)

## Related decisions

- Preserves ADR-0001 (v1 WAL store contract) and ADR-0004 (schema/reporting separation); relies on ADR-0005 (deny-by-default Seatbelt, disposable tmux sockets) for the isolation both features inherit.
