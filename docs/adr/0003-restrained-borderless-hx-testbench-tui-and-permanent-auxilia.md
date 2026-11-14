# ADR-0003: Restrained borderless hx:testbench TUI and permanent auxiliary tmux window semantics

- Status: accepted
- Date: 2026-11-14
- Deciders: Sein, Carbon (lead architect)
- Supersedes: none
- Superseded by: none

## Context and problem

Ongoing delegated work and multi-suite test/benchmark runs need an interactive terminal surface inside `hermes-tmux` (`hx:testbench`, focused via `hermes-tmux testbench`) so a human or PTY test harness can monitor active/queued/historical runs, inspect sanitized failure logs and reproduction commands, compare benchmarks, and explicitly trigger or cancel TestBench-owned runs without creating a second source of truth or breaking `hermes-tmux`'s strict ownership and focus-preservation guarantees.

## Constraints and requirements

- Consume TestBench's authoritative `state.db` and `runs/<id>/` artifacts directly; never maintain a separate run store or busy-poll.
- Preserve `hermes-tmux` permanent auxiliary-window invariants (`~/Code/hermes-tmux-workspace/docs/adr/0001-*.md` and `tests/test_mail_aux.py`): stable `hx:testbench` name, detached creation on `up` (`focus=False`), no window renumbering, idempotent repeated `up`, detached crash recovery in `watch.py`, explicit focus on `hermes-tmux testbench`, per-pane `@hx` ownership check on `down`, compatibility with `hx:mail` and lazy `hx:tests`, and optional opt-out (`HERMES_TMUX_TESTBENCH=0`).
- Never auto-run tests on window open; never silently trigger expensive or billable LLM-backed suites; degrade cleanly when run history is empty or `testbench` is unavailable.
- Match `hxmail` (`crates/hxmail/src/render.rs`) visual grammar: borderless 4-band layout, 256-color indexed palette (`39, 114, 214, 203, 244, 240, 238`), `NO_COLOR=1` and ASCII glyph fallback, zero decorative boxes or spinners.

## Alternatives considered

| Alternative | Why it was weighed | Why it lost |
|---|---|---|
| Lazy-only `hx:tb` window created only on first `hermes-tmux testbench` | Suggested by Opus advisor to save ~15 status-bar columns when idle | Conflicts with explicit milestone requirement for permanent `hx:testbench` auxiliary-window startup, crash recovery, and identity; tab-bar budget concern is resolved via `HERMES_TMUX_TESTBENCH=0` opt-out and `tmuxio.new_window`'s non-renumbering `after_window` insertion (`tmuxio.py:353-364`) |
| Four separate numbered screens (`1..4`) with `Enter` confirming actions and `q` quitting at top level | Initial draft layout | Opus critique demonstrated three UX defects: (1) double-tapping `Enter` to inspect can accidentally confirm a run; (2) pressing `q` at top level in a permanent window triggers `watch.py` crash-restore loops; (3) 4 disconnected screens hide failing/running work behind mode switches |
| Curses / third-party Python TUI framework (`textual` / `urwid`) | Built-in widgets and layout | Violates zero-dependency stdlib constraint (`textual`/`urwid` absent in system Python), adds heavy terminal mouse/escape overhead, and diverges from `hxmail`/`hermes-tmux` visual restraint |

## Decision

1. **`hermes-tmux` lifecycle integration (`~/Code/hermes-tmux-workspace`)**:
   - Add `"testbench"` to `AUX_ROLES` and generalize permanent auxiliary window creation/recovery to `PERMANENT_AUX_ROLES` (`("mail", "testbench")`, gated on `HERMES_TMUX_TESTBENCH != "0"`).
   - `cmd_up` creates `hx:testbench` detached (`focus=False`) chained after `hx:mail` (`after_window=mail_wid or window_id`); `tmuxio.new_window` inserts into the next free index or appends without renumbering any user or worker window.
   - `watch.py` restores any missing permanent aux window detached (`focus=False`), and `ensure_aux_window` isolates `TESTBENCH_HOME=$HERMES_HOME/cache/testbench` whenever `HERMES_HOME` is non-default. Existing `hx:tests` (`hermes-tmux tests`) remains untouched.

2. **`testbench tui` design (`hermes_testbench/tui.py`)**:
   - **4-band borderless frame**: Row 0 left-right title (`TESTBENCH · <project|ALL>` + counts/store status), Rows `1..h-3` body, Row `h-2` contextual key hints, Row `h-1` minibuffer/confirmation line.
   - **Unified priority-ranked view + responsive detail**: Active (`RUNNING`/`QUEUED`) jobs first (elapsed vs baseline e.g. `4.2s / ~4.9s`, no spinners), followed by projects/recent runs ranked with failures/errors surfaced inline (fixed-width status words `RUN `, `QUEU`, `PASS`, `FAIL`, `SKIP`, `BLKD`, `TOUT`, `CNCL`, `ERR ` so `NO_COLOR=1` stays scannable). At `width >= 120`, renders a clean borderless two-column master/detail split; at `width < 120` (e.g. `80x24`), `Enter` opens a full-screen detail sheet (failure summary, scrollable sanitized log, benchmark comparison table, evidence paths, and reproduction command).
   - **Safe keybindings**: `j`/`k`/arrows move, `g`/`G` top/bottom, `Tab` switch focus/mode (`Overview` / `History` / `Projects` / `Benchmarks`), `p` project filter picker, `/` text filter, `Enter` inspect detail, `l` full log sheet, `y` copy repro command (`tmux set-buffer` when inside tmux), `r` select profile to run -> requires explicit `y` in minibuffer (never `Enter`), `R` rerun selected -> requires `y`, `x` cancel owned running job -> requires `y`, `?` help, `q`/`Esc` closes active sheet/prompt (at top level in permanent tmux mode, `q` flashes `"press Q to exit TUI, or switch tmux windows"` rather than exiting into a `watch.py` restore loop; `Q` or `--quit-on-q` exits cleanly).
   - **Refresh**: `select.kqueue` hint on `$TESTBENCH_HOME` + `PRAGMA data_version` check + 1 Hz tick only while a run is `RUNNING`/`QUEUED` + `SIGWINCH` immediate resize redraw.

## Rationale

Independent critique (opus, 2026-11-14, `delivery_id=90eba6086c27b546687b288d90123d27a21da877160df0b46da5f4147c6ab655`) directly improved four interaction details adopted here: separating `Enter` (inspect) from `y` (confirm execution/cancel), preventing top-level `q` from fighting `watch.py` crash recovery, replacing `state.db-wal`-only kqueue watching with `kqueue` + `PRAGMA data_version` so WAL checkpoint deletion cannot freeze the display, and using a priority-ranked view with responsive `>=120`-col split and fixed-width status words. Carbon retained the explicit `hx:testbench` permanent window semantics required by the specification, adding `HERMES_TMUX_TESTBENCH=0` opt-out.

## Tradeoffs and consequences

- **Easy**: Testable in-process (pure `render_screen(state, width, height)` function) AND over a real PTY/tmux pane (`pyte` + `tmux capture-pane`); zero external TUI dependencies; immune to accidental test execution or `q`-induced respawn churn.
- **Harder**: Adds one permanent tab (`hx:testbench`, 12 chars) alongside `agents` and `hx:mail` by default (~30 cols total of a 188-col bar, leaving ~158 cols for user + up to 8 worker tabs; opt-out via `HERMES_TMUX_TESTBENCH=0`).

## Supporting evidence

- `02-tmux-workspace.md` §2-5: `cli.py:324-326`, `cli.py:547-627`, `tmuxio.py:353-364`, `watch.py:1078-1092`, `tests/test_mail_aux.py`.
- `03-hxmail-hte.md` §4: `crates/hxmail/src/render.rs:16-179` palette and 4-band borderless frame.
- Opus 5.5 consultation receipt: `/Users/sein/.hermes/profiles/opus/runtime/bot_live_delivery/90eba6086c27b546687b288d90123d27a21da877160df0b46da5f4147c6ab655.json`.

## Verification status

verified (`/Users/sein/.local/bin/python3.14 -W default -m unittest tests.test_tui_render tests.test_tui_pty -v` — 44/44 passed; `testbench run wt-hermes-tmux --profile terminal` `tb_20261114_234118_923ce01c` — 27/27 tmux lifecycle tests passed, 2026-11-14)

## Related decisions

- ADR-0001: Core engine and WAL SQLite store (`PRAGMA data_version` and `state.db` schema).
- ADR-0002: Local stdio MCP server (shares `store.py` and detached `_worker` execution/cancellation).
- `~/Code/hermes-tmux-workspace/docs/adr/0001-read-only-side-car-over-hermes-artifacts-not-agent-hooks.md` (constrains `hermes-tmux` side-car ownership model).
