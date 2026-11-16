# Terminal tapes

A `.tape` file is a declarative script that drives a real program through a PTY
under TestBench isolation and asserts on the resulting screen. It is the local,
inspectable counterpart to the deferred VHS/browser backends: no browser, no
screen recorder, no host-side side effects.

```
Size 80 24
Command {python} -m hermes_testbench tui
WaitFor "TESTBENCH" 10000
WaitQuiet 200
ExpectScreen contains "overview"
ExpectRaw contains "\u001b[?1049h"
Type "q"
WaitFor "press Q" 5000
Key Q
```

## Directives

| Line | Meaning |
| --- | --- |
| `Size COLS ROWS` | Initial PTY geometry (default `80 24`). |
| `Command ...` | Program argv; `{python}` expands to the running interpreter. |
| `Type "..."` | Send text (JSON escapes honoured: `\n`, `\t`, `\u001b`). |
| `Key NAME` | Send one key: `Enter`, `Tab`, `Esc`, `Space`, `Backspace`, `Up/Down/Left/Right`, `Home`, `End`, `PgUp`, `PgDn`, `Ctrl-C`, or any single character. |
| `Resize COLS ROWS` | `TIOCSWINSZ` plus `SIGWINCH` to the child's process group. |
| `WaitFor "..." [ms]` | Wait until the literal reaches the screen or the raw stream (default 5000 ms). |
| `WaitQuiet [ms]` | Wait until output has been silent that long (default 250 ms). |
| `ExpectScreen contains\|excludes "..."` | Assert on the interpreted VT screen. |
| `ExpectRaw contains\|excludes "..."` | Assert on the decoded byte stream (escape sequences visible). |
| `Golden "name"` | Compare the current screen to `GOLDEN_DIR/name.screen.txt` (read-only). |

A JSON tape (`*.json`) is equivalent: `{"name", "cols", "rows", "command", "steps": [{"action": ...}]}`.

`Backend vhs`, `Vhs`, `Browser`, `Screenshot`, and `chrome|chromium|playwright` are
refused with `BLOCKED_POLICY`: *VHS backend deferred: host-side-effect isolation
not established*. Nothing is spawned when a tape is refused — parsing happens
before the PTY exists.

## Goldens

`GOLDEN_DIR/<name>.screen.txt` holds the approved screen and
`<name>.manifest.json` records `screen_sha256`, `approved_sha256`,
`tape_sha256`, `cols`, `rows`, and `approved_at`.

`testbench tape run` opens goldens **read-only**. Missing, unapproved
(`approved_sha256 != screen_sha256`, or the golden file changed after approval),
stale (`tape_sha256` or geometry drift), or different screens never rewrite the
repository: the run writes `NAME.actual.screen.txt` and `NAME.golden.diff` into
the run's evidence directory and ends `BLOCKED` (missing/unapproved/stale) or
`FAIL`/`assertion` (diff).

Approval is a separate human action:

```
testbench tape approve <evidence-dir-or-run-id> --golden NAME --golden-dir DIR --yes
```

It refuses to run inside a TestBench sandbox (`TESTBENCH_SANDBOX=1`) or without
an interactive TTY, so suites and background agents cannot self-approve.

## CLI

```
testbench tape run TAPE [--project-root P] [--golden-dir D] [--timeout S] [--json]
```

Each run persists a normal `RunRecord` (`run.json`, `junit.xml`, `repro.sh`) and
prints a final `TB-TAPE {...}` marker line that the `terminal_tape` adapter parses
into per-check counts. `discovery` maps every `*.tape` file to a `terminal_tape`
suite in the `terminal` profile; such suites may not request network access.
