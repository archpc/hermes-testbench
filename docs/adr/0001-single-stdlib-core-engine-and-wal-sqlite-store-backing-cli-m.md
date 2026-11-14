# ADR-0001: Single stdlib core engine and WAL SQLite store backing CLI, MCP, skill, and tmux TUI

- Status: accepted
- Date: 2026-11-14
- Deciders: Sein, Carbon (lead architect)
- Supersedes: none
- Superseded by: none

## Context and problem

Active repositories on this host use incompatible test and benchmark conventions: `hermes-tmux-workspace` runs 354 stdlib `unittest` cases across private `TMUX_TMPDIR` servers, `hermes-terminal-env` combines a 3-crate Cargo workspace (`hxmail`), GnuCOBOL (`listbench.cob` `-O2`/`-debug`), Clang ASan/UBSan C harnesses, and `pyte` PTY benchmarks, `apple-power-firmware` runs read-only Python verification scripts with explicit `SKIP` semantics when gitignored blobs are absent, and `agent-messaging` relies on a drift script plus Hermes unit suites. Each agent session currently rediscovers commands, parses stderr/stdout ad hoc, risks leaking tmux servers or touching live `~/.hermes` state, and leaves no persistent structured run history.

TestBench must expose five surfaces — Core library, standalone CLI (`testbench`), local stdio MCP server (`testbench mcp`), Hermes skill (`testbench`), and permanent tmux TUI (`hx:testbench`) — without duplicating execution engines, runner state, or result schemas.

## Constraints and requirements

- Zero runtime dependency on Hermes, its gateway, active sessions, or third-party Python packages; Core and CLI must run standalone.
- Pin Python >= 3.11 explicitly in installed launchers because macOS `/usr/bin/python3` is 3.9.6 (lacking `tomllib`), while `~/.local/bin/python3.14` (3.14.7) is installed.
- One authoritative state store (`$TESTBENCH_HOME/state.db`, default `~/.local/share/testbench/state.db`, `0700` dir / `0600` file) safe under concurrent CLI, MCP, detached worker, and TUI access.
- Distinguish `PASS`, `FAIL`, `SKIP`, `BLOCKED`, `TIMEOUT`, `CANCELLED`, and `ERROR` (harness/setup failure vs assertion failure), plus `QUEUED` and `RUNNING`.
- Capture stdout, stderr (including `unittest` summaries written to stderr), exit code, counts, sanitized output, failures, reproduction command, environment metadata, and optional model token/cost fields without ever rerunning a suite to count results.
- Mandatory isolation: disposable `0700` `HOME`/`XDG_*`/`TMPDIR` under `~/.hermes/cache/scratch`, scrubbed credentials and `TMUX`/`TMUX_PANE`, optional private `TMUX_TMPDIR` + `-L` socket, process-group (`setsid`) timeouts, explicit `tmux -L <sock> kill-server` + nonce sweep, and macOS `sandbox-exec` protection of `~/.hermes` and `~/Library/Application Support/hxmail`.

## Alternatives considered

| Alternative | Why it was weighed | Why it lost |
|---|---|---|
| Rust workspace (`rusqlite` + `ratatui` + `serde_json`, like `hxmail`) | Strong compile-time typing, single native binary, shared `ratatui` widgets with `hxmail` | Slower iteration across 9 heterogeneous language/script adapters, requires external crate cache for every edit, whereas Python 3.14 stdlib provides `sqlite3`, `tomllib`, `pty`, `termios`, `select.kqueue`, `fcntl`, and `xml.etree.ElementTree` with zero dependencies |
| Split Python Core/CLI/MCP + Rust TUI | Keeps Python adapter ergonomics while reusing `hxmail`'s `ratatui` code | Splits SQLite schema, status enums, and `PRAGMA user_version` migrations across two languages and two build pipelines |
| Per-run JSON files on disk with no SQLite index | Simplest write path (one directory per run) | Cross-project queries ("last verified run per project", concurrent status updates, retention pruning) scan hundreds of directories and suffer read-modify-write races |

## Decision

Implement TestBench in `/Users/sein/Code/hermes-testbench` as a single zero-dependency Python 3.11+ package (`hermes_testbench`) with launcher `~/.local/bin/testbench` pinned to `/Users/sein/.local/bin/python3.14`:

- **Authoritative store (`hermes_testbench/store.py`)**: SQLite WAL database `$TESTBENCH_HOME/state.db` (`PRAGMA journal_mode=WAL`, `synchronous=NORMAL`, `busy_timeout=5000`, `foreign_keys=ON`, strict `PRAGMA user_version` check). Every write opens an explicit `BEGIN IMMEDIATE` transaction (`isolation_level=None`) to prevent deferred-reader upgrade `SQLITE_BUSY` failures.
- **Portable evidence export**: At run settlement, `runs/<run_id>/run.json`, `junit.xml`, and `repro.sh` are written atomically (`tmp` + `os.replace`) from `state.db` alongside streamed `stdout.log`, `stderr.log`, and `sanitized.log` (capped before XML/JSON parsing).
- **Crash reconciler**: Each detached runner (`testbench _worker`) holds a non-inherited (`O_CLOEXEC`, `close_fds=True`) advisory `fcntl.flock` on `runs/<run_id>/run.lock`. On startup/query, any `QUEUED`/`RUNNING` run whose `run.lock` is unheld is reconciled to `ERROR` (`abandoned_worker`).
- **Isolation & reaper (`hermes_testbench/isolation.py`)**: Every adapter step runs in its own `setsid` process group inside a `0700` scratch sandbox (`~/.hermes/cache/scratch/tb-work-*`) with non-symlink-following cleanup, macOS `sandbox-exec` confinement (denying writes to `/Users/sein/.hermes` outside the run scratch dir and denying network by default for offline profiles), `(pgid, start_time)` verified `SIGTERM` -> `SIGKILL` reaper, explicit private `tmux -L <sock> kill-server`, and `TESTBENCH_RUN_NONCE` orphan sweep.

## Rationale

Independent critique (opus, 2026-11-14, `delivery_id=90eba6086c27b546687b288d90123d27a21da877160df0b46da5f4147c6ab655`): confirmed Python stdlib over Rust unless >10k-run TUI lag is measured, and identified seven failure modes that shaped this design: (1) Python `sqlite3` deferred-transaction `SQLITE_BUSY` ignoring `busy_timeout` -> solved with `BEGIN IMMEDIATE`; (2) `state.db` vs `run.json` divergence -> `state.db` made sole source of truth with atomic terminal `run.json` export + `run.lock` reconciler; (3) `ERROR` status added alongside `FAIL` so harness crashes never masquerade as test assertion failures; (4) reparented daemons (`tmux` servers) escaping `-pgid` -> explicit `tmux -L <sock> kill-server` + nonce sweep; (5) `PRAGMA user_version` gate for long-lived MCP processes; (6) `run.lock` close-on-exec non-inheritance; (7) pinning Python 3.14 because `/usr/bin/python3` was verified to be 3.9.6.

## Tradeoffs and consequences

- **Easy**: Zero virtualenv/wheel breakage across profiles; single schema shared by CLI, MCP, skill helper, and TUI; portable `run.json`/`junit.xml` artifacts survive DB export.
- **Harder**: No static type checker at runtime (mitigated by typed dataclasses, schema validation tests, and strict boundary input checks); TUI rendering uses our own cell-width/ANSI frame builder rather than `ratatui`.
- **Platform**: `sandbox-exec` and `select.kqueue` are macOS/BSD-specific; both degrade cleanly to POSIX env/umask/poll fallbacks on Linux when `shutil.which("sandbox-exec")` or `hasattr(select, "kqueue")` is absent.

## Supporting evidence

- Archaeology: `/Users/sein/.hermes/cache/scratch/testbench/archaeology/01-mcp-feasibility.md`, `02-tmux-workspace.md`, `03-hxmail-hte.md`, `04-messaging-and-repos.md`.
- Verified `/usr/bin/python3 --version` -> `Python 3.9.6`; `/Users/sein/.local/bin/python3.14 --version` -> `Python 3.14.7`.
- Verified `sandbox-exec` on macOS 27.2.1 blocks `(deny network*)` and `(deny file-write* (subpath "/Users/sein/.hermes"))` while allowing `(allow file-write* (subpath "<run-scratch>"))`.
- Opus 5.5 consultation receipt: `/Users/sein/.hermes/profiles/opus/runtime/bot_live_delivery/90eba6086c27b546687b288d90123d27a21da877160df0b46da5f4147c6ab655.json`.

## Verification status

verified (`/Users/sein/.local/bin/python3.14 -W default -m unittest discover -s tests -p "test_*.py"` — 181/181 passed; `testbench run ~/Code/hermes-testbench --profile regression` `tb_20261114_233234_da1f84dc` 180/180 passed; multi-repo runs across `hermes-terminal-env`, `hermes-tmux-workspace`, `apple-power-firmware`, and `agent-messaging`, 2026-11-14)

## Related decisions

- ADR-0002: Local stdio MCP server and project trust/sandbox boundary (depends on this core engine and store).
- ADR-0003: `hx:testbench` TUI and `hermes-tmux` permanent auxiliary-window integration (consumes `state.db` read-only and invokes Core APIs).
