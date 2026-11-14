# Hermes TestBench Architecture & Module Ownership

Hermes TestBench (`hermes_testbench`) is a zero-dependency Python 3.11+ testing, benchmarking, isolation, and verification engine with five thin interfaces over one execution core and one SQLite WAL store:

1. **TestBench Core (`hermes_testbench/`)**
   - `schema.py` — Pure data contracts ONLY: dataclasses (`RunRecord`, `SuiteResult`, `CaseResult`, `FailureRecord`, `ProjectSpec`, `SuiteSpec`), status enums (`QUEUED`, `RUNNING`, `PASS`, `FAIL`, `SKIP`, `BLOCKED`, `TIMEOUT`, `CANCELLED`, `ERROR`), profile constants, and dict conversion. Zero regex, XML, I/O, or formatting logic.
   - `sanitize.py` — Untrusted text and secret defense ONLY: `redact_secrets`, chunk-safe `StreamRedactor` (applied in RAM before any output hits disk), `strip_terminal_escapes`, `sanitize_terminal_text` (OSC 52/8, CSI, DCS, C0/C1, NUL, bidi, Unicode tags), and `sanitize_for_agent` (bounded, framed untrusted excerpts).
   - `reporting.py` — Output formats ONLY: `render_junit_xml`, `parse_junit_xml_bounded` (XXE/size-guarded), `render_run_json`, `format_terminal_summary`, and `format_repro_script`.
   - `store.py` — Authoritative SQLite WAL store (`$TESTBENCH_HOME/state.db`, `0700` dir / `0600` files + Time Machine exclusion), `PRAGMA user_version` check, `BEGIN IMMEDIATE` write transactions, `run.lock` orphan reconciler, project registry/trust pins, retention pruning, and atomic `run.json` / `junit.xml` / `repro.sh` export via `reporting.py`.
   - `isolation.py` — Disposable `0700` sandbox (`HOME`, `XDG_*`, `TMPDIR` under `~/.hermes/cache/scratch`), env credential/tmux scrubber, macOS `sandbox-exec` Seatbelt profile generator (denying `~/.ssh`, `~/.hermes/.env`, `auth.json`, writes to `~/.hermes` outside scratch, and network by default), `setsid` process-group runner pumping stdout/stderr through `StreamRedactor` in RAM before writing `0600` logs, `(pgid, start_time)` verified `SIGTERM`->`SIGKILL` cancellation/timeout reaper, `tmux -L <sock> kill-server`, `TESTBENCH_RUN_NONCE` sweep, and non-symlink-following directory cleanup.
   - `discovery.py` — Zero-config repository capability detection + `.testbench.toml` / external registry loader (`$TESTBENCH_HOME/projects/<slug>.toml`) + trust SHA-256 verification.
   - `adapters/` — Native runner adapters (single-pass stdout+stderr parsing, exact exit status preservation, zero duplicate reruns): `cargo`, `python_runner` (`unittest` stderr + `pytest`), `cobol` (GnuCOBOL `-O2`/`-debug`), `c_sanitizer` (`clang -fsanitize=address,undefined`), `shell`, `sqlite_probe`, `pty_tmux`, `re_project`, `agent_eval` (explicit opt-in + token cap, `BLOCKED` by default).
   - `fixtures.py` — Deterministic generators & fault injectors: Unicode boundaries, malformed inputs, SQLite WAL contention, hostile terminal escapes, PTY resize, signals/crashes, repeated init, tmux lifecycle, async `agent-messaging` receipts/contracts, and cross-language fixture DBs.
   - `benchmark.py` — Workload runner (warmup + N rounds), environment metadata, `min`/`p50`/`p95`/`max`/`iqr`/`mad` + median-of-medians statistics, and baseline comparison without arbitrary pass/fail thresholds.
   - `engine.py` — Single orchestrator invoked by CLI, detached `_worker`, MCP, and TUI.

2. **Standalone CLI (`hermes_testbench/cli.py` + `bin/testbench` -> `~/.local/bin/testbench`)**
   - `discover`, `run`, `status`, `results`, `reproduce`, `cancel`, `bench`, `projects`, `trust`, `prune`, `fixtures`, `tui`, `mcp`, `_worker`.

3. **Local Stdio MCP Server (`hermes_testbench/mcp_server.py`)**
   - Read-only (`readOnlyHint: true`): `testbench_discover`, `testbench_status`, `testbench_results`.
   - Mutating (`readOnlyHint: false`): `testbench_run`, `testbench_reproduce`, `testbench_cancel`, `testbench_benchmark`.
   - Exports only structured metadata + `sanitize_for_agent` excerpts + `sanitized.log`/`run.json` paths.

4. **Interactive TUI (`hermes_testbench/tui_render.py` + `hermes_testbench/tui.py`) + `hermes-tmux` (`hx:testbench`)**
   - Borderless 4-band TUI consuming `store.py` (`kqueue` + `PRAGMA data_version` + `SIGWINCH`), explicit `y` confirmations, responsive split/sheet layout.

5. **Hermes Skill (`~/.hermes/skills/software-development/testbench/SKILL.md`)**
   - Concise orchestration guide mapping natural requests to `mcp_testbench_*` tools and `testbench` CLI.
