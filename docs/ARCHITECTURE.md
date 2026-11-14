# Hermes TestBench Architecture & Module Ownership

Hermes TestBench (`hermes_testbench`) is a zero-dependency Python 3.11+ testing, benchmarking, isolation, and verification engine with five thin interfaces over one execution core and one SQLite WAL store:

1. **TestBench Core (`hermes_testbench/`)**
   - `schema.py` — Canonical dataclasses, status enums (`QUEUED`, `RUNNING`, `PASS`, `FAIL`, `SKIP`, `BLOCKED`, `TIMEOUT`, `CANCELLED`, `ERROR`), profile names (`unit`, `integration`, `terminal`, `security`, `regression`, `benchmark`, `agent-eval`), JSON/JUnit serializers, and sanitizer helpers.
   - `store.py` — Authoritative SQLite WAL store (`$TESTBENCH_HOME/state.db`), `PRAGMA user_version` check, `BEGIN IMMEDIATE` write transactions, `run.lock` orphan reconciler, project registry/trust pins, retention pruning, and atomic `run.json` / `junit.xml` / `repro.sh` evidence export.
   - `isolation.py` — Disposable `0700` sandbox (`HOME`, `XDG_*`, `TMPDIR` under `~/.hermes/cache/scratch`), env credential/tmux scrubber, private `TMUX_TMPDIR` + `-L` socket lifecycle, macOS `sandbox-exec` Seatbelt profile generator, `setsid` process-group runner, `(pgid, start_time)` verified `SIGTERM`->`SIGKILL` cancellation/timeout reaper, `tmux kill-server`, `TESTBENCH_RUN_NONCE` sweep, and non-symlink-following directory cleanup.
   - `discovery.py` — Zero-config repository capability detection + `.testbench.toml` / external registry loader (`$TESTBENCH_HOME/projects/<slug>.toml`) + trust hash verification.
   - `adapters/` — Native runner adapters (single-pass stdout+stderr capture, exact exit status preservation, zero duplicate reruns):
     - `cargo` (Rust `cargo test --offline`, libtest count & failure parser, isolated `CARGO_TARGET_DIR` or shared cache option)
     - `python_runner` (`unittest` stderr summary parser + `pytest` summary/failure parser, explicit interpreter resolution)
     - `cobol` (GnuCOBOL `cobc` `-O2` / `-debug` + harness execution)
     - `c_sanitizer` (`clang -fsanitize=address,undefined` compile + run)
     - `shell` / `custom` (argv-only scripts & verifiers, `SKIP:` marker detection, Seatbelt + trust enforcement)
     - `sqlite_probe` (WAL integrity, `busy_timeout` contention, mode/permission checks)
     - `pty_tmux` (real PTY geometry `TIOCSWINSZ`, `SIGWINCH` resize, VT/ANSI escape hygiene, private tmux socket harness)
     - `agent_eval` (explicit `--allow-billable` + `--max-tokens` opt-in fresh-session behavioral probes; `BLOCKED` by default)
   - `fixtures.py` — Deterministic generators & fault injectors: Unicode boundaries, malformed inputs, SQLite WAL contention, hostile terminal escapes (`OSC 52`, `OSC 8`, `DCS`, `CSI`, `C1`, `NUL`, bidi), PTY resize, signals/crashes, repeated init, tmux lifecycle, async `agent-messaging` receipts/contracts, and cross-language fixture DBs.
   - `benchmark.py` — Workload runner (warmup + N rounds), environment metadata (CPU, OS, loadavg, tool versions, git revision), `min`/`p50`/`p95`/`max`/`iqr`/`mad` + median-of-medians statistics, and baseline comparison without arbitrary pass/fail thresholds.
   - `engine.py` — Single orchestrator invoked by CLI, detached `_worker`, MCP, and TUI (`discover`, `enqueue_or_run`, `run_sync`, `spawn_async_worker`, `cancel_run`, `reproduce_run`, `run_benchmark`, `compare_benchmarks`).

2. **Standalone CLI (`hermes_testbench/cli.py` + `bin/testbench` -> `~/.local/bin/testbench`)**
   - `discover`, `run`, `status`, `results`, `reproduce`, `cancel`, `bench`, `projects`, `trust`, `prune`, `fixtures`, `tui`, `mcp`, `_worker`.

3. **Local Stdio MCP Server (`hermes_testbench/mcp_server.py`)**
   - `testbench_discover`, `testbench_status`, `testbench_results` (`readOnlyHint: true`)
   - `testbench_run`, `testbench_reproduce`, `testbench_cancel`, `testbench_benchmark` (`readOnlyHint: false`)

4. **Interactive TUI (`hermes_testbench/tui.py`) + `hermes-tmux` (`hx:testbench`)**
   - Borderless 4-band TUI consuming `store.py` (`kqueue` + `PRAGMA data_version` + `SIGWINCH`), explicit `y` confirmations, responsive split/sheet layout.

5. **Hermes Skill (`~/.hermes/skills/software-development/testbench/SKILL.md`)**
   - Concise orchestration guide mapping natural requests to `mcp_testbench_*` tools and `testbench` CLI.
