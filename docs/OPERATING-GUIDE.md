# Hermes TestBench Operating Guide

## 1. Execution & Isolation Model

Every suite executed by `TestBenchEngine` runs through `isolation.run_isolated_command`:

- **Process group & cancellation**: Child starts with `start_new_session=True` (`pgid == pid`). Timeouts and `testbench cancel <run_id>` check `(pgid, process_start_fingerprint)` before sending `SIGTERM` followed by bounded `SIGKILL` to the entire process group, preventing recycled-PID kills and grandchild leaks.
- **Filesystem & environment**: Each suite gets a fresh `0700` sandbox directory under `~/.hermes/cache/scratch/tbw-<nonce>` with isolated `HOME`, `XDG_*`, `TMPDIR`, `HERMES_HOME`, `HXMAIL_DB`, `CARGO_HOME` (symlinking read-only crate caches without `credentials.toml`), and `CARGO_TARGET_DIR`. Live `TMUX`, `SSH_AUTH_SOCK`, `TESTBENCH_HOME`, and credential environment variables (`*API_KEY*`, `*SECRET*`, `*TOKEN*`, `*_KEY`, etc.) are scrubbed.
- **macOS Seatbelt (`sandbox-exec`)**: Denies outbound internet and `localhost:9119` (Hermes dashboard) while allowing UNIX domain sockets and ephemeral `127.0.0.1` loopback mock servers when `allow_network = false`; denies reading `~/.ssh`, `~/.aws`, `~/.gnupg`, `~/.cargo/credentials*`, `~/.hermes/.env`, `~/.hermes/auth.json`, `~/.hermes/mcp-tokens`, and `~/Library/Application Support/hxmail`; denies writes to `~/.hermes` (except `~/.hermes/cache/scratch`), `~/.cargo`, `~/.rustup`, and the target repository checkout when `allow_repo_write = false`.
- **In-stream secret redaction (`ADR-0004`)**: Raw pipe/PTY bytes pass through `StreamRedactor` (carrying a 256-char sliding overlap across arbitrary read boundaries) BEFORE writing to `*.stdout.log`, `*.stderr.log`, or `*.sanitized.log` (`0600`). Secrets never land unredacted on disk.

## 2. Project Onboarding & `.testbench.toml`

Unfamiliar repositories work with zero configuration (`testbench discover <repo>`). When custom profiles or multi-step suites are needed, add an optional `.testbench.toml`:

```toml
[project]
name = "example-project"
languages = ["rust", "python"]

[[suite]]
name = "fast-unit"
adapter = "cargo"
profile = "unit"
command = ["cargo", "test", "--locked", "--offline"]
timeout_s = 180.0

[[suite]]
name = "llm-eval"
adapter = "agent_eval"
profile = "agent-eval"
command = ["python3", "scripts/eval_probe.py"]
billable = true
allow_network = true
```

Security gates enforced before process spawn:
1. Untrusted `.testbench.toml` files requesting `allow_network = true` or `allow_repo_write = true` return `BLOCKED` (`untrusted_manifest`) until approved via `testbench trust <repo>` (which pins the manifest's SHA-256).
2. Suites with `billable = true` return `BLOCKED` unless invoked from CLI with `--allow-billable --max-tokens <N>` (`N > 0`). TUI (`hx:testbench`) refuses billable profiles unconditionally.

## 3. `hx:testbench` TUI & `hermes-tmux` Integration

- Permanent window `hx:testbench` opens detached beside `agents` and `hx:mail` on `hermes-tmux up` (`HERMES_TMUX_TESTBENCH=0` opts out of automatic creation while keeping `hermes-tmux testbench` available on demand).
- Keys:
  - `j` / `k` / `Up` / `Down`, `g` / `G`: navigate rows (or scroll detail logs when detail sheet is open).
  - `Tab` or `1`..`4`: switch view (`overview`, `history`, `projects`, `benchmarks`).
  - `Enter`: open/close full-screen log & evidence inspector (`< 120` cols; `>= 120` cols shows live 2-column split). `Enter` never confirms execution or cancellation.
  - `p`: filter by project; `/`: text filter; `y`: copy reproduction command to tmux buffer / clipboard.
  - `r`: select profile to run; `R`: rerun selected run; `x`: cancel selected TestBench-owned active run (each requires explicit `y` in minibuffer).
  - `q`: close active overlay; at top level in permanent-tab mode, `Q` (`Shift-Q`) exits TUI so muscle-memory `q` does not kill the permanent pane.

## 4. Retention & Maintenance

Run artifacts live under `~/.local/share/testbench` (excluded from Time Machine via `tmutil addexclusion` on macOS). Prune historical unpinned artifacts safely with:

```sh
testbench prune --keep-last 50 --dry-run
testbench prune --keep-last 50
```

## 5. Optional Local MCP Registration

`testbench mcp` speaks stdio MCP JSON-RPC 2.0 (`2024-11-05` through `2025-11-25`, verified against Hermes's `mcp` 2.0 SDK client). It remains unregistered in live `~/.hermes/config.yaml` by default so CLI + skill workflows work with zero gateway config changes; to enable MCP tools (`testbench_discover`, `testbench_run`, `testbench_status`, `testbench_results`, `testbench_reproduce`, `testbench_cancel`, `testbench_benchmark`), register `command: ~/.local/bin/testbench`, `args: ["mcp"]` under `mcp_servers.testbench` and include `mcp-testbench` if `platform_toolsets.cli` uses an explicit MCP allowlist.
