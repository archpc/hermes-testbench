# Hermes TestBench Operating Guide

## 1. Execution & Isolation Model

Every suite executed by `TestBenchEngine` runs through `isolation.run_isolated_command`:

- **Process group & cancellation**: Child starts with `start_new_session=True` (`pgid == pid`). Timeouts and `testbench cancel <run_id>` check `(pgid, process_start_fingerprint)` before sending `SIGTERM` followed by bounded `SIGKILL` to the entire process group, preventing recycled-PID kills and grandchild leaks.
- **Filesystem & environment**: Each suite gets a fresh `0700` sandbox directory under `~/.hermes/cache/scratch/tbw-<nonce>` with isolated `HOME`, `XDG_*`, `TMPDIR`, `HERMES_HOME`, `HXMAIL_DB`, `CARGO_HOME` (symlinking read-only crate caches without `credentials.toml`), and `CARGO_TARGET_DIR`. Live `TMUX`, `SSH_AUTH_SOCK`, `TESTBENCH_HOME`, and credential environment variables (`*API_KEY*`, `*SECRET*`, `*TOKEN*`, `*_KEY`, etc.) are scrubbed.
- **macOS Seatbelt (`sandbox-exec`)**: Denies outbound internet and `localhost:9119` (Hermes dashboard) while allowing UNIX domain sockets and ephemeral `127.0.0.1` loopback mock servers when `allow_network = false`; denies reading `~/.ssh`, `~/.aws`, `~/.gnupg`, `~/.cargo/credentials*`, `~/.hermes/.env`, `~/.hermes/auth.json`, `~/.hermes/mcp-tokens`, and `~/Library/Application Support/hxmail`; denies writes to `~/.hermes` (except `~/.hermes/cache/scratch`), `~/.cargo`, `~/.rustup`, and the target repository checkout when `allow_repo_write = false`.
- **In-stream secret redaction (`ADR-0004`)**: Raw pipe/PTY bytes pass through `StreamRedactor` (carrying a 256-char sliding overlap across arbitrary read boundaries) BEFORE writing to `*.stdout.log`, `*.stderr.log`, or `*.sanitized.log` (`0600`). Secrets never land unredacted on disk.

### 1.1 Platform isolation: macOS and Linux are not equivalent

**On Linux, TestBench does not confine the filesystem or the network.** Filesystem and network confinement come only from macOS Seatbelt (`sandbox-exec`). Linux has no `sandbox-exec`, so the suite runs as your user with no confinement wrapper: it can read `~/.ssh`, write anywhere your user can write (including the repository checkout and the real `~/.hermes`), and reach the internet, localhost services, and your live tmux socket. `allow_network = false` and `allow_repo_write = false` are recorded but not enforced there. Run untrusted code on Linux only inside a container or VM you control.

The run record shows which case applied: each suite's `isolation_meta.seatbelt` is `active` (confined), `inherited_outer` (already inside a Seatbelt), or `unavailable` (no confinement). `testbench --version` prints `sandbox seatbelt` or `sandbox none (env and process-group isolation only)`.

| Control | macOS, Seatbelt active | Linux |
|---|---|---|
| Writes outside the run sandbox (home, `~/.hermes`, `state.db`, repo when `allow_repo_write = false`) | Denied | **Not enforced** |
| Reads of credential files (`~/.ssh`, `~/.aws`, `~/.gnupg`, `~/.config/gh`, `~/.netrc`, Keychains, `~/.hermes/auth.json`, `.env`) | Denied | **Not enforced** |
| Network with `allow_network = false` (internet, DNS, `localhost:9119/9222/11434`) | Denied; `127.0.0.1` loopback mocks allowed | **Not enforced** |
| Live tmux socket, `mDNSResponder`, `docker.sock` | Denied | **Not enforced** |
| Credential env scrubbing (`*TOKEN*`, `*SECRET*`, `*API_KEY*`, `SSH_AUTH_SOCK`, `TMUX`, ...) | Enforced | Enforced |
| Disposable `HOME`/`XDG_*`/`TMPDIR`/`TESTBENCH_HOME` | Enforced | Enforced (redirection only: absolute paths reach the real files) |
| `setsid` process group, fingerprinted `SIGTERM` then `SIGKILL` on timeout/cancel | Enforced | Enforced (`ps -o lstart=` from procps; BusyBox `ps` disables the fingerprint check) |
| `setsid`-survivor sweep on timeout, cancel, or non-zero exit | `lsof +D` and `ps` | Best effort: skipped when `lsof` or `/bin/ps` is absent |
| Background processes left in the suite's own process group after it exits (any status) | Terminated (`SIGTERM`, then `SIGKILL`) | Same |
| Private tmux socket and `kill-server` | Enforced | Enforced |
| In-stream secret redaction before disk | Enforced | Enforced |
| Trust gate (untrusted manifest requesting network or repo writes is `BLOCKED`) | Enforced | Gate enforced, but the default it protects (no network, read-only repo) does not exist on Linux |
| Billable gate, MCP scoped roots | Enforced | Enforced |
| TUI refresh | `kqueue` | `select` polling, same results |

Env scrubbing matches names, not values: a secret in a variable whose name matches no pattern (for example `MYDBPASS`) is passed through on both platforms.

Process cleanup on a clean exit covers only the suite's own process group: TestBench leaves the exited leader unreaped (`waitid(WNOWAIT)`) so its process-group id stays reserved, then signals that group. Where `os.waitid` is unavailable (macOS on Python < 3.13) the leader is reaped first; the group id then stays reserved only while a member is alive, and the group is re-checked for live members immediately before each signal. A descendant that called `setsid()` or `setpgid()` into a new group is outside it; on a clean exit such a process survives unless it was started within a timed-out, cancelled, or failed step (which also runs the sandbox-directory sweep).

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

## 5. Installed CLI vs Development Checkouts

The installed CLI runs an immutable release snapshot, never a working checkout. A checkout or worktree under `~/Code/hermes-testbench*` can change branch, carry uncommitted edits, or be deleted without affecting `~/.local/bin/testbench`, the `hx:testbench` TUI, detached workers, or MCP.

Layout:

```
~/.local/opt/testbench/releases/<sha>/   git archive of one commit (read-only by convention)
~/.local/opt/testbench/current           symlink -> releases/<sha>
~/.local/bin/testbench                   symlink -> ~/.local/opt/testbench/current/bin/testbench
```

`bin/testbench` resolves its own path, so the CLI, `_worker` (`PYTHONPATH=repo_package_root()`), and the TUI import from `releases/<sha>`. A worker spawned before a switch keeps the release it started from. The store (`~/.local/share/testbench`, `SCHEMA_VERSION`) is shared across releases; check `SCHEMA_VERSION` in `hermes_testbench/schema.py` before deploying, and never switch across a schema change without a migration.

Deploy (from a clean, merged commit; `S` = a checkout, `R` = the commit):

```sh
S=~/Code/hermes-testbench; R=$(git -C "$S" rev-parse --short HEAD)
O=~/.local/opt/testbench; mkdir -p "$O/releases"
test -z "$(git -C "$S" status --porcelain)"                      # refuse dirty trees
mkdir "$O/releases/$R.tmp" && git -C "$S" archive "$R" | tar -x -C "$O/releases/$R.tmp"
mv "$O/releases/$R.tmp" "$O/releases/$R" && echo "$R" > "$O/releases/$R/RELEASE"
testbench --home "$(mktemp -d ~/.hermes/cache/scratch/tb-predeploy.XXXX)" run "$O/releases/$R" --profile unit   # smoke from the release
ln -s "releases/$R" "$O/current.new" && mv -h "$O/current.new" "$O/current"                                   # atomic switch
```

One-time launcher switch (replaces the checkout symlink; atomic):

```sh
ln -s ~/.local/opt/testbench/current/bin/testbench ~/.local/bin/testbench.new && mv -h ~/.local/bin/testbench.new ~/.local/bin/testbench
```

Rollback: point `current` at the previous release with the same `ln -s ... current.new && mv -h` pair. Rolling back the one-time launcher switch: `ln -s ~/Code/hermes-testbench/bin/testbench ~/.local/bin/testbench.new && mv -h ~/.local/bin/testbench.new ~/.local/bin/testbench`.

TUI restart after a switch or rollback. The running TUI keeps the code it loaded; respawn only its own pane, found by window name, never by index:

```sh
P=$(tmux list-panes -a -F '#{window_name} #{pane_id}' | awk '$1=="hx:testbench"{print $2; exit}')
tmux respawn-pane -k -t "$P" "$HOME/.local/bin/testbench tui"
```

`respawn-pane -k` keeps the window id and name, so `hermes-tmux` ownership and restore logic are unaffected. In-flight async runs are not interrupted. Retain the last three releases; delete older ones only when no `_worker` process references them (`ps -axo command | grep releases/<sha>`).

## 6. Optional Local MCP Registration

`testbench mcp` speaks stdio MCP JSON-RPC 2.0 (`2024-11-05` through `2025-11-25`, verified against Hermes's `mcp` 2.0 SDK client). It remains unregistered in live `~/.hermes/config.yaml` by default so CLI + skill workflows work with zero gateway config changes; to enable MCP tools (`testbench_discover`, `testbench_run`, `testbench_status`, `testbench_results`, `testbench_reproduce`, `testbench_cancel`, `testbench_benchmark`), register `command: ~/.local/bin/testbench`, `args: ["mcp"]` under `mcp_servers.testbench` and include `mcp-testbench` if `platform_toolsets.cli` uses an explicit MCP allowlist.
