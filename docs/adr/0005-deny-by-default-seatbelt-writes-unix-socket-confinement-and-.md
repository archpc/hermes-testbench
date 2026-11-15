# ADR-0005: Deny-by-default Seatbelt writes, Unix-socket confinement, and fingerprinted cancellation

- Status: accepted
- Date: 2026-11-14
- Deciders: Sein, Carbon (lead architect)
- Supersedes: none
- Superseded by: none

## Context and problem

Post-implementation review by Opus 5.5 (`delivery_id=2e63dfeef0079389a0e0066facc1e33f02bac918b538359926d9dd5231e4e6df`) and an independent adversarial review subagent (`deleg_f0276bbf`, reproducible probes A-E) identified six substantive security, lifecycle, and reporting gaps in the initial ADR-0001/0002/0004 implementation:

1. **Write-denylist Seatbelt gap (Opus)**: `(allow default)` with a selective `(deny file-write* ...)` left `~/Library/LaunchAgents`, `~/.zshrc`, `~/.ssh/authorized_keys`, `~/.local/bin/testbench`, and `~/.local/share/testbench/state.db` writable from untrusted suites.
2. **Unix-socket & DNS channel (Opus)**: Blanket `(allow network* (local unix))` permitted connecting to the live tmux socket (`/private/tmp/tmux-<uid>/default`) and `/private/var/run/mDNSResponder`.
3. **Read-confinement gap (R3 + Opus)**: `~/.hermes/config.yaml`, `state.db`, `sessions/`, `profiles/`, `memories/`, `~/.config/gh`, and `~/Library/Keychains` remained readable under Seatbelt.
4. **Ambient `pgid` cancellation hazard (R1, Probe A)**: `run_sync` recorded `os.getpgrp()` without `worker_start_time`, allowing `cancel()` to `SIGTERM` the caller's ambient shell job group.
5. **`setsid()` descendant escape (R2, Probe B)**: Double-forked `os.setsid()` children survived process-group termination on timeout.
6. **JUnit attribute mismatch & unclosed PEM buffering (R5, R6)**: JUnit suite attributes could diverge from emitted `<error>`/`<skipped>` elements on `TIMEOUT`/`BLOCKED`, and `StreamRedactor` buffered unbounded single-line streams inside unclosed PEM headers.

## Constraints and requirements

- Sandboxed native runners (`Cargo`, `unittest`, `pytest`, `GnuCOBOL`, `clang`, disposable `tmux -L <sock>`, and `127.0.0.1:0` loopback mock servers) must continue working offline with zero user prompts.
- Note macOS XNU `AF_UNIX` nuance: `tmux` calls `chdir("$TMUX_TMPDIR/tmux-<uid>")` before relative `bind("<sock>")` and `connect("<sock>")`, so `(remote unix-socket (subpath ...))` allowlists fail on relative `sun_path`; vnode read/write denies (`(deny file-write*)` + `(deny file-read* (subpath "/private/tmp/tmux-<uid>"))`) enforce path resolution regardless of `chdir`.
- `cancel()` must never signal the caller's process group or any PGID lacking a matching `ps -o lstart=` start-time fingerprint.

## Alternatives considered

| Alternative | Why it was weighed | Why it lost |
|---|---|---|
| Expand selective `(deny file-write* ...)` denylist | Smaller diff to `isolation.py` | Enumerating every persistence location (`LaunchAgents`, shell dotfiles, cron, sibling repos, `state.db`) is inherently incomplete compared to `(deny file-write*)` |
| `(allow network-outbound (remote unix-socket (subpath "<sandbox>")))` allowlist only | Worked for Python absolute-path sockets | Broke `tmux -L <sock>` (`Operation not permitted`) because `tmux` `chdir`s into `tmux-<uid>` and passes a relative `sun_path` to `connect(2)` |
| Spawn a new wrapper subprocess for every `run_sync` call | Gives `run_sync` its own PGID | Redundant because `run_isolated_command` already spawns every suite in its own `setsid` session; recording the active suite's `(pid, pgid, start_fp)` via `on_spawn` + `cancel.flag` achieves full isolation with zero extra wrapper processes |

## Decision

1. **Deny-by-default file writes**: Replace selective write denies with `(deny file-write*)`, explicitly allowing only `<resolved_sb>`, `<extra_writable_dirs>` (disposable `tmux_tmpdir`), `~/.hermes/cache/scratch/hxlab`, `<resolved_cwd>` only when `allow_repo_write=True`, `~/.cache/uv` only when `argv[0]` is `uv`, and explicit `/dev` character devices (`null`, `zero`, `tty`, `ptmx`, `urandom`, `random`, `dtracehelper`, `ttys*`, `fd/*`). Set `TESTBENCH_HOME` inside `build_isolated_env` to `<sandbox>/xd/testbench`.
2. **Unix-socket, localhost, and read confinement**: Block `/private/tmp/tmux-<uid>` and `/private/var/tmp/tmux-<uid>` via `(deny file-read*)` (defeating both absolute and `chdir`-relative `connect()`), deny `network-outbound` to `/private/var/run/mDNSResponder`, `/private/tmp/tmux-<uid>`, `/private/var/run/docker.sock`, and `localhost:{9119,9222,11434}`, and expand `file-read-data` denies to cover `~/.hermes/{config.yaml,state.db*,sessions,profiles,memories,logs}`, `~/.config/gh`, `~/.docker`, `~/.netrc`, `~/.npmrc`, and `~/Library/Keychains`.
3. **Fingerprinted cancellation & survivor sweep**: Never record caller `os.getpgrp()` in `run_sync`; record child suite `(pid, pgid, start_fp)` via `on_spawn` (and worker `start_fp` in `spawn_async`), require `wpgid != os.getpgrp()` and `live_fp == expected_start_time` in `cancel()`/`terminate_process_group()`, fence `_populate_and_finalize` on `cancel.flag`, and sweep `setsid()` survivors holding `<sandbox_dir>` via `lsof +D` / `ps` on timeout/cancel/nonzero exit.
4. **Self-consistent JUnit & bounded PEM redaction**: Derive JUnit `<testsuite>`/`<testsuites>` counts directly from emitted child elements, and cap `StreamRedactor` buffer to `OVERLAP_TAIL` bytes inside unclosed PEM blocks.

## Rationale

Combines kernel-enforced write confinement and vnode-resolved Unix socket protection with deterministic PGID start-time verification, eliminating every demonstrated escape and process-leak vector without adding runtime dependencies.

## Tradeoffs and consequences

- Suites that legitimately write outside `$TMPDIR`/`$HOME` (such as code generators writing into the repository tree) must declare `allow_repo_write = true` in a trusted `.testbench.toml`.
- `lsof +D` / `ps` survivor sweeps add ~25ms on timeout/cancel/failed steps (skipped on clean `exit_code == 0` runs so sub-5ms microbenchmarks remain unaffected).

## Supporting evidence

- Opus 5.5 critique receipt: `/Users/sein/.hermes/profiles/opus/runtime/bot_live_delivery/2e63dfeef0079389a0e0066facc1e33f02bac918b538359926d9dd5231e4e6df.json`.
- Adversarial review report: `/Users/sein/.hermes/cache/scratch/testbench/adversarial-review.md` (`R1..R6`).
- Regression suite: `tests/test_adversarial_and_opus_fixes.py` (7 end-to-end regression tests).

## Verification status

verified (`/Users/sein/.local/bin/python3.14 -W default -m unittest discover -s tests -p "test_*.py"` — 188/188 passed, 2026-11-14)

## Related decisions

- Refines ADR-0001, ADR-0002, ADR-0003, and ADR-0004.
