# ADR-0002: Local stdio MCP server with project trust registry and Seatbelt sandbox over in-process plugins

- Status: accepted
- Date: 2026-11-14
- Deciders: Sein, Carbon (lead architect)
- Supersedes: none
- Superseded by: none

## Context and problem

Agents (Hermes on DeepSeek, Carbon, Opus, Claude Code, Codex) need a structured interface to discover tests, launch profiles, poll status, inspect results, reproduce failures, run benchmarks, and cancel owned jobs without scraping raw terminal output or dumping megabytes of logs into LLM context. Three agent integration mechanisms exist in Hermes (`/Users/sein/.hermes/hermes-agent`): (1) a local stdio MCP server (`mcp_servers` + `platform_toolsets.cli`), (2) an in-process Python plugin (`ctx.register_tool` in `plugins/`), or (3) direct shell `terminal(command="testbench ...")` invocation guided only by `SKILL.md`.

Additionally, exposing test execution as an MCP tool creates an execution-escalation boundary: running even auto-discovered native suites (`conftest.py`, `build.rs`, `.testbench.toml` custom commands) executes repository code.

## Constraints and requirements

- Must work in ordinary Hermes CLI/tmux sessions (not only `Bot Chat`), and remain portable to non-Hermes MCP clients.
- Must not start an always-running network daemon; local stdio only.
- Long-running test runs launched through MCP must survive individual tool-call completion and MCP server restarts without hanging stdio JSON-RPC pipes or leaking runaway processes.
- Separate read-only inspection (`testbench_discover`, `testbench_status`, `testbench_results`) from execution/cancellation (`testbench_run`, `testbench_reproduce`, `testbench_cancel`, `testbench_benchmark`).
- Never turn the MCP server into an arbitrary shell-execution endpoint or allow self-escalation through agent-writable config/skill directories (`~/.hermes/skills`).

## Alternatives considered

| Alternative | Why it was weighed | Why it lost |
|---|---|---|
| In-process Hermes plugin (`ctx.register_tool`) | ~40 LOC, direct access to Hermes internals | Runs unsandboxed inside the Hermes agent process (`01-mcp-feasibility.md` §5.2), coupled to Hermes's internal Python 3.11 venv, non-portable to Claude/Codex/external MCP clients, and dies with the agent turn |
| Direct CLI invocation + `SKILL.md` only | Zero server code | Forces shell command approval/parsing on every call, provides no typed JSON schema in `tool_search`/`tool_describe`, and encourages ad-hoc flag invention or raw log dumping |
| HTTP/SSE localhost MCP daemon | Natural fit for background workers | Requires persistent port management, bearer auth, launchd supervision, and new attack surface on localhost for no multi-host requirement |
| Manual `testbench trust` required before *any* repo (including dirty worktrees) can run via MCP | Proposed by Opus advisor for strict code-execution gating | Blocks core workflow ("enter an unfamiliar repository, discover its tests, verify in-progress refactors with minimal configuration") because refactors have dirty worktrees by definition |

## Decision

Expose `testbench mcp` as a stdlib-only newline-delimited JSON-RPC 2.0 stdio MCP server (`hermes_testbench/mcp_server.py`), optionally registered in `~/.hermes/config.yaml` under `mcp_servers.testbench` (with `mcp-testbench` in `platform_toolsets.cli` when explicit toolset allowlisting is active, while CLI and skill workflows operate with zero live config edits):

- **Protocol & tool surface**: Negotiates `HANDSHAKE_PROTOCOL_VERSIONS` (`2024-11-05` .. `2025-11-25`), handles `initialize`, `notifications/initialized`, `notifications/cancelled`, `ping`, `tools/list`, and `tools/call`, exits cleanly on stdin EOF, and logs exclusively to stderr. Exposes 7 tools:
  - Read-only (`annotations.readOnlyHint: true`): `testbench_discover`, `testbench_status`, `testbench_results`.
  - Mutating (`annotations.readOnlyHint: false`): `testbench_run`, `testbench_reproduce`, `testbench_cancel`, `testbench_benchmark`.
- **Scoped authorization & Seatbelt confinement**:
  - MCP never accepts raw shell/command strings.
  - Default scoped workspace roots allow `/Users/sein/Code` and `/Users/sein/.hermes/artifacts` (canonicalized via `Path.resolve(strict=True)`), plus projects explicitly registered via CLI (`testbench projects register` / `testbench trust`). `~/.hermes` (including `~/.hermes/skills`, `plugins`, `config.yaml`, `.env`) is **excluded** so an agent cannot write a skill/manifest and execute it via MCP.
  - Built-in native adapters (`cargo`, `unittest`, `pytest`, `cobc`, `clang`, read-only verifiers) run inside macOS `sandbox-exec` denying writes to `~/.hermes` (except the run's scratch dir), denying reads of `~/.ssh`/`~/.aws`/`~/.gnupg`/`~/.hermes/.env`/`auth.json`, and denying network by default.
  - Custom arbitrary command suites in `.testbench.toml` require either matching CLI trust (`testbench trust <repo>` pinning `.testbench.toml` SHA-256; MCP itself has no trust-granting tool) OR strict read-only-workspace + network-deny Seatbelt confinement; untrusted host-mutating or billable suites return `BLOCKED` (`reason="untrusted_custom_suite"` or `"billable_opt_in_required"`).
- **Async detached worker**: `testbench_run` and `testbench_benchmark` spawn `testbench _worker --run-id <id>` with `start_new_session=True`, `stdin/stdout/stderr=DEVNULL`, and `close_fds=True` (empirically verified in `01-mcp-feasibility.md` §3 Probe A to escape Hermes's `_stdio_pgids` sweep without hanging stdio), wait up to bounded `wait_up_to_s` (default `3.0`, max `30.0`) so fast unit runs settle in one call while longer suites return `{status: "RUNNING", run_id, evidence_dir}`, and return sanitized, bounded (`<= 2000` chars) failure excerpts labelled as untrusted test output.

## Rationale

Empirical wire probes (`01-mcp-feasibility.md` §3-4) proved a ~150-LOC stdlib stdio server interoperates cleanly with Hermes's `mcp` 2.0.0 client in ordinary CLI sessions via `tool_search`/`tool_describe`/`tool_call`. Independent critique (opus, 2026-11-14, `delivery_id=90eba6086c27b546687b288d90123d27a21da877160df0b46da5f4147c6ab655`) caught that `~/.hermes/skills` on an MCP allowlist would permit self-escalation, that `.testbench.toml` custom commands require CLI-only hash trust, that `sandbox-exec` is needed to stop hardcoded paths bypassing `HOME` scrubbing, and that MCP log excerpts must be truncated and marked untrusted against prompt injection. Carbon accepted all four defenses while permitting built-in native adapters in `~/Code` and `~/.hermes/artifacts` (including dirty worktrees) inside Seatbelt confinement so refactor verification works out of the box.

## Tradeoffs and consequences

- **Easy**: Works across DeepSeek, Carbon, Opus, and non-Hermes MCP clients; fast suites return settled in 1 call, slow suites run asynchronously and survive session/server restarts.
- **Harder**: In Hermes CLI sessions, MCP tools sit behind progressive disclosure (`tool_search` -> `tool_describe` -> `tool_call`), costing 1-2 bridge calls on first use per session; `SKILL.md` documents exact tool names so `tool_describe` can be called directly without a search round-trip.
- **Boundary**: Custom suites that genuinely require live network or writes outside sandbox/repo must either be run from the CLI directly or explicitly trusted via `testbench trust <repo>` from a human/CLI shell.

## Supporting evidence

- `01-mcp-feasibility.md` §1-6: verified `tools/mcp_tool_transport.py:306-341`, `tools/mcp_tool_registration.py:52-59`, `hermes_cli/tools_config.py:734-753`, and Probe A/B against `mcp` 2.0.0 (`HANDSHAKE_PROTOCOL_VERSIONS`).
- Opus 5.5 consultation receipt: `/Users/sein/.hermes/profiles/opus/runtime/bot_live_delivery/90eba6086c27b546687b288d90123d27a21da877160df0b46da5f4147c6ab655.json`.

## Verification status

verified (`/Users/sein/.local/bin/python3.14 -W default -m unittest tests.test_mcp_server tests.test_isolation -v` — 13/13 passed including live `mcp` 2.0.0 SDK client handshake, `testbench_discover`, `testbench_run`, `testbench_results`, `~/.hermes/skills` refusal, and detached worker survival across server EOF, 2026-11-14)

## Related decisions

- ADR-0001: Single stdlib core engine and WAL SQLite store (provides `store.py`, `isolation.py`, and `_worker`).
- ADR-0003: `hx:testbench` TUI and `hermes-tmux` integration (observes and cancels the same MCP-launched runs).
