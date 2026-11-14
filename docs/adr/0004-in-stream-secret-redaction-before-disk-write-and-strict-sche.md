# ADR-0004: In-stream secret redaction before disk write and strict schema-sanitize-reporting separation

- Status: accepted
- Date: 2026-11-14
- Deciders: Sein, Carbon (lead architect)
- Supersedes: none
- Superseded by: none

## Context and problem

Two design flaws surfaced immediately when scaffolding TestBench's core contract:

1. **Secret persistence vs post-hoc redaction**: Streaming raw child `stdout`/`stderr` directly to `stdout.log`/`stderr.log` and only redacting afterward when building `sanitized.log` or MCP responses allows leaked credentials to land on disk in persistent run directories, backups, or crash remnants.
2. **Module coupling in `schema.py`**: Co-locating canonical dataclasses (`RunRecord`, `SuiteResult`) with terminal escape stripping, secret regex redaction, and JUnit XML serialization makes `schema.py` a god-module that drags regexes, XML parsers, and presentation rules into every consumer.

## Constraints and requirements

- Secrets must be prevented from entering test subprocesses where possible, and must never reach persistent disk unredacted during normal operation.
- Adapters still need separated `stdout` and `stderr` streams (e.g., Python `unittest` writes counts to `stderr`) and local VT/PTY tests occasionally inspect ANSI escape sequences without needing plaintext secrets.
- Agents (`mcp_server.py`, `SKILL.md`, JSON summaries) must only receive bounded, escape-stripped, secret-redacted text marked as untrusted output.
- Core contracts (`schema.py`), untrusted-text sanitization (`sanitize.py`), and output format serialization (`reporting.py`) must have single, non-overlapping responsibilities.

## Alternatives considered

| Alternative | Why it was weighed | Why it lost |
|---|---|---|
| Write raw `stdout.log`/`stderr.log` first, redact only when generating `sanitized.log` and MCP responses | Simplest `subprocess` file-descriptor redirection (`stdout=file`) | Writes plaintext secrets to disk before redaction runs; crash or backup mid-run preserves unredacted secrets |
| Discard `stdout.log`/`stderr.log` completely and keep only merged escape-stripped `sanitized.log` | Maximum minimization on disk | Destroys stdout-vs-stderr separation required by `unittest`/compiler adapters and strips benign SGR/CSI needed when debugging local PTY/TUI tests |
| Keep dataclasses, sanitizers, and JUnit XML together in `schema.py` | Fewer files | Couples pure data contracts to regexes, XML bounds, and presentation formatting across every future module |

## Decision

1. **Three-stage secret & output policy**:
   - **Pre-spawn prevention (`isolation.py`)**: Scrub credential-bearing env vars (`*API_KEY*`, `*TOKEN*`, `*SECRET*`, `*PASSWORD*`, `*PASSWD*`, `*CREDENTIAL*`, provider prefixes, `HERMES_SPAWN`) and apply macOS `sandbox-exec` read-denial on `~/.ssh`, `~/.aws`, `~/.gnupg`, `~/.hermes/.env`, `~/.hermes/auth.json`, and `~/Library/Application Support/hxmail`.
   - **In-memory stream redaction before disk write (`sanitize.StreamRedactor` + `isolation.py`)**: Child pipes are drained in memory through a line-buffered `StreamRedactor` (handling chunk-split tokens across `os.read` boundaries) **before** bytes are written to `0600` `stdout.log` and `stderr.log` inside a `0700` `runs/<run_id>/` directory excluded from Time Machine (`com.apple.metadata:com_apple_backup_excludeItem`). Unredacted raw output is **never** persisted to `$TESTBENCH_HOME`. `sanitized.log` additionally strips all ANSI/VT escapes, C0/C1 controls, NULs, Unicode bidi overrides, and tag characters.
   - **Agent export boundary (`sanitize.sanitize_for_agent`, `mcp_server.py`, `reporting.py`)**: `state.db` summaries, `run.json`, `junit.xml`, and all MCP tool responses export only escape-stripped + secret-redacted text bounded to `<= 2000` chars per excerpt and explicitly labelled `[untrusted test output]`, with file pointers to `sanitized.log` and `run.json` (never `stdout.log`/`stderr.log`).

2. **Strict module boundary split**:
   - `hermes_testbench/schema.py` — Pure data contracts, enums, IDs, and dict conversion only (zero regex, XML, I/O, or formatting imports).
   - `hermes_testbench/sanitize.py` — `redact_secrets`, `StreamRedactor`, `strip_terminal_escapes`, `sanitize_terminal_text`, and `sanitize_for_agent`.
   - `hermes_testbench/reporting.py` — `render_junit_xml`, `parse_junit_xml_bounded`, `render_run_json`, `format_terminal_summary`, and `format_repro_script`.

## Rationale

Separating prevention, disk persistence, and agent export closes the disk-leakage window without sacrificing separated stdout/stderr adapter parsing. Splitting `schema.py`, `sanitize.py`, and `reporting.py` keeps the dependency DAG acyclic: `schema` -> `{sanitize, reporting, store}` -> `isolation` -> `{adapters, engine}` -> `{cli, mcp_server, tui}`.

## Tradeoffs and consequences

- **Easy**: Every module importing `schema.py` gets pure dataclasses with zero side effects; secrets accidentally echoed by a buggy test are scrubbed in RAM before touching the SSD.
- **Harder**: `isolation.py` must pump stdout/stderr pipes via `select`/`selectors` through `StreamRedactor` rather than passing raw file descriptors to `Popen`, adding ~50 lines of stream-pump code and bounded memory buffering per incomplete line.
- **Limitation**: Pattern-based `StreamRedactor` catches structured keys, tokens, bearers, URLs with userinfo, and PEM blocks, but cannot recognize an arbitrary unlabelled high-entropy passphrase printed without context; pre-spawn env scrubbing + Seatbelt file-read denial is the primary defense against real host credentials entering the child at all.

## Supporting evidence

- User architectural steer (2026-11-14): secret prevention vs disk storage vs agent export, and `schema.py` / `sanitize.py` / `reporting.py` separation.
- Precedent on this host: `~/Code/hermes-terminal-env/docs/OPERATING-GUIDE.md` (`0700` dir, `0600` DB, `com_apple_backup_excludeItem` Time Machine exclusion, two-stage ingest + draw-time sanitization).

## Verification status

verified (`/Users/sein/.local/bin/python3.14 -W default -m unittest tests.test_schema tests.test_sanitize tests.test_reporting tests.test_isolation -v` — 30/30 passed including AST import isolation on `schema.py` and chunk-split `StreamRedactor` pre-disk redaction, 2026-11-14)

## Related decisions

- ADR-0001: Refines the output capture and module layout of the TestBench Core.
- ADR-0002: Governs what `mcp_server.py` exports to LLM callers.
