# Hermes TestBench

Reusable testing, benchmarking, and verification foundation for multi-language repositories. One execution engine, one persisted result format, four surfaces: standalone CLI (`testbench`), first-class Hermes skill (`testbench`), local stdio MCP server (`testbench mcp`), and permanent `hx:testbench` tmux TUI (`hermes-tmux testbench`).

Zero third-party runtime dependencies (Python 3.11+ stdlib). macOS and Linux.

> **Linux has no filesystem or network confinement.** On macOS, suites run inside a Seatbelt (`sandbox-exec`) profile that denies writes outside the run sandbox, credential-file reads, and network access. Linux has no equivalent here: suites get environment scrubbing, disposable `HOME`/`TMPDIR`, and process-group cleanup only, and can read, write, and connect anywhere your user can. Do not run untrusted code with TestBench on Linux outside a container or VM. Details: `docs/OPERATING-GUIDE.md` §1.1.

## Three Questions Answered

1. **Does it work?** Native runners (`Cargo`, `unittest`, `pytest`, `GnuCOBOL`, `clang` ASan/UBSan, SQLite `PRAGMA integrity_check`, PTY/tmux harnesses, and custom verifier scripts) execute once inside isolated `setsid` process groups, disposable `0700` `HOME`/`XDG`/`TMPDIR` overlays, disposable tmux sockets, and macOS Seatbelt confinement (macOS only; see the Linux note above).
2. **How do we know?** Every run persists a structured row in `~/.local/share/testbench/state.db` (`WAL` mode) plus `runs/<run_id>/run.json`, `junit.xml`, and in-stream secret-redacted `*.sanitized.log` files (`ADR-0004`). Historical results are inspectable without re-running suites.
3. **Can we prove it again?** Every run captures git SHA (`+dirty` on uncommitted tracked edits), exact argv, environment provenance, and deterministic reproduction commands (`testbench reproduce <run_id>`).

## Quick Start

```sh
# Install from a checkout (isolated tool venv; the worker runs under that venv's Python)
uv tool install .            # or: pipx install .
testbench --version          # shows Python, schema, and sandbox backend (seatbelt | none)

# Discover suites in any repository (zero-config or .testbench.toml)
testbench discover ~/Code/hermes-terminal-env

# Execute a profile safely in isolation
testbench run ~/Code/hermes-terminal-env --profile terminal
testbench run . --profile unit

# Inspect historical results without re-running tests
testbench status --limit 15
testbench results --last --logs
testbench results <run_id> --format junit

# Hunt flakes: N independent attempts, classified per suite (and per case)
testbench flake run . --profile unit --repeats 5        # exit 0 only if all always_pass
testbench flake analyze --project myproj --revision abc1234 --json
testbench flake analyze --group fg_<run_id>            # one recorded group, no re-run

# Print or execute deterministic failure reproduction
testbench reproduce <run_id> --print-only
testbench reproduce <run_id>

# Run & compare repeatable benchmarks (separate from pass/fail gates)
testbench bench run . --name tui-100f --warmup 1 --repeats 5 -- python3.14 -c "..."
testbench bench compare <baseline_run_id> <candidate_run_id>

# Verify offline agent-messaging contract (zero LLM calls, zero gateway restarts)
testbench messaging-verify

# Focus (or open) permanent hx:testbench TUI inside hermes-tmux
hermes-tmux testbench
```

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), [docs/OPERATING-GUIDE.md](docs/OPERATING-GUIDE.md), and validated ADRs in [docs/adr/README.md](docs/adr/README.md):

- **ADR-0001**: Single stdlib core engine and WAL SQLite store backing CLI, MCP, skill, and TUI.
- **ADR-0002**: Local stdio MCP server with project trust registry and Seatbelt sandbox.
- **ADR-0003**: Restrained borderless `hx:testbench` TUI and permanent auxiliary window reuse.
- **ADR-0004**: In-stream secret redaction before disk write and strict schema/sanitize/reporting separation.

## Self-Verification

```sh
python3 -W default -m unittest discover -s tests -p "test_*.py" -v
testbench run ~/Code/hermes-testbench --profile regression
python3.14 ~/.hermes/skills/software-development/architecture-decisions/scripts/adr.py check --strict
```
