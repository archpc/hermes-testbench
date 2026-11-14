# Hermes TestBench

Reusable testing, benchmarking, and verification foundation for multi-language repositories. One execution engine, one persisted result format, four surfaces: standalone CLI (`testbench`), first-class Hermes skill (`testbench`), local stdio MCP server (`testbench mcp`), and permanent `hx:testbench` tmux TUI (`hermes-tmux testbench`).

Zero third-party runtime dependencies (Python 3.11+ stdlib; tested on Python 3.14.7).

## Three Questions Answered

1. **Does it work?** Native runners (`Cargo`, `unittest`, `pytest`, `GnuCOBOL`, `clang` ASan/UBSan, SQLite `PRAGMA integrity_check`, PTY/tmux harnesses, and custom verifier scripts) execute once inside isolated `setsid` process groups, disposable `0700` `HOME`/`XDG`/`TMPDIR` overlays, disposable tmux sockets, and macOS Seatbelt confinement.
2. **How do we know?** Every run persists a structured row in `~/.local/share/testbench/state.db` (`WAL` mode) plus `runs/<run_id>/run.json`, `junit.xml`, and in-stream secret-redacted `*.sanitized.log` files (`ADR-0004`). Historical results are inspectable without re-running suites.
3. **Can we prove it again?** Every run captures git SHA (`+dirty` on uncommitted tracked edits), exact argv, environment provenance, and deterministic reproduction commands (`testbench reproduce <run_id>`).

## Quick Start

```sh
# Install CLI symlink (already wired on Honk)
ln -sf ~/Code/hermes-testbench/bin/testbench ~/.local/bin/testbench

# Discover suites in any repository (zero-config or .testbench.toml)
testbench discover ~/Code/hermes-terminal-env

# Execute a profile safely in isolation
testbench run ~/Code/hermes-terminal-env --profile terminal
testbench run . --profile unit

# Inspect historical results without re-running tests
testbench status --limit 15
testbench results --last --logs
testbench results <run_id> --format junit

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
/Users/sein/.local/bin/python3.14 -W default -m unittest discover -s tests -p "test_*.py" -v
testbench run ~/Code/hermes-testbench --profile regression
python3.14 ~/.hermes/skills/software-development/architecture-decisions/scripts/adr.py check --strict
```
