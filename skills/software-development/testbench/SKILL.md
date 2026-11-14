---
name: testbench
description: "Run isolated tests, benchmarks, and repros via TestBench."
version: 0.1.0
author: Sein, Hermes Agent
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [testing, verification, benchmarking, isolation, reproduction, tui]
    related_skills: [architecture-decisions, test-driven-development, systematic-debugging, verification-before-completion]
---

# Hermes TestBench

Standalone testing, benchmarking, and verification foundation for multi-language repositories. Executes native project suites (`Cargo`, `unittest`, `pytest`, `GnuCOBOL`, `clang` ASan/UBSan, SQLite integrity, PTY/tmux harnesses, and verifier scripts) inside disposable process-group, `HOME`/`TMPDIR`, tmux-socket, and macOS Seatbelt sandboxes while persisting sanitized evidence, JUnit/JSON reports, and reproduction commands.

## When to Use

- Entering an unfamiliar repository and discovering how its tests run.
- Verifying implementations or refactors with persisted, auditable evidence.
- Reproducing a test failure or inspecting historical results without re-running expensive suites.
- Running adversarial security, escape, sanitizer, or SQLite contention checks.
- Measuring and comparing repeatable benchmark workloads separate from pass/fail correctness gates.
- Verifying `agent-messaging` contracts offline without live LLM calls or gateway restarts.
- Opening or focusing the live `hx:testbench` monitoring tab inside `hermes-tmux`.

Don't use for: replacing a repository's native test framework, or claiming verification without inspecting TestBench's exit status and persisted run summary.

## Prerequisites

- `testbench` CLI on `PATH` (`~/.local/bin/testbench`).
- Optional: `hermes-tmux` inside tmux for the `hx:testbench` interactive monitor.

## How to Run

Invoke `testbench` through the `terminal` tool:

```
terminal(command="testbench discover .")
terminal(command="testbench run . --profile unit", timeout=300)
terminal(command="testbench results --last --logs")
```

## Quick Reference

```
testbench discover [path] [--json]
testbench run [path] [--profile unit|integration|terminal|security|regression] [--suite NAME] [--async] [--json]
testbench status [run_id] [--project NAME] [--json]
testbench results [run_id] [--last] [--project NAME] [--json | --format junit | --logs]
testbench reproduce <run_id> [--print-only]
testbench cancel <run_id>
testbench bench run [path] --name WORKLOAD [--warmup 1] [--repeats 5] -- <cmd...>
testbench bench compare <baseline_run_id> <candidate_run_id> [--json]
testbench messaging-verify [--json]
testbench trust [path]
hermes-tmux testbench
```

## Procedure

1. **Inspect existing runs before re-running.** Call `terminal(command="testbench status --limit 10")`. If a matching run for the current git revision already exists, inspect it with `terminal(command="testbench results <run_id> --logs")` instead of re-executing an expensive suite.
2. **Discover project suites.** Call `terminal(command="testbench discover <repo>")` to inspect detected languages, profiles, isolation flags (`tmux`, `pty`, `billable-opt-in`), and manifest trust state.
3. **Execute the narrowest sufficient profile.** Run `unit` during tight TDD loops, `terminal` for PTY/tmux changes, `security` for sanitizers/confinement, and `regression` before final sign-off. For suites over 60 seconds, pass `--async` and check `testbench status <run_id>` rather than sleep-looping.
4. **Reproduce and diagnose failures.** Inspect sanitized failure excerpts with `terminal(command="testbench results <run_id> --logs")` and extract or replay exact commands with `terminal(command="testbench reproduce <run_id> --print-only")`.
5. **Benchmark separately from correctness.** Measure workloads with `testbench bench run` and compare `p50_ms`, `p95_ms`, `iqr_ms`, and environment drift via `testbench bench compare`. Never apply arbitrary pass/fail percentage thresholds to benchmark noise.

## Pitfalls

- **Billable suites are blocked by default.** Profiles marked `billable = true` (`agent-eval`) return `BLOCKED` unless `--allow-billable --max-tokens <N>` is explicitly passed with user authorization. Never silently trigger model-backed evaluations.
- **Untrusted `.testbench.toml` confinement.** Repository manifests requesting `allow_network = true` or `allow_repo_write = true` return `BLOCKED` until explicitly approved via `testbench trust <repo>`.
- **Pipe exit codes.** Never pipe `testbench run` through a shell filter; read `testbench`'s direct exit code and inspect logs afterward with `testbench results <run_id> --logs` or `read_file`.
- **Owned cancellation only.** `testbench cancel <run_id>` only terminates TestBench-owned run locks and process groups; never kill arbitrary PIDs by hand.

## Verification

Every verification report backed by TestBench must cite:
1. Exact `run_id` and git `revision` (including `+dirty` when uncommitted tracked changes exist).
2. Status (`PASS`, `FAIL`, `SKIP`, `BLOCKED`, `TIMEOUT`, `CANCELLED`) and exact counts (`pass`, `fail`, `skip`).
3. Evidence directory path (`runs/<run_id>/run.json` and `junit.xml`).
