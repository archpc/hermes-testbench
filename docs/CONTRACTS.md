# Hermes TestBench Internal Module Contracts

Locked shared interfaces across `hermes_testbench` modules so parallel worktrees remain 100% disjoint:

- `hermes_testbench/schema.py` (committed): `RunStatus`, `ProfileName`, `FailureKind`, `SuiteSpec`, `ProjectSpec`, `FailureRecord`, `CaseResult`, `SuiteResult`, `RunRecord`, `utc_now_iso`, `new_run_id`, `run_to_dict`, `run_from_dict`.
- `hermes_testbench/sanitize.py` (committed): `redact_secrets`, `StreamRedactor`, `strip_terminal_escapes`, `sanitize_terminal_text`, `sanitize_for_agent`.
- `hermes_testbench/reporting.py` (committed): `render_junit_xml`, `parse_junit_xml_bounded`, `render_run_json`, `format_terminal_summary`, `format_repro_script`.
- `hermes_testbench/store.py` (committed): `TestBenchStore`, `SchemaMismatchError`, `default_testbench_home`, `atomic_write_file`.
- `hermes_testbench/isolation.py` (committed): `IsolatedExecResult`, `run_isolated_command`, `build_isolated_env`, `terminate_process_group`, `can_apply_seatbelt`, `real_user_home`, `default_scratch_root`.

## Disjoint Worktree Ownership

1. `feat/discovery-and-adapters`:
   - `hermes_testbench/discovery.py`: `detect_git_metadata(root: Path) -> tuple[str, str]`, `discover_project(root: Path, *, store: Optional[TestBenchStore] = None, external_manifest: Optional[Path] = None) -> ProjectSpec`.
   - `hermes_testbench/adapters/__init__.py` (+ per-runner modules): `execute_suite(suite: SuiteSpec, *, project_root: Path, evidence_dir: Path, run_id: str, trusted_manifest: bool = False, allow_billable: bool = False, max_tokens: int = 0, cancel_flag_path: Optional[Path] = None, scratch_root: Optional[Path] = None) -> tuple[SuiteResult, Optional[dict[str, Any]]]`.
   - `tests/test_discovery_and_adapters.py`

2. `feat/fixtures-bench-messaging`:
   - `hermes_testbench/fixtures.py`: deterministic corpus/fixture builders and fault probes (Unicode, malformed inputs, SQLite WAL contention, terminal escapes, PTY resize, signals/crashes, repeated init, tmux lifecycle, cross-language SQLite mail schema).
   - `hermes_testbench/benchmark.py`: `collect_environment_metadata`, `compute_stats`, `run_workload`, `compare_benchmarks`, `format_benchmark_comparison`.
   - `hermes_testbench/messaging_suite.py`: `run_agent_messaging_verification` covering queued vs settled, attribution, provider failures, invalid targets, duplicate sends, sender restarts, notification handling.
   - `tests/test_fixtures_and_bench.py`, `tests/test_messaging_suite.py`

3. `feat/tui-renderer`:
   - `hermes_testbench/tui_render.py`: `TuiState`, `TuiAction`, `Palette`, `display_width`, `fit_cells`, `handle_key`, `render_frame`, `format_ansi_frame`.
   - `tests/test_tui_render.py`

4. `main` (Carbon parent):
   - `hermes_testbench/engine.py`, `hermes_testbench/cli.py`, `hermes_testbench/__main__.py`, `hermes_testbench/mcp_server.py`, `hermes_testbench/tui.py`, `bin/testbench`.
   - `tests/test_engine_and_cli.py`, `tests/test_mcp_server.py`, `tests/test_tui_pty.py`.
