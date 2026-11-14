"""Canonical pure data contracts for Hermes TestBench.

Zero regex, XML, I/O, formatting, or runner dependencies (ADR-0004).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import os
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = 1


class RunStatus:
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"
    BLOCKED = "BLOCKED"
    TIMEOUT = "TIMEOUT"
    CANCELLED = "CANCELLED"
    ERROR = "ERROR"

    TERMINAL = frozenset({PASS, FAIL, SKIP, BLOCKED, TIMEOUT, CANCELLED, ERROR})
    ACTIVE = frozenset({QUEUED, RUNNING})
    ALL = TERMINAL | ACTIVE


class ProfileName:
    UNIT = "unit"
    INTEGRATION = "integration"
    TERMINAL = "terminal"
    SECURITY = "security"
    REGRESSION = "regression"
    BENCHMARK = "benchmark"
    AGENT_EVAL = "agent-eval"

    STANDARD = (UNIT, INTEGRATION, TERMINAL, SECURITY, REGRESSION, BENCHMARK, AGENT_EVAL)


class FailureKind:
    ASSERTION = "assertion"
    HARNESS_ERROR = "harness_error"
    SPAWN_ERROR = "spawn_error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    BLOCKED_POLICY = "blocked_policy"
    ABANDONED_WORKER = "abandoned_worker"
    UNTRUSTED_MANIFEST = "untrusted_manifest"


@dataclass
class SuiteSpec:
    name: str
    adapter: str
    profile: str
    command: List[str] = field(default_factory=list)
    cwd: str = "."
    timeout_s: float = 120.0
    needs_tmux: bool = False
    needs_pty: bool = False
    allow_network: bool = False
    allow_repo_write: bool = False
    billable: bool = False
    custom_from_manifest: bool = False
    env_overrides: Dict[str, str] = field(default_factory=dict)
    required_paths: List[str] = field(default_factory=list)
    required_binaries: List[str] = field(default_factory=list)
    description: str = ""
    warmup_runs: int = 1
    repeat_runs: int = 5


@dataclass
class ProjectSpec:
    name: str
    root: str
    languages: List[str] = field(default_factory=list)
    suites: List[SuiteSpec] = field(default_factory=list)
    manifest_path: Optional[str] = None
    manifest_sha256: Optional[str] = None
    trusted: bool = False
    revision: str = ""
    branch: str = ""


@dataclass
class FailureRecord:
    suite: str
    case_name: str
    kind: str = "assertion"
    message: str = ""
    file: Optional[str] = None
    line: Optional[int] = None
    excerpt: str = ""


@dataclass
class CaseResult:
    name: str
    suite: str
    status: str
    duration_ms: float = 0.0
    message: str = ""
    file: Optional[str] = None
    line: Optional[int] = None


@dataclass
class SuiteResult:
    name: str
    adapter: str
    profile: str
    command: List[str]
    cwd: str
    status: str
    exit_code: Optional[int] = None
    signal_num: Optional[int] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    duration_ms: float = 0.0
    total_count: int = 0
    pass_count: int = 0
    fail_count: int = 0
    skip_count: int = 0
    blocked_count: int = 0
    error_count: int = 0
    stdout_path: Optional[str] = None
    stderr_path: Optional[str] = None
    sanitized_path: Optional[str] = None
    summary_excerpt: str = ""
    repro_command: str = ""
    failures: List[FailureRecord] = field(default_factory=list)
    cases: List[CaseResult] = field(default_factory=list)
    isolation_meta: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunRecord:
    run_id: str
    project: str
    project_root: str
    revision: str
    profile: str
    status: str
    branch: str = ""
    failure_kind: Optional[str] = None
    commands: List[List[str]] = field(default_factory=list)
    created_at: str = ""
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    duration_ms: Optional[float] = None
    exit_code: Optional[int] = None
    total_count: int = 0
    pass_count: int = 0
    fail_count: int = 0
    skip_count: int = 0
    blocked_count: int = 0
    error_count: int = 0
    environment: Dict[str, Any] = field(default_factory=dict)
    evidence_dir: str = ""
    run_json_path: Optional[str] = None
    junit_xml_path: Optional[str] = None
    repro_path: Optional[str] = None
    summary: str = ""
    first_failure: str = ""
    repro_instructions: str = ""
    suites: List[SuiteResult] = field(default_factory=list)
    model_usage: Optional[Dict[str, Any]] = None
    delegation_id: Optional[str] = None
    worker_pid: Optional[int] = None
    worker_pgid: Optional[int] = None
    worker_start_time: Optional[float] = None
    run_nonce: str = ""
    pinned: bool = False


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"tb_{stamp}_{os.urandom(4).hex()}"


def run_to_dict(run: RunRecord) -> Dict[str, Any]:
    return asdict(run)


def _suite_from_dict(raw: Dict[str, Any]) -> SuiteResult:
    failures = [FailureRecord(**f) for f in (raw.get("failures") or [])]
    cases = [CaseResult(**c) for c in (raw.get("cases") or [])]
    copy = dict(raw)
    copy["failures"] = failures
    copy["cases"] = cases
    return SuiteResult(**copy)


def run_from_dict(data: Dict[str, Any]) -> RunRecord:
    raw = dict(data)
    raw["suites"] = [_suite_from_dict(s) for s in (raw.get("suites") or [])]
    if raw.get("model_usage") is None:
        raw["model_usage"] = None
    return RunRecord(**raw)
