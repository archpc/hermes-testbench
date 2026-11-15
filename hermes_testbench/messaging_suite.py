"""Offline, deterministic verification of the ``agent-messaging`` contract.

Zero live LLM calls and zero live gateway restarts. The suite reads the Hermes source
tree read-only, inspects the artifact's evidence receipts, and drives the artifact's
``scripts/check-wiring.py`` against a synthetic ``--hermes-home`` fixture.

``run_agent_messaging_verification`` returns ``SuiteResult`` objects:
  * ``agent-messaging:contract`` — one case per required behavior (a-g) plus an explicit
    SKIP for mid-flight live gateway restart (blocked by the in-gateway restart guard).
  * ``agent-messaging:wiring`` — synthetic wired/unwired ``--hermes-home`` fixtures when
    ``artifact_root`` exists.
  * ``agent-messaging:pytest`` — only when ``run_hermes_pytest=True``; a SKIP when the
    hermetic bot-mode pytest files cannot be run.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .sanitize import redact_secrets, sanitize_terminal_text
from .schema import CaseResult, FailureKind, FailureRecord, RunStatus, SuiteResult, utc_now_iso
from .store import atomic_write_file

CONTRACT_SUITE = "agent-messaging:contract"
WIRING_SUITE = "agent-messaging:wiring"
PYTEST_SUITE = "agent-messaging:pytest"

_DELIVERY_ID_RE = re.compile(r"[0-9a-f]{32,64}")

_TERMINAL_STATUS_NAMES = frozenset({"settled", "failed", "cancelled", "ambiguous"})

_EXPECTED_REASONS = frozenset(
    {
        "runtime_offline",
        "queued_expired",
        "delivery_timeout",
        "agent_blocked",
        "cancelled",
        "provider_auth_or_access",
        "provider_quota_limit",
        "provider_rate_limit",
        "provider_server_error",
        "context_overflow",
        "missing_config",
        "model_unavailable",
        "target_scope_unresolved",
        "unknown",
    }
)

# Name -> (title, behavior family) for every required contract behavior.
_REQUIRED_CASES: Tuple[str, ...] = (
    "a1_ack_is_queued_handoff_with_delivery_id",
    "a2_ack_documented_as_handoff_not_receipt",
    "a3_terminal_status_vocabulary_includes_settled",
    "a4_evidence_settled_receipt_carries_reply",
    "b1_server_side_attribution_prefix",
    "b2_author_envelope_is_bot_typed",
    "b3_peer_author_envelope_host_qualified",
    "b4_evidence_author_envelope",
    "c1_closed_failure_reason_vocabulary",
    "c2_provider_quota_limit_is_not_auto_retryable",
    "c3_rate_limit_and_runtime_offline_are_retryable",
    "c4_ambiguous_means_do_not_resend",
    "c5_held_not_sent_outside_all_reasons",
    "c6_evidence_quota_failure_is_typed",
    "d1_structured_refusal_payload_with_live_roster",
    "d2_ambiguous_target_produces_disambiguation_forms",
    "e1_bot_turn_duplicate_send_is_refused",
    "e2_hop_limit_enforced",
    "e3_human_turn_gap_documented",
    "f1_pinned_live_intent_file",
    "f2_retry_inspects_and_never_retransports",
    "f3_restart_regression_test_exists",
    "g1_completion_notification_is_the_settle_channel",
    "g2_durable_process_complete_transcript_row",
    "g3_durable_receipt_directory",
    "g4_truncated_notification_falls_back_to_receipt_file",
)

RESTART_SKIP_CASE = "h1_midflight_gateway_restart_unverified"


def delivery_receipt_path(profile_home: Path, delivery_id: str) -> Path:
    """Return the durable receipt location for a validated delivery id."""
    if not isinstance(delivery_id, str) or _DELIVERY_ID_RE.fullmatch(delivery_id) is None:
        raise ValueError("delivery id must be 32 to 64 lowercase hex characters")
    return (
        Path(profile_home).resolve()
        / "runtime"
        / "bot_live_delivery"
        / f"{delivery_id}.json"
    )


def settlement_from_notification(
    text: str, *, profile_home: Path, delivery_id: str
) -> Dict[str, Any]:
    """Resolve settlement from the notification text, then the durable receipt file."""
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except (ValueError, TypeError):
            continue
        if isinstance(payload, dict) and payload.get("delivery_id") == delivery_id:
            resolved = dict(payload)
            resolved["source"] = "notification"
            return resolved

    receipt_path = delivery_receipt_path(profile_home, delivery_id)
    if receipt_path.is_file():
        try:
            payload = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            payload = None
        if isinstance(payload, dict):
            resolved = dict(payload)
            resolved["delivery_id"] = delivery_id
            resolved.setdefault("status", "ambiguous")
            resolved["source"] = "receipt_file"
            return resolved

    return {
        "delivery_id": delivery_id,
        "status": "ambiguous",
        "source": "missing",
    }


# --------------------------------------------------------------------------- helpers


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _parse_frozenset(text: str, name: str) -> Optional[str]:
    match = re.search(rf"{name}\s*=\s*frozenset\(\{{(.*?)\}}\)", text, re.S)
    return match.group(1) if match else None


def _declared_strings(text: str) -> Dict[str, str]:
    return {
        match.group(1): match.group(2)
        for match in re.finditer(r"^([A-Z][A-Z0-9_]*)\s*=\s*\"([^\"]*)\"", text, re.M)
    }


def _receipts(artifact_root: Optional[Path], evidence_dir: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    candidates: List[Path] = []
    if artifact_root is not None:
        candidates.append(Path(artifact_root) / "evidence")
    candidates.append(Path(evidence_dir))
    seen: set = set()
    for directory in candidates:
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.json")):
            if path in seen:
                continue
            seen.add(path)
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(payload, dict):
                out.append(payload)
    return out


def _result(
    name: str, ok: bool, message: str, suite: str = CONTRACT_SUITE
) -> CaseResult:
    return CaseResult(
        name=name,
        suite=suite,
        status=RunStatus.PASS if ok else RunStatus.FAIL,
        message="" if ok else message,
    )


def _skipped(name: str, message: str, suite: str = CONTRACT_SUITE) -> CaseResult:
    return CaseResult(name=name, suite=suite, status=RunStatus.SKIP, message=message)


def _status_for(pass_count: int, fail_count: int, skip_count: int, error_count: int) -> str:
    if fail_count or error_count:
        return RunStatus.FAIL
    if pass_count == 0 and skip_count:
        return RunStatus.SKIP
    return RunStatus.PASS


def _suite(
    name: str,
    *,
    adapter: str,
    profile: str,
    cwd: Path,
    cases: List[CaseResult],
    command: Optional[Sequence[str]] = None,
    summary: str = "",
    repro_command: str = "testbench messaging-verify",
    evidence_dir: Optional[Path] = None,
    isolation_meta: Optional[Dict[str, Any]] = None,
) -> SuiteResult:
    pass_count = sum(1 for c in cases if c.status == RunStatus.PASS)
    fail_count = sum(1 for c in cases if c.status == RunStatus.FAIL)
    skip_count = sum(1 for c in cases if c.status == RunStatus.SKIP)
    error_count = sum(1 for c in cases if c.status == RunStatus.ERROR)
    failures = [
        FailureRecord(
            suite=name,
            case_name=c.name,
            kind=FailureKind.ASSERTION if c.status == RunStatus.FAIL else FailureKind.HARNESS_ERROR,
            message=c.message or c.status,
        )
        for c in cases
        if c.status in (RunStatus.FAIL, RunStatus.ERROR)
    ]
    sanitized_path: Optional[str] = None
    if evidence_dir is not None:
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "messaging"
        log_file = Path(evidence_dir) / f"{slug}.sanitized.log"
        log_lines = [
            f"{c.status} {c.name}" + (f": {c.message}" if c.message else "")
            for c in cases
        ]
        if summary:
            log_lines.append(summary)
        clean_log = sanitize_terminal_text(redact_secrets("\n".join(log_lines) + "\n"))
        atomic_write_file(log_file, clean_log, mode=0o600)
        sanitized_path = str(log_file)
    meta = {
        "mode": "offline_in_process",
        "allow_network": False,
        "allow_repo_write": False,
    }
    if isolation_meta:
        meta.update(isolation_meta)
    return SuiteResult(
        name=name,
        adapter=adapter,
        profile=profile,
        command=list(command or ["testbench", "messaging-verify"]),
        cwd=str(cwd),
        status=_status_for(pass_count, fail_count, skip_count, error_count),
        started_at=utc_now_iso(),
        finished_at=utc_now_iso(),
        total_count=len(cases),
        pass_count=pass_count,
        fail_count=fail_count,
        skip_count=skip_count,
        error_count=error_count,
        sanitized_path=sanitized_path,
        summary_excerpt=summary,
        repro_command=repro_command,
        failures=failures,
        cases=cases,
        isolation_meta=meta,
    )


# ----------------------------------------------------------------------- contract


def _contract_cases(
    hermes_agent_root: Optional[Path],
    artifact_root: Optional[Path],
    evidence_dir: Path,
) -> List[CaseResult]:
    if hermes_agent_root is None or not Path(hermes_agent_root).is_dir():
        message = "source tree absent: the Hermes source checkout was not provided"
        return [_skipped(name, message) for name in (*_REQUIRED_CASES, RESTART_SKIP_CASE)]

    root = Path(hermes_agent_root)
    dm_src = _read_text(root / "tools" / "bot_mode_dm.py")
    reasons_src = _read_text(root / "tools" / "bot_failure_reasons.py")
    delivery_src = _read_text(root / "tools" / "bot_live_delivery.py")
    chain_src = _read_text(root / "agent" / "bot_chain.py")
    regression_src = _read_text(root / "tests" / "tools" / "test_bot_mode_dm.py")

    terminal_block = _parse_frozenset(delivery_src, "_TERMINAL")
    declared = _declared_strings(reasons_src)
    all_reasons = _parse_frozenset(reasons_src, "ALL_REASONS")
    auto_retry = _parse_frozenset(reasons_src, "AUTO_RETRYABLE")

    receipts = _receipts(artifact_root, evidence_dir)
    settled = [r for r in receipts if r.get("status") == "settled"]
    failed = [r for r in receipts if r.get("status") == "failed"]
    readme = ""
    if artifact_root is not None:
        readme = _read_text(Path(artifact_root) / "README.md")
    artifact_ready = artifact_root is not None and Path(artifact_root).is_dir()

    cases: List[CaseResult] = []

    cases.append(
        _result(
            "a1_ack_is_queued_handoff_with_delivery_id",
            '"status": "queued"' in dm_src and '"delivery_id"' in dm_src,
            "ack payload must carry status=queued and delivery_id",
        )
    )
    cases.append(
        _result(
            "a2_ack_documented_as_handoff_not_receipt",
            "not a delivery receipt" in dm_src,
            "ack detail must document that queued is a handoff, not a delivery receipt",
        )
    )
    cases.append(
        _result(
            "a3_terminal_status_vocabulary_includes_settled",
            terminal_block is not None and "settled" in terminal_block,
            "terminal status vocabulary must include settled",
        )
    )
    if not artifact_ready:
        cases.append(_skipped("a4_evidence_settled_receipt_carries_reply", "artifact evidence absent"))
    else:
        ok = any(str(r.get("reply") or "").strip() for r in settled)
        cases.append(
            _result(
                "a4_evidence_settled_receipt_carries_reply",
                ok,
                "settled evidence receipt must carry the reply text",
            )
        )

    cases.append(
        _result(
            "b1_server_side_attribution_prefix",
            "Message from" in dm_src and "(@{" in dm_src,
            "sender attribution prefix must be built server-side",
        )
    )
    cases.append(
        _result(
            "b2_author_envelope_is_bot_typed",
            '"is_bot": True' in dm_src and '"bot:' in dm_src,
            "author envelope must be bot-typed with a bot: id",
        )
    )
    cases.append(
        _result(
            "b3_peer_author_envelope_host_qualified",
            "bot_author_id(" in dm_src and "local_origin()" in dm_src,
            "peer author envelope must be host-qualified via bot_author_id/local_origin",
        )
    )
    if not artifact_ready:
        cases.append(_skipped("b4_evidence_author_envelope", "artifact evidence absent"))
    else:
        ok = any(
            isinstance(r.get("author"), dict)
            and r["author"].get("is_bot") is True
            and str(r["author"].get("id", "")).startswith("bot:")
            for r in receipts
        )
        cases.append(
            _result(
                "b4_evidence_author_envelope",
                ok,
                "evidence receipt author envelope must be bot-typed with a bot: id",
            )
        )

    literal_reasons = {v for v in declared.values()}
    cases.append(
        _result(
            "c1_closed_failure_reason_vocabulary",
            _EXPECTED_REASONS <= literal_reasons,
            "failure reason vocabulary is missing: "
            + ", ".join(sorted(_EXPECTED_REASONS - literal_reasons)),
        )
    )
    quota_ok = auto_retry is not None and "PROVIDER_QUOTA_LIMIT" not in auto_retry
    cases.append(
        _result(
            "c2_provider_quota_limit_is_not_auto_retryable",
            quota_ok,
            "provider_quota_limit must not be auto-retryable; AUTO_RETRYABLE="
            + (auto_retry.strip() if auto_retry is not None else "<unparsed>"),
        )
    )
    retry_ok = (
        auto_retry is not None
        and "PROVIDER_RATE_LIMIT" in auto_retry
        and "RUNTIME_OFFLINE" in auto_retry
    )
    cases.append(
        _result(
            "c3_rate_limit_and_runtime_offline_are_retryable",
            retry_ok,
            "provider_rate_limit and runtime_offline must remain auto-retryable",
        )
    )
    cases.append(
        _result(
            "c4_ambiguous_means_do_not_resend",
            '"status": "ambiguous"' in dm_src and "Do not resend" in dm_src,
            "ambiguous outcome must instruct the caller not to resend",
        )
    )
    held_ok = (
        declared.get("HELD_NOT_SENT") == "runtime_coordination_required"
        and all_reasons is not None
        and "HELD_NOT_SENT" not in all_reasons
    )
    cases.append(
        _result(
            "c5_held_not_sent_outside_all_reasons",
            held_ok,
            "runtime_coordination_required must stay outside ALL_REASONS",
        )
    )
    if not artifact_ready:
        cases.append(_skipped("c6_evidence_quota_failure_is_typed", "artifact evidence absent"))
    else:
        ok = any(
            r.get("reason") == "provider_quota_limit" and r.get("status") == "failed"
            for r in failed
        )
        cases.append(
            _result(
                "c6_evidence_quota_failure_is_typed",
                ok,
                "failed evidence receipt must be typed provider_quota_limit",
            )
        )

    cases.append(
        _result(
            "d1_structured_refusal_payload_with_live_roster",
            '"teammates"' in dm_src and '"peers"' in dm_src and '"error"' in dm_src,
            "refusal payload must carry error plus live roster/peer listings",
        )
    )
    cases.append(
        _result(
            "d2_ambiguous_target_produces_disambiguation_forms",
            'match == "ambiguous"' in dm_src and "disambiguate with one of" in dm_src,
            "ambiguous targets must return disambiguation forms",
        )
    )

    cases.append(
        _result(
            "e1_bot_turn_duplicate_send_is_refused",
            'if bot_turn and ask in state["asks"]:' in chain_src
            and "You already sent this exact message" in chain_src,
            "duplicate-send guard must be scoped to bot turns",
        )
    )
    cases.append(
        _result(
            "e2_hop_limit_enforced",
            "hop > settings.max_hops" in chain_src,
            "bot-to-bot hop limit must be enforced",
        )
    )
    if not artifact_ready:
        cases.append(_skipped("e3_human_turn_gap_documented", "artifact README absent"))
    else:
        ok = "known gap" in readme.lower() and "bot-turn only" in readme.lower()
        cases.append(
            _result(
                "e3_human_turn_gap_documented",
                ok,
                "artifact README must document the human-turn dedupe gap",
            )
        )

    cases.append(
        _result(
            "f1_pinned_live_intent_file",
            "O_EXCL" in dm_src and ".live.json" in dm_src,
            "live admission must pin intent with an O_EXCL intent file",
        )
    )
    cases.append(
        _result(
            "f2_retry_inspects_and_never_retransports",
            "never change transport" in dm_src,
            "retry path must inspect pinned intent and never re-transport",
        )
    )
    cases.append(
        _result(
            "f3_restart_regression_test_exists",
            "never_reexecutes_failed_admission" in regression_src,
            "a restart regression test for failed admission must exist",
        )
    )

    cases.append(
        _result(
            "g1_completion_notification_is_the_settle_channel",
            '"reply_delivery"' in dm_src and '"notification"' in dm_src and "notify_on_complete" in dm_src,
            "completion notification must be the settle channel",
        )
    )
    cases.append(
        _result(
            "g2_durable_process_complete_transcript_row",
            "process_complete" in dm_src,
            "processor completion must persist a process_complete transcript row",
        )
    )
    cases.append(
        _result(
            "g3_durable_receipt_directory",
            'DELIVERY_DIR_NAME = "bot_live_delivery"' in delivery_src
            and '"runtime"' in delivery_src,
            "durable receipts must live under runtime/bot_live_delivery",
        )
    )

    # g4 exercises this module's own fallback against a synthetic durable receipt.
    delivery_id = "ab" * 32
    evidence_dir.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix="tb-g4-", dir=str(evidence_dir)) as tmp:
            profile_home = Path(tmp) / "profile"
            receipt = delivery_receipt_path(profile_home, delivery_id)
            receipt.parent.mkdir(parents=True, exist_ok=True)
            receipt.write_text(
                json.dumps({"status": "settled", "delivery_id": delivery_id, "reply": "ok"}),
                encoding="utf-8",
            )
            fallback = settlement_from_notification(
                "runner exited; [output truncated at 4096 chars]",
                profile_home=profile_home,
                delivery_id=delivery_id,
            )
        g4_ok = fallback.get("source") == "receipt_file" and fallback.get("status") == "settled"
        g4_message = "truncated notification must fall back to the durable receipt file"
    except OSError as exc:  # pragma: no cover - scratch unwritable
        g4_ok = False
        g4_message = f"receipt fallback probe could not run: {exc}"
    cases.append(_result("g4_truncated_notification_falls_back_to_receipt_file", g4_ok, g4_message))

    cases.append(
        _skipped(
            RESTART_SKIP_CASE,
            "mid-flight live gateway restart not verified: blocked by the in-gateway restart guard",
        )
    )
    return cases


# ------------------------------------------------------------------------- wiring


def _run_wiring_script(script: Path, home: Path) -> Tuple[int, str]:
    try:
        proc = subprocess.run(
            [sys.executable, str(script), "--hermes-home", str(home)],
            capture_output=True,
            text=True,
            timeout=120.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return 127, f"wiring script could not run: {exc}"
    return int(proc.returncode), (proc.stdout or "") + (proc.stderr or "")


def _wiring_suite(artifact_root: Path, evidence_dir: Path) -> Optional[SuiteResult]:
    root = Path(artifact_root)
    script = root / "scripts" / "check-wiring.py"
    if not script.is_file():
        return None

    from contextlib import closing
    import sqlite3
    cases: List[CaseResult] = []
    evidence_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tb-wiring-", dir=str(evidence_dir)) as tmp:
        base = Path(tmp)
        wired = base / "wired-home"
        (wired / "profiles" / "opus").mkdir(parents=True, exist_ok=True)
        (wired / "profiles" / "opus" / "config.yaml").write_text(
            "model:\n  default: anthropic/claude-opus-5.5\n", encoding="utf-8"
        )
        (wired / "profile.yaml").write_text("ui_meta:\n  hermes-bots:\n    enabled: true\n", encoding="utf-8")
        for db_dir in (wired, wired / "profiles" / "opus"):
            with closing(sqlite3.connect(str(db_dir / "state.db"))) as conn:
                conn.execute("CREATE TABLE sessions (id TEXT, title TEXT, started_at INTEGER)")
                conn.execute("INSERT INTO sessions VALUES ('s1', 'Bot Chat', 1)")
                conn.commit()
        unwired = base / "unwired-home"
        unwired.mkdir(parents=True, exist_ok=True)

        rc_wired, out_wired = _run_wiring_script(script, wired)
        cases.append(
            _result(
                "w1_synthetic_wired_home_passes",
                rc_wired == 0 and "wiring OK" in out_wired,
                f"synthetic wired home was not accepted (rc={rc_wired}): {out_wired.strip()[:400]}",
                suite=WIRING_SUITE,
            )
        )

        rc_unwired, out_unwired = _run_wiring_script(script, unwired)
        cases.append(
            _result(
                "w2_missing_bot_mode_marker_fails",
                rc_unwired != 0 and "advisor profile missing" in out_unwired,
                f"unwired home was not flagged (rc={rc_unwired}): {out_unwired.strip()[:400]}",
                suite=WIRING_SUITE,
            )
        )

    return _suite(
        WIRING_SUITE,
        adapter="wiring",
        profile="integration",
        cwd=root,
        cases=cases,
        command=[sys.executable, str(script)],
        summary="wiring: artifact check-wiring.py exercised against synthetic wired and unwired homes",
        evidence_dir=evidence_dir,
    )


# ------------------------------------------------------------------------- pytest


def _pytest_suite(hermes_agent_root: Optional[Path], evidence_dir: Optional[Path] = None) -> SuiteResult:
    root = Path(hermes_agent_root) if hermes_agent_root is not None else None
    command: List[str] = []
    if root is not None:
        command = [str(root / "venv" / "bin" / "python"), "-m", "pytest", "-q", "tests/tools/test_bot_mode_dm.py"]
    if root is None or not (root / "venv" / "bin" / "python").is_file():
        case = _skipped(
            "pytest_bot_mode_dm",
            "hermetic bot-mode pytest files cannot run: source-tree venv/bin/python absent",
            suite=PYTEST_SUITE,
        )
        return _suite(
            PYTEST_SUITE,
            adapter="pytest",
            profile="integration",
            cwd=root or Path("."),
            cases=[case],
            command=command,
            evidence_dir=evidence_dir,
            summary="pytest: skipped (no source-tree venv/bin/python)",
        )

    try:
        proc = subprocess.run(
            command,
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=900.0,
            check=False,
        )
        rc = int(proc.returncode)
        output = (proc.stdout or "") + (proc.stderr or "")
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover
        rc = 127
        output = f"pytest could not run: {exc}"

    case = _result(
        "pytest_bot_mode_dm",
        rc == 0,
        f"pytest exit={rc}: {output.strip()[:600]}",
        suite=PYTEST_SUITE,
    )
    res = _suite(
        PYTEST_SUITE,
        adapter="pytest",
        profile="integration",
        cwd=root,
        cases=[case],
        command=command,
        evidence_dir=evidence_dir,
        summary=f"pytest: hermetic bot-mode tests exit={rc}",
    )
    if evidence_dir is not None and output.strip():
        log_file = Path(evidence_dir) / "agent-messaging_pytest.sanitized.log"
        clean_out = sanitize_terminal_text(redact_secrets(output))
        atomic_write_file(log_file, clean_out, mode=0o600)
        res.sanitized_path = str(log_file)
    return res


# ---------------------------------------------------------------------- entrypoint


def run_agent_messaging_verification(
    evidence_dir: Path,
    *,
    artifact_root: Optional[Path] = None,
    hermes_agent_root: Optional[Path] = None,
    run_hermes_pytest: bool = False,
) -> List[SuiteResult]:
    """Run the offline agent-messaging verification suites in deterministic order."""
    evidence = Path(evidence_dir)
    evidence.mkdir(parents=True, exist_ok=True)

    suites: List[SuiteResult] = []

    repro = "testbench messaging-verify --hermes-pytest" if run_hermes_pytest else "testbench messaging-verify"
    contract_cases = _contract_cases(hermes_agent_root, artifact_root, evidence)
    contract_cwd = Path(hermes_agent_root) if hermes_agent_root is not None else evidence
    suites.append(
        _suite(
            CONTRACT_SUITE,
            adapter="contract",
            profile="integration",
            cwd=contract_cwd,
            cases=contract_cases,
            command=repro.split(),
            repro_command=repro,
            evidence_dir=evidence,
            summary=(
                f"contract: {len(_REQUIRED_CASES)} required behaviors inspected offline; "
                "mid-flight gateway restart explicitly skipped"
            ),
        )
    )

    if artifact_root is not None and Path(artifact_root).is_dir():
        wiring = _wiring_suite(Path(artifact_root), evidence)
        if wiring is not None:
            wiring.repro_command = repro
            suites.append(wiring)

    if run_hermes_pytest:
        pt_suite = _pytest_suite(hermes_agent_root, evidence_dir=evidence)
        pt_suite.repro_command = repro
        suites.append(pt_suite)

    return suites
