"""Declarative terminal tapes: scripted PTY replay with explicitly approved goldens.

A tape is a line-oriented script (``.tape``) or a JSON document describing a real
program driven through :class:`pty_session.PtySession`. Replay is deterministic:
every wait has an explicit condition and every assertion is computed from the
final VT screen or the captured byte stream.

Goldens are read-only during ``tape run``. A golden that is missing, unapproved,
stale, or different never rewrites the repository: the run writes
``NAME.actual.screen.txt`` and ``NAME.golden.diff`` into the evidence directory
and fails. Approving is a separate, interactive, human-gated action
(:func:`approve_golden`), refused inside a sandbox and without a TTY so suites
and background agents can never silently self-approve.

VHS/browser/screenshot backends are refused with ``FailureKind.BLOCKED_POLICY``:
host-side-effect isolation for them is not established.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import difflib
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .pty_session import PtySession, PtyTimeout
from .sanitize import StreamRedactor, sanitize_terminal_text
from .schema import (
    FailureKind,
    FailureRecord,
    ProfileName,
    RunRecord,
    RunStatus,
    SuiteResult,
    new_run_id,
    utc_now_iso,
)
from .store import TestBenchStore, atomic_write_file

TAPE_VERSION = 1
VHS_DEFERRAL_MESSAGE = "VHS backend deferred: host-side-effect isolation not established"

_BLOCKED_BACKENDS = frozenset(
    {
        "vhs",
        "vhs-cli",
        "browser",
        "chrome",
        "chromium",
        "headless-chrome",
        "playwright",
        "screenshot",
        "web",
    }
)

DEFAULT_WAITFOR_MS = 5000
DEFAULT_WAITQUIET_MS = 250
DEFAULT_TIMEOUT_S = 120.0
_GOLDEN_QUIET_MS = 150


class TapeError(ValueError):
    """Base class for tape parse/replay errors (ValueError keeps CLI friendly)."""

    failure_kind = FailureKind.HARNESS_ERROR


class TapeParseError(TapeError):
    """Malformed tape script."""


class TapePolicyError(TapeError):
    """A requested backend violates TestBench policy."""

    failure_kind = FailureKind.BLOCKED_POLICY


class TapeAssertionError(TapeError):
    """An expectation or golden comparison did not hold."""

    failure_kind = FailureKind.ASSERTION


class TapeTimeout(TapeError):
    """A wait exceeded its deadline."""

    failure_kind = FailureKind.TIMEOUT


class GoldenApprovalRefused(TapeError):
    """Golden approval was refused (no TTY, sandboxed, or declined)."""


@dataclass
class TapeStep:
    action: str
    text: str = ""
    mode: str = "contains"
    cols: int = 0
    rows: int = 0
    timeout_ms: int = 0
    quiet_ms: int = 0
    name: str = ""


@dataclass
class TapeScript:
    name: str
    cols: int
    rows: int
    command: List[str]
    steps: List[TapeStep]
    path: Path
    sha256: str
    source_format: str = "dsl"

    def golden_step(self) -> Optional[TapeStep]:
        for step in self.steps:
            if step.action == "golden":
                return step
        return None

    def expectations(self) -> List[TapeStep]:
        return [s for s in self.steps if s.action in ("expectscreen", "expectraw", "golden")]


# --------------------------------------------------------------------------- parse


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", (name or "tape").strip())[:96] or "tape"


def _unquote(token: str) -> str:
    """Decode a quoted tape argument, honouring JSON escapes when quoted."""
    token = token.strip()
    if len(token) >= 2 and token[0] == '"' and token[-1] == '"':
        try:
            return json.loads(token)
        except json.JSONDecodeError as exc:
            raise TapeParseError(f"invalid quoted literal {token!r}: {exc}") from exc
    return token


def _split_first_arg(text: str) -> Tuple[str, str]:
    """Split ``"quoted literal" rest`` into ``(literal, rest)`` keeping escapes."""
    text = text.strip()
    if not text:
        return "", ""
    if text.startswith('"'):
        index = 1
        while index < len(text):
            if text[index] == "\\":
                index += 2
                continue
            if text[index] == '"':
                return text[: index + 1], text[index + 1 :].strip()
            index += 1
        raise TapeParseError(f"unterminated quoted literal: {text!r}")
    head, _, tail = text.partition(" ")
    return head, tail.strip()


def _as_int(token: str, what: str) -> int:
    try:
        return int(token)
    except (TypeError, ValueError) as exc:
        raise TapeParseError(f"{what} must be an integer, got {token!r}") from exc


def _require_backend_allowed(backend: str, line_no: int) -> None:
    if backend.strip().lower() in _BLOCKED_BACKENDS:
        raise TapePolicyError(f"line {line_no}: {VHS_DEFERRAL_MESSAGE} (requested backend {backend!r})")


def _parse_dsl(raw: str, path: Path) -> TapeScript:
    name = path.stem
    cols, rows = 80, 24
    command: List[str] = []
    steps: List[TapeStep] = []
    golden_seen = False
    for line_no, raw_line in enumerate(raw.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        head, _, rest = line.partition(" ")
        action = head.strip().lower()
        rest = rest.strip()

        if action in ("vhs", "browser", "screenshot"):
            raise TapePolicyError(f"line {line_no}: {VHS_DEFERRAL_MESSAGE} (requested backend {action!r})")
        if action in ("backend", "backend_"):
            _require_backend_allowed(_unquote(rest), line_no)
            continue
        if action == "size":
            parts = rest.split()
            if len(parts) != 2:
                raise TapeParseError(f"line {line_no}: Size requires COLS ROWS")
            cols, rows = _as_int(parts[0], "Size cols"), _as_int(parts[1], "Size rows")
            if cols <= 0 or rows <= 0:
                raise TapeParseError(f"line {line_no}: Size must be positive")
        elif action == "command":
            if not rest:
                raise TapeParseError(f"line {line_no}: Command requires an argv")
            argv = shlex.split(rest)
            command = [sys.executable if arg == "{python}" else arg for arg in argv]
        elif action == "name":
            if not rest:
                raise TapeParseError(f"line {line_no}: Name requires a value")
            name = _unquote(rest)
        elif action == "type":
            steps.append(TapeStep(action="type", text=_unquote(rest)))
        elif action == "key":
            if not rest:
                raise TapeParseError(f"line {line_no}: Key requires a key name")
            token, _tail = _split_first_arg(rest)
            steps.append(TapeStep(action="key", name=_unquote(token)))
        elif action == "resize":
            parts = rest.split()
            if len(parts) != 2:
                raise TapeParseError(f"line {line_no}: Resize requires COLS ROWS")
            steps.append(
                TapeStep(
                    action="resize",
                    cols=_as_int(parts[0], "Resize cols"),
                    rows=_as_int(parts[1], "Resize rows"),
                )
            )
        elif action == "waitfor":
            token, tail = _split_first_arg(rest)
            if not token:
                raise TapeParseError(f"line {line_no}: WaitFor requires a literal")
            text = _unquote(token)
            timeout_ms = _as_int(tail, "WaitFor timeout_ms") if tail else DEFAULT_WAITFOR_MS
            steps.append(TapeStep(action="waitfor", text=text, timeout_ms=timeout_ms))
        elif action == "waitquiet":
            tokens = rest.split()
            quiet_ms = _as_int(tokens[0], "WaitQuiet quiet_ms") if tokens else DEFAULT_WAITQUIET_MS
            steps.append(TapeStep(action="waitquiet", quiet_ms=quiet_ms))
        elif action in ("expectscreen", "expectraw"):
            mode_token, tail = _split_first_arg(rest)
            if not mode_token:
                raise TapeParseError(f"line {line_no}: {head} requires contains|excludes and a literal")
            mode = mode_token.lower()
            if mode not in ("contains", "excludes"):
                raise TapeParseError(f"line {line_no}: {head} mode must be contains or excludes, got {mode!r}")
            if not tail:
                raise TapeParseError(f"line {line_no}: {head} requires a literal")
            steps.append(TapeStep(action=action, mode=mode, text=_unquote(tail)))
        elif action == "golden":
            if golden_seen:
                raise TapeParseError(f"line {line_no}: a tape may declare only one Golden")
            golden_seen = True
            if not rest:
                raise TapeParseError(f"line {line_no}: Golden requires a name")
            token, _tail = _split_first_arg(rest)
            steps.append(TapeStep(action="golden", name=_unquote(token)))
        else:
            raise TapeParseError(
                f"line {line_no}: unknown tape action {head!r}; "
                "expected Size, Command, Type, Key, Resize, WaitFor, WaitQuiet, "
                "ExpectScreen, ExpectRaw, Golden"
            )
    if not command:
        raise TapeParseError(f"tape {path} has no Command step")
    if cols <= 0 or rows <= 0:
        raise TapeParseError(f"tape {path} has invalid geometry {cols}x{rows}")
    return TapeScript(
        name=name,
        cols=cols,
        rows=rows,
        command=command,
        steps=steps,
        path=path,
        sha256=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
    )


_JSON_STEP_ACTIONS = {
    "type": "type",
    "key": "key",
    "resize": "resize",
    "waitfor": "waitfor",
    "wait_for": "waitfor",
    "waitquiet": "waitquiet",
    "wait_quiet": "waitquiet",
    "expectscreen": "expectscreen",
    "expect_screen": "expectscreen",
    "expectraw": "expectraw",
    "expect_raw": "expectraw",
    "golden": "golden",
}


def _parse_json(raw: str, path: Path) -> TapeScript:
    try:
        doc = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise TapeParseError(f"invalid tape JSON {path}: {exc}") from exc
    if not isinstance(doc, dict):
        raise TapeParseError(f"tape JSON {path} must be an object")
    backend = doc.get("backend")
    if isinstance(backend, str):
        _require_backend_allowed(backend, 0)
    name = str(doc.get("name") or path.stem)
    cols = int(doc.get("cols") or 80)
    rows = int(doc.get("rows") or 24)
    if cols <= 0 or rows <= 0:
        raise TapeParseError(f"tape JSON {path} has invalid geometry {cols}x{rows}")
    raw_command = doc.get("command")
    if not isinstance(raw_command, list) or not raw_command or any(not isinstance(a, str) for a in raw_command):
        raise TapeParseError(f"tape JSON {path} requires command as a non-empty list of strings")
    command = [sys.executable if arg == "{python}" else arg for arg in raw_command]
    steps: List[TapeStep] = []
    entries = doc.get("steps")
    if entries is None:
        entries = []
    if not isinstance(entries, list):
        raise TapeParseError(f"tape JSON {path} steps must be a list")
    golden_seen = False
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise TapeParseError(f"tape JSON {path} step #{index + 1} must be an object")
        action_raw = str(entry.get("action") or entry.get("do") or "").strip().lower()
        if action_raw in _BLOCKED_BACKENDS:
            raise TapePolicyError(f"{VHS_DEFERRAL_MESSAGE} (requested backend {action_raw!r})")
        action = _JSON_STEP_ACTIONS.get(action_raw)
        if action is None:
            raise TapeParseError(f"tape JSON {path} step #{index + 1} has unknown action {action_raw!r}")
        if action == "type":
            steps.append(TapeStep(action="type", text=str(entry.get("text", ""))))
        elif action == "key":
            steps.append(TapeStep(action="key", name=str(entry.get("key", ""))))
        elif action == "resize":
            steps.append(TapeStep(action="resize", cols=int(entry.get("cols", 0)), rows=int(entry.get("rows", 0))))
        elif action == "waitfor":
            steps.append(
                TapeStep(
                    action="waitfor",
                    text=str(entry.get("text", "")),
                    timeout_ms=int(entry.get("timeout_ms", DEFAULT_WAITFOR_MS)),
                )
            )
        elif action == "waitquiet":
            steps.append(TapeStep(action="waitquiet", quiet_ms=int(entry.get("quiet_ms", DEFAULT_WAITQUIET_MS))))
        elif action in ("expectscreen", "expectraw"):
            mode = str(entry.get("mode", "contains")).lower()
            if mode not in ("contains", "excludes"):
                raise TapeParseError(f"tape JSON {path} step #{index + 1} mode must be contains or excludes")
            steps.append(TapeStep(action=action, mode=mode, text=str(entry.get("text", ""))))
        elif action == "golden":
            if golden_seen:
                raise TapeParseError(f"tape JSON {path} declares more than one Golden")
            golden_seen = True
            steps.append(TapeStep(action="golden", name=str(entry.get("name", name))))
    return TapeScript(
        name=name,
        cols=cols,
        rows=rows,
        command=command,
        steps=steps,
        path=path,
        sha256=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        source_format="json",
    )


def load_tape(path: Path) -> TapeScript:
    p = Path(path).expanduser()
    if not p.is_file():
        raise TapeParseError(f"tape file not found: {p}")
    try:
        raw = p.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise TapeParseError(f"cannot read tape {p}: {exc}") from exc
    resolved = p.resolve()
    if resolved.suffix.lower() == ".json":
        return _parse_json(raw, resolved)
    return _parse_dsl(raw, resolved)


# ------------------------------------------------------------------------- screen


def normalized_screen(text: str) -> str:
    """Canonical screen text: right-stripped lines, single trailing newline."""
    lines = [line.rstrip() for line in (text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines) + "\n"


def screen_sha256(text: str) -> str:
    return hashlib.sha256(normalized_screen(text).encode("utf-8")).hexdigest()


@dataclass
class GoldenCheck:
    status: str  # pass | missing | unapproved | stale | diff
    detail: str = ""
    expected_text: str = ""
    screen_path: Optional[Path] = None
    manifest_path: Optional[Path] = None

    @property
    def ok(self) -> bool:
        return self.status == "pass"


def golden_paths(golden_dir: Path, name: str) -> Tuple[Path, Path]:
    base = Path(golden_dir)
    slug = _slug(name)
    return base / f"{slug}.screen.txt", base / f"{slug}.manifest.json"


def _read_golden_manifest(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def check_golden(
    actual_screen: str,
    *,
    golden_dir: Path,
    name: str,
    tape_sha256: str,
    cols: int,
    rows: int,
) -> GoldenCheck:
    """Compare ``actual_screen`` to an approved golden without touching it."""
    screen_path, manifest_path = golden_paths(golden_dir, name)
    if not screen_path.is_file():
        return GoldenCheck(
            status="missing",
            detail=(
                f"golden missing: {screen_path} does not exist; "
                f"run the tape to capture evidence, then approve with `testbench tape approve`"
            ),
            screen_path=screen_path,
            manifest_path=manifest_path,
        )
    manifest = _read_golden_manifest(manifest_path)
    if manifest is None:
        return GoldenCheck(
            status="unapproved",
            detail=f"golden manifest missing or unreadable: {manifest_path}",
            screen_path=screen_path,
            manifest_path=manifest_path,
        )
    approved_sha = str(manifest.get("approved_sha256") or "")
    recorded_sha = str(manifest.get("screen_sha256") or "")
    if not approved_sha or approved_sha != recorded_sha:
        return GoldenCheck(
            status="unapproved",
            detail=(
                f"golden not approved: manifest approved_sha256={approved_sha or '-'} "
                f"!= screen_sha256={recorded_sha or '-'}"
            ),
            screen_path=screen_path,
            manifest_path=manifest_path,
        )
    if str(manifest.get("tape_sha256") or "") != tape_sha256:
        return GoldenCheck(
            status="stale",
            detail=(
                "golden stale: tape changed since approval "
                f"(manifest tape_sha256={str(manifest.get('tape_sha256') or '-')[:12]}, "
                f"current={tape_sha256[:12]})"
            ),
            screen_path=screen_path,
            manifest_path=manifest_path,
        )
    if int(manifest.get("cols") or 0) != int(cols) or int(manifest.get("rows") or 0) != int(rows):
        return GoldenCheck(
            status="stale",
            detail=(
                "golden stale: geometry changed since approval "
                f"(manifest {manifest.get('cols')}x{manifest.get('rows')}, current {cols}x{rows})"
            ),
            screen_path=screen_path,
            manifest_path=manifest_path,
        )
    try:
        expected_raw = screen_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return GoldenCheck(status="unapproved", detail=f"golden unreadable: {exc}", screen_path=screen_path)
    file_sha = screen_sha256(expected_raw)
    if file_sha != approved_sha:
        return GoldenCheck(
            status="unapproved",
            detail=(
                "golden file changed after approval: on-disk screen_sha256="
                f"{file_sha[:12]} != approved_sha256={approved_sha[:12]}"
            ),
            expected_text=expected_raw,
            screen_path=screen_path,
            manifest_path=manifest_path,
        )
    expected_text = normalized_screen(expected_raw)
    if expected_text != normalized_screen(actual_screen):
        return GoldenCheck(
            status="diff",
            detail=f"screen differs from approved golden {screen_path}",
            expected_text=expected_text,
            screen_path=screen_path,
            manifest_path=manifest_path,
        )
    return GoldenCheck(status="pass", expected_text=expected_text, screen_path=screen_path, manifest_path=manifest_path)


# ------------------------------------------------------------------------- replay


@dataclass
class TapeOutcome:
    name: str
    status: str
    failure_kind: Optional[str] = None
    message: str = ""
    checks: List[Dict[str, Any]] = field(default_factory=list)
    screen_text: str = ""
    raw_text: str = ""
    golden: Optional[Dict[str, Any]] = None
    exit_code: Optional[int] = None
    duration_ms: float = 0.0
    evidence_dir: str = ""
    screen_path: str = ""
    actual_screen_path: str = ""
    raw_log_path: str = ""
    diff_path: str = ""
    seatbelt: str = "disabled"
    tape_path: str = ""
    tape_sha256: str = ""
    cols: int = 80
    rows: int = 24
    truncated: bool = False

    @property
    def total(self) -> int:
        return len(self.checks) or 1

    @property
    def passed(self) -> int:
        return sum(1 for c in self.checks if c.get("ok"))

    @property
    def failed(self) -> int:
        return sum(1 for c in self.checks if not c.get("ok"))


def _sanitized_stream(raw: str) -> str:
    redactor = StreamRedactor()
    return sanitize_terminal_text(redactor.feed(raw) + redactor.flush())


def _write_evidence(path: Path, text: str) -> str:
    atomic_write_file(path, text, mode=0o600)
    return str(path)


def _unified_diff(expected: str, actual: str, *, label: str) -> str:
    expected_lines = normalized_screen(expected).splitlines(keepends=True)
    actual_lines = normalized_screen(actual).splitlines(keepends=True)
    diff = list(
        difflib.unified_diff(
            expected_lines,
            actual_lines,
            fromfile=f"golden/{label}",
            tofile=f"actual/{label}",
            lineterm="\n",
        )
    )
    return "".join(diff)


def _expectation_report(kind: str, mode: str, literal: str, actual: str) -> str:
    header = [f"expectation failed: {kind} {mode} {literal!r}", "", ""]
    expected_lines = [line + "\n" for line in literal.split("\n")]
    actual_lines = [line + "\n" for line in normalized_screen(actual).splitlines()]
    diff = difflib.unified_diff(
        expected_lines,
        actual_lines,
        fromfile="expected",
        tofile="actual",
        lineterm="\n",
    )
    return "\n".join(header) + "".join(diff)


def replay_tape(
    script: TapeScript,
    *,
    project_root: Path,
    evidence_dir: Path,
    golden_dir: Optional[Path] = None,
    use_seatbelt: bool = True,
    scratch_root: Optional[Path] = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> TapeOutcome:
    """Drive ``script`` through a real isolated PTY and evaluate every assertion."""
    project_root = Path(project_root).expanduser().resolve()
    evidence = Path(evidence_dir).expanduser().resolve()
    evidence.mkdir(parents=True, exist_ok=True)
    os.chmod(str(evidence), 0o700)
    golden_dir = Path(golden_dir).expanduser().resolve() if golden_dir else script.path.parent / "golden"
    slug = _slug(script.name)

    outcome = TapeOutcome(
        name=script.name,
        status=RunStatus.PASS,
        evidence_dir=str(evidence),
        tape_path=str(script.path),
        tape_sha256=script.sha256,
        cols=script.cols,
        rows=script.rows,
    )
    checks: List[Dict[str, Any]] = []
    session: Optional[PtySession] = None
    failure: Optional[TapeError] = None
    t0 = time.monotonic()
    deadline = t0 + max(1.0, timeout_s)

    try:
        session = PtySession(
            script.command,
            cwd=project_root,
            cols=script.cols,
            rows=script.rows,
            run_id=f"tape_{slug}",
            use_seatbelt=use_seatbelt,
            scratch_root=scratch_root,
        )
        outcome.seatbelt = session.seatbelt_state
        for index, step in enumerate(script.steps, start=1):
            if time.monotonic() > deadline:
                raise TapeTimeout(f"tape exceeded its {timeout_s}s budget before step #{index}")
            action = step.action
            if action == "type":
                session.send_text(step.text)
            elif action == "key":
                session.send_key(step.name)
            elif action == "resize":
                if step.cols <= 0 or step.rows <= 0:
                    raise TapeParseError(f"step #{index}: Resize requires positive COLS ROWS")
                session.resize(step.cols, step.rows)
                outcome.cols, outcome.rows = step.cols, step.rows
            elif action == "waitfor":
                try:
                    session.wait_for(step.text, step.timeout_ms or DEFAULT_WAITFOR_MS)
                except PtyTimeout as exc:
                    raise TapeTimeout(f"step #{index}: {exc}") from exc
            elif action == "waitquiet":
                session.wait_quiet(step.quiet_ms or DEFAULT_WAITQUIET_MS)
            elif action in ("expectscreen", "expectraw"):
                actual = session.screen_text() if action == "expectscreen" else session.text
                hit = step.text in actual
                ok = hit if step.mode == "contains" else not hit
                checks.append(
                    {
                        "kind": action,
                        "mode": step.mode,
                        "text": step.text,
                        "ok": ok,
                        "detail": "" if ok else f"{action} {step.mode} {step.text!r} failed",
                    }
                )
                if not ok:
                    raise TapeAssertionError(f"step #{index}: {action} {step.mode} {step.text!r} did not hold")
            elif action == "golden":
                session.wait_quiet(_GOLDEN_QUIET_MS, timeout_ms=3000)
                actual = session.screen_text()
                check = check_golden(
                    actual,
                    golden_dir=golden_dir,
                    name=step.name or script.name,
                    tape_sha256=script.sha256,
                    cols=session.cols,
                    rows=session.rows,
                )
                outcome.cols, outcome.rows = session.cols, session.rows
                outcome.golden = {
                    "name": step.name or script.name,
                    "status": check.status,
                    "detail": check.detail,
                    "screen_path": str(check.screen_path or ""),
                    "manifest_path": str(check.manifest_path or ""),
                }
                checks.append(
                    {
                        "kind": "golden",
                        "name": step.name or script.name,
                        "ok": check.ok,
                        "status": check.status,
                        "detail": check.detail,
                    }
                )
                if check.status == "diff":
                    raise TapeAssertionError(
                        f"step #{index}: {check.detail}; see {evidence / (slug + '.golden.diff')}"
                    )
                if check.status != "pass":
                    raise TapePolicyError(f"step #{index}: {check.detail}")
            else:  # pragma: no cover - parser rejects unknown actions
                raise TapeParseError(f"step #{index}: unsupported action {action!r}")
        # Give a still-running program a moment to flush its last frame.
        session.wait_quiet(_GOLDEN_QUIET_MS, timeout_ms=2000)
    except TapeError as exc:
        failure = exc
    except OSError as exc:
        failure = TapeError(f"pty transport error: {exc}")
    finally:
        if session is not None:
            try:
                outcome.exit_code = session.close()
            except Exception as exc:  # never mask the assertion that already failed
                if failure is None:
                    failure = TapeError(f"pty teardown error: {exc}")
            outcome.screen_text = session.screen_text()
            outcome.raw_text = session.text
            outcome.truncated = session.truncated
            outcome.seatbelt = session.seatbelt_state

    if failure is not None:
        kind = getattr(failure, "failure_kind", FailureKind.HARNESS_ERROR)
        if kind == FailureKind.BLOCKED_POLICY:
            outcome.status = RunStatus.BLOCKED
        elif kind == FailureKind.TIMEOUT:
            outcome.status = RunStatus.TIMEOUT
        else:
            outcome.status = RunStatus.FAIL
        outcome.failure_kind = kind
        outcome.message = str(failure)
    outcome.checks = checks
    outcome.duration_ms = round((time.monotonic() - t0) * 1000.0, 2)

    # ---- persisted evidence (always sanitized) -----------------------------
    safe_screen = _sanitized_stream(outcome.screen_text)
    safe_raw = _sanitized_stream(outcome.raw_text)
    outcome.screen_path = _write_evidence(evidence / f"{slug}.screen.txt", safe_screen)
    outcome.raw_log_path = _write_evidence(evidence / f"{slug}.raw.log", safe_raw)
    if outcome.status != RunStatus.PASS:
        outcome.actual_screen_path = _write_evidence(evidence / f"{slug}.actual.screen.txt", safe_screen)
        failure_kind = outcome.failure_kind
        golden = outcome.golden or {}
        if failure_kind == FailureKind.ASSERTION and golden.get("status") == "diff":
            expected_text = ""
            screen_path = golden.get("screen_path") or ""
            if screen_path and Path(screen_path).is_file():
                expected_text = Path(screen_path).read_text(encoding="utf-8", errors="replace")
            diff_text = _unified_diff(
                _sanitized_stream(expected_text),
                safe_screen,
                label=f"{slug}.screen.txt",
            )
            if diff_text:
                outcome.diff_path = _write_evidence(evidence / f"{slug}.golden.diff", diff_text)
        elif failure_kind == FailureKind.ASSERTION:
            failed_check = next((c for c in checks if not c.get("ok")), None)
            if failed_check is not None:
                report = _expectation_report(
                    str(failed_check.get("kind")),
                    str(failed_check.get("mode")),
                    str(failed_check.get("text")),
                    safe_screen if failed_check.get("kind") == "expectscreen" else safe_raw,
                )
                outcome.diff_path = _write_evidence(evidence / f"{slug}.expect.diff", report)
    meta = {
        "tape_version": TAPE_VERSION,
        "name": script.name,
        "tape_path": str(script.path),
        "tape_sha256": script.sha256,
        "cols": outcome.cols,
        "rows": outcome.rows,
        "status": outcome.status,
        "failure_kind": outcome.failure_kind,
        "checks": checks,
        "golden": outcome.golden,
        "seatbelt": outcome.seatbelt,
        "source_format": script.source_format,
        "captured_at": utc_now_iso(),
    }
    atomic_write_file(evidence / "tape.meta.json", json.dumps(meta, indent=2, sort_keys=True) + "\n", mode=0o600)
    return outcome


# ---------------------------------------------------------------------- run record


def _aggregate(statuses: Sequence[str]) -> Tuple[str, Optional[str]]:
    if RunStatus.FAIL in statuses:
        return RunStatus.FAIL, FailureKind.ASSERTION
    if RunStatus.ERROR in statuses:
        return RunStatus.ERROR, FailureKind.HARNESS_ERROR
    if RunStatus.TIMEOUT in statuses:
        return RunStatus.TIMEOUT, FailureKind.TIMEOUT
    if RunStatus.BLOCKED in statuses:
        return RunStatus.BLOCKED, FailureKind.BLOCKED_POLICY
    if RunStatus.CANCELLED in statuses:
        return RunStatus.CANCELLED, FailureKind.CANCELLED
    if RunStatus.SKIP in statuses:
        return RunStatus.SKIP, None
    return RunStatus.PASS, None


def run_tape(
    store: TestBenchStore,
    tape: Any,
    *,
    project_root: Optional[Path] = None,
    golden_dir: Optional[Path] = None,
    use_seatbelt: bool = True,
    scratch_root: Optional[Path] = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    profile: str = ProfileName.TERMINAL,
) -> RunRecord:
    """Replay a tape and persist a real :class:`RunRecord` with all artifacts."""
    from .discovery import detect_git_metadata

    script = tape if isinstance(tape, TapeScript) else load_tape(Path(tape))
    root = Path(project_root).expanduser().resolve() if project_root else script.path.parent
    if not root.is_dir():
        raise TapeError(f"project root is not a directory: {root}")
    gold = Path(golden_dir).expanduser().resolve() if golden_dir else script.path.parent / "golden"
    revision, branch = detect_git_metadata(root)
    rid = new_run_id()
    now = utc_now_iso()
    run = RunRecord(
        run_id=rid,
        project=root.name or "tape",
        project_root=str(root),
        revision=revision,
        branch=branch,
        profile=profile,
        status=RunStatus.RUNNING,
        commands=[["testbench", "tape", "run", str(script.path)]],
        created_at=now,
        started_at=now,
        environment={
            "adapter": "terminal_tape",
            "tape_path": str(script.path),
            "tape_sha256": script.sha256,
            "golden_dir": str(gold),
            "cols": script.cols,
            "rows": script.rows,
        },
    )
    store.create_run(run)
    evidence = Path(run.evidence_dir or store.run_dir(rid))
    t0 = time.monotonic()
    outcome = replay_tape(
        script,
        project_root=root,
        evidence_dir=evidence,
        golden_dir=gold,
        use_seatbelt=use_seatbelt,
        scratch_root=scratch_root,
        timeout_s=timeout_s,
    )

    total = outcome.total
    passed = outcome.passed
    failed = outcome.failed
    tape_argv = ["testbench", "tape", "run", str(script.path), "--golden-dir", str(gold)]
    suite = SuiteResult(
        name=script.name,
        adapter="terminal_tape",
        profile=profile,
        command=tape_argv,
        cwd=str(root),
        status=outcome.status,
        exit_code=outcome.exit_code,
        started_at=now,
        finished_at=utc_now_iso(),
        duration_ms=outcome.duration_ms,
        total_count=total,
        pass_count=passed if outcome.status == RunStatus.PASS else 0,
        fail_count=failed if outcome.status in (RunStatus.FAIL, RunStatus.TIMEOUT) else 0,
        blocked_count=1 if outcome.status == RunStatus.BLOCKED else 0,
        stdout_path=outcome.raw_log_path,
        sanitized_path=outcome.screen_path,
        summary_excerpt=(outcome.message or f"{passed}/{total} tape checks passed")[:800],
        repro_command=shlex.join(tape_argv),
        isolation_meta={
            "seatbelt": outcome.seatbelt,
            "pty": True,
            "cols": outcome.cols,
            "rows": outcome.rows,
            "truncated": outcome.truncated,
            "golden": (outcome.golden or {}).get("status", ""),
        },
    )
    if outcome.status == RunStatus.PASS:
        suite.fail_count = 0
        suite.pass_count = max(1, passed)
        suite.total_count = max(1, total)
    else:
        suite.pass_count = passed
        suite.failures = [
            FailureRecord(
                suite=script.name,
                case_name="terminal_tape",
                kind=outcome.failure_kind or FailureKind.ASSERTION,
                message=outcome.message[:800],
                excerpt=outcome.screen_text[-800:],
            )
        ]
    suite.metrics = {"checks": outcome.checks, "program_argv": list(script.command)}

    status, fkind = _aggregate([outcome.status])
    run.suites = [suite]
    run.status = status
    run.failure_kind = fkind if status != RunStatus.PASS else None
    run.duration_ms = round((time.monotonic() - t0) * 1000.0, 2)
    run.total_count = suite.total_count
    run.pass_count = suite.pass_count
    run.fail_count = suite.fail_count
    run.blocked_count = suite.blocked_count
    run.exit_code = 0 if status == RunStatus.PASS else 1
    run.first_failure = (f"terminal_tape: {outcome.message}" if outcome.message else "")[:300]
    run.repro_instructions = suite.repro_command
    run.summary = (
        f"{status}: {suite.pass_count}/{suite.total_count} tape checks passed, "
        f"{suite.fail_count} failed, {suite.blocked_count} blocked"
    )
    run.finished_at = utc_now_iso()
    return store.finalize_run(run)


# ------------------------------------------------------------------------- approve


def _assert_interactive_approval() -> None:
    """Refuse approvals that could be issued by a suite or a background agent."""
    if os.environ.get("TESTBENCH_SANDBOX") == "1":
        raise GoldenApprovalRefused(
            "refusing to approve goldens inside a TestBench sandbox (TESTBENCH_SANDBOX=1)"
        )
    try:
        is_tty = bool(sys.stdin.isatty())
    except (AttributeError, ValueError, OSError):
        is_tty = False
    if not is_tty:
        raise GoldenApprovalRefused(
            "refusing to approve goldens without an interactive TTY (stdin is not a tty); "
            "run `testbench tape approve` from a terminal"
        )


def _resolve_evidence_dir(target: Any, store: Optional[TestBenchStore]) -> Path:
    raw = str(target)
    if store is not None:
        rec = store.get_run(raw)
        if rec is not None and rec.evidence_dir:
            return Path(rec.evidence_dir)
    candidate = Path(raw).expanduser()
    if candidate.is_dir():
        return candidate.resolve()
    if candidate.is_file():
        return candidate.resolve().parent
    raise TapeError(f"no evidence directory or known run_id found for {raw!r}")


def approve_golden(
    target: Any,
    *,
    golden_name: str = "",
    golden_dir: Path,
    store: Optional[TestBenchStore] = None,
    assume_yes: bool = False,
) -> Dict[str, Any]:
    """Promote a run's captured screen to an approved golden (human gated)."""
    _assert_interactive_approval()
    evidence = _resolve_evidence_dir(target, store)
    meta_path = evidence / "tape.meta.json"
    if not meta_path.is_file():
        raise TapeError(f"no tape.meta.json in {evidence}; run the tape first")
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TapeError(f"unreadable tape.meta.json in {evidence}: {exc}") from exc
    if not isinstance(meta, dict):
        raise TapeError(f"invalid tape.meta.json in {evidence}")
    name = (golden_name or str(meta.get("name") or "")).strip()
    if not name:
        raise TapeError("no golden name available; pass --golden NAME")
    slug = _slug(name)
    tape_slug = _slug(str(meta.get("name") or ""))
    candidates = [
        evidence / f"{slug}.actual.screen.txt",
        evidence / f"{slug}.screen.txt",
    ]
    if tape_slug and tape_slug != slug:
        candidates.extend(
            [
                evidence / f"{tape_slug}.actual.screen.txt",
                evidence / f"{tape_slug}.screen.txt",
            ]
        )
    candidates.extend(sorted(evidence.glob("*.actual.screen.txt")))
    candidates.extend(sorted(evidence.glob("*.screen.txt")))
    actual_path = next((c for c in candidates if c.is_file()), None)
    if actual_path is None:
        raise TapeError(
            f"no captured screen for {name!r} in {evidence}; run `testbench tape run` first"
        )
    try:
        actual_text = actual_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise TapeError(f"unreadable capture {actual_path}: {exc}") from exc
    text = normalized_screen(actual_text)
    digest = screen_sha256(text)
    tape_sha = str(meta.get("tape_sha256") or "")
    if not tape_sha:
        raise TapeError(f"tape.meta.json in {evidence} has no tape_sha256")
    cols = int(meta.get("cols") or 0)
    rows = int(meta.get("rows") or 0)
    if not assume_yes:
        try:
            answer = input(f"Approve golden {name!r} ({cols}x{rows}, sha256 {digest[:12]})? [y/N] ")
        except (EOFError, OSError, KeyboardInterrupt) as exc:
            raise GoldenApprovalRefused("approval prompt aborted (no interactive confirmation)") from exc
        if answer.strip().lower() not in ("y", "yes"):
            raise GoldenApprovalRefused("approval declined")
    screen_path, manifest_path = golden_paths(Path(golden_dir), name)
    screen_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_file(screen_path, text, mode=0o644)
    manifest = {
        "name": name,
        "schema_version": TAPE_VERSION,
        "screen_sha256": digest,
        "approved_sha256": digest,
        "tape_sha256": tape_sha,
        "tape_path": str(meta.get("tape_path") or ""),
        "cols": cols,
        "rows": rows,
        "approved_at": utc_now_iso(),
    }
    atomic_write_file(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n", mode=0o644)
    return {
        "approved": True,
        "name": name,
        "screen_path": str(screen_path),
        "manifest_path": str(manifest_path),
        "screen_sha256": digest,
        "tape_sha256": tape_sha,
        "cols": cols,
        "rows": rows,
        "evidence_dir": str(evidence),
    }


# ------------------------------------------------------------------------ marker


def tape_marker(outcome: TapeOutcome) -> str:
    """One-line machine payload consumed by the ``terminal_tape`` adapter."""
    payload = {
        "adapter": "terminal_tape",
        "name": outcome.name,
        "status": outcome.status,
        "failure_kind": outcome.failure_kind,
        "message": outcome.message[:400],
        "checks": [
            {
                "kind": c.get("kind"),
                "mode": c.get("mode"),
                "name": c.get("name"),
                "ok": bool(c.get("ok")),
                "status": c.get("status"),
                "detail": str(c.get("detail") or "")[:300],
            }
            for c in outcome.checks
        ],
        "total": outcome.total,
        "passed": outcome.passed,
        "failed": outcome.failed,
        "evidence_dir": outcome.evidence_dir,
    }
    return "TB-TAPE " + json.dumps(payload, sort_keys=True)


def tape_marker_from_run(run: RunRecord) -> str:
    """Machine payload reconstructed from a persisted tape run (adapter input)."""
    suite = run.suites[0] if run.suites else None
    checks = list((suite.metrics or {}).get("checks") or []) if suite else []
    payload = {
        "adapter": "terminal_tape",
        "name": suite.name if suite else "tape",
        "status": run.status,
        "failure_kind": run.failure_kind,
        "message": (suite.summary_excerpt if suite else run.first_failure)[:400],
        "checks": [
            {
                "kind": c.get("kind"),
                "mode": c.get("mode"),
                "name": c.get("name"),
                "ok": bool(c.get("ok")),
                "status": c.get("status"),
                "detail": str(c.get("detail") or "")[:300],
            }
            for c in checks
        ],
        "total": run.total_count or 1,
        "passed": run.pass_count,
        "failed": run.fail_count,
        "evidence_dir": run.evidence_dir,
    }
    return "TB-TAPE " + json.dumps(payload, sort_keys=True)
