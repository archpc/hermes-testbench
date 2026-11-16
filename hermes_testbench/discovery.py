"""Project capability discovery: git metadata, manifests, and zero-config detection.

Pure stdlib. Never writes inside the examined project tree (ADR-0004).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib
from typing import Any, Dict, List, Optional, Tuple

from .schema import ProfileName, ProjectSpec, SuiteSpec
from .store import TestBenchStore

MAX_MANIFEST_BYTES = 256 * 1024

def default_interpreter() -> Path:
    """Interpreter for discovered Python suites when a project has no venv.

    ``$TESTBENCH_PYTHON`` when it names an executable file, else the interpreter running
    TestBench (>=3.11 by construction: this module imports ``tomllib``).
    """
    override = os.environ.get("TESTBENCH_PYTHON", "").strip()
    if override:
        cand = Path(override).expanduser()
        if cand.is_file() and os.access(str(cand), os.X_OK):
            return cand
    return Path(sys.executable)


PYTHON_TOKEN = "{python}"


_UNVERSIONED: Tuple[str, str] = ("unversioned", "")


def _git(root: Path, *args: str, timeout_s: float = 10.0) -> Optional[str]:
    """Run git read-only against ``root``; return stdout or None on any failure."""
    env = dict(os.environ)
    env["GIT_OPTIONAL_LOCKS"] = "0"
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    env["LC_ALL"] = "C"
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
            check=False,
            env=env,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


def detect_git_metadata(root: Path) -> Tuple[str, str]:
    """Return ``(revision, branch)`` for ``root`` or ``("unversioned", "")``.

    Revision is the short SHA, suffixed ``+dirty`` when tracked files carry
    uncommitted changes (untracked files are ignored). Branch falls back to
    ``HEAD`` for detached checkouts. A broken ``.git`` pointer or a directory
    that is merely *inside* some other repository degrades to ``unversioned``.
    """
    try:
        resolved = Path(root).expanduser().resolve(strict=True)
    except OSError:
        return _UNVERSIONED
    if not resolved.is_dir():
        return _UNVERSIONED
    git_entry = resolved / ".git"
    if not (git_entry.is_dir() or git_entry.is_file()):
        return _UNVERSIONED
    if shutil.which("git") is None:
        return _UNVERSIONED

    sha_out = _git(resolved, "rev-parse", "--short", "HEAD")
    sha = (sha_out or "").strip()
    if not sha:
        return _UNVERSIONED
    # Refuse a repository whose (possibly broken) pointer resolves elsewhere.
    top_out = _git(resolved, "rev-parse", "--show-toplevel")
    top = (top_out or "").strip()
    if top:
        try:
            if Path(top).resolve(strict=True) != resolved:
                return _UNVERSIONED
        except OSError:
            return _UNVERSIONED

    branch_out = _git(resolved, "rev-parse", "--abbrev-ref", "HEAD")
    branch = (branch_out or "").strip() or "HEAD"

    status_out = _git(resolved, "status", "--porcelain=v1", "--untracked-files=no")
    dirty = bool((status_out or "").strip())
    return (f"{sha}+dirty" if dirty else sha, branch)


KNOWN_ADAPTERS = frozenset(
    {
        "agent_eval",
        "c_sanitizer",
        "cargo",
        "cargo_mutants",
        "cobol",
        "mutmut",
        "pty_tmux",
        "python_pytest",
        "python_unittest",
        "re_project",
        "shell",
        "sqlite_probe",
    }
)

_MANIFEST_SUITE_FIELDS = frozenset(
    {
        "name",
        "adapter",
        "profile",
        "command",
        "cwd",
        "timeout_s",
        "needs_tmux",
        "needs_pty",
        "allow_network",
        "allow_repo_write",
        "billable",
        "custom_from_manifest",
        "env_overrides",
        "required_paths",
        "required_binaries",
        "description",
        "warmup_runs",
        "repeat_runs",
        "mutation_score_gate",
    }
)


def _as_str_list(value: Any, what: str) -> List[str]:
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(v, str) for v in value):
        raise ValueError(f"{what} must be a list of strings, got {value!r}")
    return list(value)


def _as_bool(value: Any, what: str, default: bool = False) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError(f"{what} must be a boolean, got {value!r}")
    return value


def _as_float(value: Any, what: str, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{what} must be a number, got {value!r}")
    if float(value) <= 0:
        raise ValueError(f"{what} must be positive, got {value!r}")
    return float(value)


def _as_int(value: Any, what: str, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{what} must be an integer, got {value!r}")
    return int(value)


def _as_gate(value: Any, what: str) -> Optional[float]:
    """A mutation score gate is a fraction in ``[0, 1]`` or absent."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{what} must be a number between 0 and 1, got {value!r}")
    gate = float(value)
    if not 0.0 <= gate <= 1.0:
        raise ValueError(f"{what} must be between 0 and 1, got {value!r}")
    return gate


def _read_manifest_text(path: Path) -> bytes:
    if path.is_symlink():
        raise ValueError(f"manifest must not be a symlink: {path}")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read manifest {path}: {exc}") from exc
    if len(raw) > MAX_MANIFEST_BYTES:
        raise ValueError(
            f"manifest {path} is {len(raw)} bytes, exceeding the {MAX_MANIFEST_BYTES} byte limit"
        )
    return raw


def _parse_manifest_bytes(raw: bytes, path: Path) -> Dict[str, Any]:
    try:
        data = tomllib.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"invalid TOML in manifest {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"manifest {path} must contain a TOML table at the root")
    return data


def _suite_from_manifest(
    entry: Any, index: int, *, custom_from_manifest: bool, manifest: Path
) -> SuiteSpec:
    where = f"manifest {manifest} suite #{index + 1}"
    if not isinstance(entry, dict):
        raise ValueError(f"{where} must be a table")

    unknown = set(entry) - _MANIFEST_SUITE_FIELDS
    if "env" in entry and "env_overrides" not in entry:
        entry = {**entry, "env_overrides": entry["env"]}
        unknown.discard("env")
    if unknown:
        raise ValueError(f"{where} has unknown keys: {', '.join(sorted(unknown))}")

    name = entry.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{where} requires a non-empty 'name'")
    adapter = entry.get("adapter")
    if not isinstance(adapter, str) or adapter not in KNOWN_ADAPTERS:
        raise ValueError(
            f"{where} adapter {adapter!r} is not one of: {', '.join(sorted(KNOWN_ADAPTERS))}"
        )
    profile = entry.get("profile", ProfileName.UNIT)
    if profile not in ProfileName.STANDARD:
        raise ValueError(
            f"{where} profile {profile!r} is not one of: {', '.join(ProfileName.STANDARD)}"
        )

    env_raw = entry.get("env_overrides")
    if env_raw is None:
        env: Dict[str, str] = {}
    elif isinstance(env_raw, dict):
        env = {}
        for key, val in env_raw.items():
            if not isinstance(key, str) or not isinstance(val, str):
                raise ValueError(f"{where} env_overrides must map strings to strings")
            env[key] = val
    else:
        raise ValueError(f"{where} env_overrides must be a table")

    cwd = entry.get("cwd", ".")
    if not isinstance(cwd, str) or not cwd:
        raise ValueError(f"{where} cwd must be a non-empty string")

    description = entry.get("description", "")
    if not isinstance(description, str):
        raise ValueError(f"{where} description must be a string")

    return SuiteSpec(
        name=name.strip(),
        adapter=adapter,
        profile=profile,
        command=[
            str(default_interpreter()) if i == 0 and arg == PYTHON_TOKEN else arg
            for i, arg in enumerate(_as_str_list(entry.get("command"), f"{where} command"))
        ],
        cwd=cwd,
        timeout_s=_as_float(entry.get("timeout_s"), f"{where} timeout_s", 120.0),
        needs_tmux=_as_bool(entry.get("needs_tmux"), f"{where} needs_tmux"),
        needs_pty=_as_bool(entry.get("needs_pty"), f"{where} needs_pty"),
        allow_network=_as_bool(entry.get("allow_network"), f"{where} allow_network"),
        allow_repo_write=_as_bool(entry.get("allow_repo_write"), f"{where} allow_repo_write"),
        billable=_as_bool(entry.get("billable"), f"{where} billable"),
        custom_from_manifest=custom_from_manifest,
        env_overrides=env,
        required_paths=_as_str_list(entry.get("required_paths"), f"{where} required_paths"),
        required_binaries=_as_str_list(
            entry.get("required_binaries"), f"{where} required_binaries"
        ),
        description=description,
        warmup_runs=_as_int(entry.get("warmup_runs"), f"{where} warmup_runs", 1),
        repeat_runs=_as_int(entry.get("repeat_runs"), f"{where} repeat_runs", 5),
        mutation_score_gate=_as_gate(
            entry.get("mutation_score_gate"), f"{where} mutation_score_gate"
        ),
    )


def _suites_from_manifest(
    data: Dict[str, Any], *, custom_from_manifest: bool, manifest: Path
) -> List[SuiteSpec]:
    entries = data.get("suite", data.get("suites", []))
    if entries is None:
        entries = []
    if isinstance(entries, dict):
        entries = [entries]
    if not isinstance(entries, list):
        raise ValueError(f"manifest {manifest} 'suite' must be an array of tables")
    suites = [
        _suite_from_manifest(entry, idx, custom_from_manifest=custom_from_manifest, manifest=manifest)
        for idx, entry in enumerate(entries)
    ]
    seen: set[str] = set()
    for suite in suites:
        if suite.name in seen:
            raise ValueError(f"manifest {manifest} declares duplicate suite name {suite.name!r}")
        seen.add(suite.name)
    return suites


def _interpreter_for(root: Path) -> Path:
    for candidate in (root / "venv" / "bin" / "python", root / ".venv" / "bin" / "python"):
        if candidate.is_file() and os.access(str(candidate), os.X_OK):
            return candidate
    return default_interpreter()


def _has_module(python: Path, module: str) -> bool:
    try:
        proc = subprocess.run(
            [str(python), "-c", f"import {module}"],
            capture_output=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


def discover_project(
    root: Path,
    *,
    store: Optional[TestBenchStore] = None,
    external_manifest: Optional[Path] = None,
) -> ProjectSpec:
    """Detect the project layout and the suites it can run."""
    try:
        resolved = Path(root).expanduser().resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"project root does not exist: {root}") from exc
    if not resolved.is_dir():
        raise ValueError(f"project root is not a directory: {resolved}")

    revision, branch = detect_git_metadata(resolved)

    manifest_path: Optional[Path] = None
    custom_from_manifest = False
    trusted = False
    if external_manifest is not None:
        manifest_path = Path(external_manifest).expanduser()
        try:
            manifest_path = manifest_path.resolve(strict=True)
        except OSError as exc:
            raise ValueError(f"external manifest does not exist: {external_manifest}") from exc
        custom_from_manifest = False
        trusted = False
    else:
        local = resolved / ".testbench.toml"
        if local.is_symlink() or local.is_file():
            manifest_path = local
            custom_from_manifest = True
            trusted = bool(store.is_manifest_trusted(resolved)) if store is not None else False

    manifest_digest: Optional[str] = None
    data: Dict[str, Any] = {}
    suites: List[SuiteSpec] = []
    if manifest_path is not None:
        raw = _read_manifest_text(manifest_path)
        manifest_digest = hashlib.sha256(raw).hexdigest()
        data = _parse_manifest_bytes(raw, manifest_path)
        suites = _suites_from_manifest(
            data, custom_from_manifest=custom_from_manifest, manifest=manifest_path
        )

    project_meta = data.get("project") or {}
    if not isinstance(project_meta, dict):
        raise ValueError(f"manifest {manifest_path} 'project' must be a table")
    name = project_meta.get("name") or resolved.name
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"manifest {manifest_path} project.name must be a non-empty string")
    languages = _as_str_list(project_meta.get("languages"), "project.languages")

    if manifest_path is None:
        suites, detected_languages = _detect_zero_config(resolved)
        languages = detected_languages

    return ProjectSpec(
        name=name.strip(),
        root=str(resolved),
        languages=sorted(set(languages)),
        suites=suites,
        manifest_path=str(manifest_path) if manifest_path is not None else None,
        manifest_sha256=manifest_digest,
        trusted=trusted,
        revision=revision,
        branch=branch,
    )


def _python_suite(root: Path) -> Optional[SuiteSpec]:
    tests_dir = root / "tests"
    has_tests = tests_dir.is_dir() and any(tests_dir.glob("test_*.py"))
    has_pyproject = (root / "pyproject.toml").is_file()
    if not (has_tests or has_pyproject):
        return None

    python = _interpreter_for(root)
    needs_tmux = (root / "hermes_tmux").exists() or (
        tests_dir / "test_tmux_integration.py"
    ).is_file()

    needs_pyte = False
    if has_tests and not _has_module(python, "pyte") and shutil.which("uv"):
        import re as _re
        for tf in tests_dir.glob("test_*.py"):
            try:
                src = tf.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if _re.search(r"^[ \t]*(?:import|from)[ \t]+pyte\b", src, _re.M):
                needs_pyte = True
                break

    if needs_pyte:
        command = ["uv", "run", "--offline", "--with", "pyte", "python", "-m", "unittest", "discover", "-s", "tests", "-v"]
        adapter, name = "python_unittest", "python-unittest"
        description = "unittest discovery via uv offline with pyte"
    elif _has_module(python, "pytest"):
        command = [str(python), "-m", "pytest", "-q"]
        if has_tests:
            command.append("tests")
        adapter, name = "python_pytest", "python-pytest"
        description = "pytest suite discovered from tests/ or pyproject.toml"
    else:
        command = [str(python), "-m", "unittest", "discover"]
        if has_tests:
            command.extend(["-s", "tests"])
        command.append("-v")
        adapter, name = "python_unittest", "python-unittest"
        description = "unittest discovery (pytest unavailable in the chosen interpreter)"

    return SuiteSpec(
        name=name,
        adapter=adapter,
        profile=ProfileName.UNIT,
        command=command,
        timeout_s=600.0,
        needs_tmux=needs_tmux,
        description=description,
    )


def _rust_suite(root: Path) -> Optional[SuiteSpec]:
    if not (root / "Cargo.toml").is_file():
        return None
    cmd = ["cargo", "test"]
    if (root / "Cargo.lock").is_file():
        cmd.append("--locked")
    cmd.extend(["--offline", "--workspace"])
    return SuiteSpec(
        name="cargo-unit",
        adapter="cargo",
        profile=ProfileName.UNIT,
        command=cmd,
        timeout_s=900.0,
        allow_repo_write=False,
        required_binaries=["cargo"],
        description="cargo workspace unit tests (offline)",
    )


_NOISE_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".mypy_cache",
        ".pytest_cache",
        ".tox",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
        "target",
        "vendor",
        "venv",
    }
)

MAX_WALK_FILES = 20000

_COBOL_SHELL_SCRIPT = """set -u
rc=0
for src in "$@"; do
  name=$(printf '%s' "$src" | tr -c 'A-Za-z0-9._-' '_')
  bin="$TMPDIR/cobol_$name"
  log="$TMPDIR/cobol_$name.compile.log"
  if ! cobc -O2 -debug -x -o "$bin" "$src" >"$log" 2>&1; then
    echo "TB-FAIL cobol:$src compile"
    sed -n '1,20p' "$log" | sed 's/^/  | /'
    err=$(grep -m1 ': error: ' "$log" || true)
    echo "  compile failed: ${err:-see compiler output above}"
    rc=1
    continue
  fi
  out="$TMPDIR/cobol_$name.stdout"
  "$bin" </dev/null >"$out" 2>"$TMPDIR/cobol_$name.stderr"
  status=$?
  if [ "$status" -ne 0 ]; then
    echo "TB-FAIL cobol:$src rc=$status"
    sed -n '1,10p' "$TMPDIR/cobol_$name.stderr" | sed 's/^/  | /'
    echo "  process exited with status $status"
    rc=1
    continue
  fi
  if [ -f "$src.expected" ]; then
    if cmp -s "$src.expected" "$out"; then
      echo "TB-PASS cobol:$src"
    else
      echo "TB-FAIL cobol:$src output"
      diff -u "$src.expected" "$out" | sed -n '1,24p' | sed 's/^/  | /'
      echo "  output differs from $src.expected: expected [$(sed -n '1p' "$src.expected")] got [$(sed -n '1p' "$out")]"
      rc=1
    fi
  else
    echo "TB-PASS cobol:$src"
  fi
done
exit $rc
"""

_C_SANITIZER_SCRIPT = """set -u
rc=0
for src in "$@"; do
  name=$(printf '%s' "$src" | tr -c 'A-Za-z0-9._-' '_')
  bin="$TMPDIR/asan_$name"
  dir=$(dirname "$src")
  extra_c=""
  for sib in "$dir"/*.c; do
    [ -f "$sib" ] || continue
    [ "$sib" = "$src" ] && continue
    if ! grep -Eq 'int[[:space:]]+main[[:space:]]*\\(' "$sib"; then
      extra_c="$extra_c $sib"
    fi
  done
  libs=""
  if grep -Fq 'sqlite3.h' "$src"; then
    libs="-lsqlite3"
  fi
  if ! clang -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -o "$bin" "$src" $extra_c $libs; then
    echo "TB-FAIL c-asan:$src compile"
    rc=1
    continue
  fi
  if "$bin"; then
    echo "TB-PASS c-asan:$src"
  else
    status=$?
    echo "TB-FAIL c-asan:$src rc=$status"
    rc=1
  fi
done
exit $rc
"""

_SQLITE_PROBE_SCRIPT = """
import json
import os
import sqlite3
import stat
import sys

db_paths = sys.argv[1:]
passed = 0
failed = 0
for db in db_paths:
    try:
        st = os.stat(db)
    except OSError as exc:
        print(f"TB-FAIL sqlite:{db} stat: {exc}")
        failed += 1
        continue
    mode = stat.S_IMODE(st.st_mode)
    if mode & 0o022:
        print(f"TB-FAIL sqlite:{db} permissions {oct(mode)} are group/world writable")
        failed += 1
        continue
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro&immutable=1", uri=True)
        try:
            row = con.execute("PRAGMA integrity_check").fetchone()
        finally:
            con.close()
    except sqlite3.Error as exc:
        print(f"TB-FAIL sqlite:{db} error: {exc}")
        failed += 1
        continue
    result = str(row[0]) if row else "no result"
    if result.lower() != "ok":
        print(f"TB-FAIL sqlite:{db} integrity_check: {result}")
        failed += 1
    else:
        print(f"TB-PASS sqlite:{db}")
        passed += 1
print(f"{passed} passed, 0 skipped, {failed} failed")
sys.exit(1 if failed else 0)
"""


def _project_files(root: Path) -> List[str]:
    """All non-noise, non-hidden project files as sorted posix relative paths."""
    rels: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            d for d in dirnames if d not in _NOISE_DIRS and not d.startswith(".")
        )
        for fname in sorted(filenames):
            if fname.startswith("."):
                continue
            full = Path(dirpath) / fname
            try:
                rel = full.relative_to(root).as_posix()
            except ValueError:
                continue
            rels.append(rel)
            if len(rels) >= MAX_WALK_FILES:
                return sorted(rels)
    return sorted(rels)


def _matching(files: List[str], *, suffix: str = "", pattern: str = "") -> List[str]:
    import fnmatch

    out: List[str] = []
    for rel in files:
        base = rel.rsplit("/", 1)[-1]
        if suffix and not base.endswith(suffix):
            continue
        if pattern and not fnmatch.fnmatch(base, pattern):
            continue
        out.append(rel)
    return out


def _cobol_suites(root: Path, files: List[str]) -> Tuple[List[SuiteSpec], bool]:
    suites: List[SuiteSpec] = []
    harness = "bench/tui/cobol/test_cobol.py"
    if any(rel == harness for rel in files):
        suites.append(
            SuiteSpec(
                name="cobol-tui-harness",
                adapter="cobol",
                profile=ProfileName.TERMINAL,
                command=[
                    "uv",
                    "run",
                    "--offline",
                    "--with",
                    "pyte",
                    "python",
                    "-m",
                    "unittest",
                    "discover",
                    "-s",
                    "bench/tui/cobol",
                    "-v",
                ],
                timeout_s=300.0,
                required_binaries=["uv"],
                description="GnuCOBOL TUI harness run through uv with pyte",
            )
        )

    sources = _matching(files, suffix=".cob") if not suites else []
    if sources:
        suites.append(
            SuiteSpec(
                name="cobol-unit",
                adapter="cobol",
                profile=ProfileName.UNIT,
                command=["/bin/sh", "-c", _COBOL_SHELL_SCRIPT, "testbench-cobol", *sources],
                timeout_s=300.0,
                required_binaries=["cobc"],
                description="compile each GnuCOBOL source into $TMPDIR and run it",
            )
        )
    return suites, bool(suites)


def _c_suites(files: List[str]) -> Tuple[List[SuiteSpec], bool]:
    sources = _matching(files, suffix=".c", pattern="*selftest*.c")
    if not sources:
        return [], False
    return (
        [
            SuiteSpec(
                name="c-sanitizer",
                adapter="c_sanitizer",
                profile=ProfileName.SECURITY,
                command=["/bin/sh", "-c", _C_SANITIZER_SCRIPT, "testbench-c-asan", *sources],
                timeout_s=300.0,
                required_binaries=["clang"],
                description="clang ASan/UBSan self-tests built into $TMPDIR",
            )
        ],
        True,
    )


_VERIFIER_SCRIPTS: Tuple[Tuple[str, str, str, str], ...] = (
    ("scripts/verify_docs.py", "verify-docs", "re_project", "python"),
    ("scripts/verify_claims.py", "verify-claims", "re_project", "python"),
    ("scripts/regress_release.py", "regress-release", "shell", "shell"),
    ("scripts/check-wiring.py", "check-wiring", "shell", "shell"),
)


_CHECK_WIRING_PROBE_SCRIPT = """
from contextlib import closing
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile

script = sys.argv[1]
rc_all = 0
with tempfile.TemporaryDirectory(prefix="tb-cw-") as tmp:
    base = Path(tmp)
    wired = base / "wired-home"
    (wired / "profiles" / "opus").mkdir(parents=True, exist_ok=True)
    (wired / "profiles" / "opus" / "config.yaml").write_text(
        "model:\\n  default: anthropic/claude-opus-5.5\\n", encoding="utf-8"
    )
    (wired / "profile.yaml").write_text("ui_meta:\\n  hermes-bots:\\n    enabled: true\\n", encoding="utf-8")
    for db_dir in (wired, wired / "profiles" / "opus"):
        with closing(sqlite3.connect(str(db_dir / "state.db"))) as conn:
            conn.execute("CREATE TABLE sessions (id TEXT, title TEXT, started_at INTEGER)")
            conn.execute("INSERT INTO sessions VALUES ('s1', 'Bot Chat', 1)")
            conn.commit()
    unwired = base / "unwired-home"
    unwired.mkdir(parents=True, exist_ok=True)

    p_ok = subprocess.run(
        [sys.executable, script, "--hermes-home", str(wired)], capture_output=True, text=True, timeout=120.0
    )
    out_ok = (p_ok.stdout or "") + (p_ok.stderr or "")
    if p_ok.returncode == 0 and "wiring OK" in out_ok:
        print("TB-PASS check-wiring:synthetic_wired_home")
    else:
        print(f"TB-FAIL check-wiring:synthetic_wired_home rc={p_ok.returncode} {out_ok.strip()[:240]}")
        rc_all = 1

    p_bad = subprocess.run(
        [sys.executable, script, "--hermes-home", str(unwired)], capture_output=True, text=True, timeout=120.0
    )
    out_bad = (p_bad.stdout or "") + (p_bad.stderr or "")
    if p_bad.returncode != 0 and "advisor profile missing" in out_bad:
        print("TB-PASS check-wiring:unwired_home_rejected")
    else:
        print(f"TB-FAIL check-wiring:unwired_home_rejected rc={p_bad.returncode} {out_bad.strip()[:240]}")
        rc_all = 1
sys.exit(rc_all)
"""


def _script_suites(root: Path, files: List[str]) -> Tuple[List[SuiteSpec], List[str]]:
    suites: List[SuiteSpec] = []
    languages: List[str] = []
    present = set(files)
    python = _interpreter_for(root)
    for rel, name, adapter, language in _VERIFIER_SCRIPTS:
        if rel not in present:
            continue
        cmd = [str(python), rel]
        try:
            src_text = (root / rel).read_text(encoding="utf-8", errors="ignore")
            if rel == "scripts/check-wiring.py" and "--hermes-home" in src_text and "hermes-bots" in src_text:
                cmd = [str(python), "-c", _CHECK_WIRING_PROBE_SCRIPT, rel]
            elif "sys.argv[1]" in src_text:
                cmd.append(".")
        except OSError:
            pass
        suites.append(
            SuiteSpec(
                name=name,
                adapter=adapter,
                profile=ProfileName.REGRESSION,
                command=cmd,
                timeout_s=300.0,
                description=f"verifier script {rel}",
            )
        )
        languages.append(language)
    return suites, languages


def _sqlite_suites(root: Path, files: List[str]) -> Tuple[List[SuiteSpec], bool]:
    dbs = [rel for rel in files if rel.endswith(".db") or rel.endswith(".sqlite")]
    if not dbs:
        return [], False
    python = _interpreter_for(root)
    return (
        [
            SuiteSpec(
                name="sqlite-probe",
                adapter="sqlite_probe",
                profile=ProfileName.INTEGRATION,
                command=[str(python), "-c", _SQLITE_PROBE_SCRIPT, *dbs],
                timeout_s=120.0,
                description="PRAGMA integrity_check plus permission audit for SQLite files",
            )
        ],
        True,
    )


def _mutation_suites(root: Path) -> List[SuiteSpec]:
    """Mutation suites for projects that ship a supported tool configuration.

    Only the configuration files are inspected: whether the binary itself is installed is a
    run-time question, and a missing tool must surface as SKIP rather than vanish.
    """
    suites: List[SuiteSpec] = []
    if (root / "Cargo.toml").is_file():
        suites.append(
            SuiteSpec(
                name="cargo-mutants",
                adapter="cargo_mutants",
                profile=ProfileName.MUTATION,
                command=["cargo-mutants", "mutants"],
                timeout_s=3600.0,
                allow_repo_write=False,
                description="cargo-mutants over a disposable staged copy (never the checkout)",
            )
        )
    mutmut_config = False
    for name, marker in (("pyproject.toml", "[tool.mutmut]"), ("setup.cfg", "[mutmut]")):
        path = root / name
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if marker in text:
            mutmut_config = True
    if mutmut_config:
        suites.append(
            SuiteSpec(
                name="mutmut",
                adapter="mutmut",
                profile=ProfileName.MUTATION,
                command=["mutmut", "run"],
                timeout_s=3600.0,
                allow_repo_write=False,
                description="mutmut over a disposable staged copy (never the checkout)",
            )
        )
    return suites


def _detect_zero_config(root: Path) -> Tuple[List[SuiteSpec], List[str]]:
    """Zero-config capability discovery for repositories with no manifest."""
    suites: List[SuiteSpec] = []
    languages: List[str] = []

    rust = _rust_suite(root)
    if rust is not None:
        suites.append(rust)
        languages.append("rust")

    python = _python_suite(root)
    if python is not None:
        suites.append(python)
        languages.append("python")

    files = _project_files(root)

    cobol_suites, has_cobol = _cobol_suites(root, files)
    if has_cobol:
        suites.extend(cobol_suites)
        languages.append("cobol")

    c_suites, has_c = _c_suites(files)
    if has_c:
        suites.extend(c_suites)
        languages.append("c")

    script_suites, script_languages = _script_suites(root, files)
    suites.extend(script_suites)
    languages.extend(script_languages)

    sqlite_suites, has_sqlite = _sqlite_suites(root, files)
    if has_sqlite:
        suites.extend(sqlite_suites)
        languages.append("sqlite")

    suites.extend(_mutation_suites(root))

    return suites, sorted(set(languages))
