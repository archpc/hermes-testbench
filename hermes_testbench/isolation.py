"""Mandatory execution isolation, Seatbelt sandboxing, process-group timeouts, and cleanup."""

from __future__ import annotations

import codecs
from dataclasses import dataclass, field
import errno
import fcntl
import os
from pathlib import Path
import pty
import pwd
import re
import selectors
import shutil
import signal
import struct
import subprocess
import termios
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from .sanitize import StreamRedactor, sanitize_terminal_text
from .store import _safe_rmtree_no_symlinks

_SCRUB_EXACT = frozenset(
    {
        "TMUX",
        "TMUX_PANE",
        "TMUX_PLUGIN_MANAGER_PATH",
        "HERMES_SPAWN",
        "HERMES_TMUX",
        "HXMAIL_DB",
        "HXMAIL_LOCAL_DIR",
        "TESTBENCH_HOME",
        "SSH_AUTH_SOCK",
        "SSH_AGENT_PID",
        "GPG_AGENT_INFO",
        "SECURITYSESSIONID",
    }
)

_SCRUB_KEY_RE = re.compile(
    r"(?i)(?:"
    r"API_KEY|SECRET|TOKEN|PASSWORD|PASSWD|CREDENTIAL|PRIVATE_KEY|COOKIE|SESSION_KEY|AUTH_KEY|_KEY$"
    r"|^(?:OPENAI|ANTHROPIC|OPENROUTER|NOUS|DEEPSEEK|GITHUB|GH|AWS|FASTMAIL|SLACK|DISCORD|TELEGRAM|HERMES_PEER)_"
    r")"
)

MAX_MEM_CAPTURE_CHARS = 4 * 1024 * 1024
MAX_DISK_LOG_BYTES = 64 * 1024 * 1024


@dataclass
class IsolatedExecResult:
    command: List[str]
    cwd: str
    exit_code: Optional[int]
    signal_num: Optional[int]
    duration_ms: float
    timed_out: bool
    cancelled: bool
    stdout_text: str
    stderr_text: str
    stdout_path: str
    stderr_path: str
    sanitized_path: str
    isolation_meta: Dict[str, Any] = field(default_factory=dict)


def real_user_home() -> Path:
    try:
        return Path(pwd.getpwuid(os.getuid()).pw_dir).resolve()
    except Exception:
        return Path.home().resolve()


def extra_bin_dirs() -> List[str]:
    """Toolchain directories appended to PATH, derived from the real user home.

    ``~/.local/bin`` is the XDG user bin dir; ``/opt/homebrew/bin`` and ``/usr/local/bin``
    are the Homebrew prefixes on Apple silicon and Intel/Linux. Missing ones are harmless.
    """
    return [str(real_user_home() / ".local" / "bin"), "/opt/homebrew/bin", "/usr/local/bin"]


def _probe_writable_dir(cand: Path) -> bool:
    try:
        cand.mkdir(parents=True, exist_ok=True)
        probe = cand / f".tbp_{os.getpid()}_{os.urandom(3).hex()}"
        os.mkdir(str(probe), 0o700)
        os.rmdir(str(probe))
        return True
    except OSError:
        return False


def default_scratch_root() -> Path:
    tb_sc = os.environ.get("TESTBENCH_SCRATCH_DIR", "").strip()
    if tb_sc:
        p = Path(tb_sc).expanduser()
        if _probe_writable_dir(p):
            return p.resolve()
    preferred = real_user_home() / ".hermes" / "cache" / "scratch"
    if _probe_writable_dir(preferred):
        return preferred.resolve()
    td = os.environ.get("TMPDIR", "").strip()
    if td:
        p = Path(td).expanduser()
        if _probe_writable_dir(p):
            return p.resolve()
    preferred.mkdir(parents=True, exist_ok=True)
    return preferred.resolve()


def short_writable_tmux_base(preferred: Path) -> Path:
    """Return a writable directory short enough for macOS 104-byte sockaddr_un."""
    tb_sc = os.environ.get("TESTBENCH_SCRATCH_DIR", "").strip()
    if tb_sc:
        p = Path(tb_sc).expanduser().resolve()
        if len(str(p)) <= 68 and _probe_writable_dir(p):
            return p
        if len(str(p.parent)) <= 68 and _probe_writable_dir(p.parent):
            return p.parent
    canonical_scratch = (real_user_home() / ".hermes" / "cache" / "scratch").resolve()
    if len(str(canonical_scratch)) <= 68 and _probe_writable_dir(canonical_scratch):
        return canonical_scratch
    pref_res = preferred.expanduser().resolve()
    if len(str(pref_res)) <= 68 and _probe_writable_dir(pref_res):
        return pref_res
    for anc in pref_res.parents:
        # Never walk into ~/.hermes root or user home root
        if anc in (real_user_home(), (real_user_home() / ".hermes").resolve()) or len(str(anc)) < 20:
            break
        if len(str(anc)) <= 68 and _probe_writable_dir(anc):
            return anc
    return pref_res


def _process_start_fingerprint(pid: int) -> Optional[str]:
    if pid <= 1:
        return None
    try:
        proc = subprocess.run(
            ["ps", "-o", "lstart=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
        if proc.returncode != 0:
            return None
        val = (proc.stdout or "").strip()
        return val or None
    except (OSError, subprocess.SubprocessError):
        return None


def pgroup_alive(pgid: int) -> bool:
    if pgid <= 1:
        return False
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def terminate_process_group(
    pgid: int,
    *,
    expected_start_time: Optional[str] = None,
    grace_s: float = 1.0,
) -> bool:
    if pgid <= 1 or not pgroup_alive(pgid):
        return True
    if expected_start_time is not None:
        live_fp = _process_start_fingerprint(pgid)
        if live_fp != expected_start_time:
            return False

    try:
        os.killpg(pgid, signal.SIGTERM)
    except OSError:
        pass

    deadline = time.monotonic() + max(0.05, grace_s)
    while time.monotonic() < deadline:
        if not pgroup_alive(pgid):
            return True
        time.sleep(0.02)

    try:
        os.killpg(pgid, signal.SIGKILL)
    except OSError:
        pass

    kill_deadline = time.monotonic() + 0.5
    while time.monotonic() < kill_deadline:
        if not pgroup_alive(pgid):
            return True
        time.sleep(0.02)
    return not pgroup_alive(pgid)


_HAS_WAITID_NOWAIT = hasattr(os, "waitid") and hasattr(os, "WNOWAIT")


def _leader_exited(proc: "subprocess.Popen[Any]") -> bool:
    """True once the suite leader (our direct child) has exited.

    With ``waitid(WNOWAIT)`` the leader stays an unreaped zombie, so its pid, and with it the
    process-group id, cannot be reused: a later ``killpg(pgid)`` reaches only that group.
    Without it (macOS on Python < 3.13) the leader is reaped here; the pgid then stays
    reserved only while some member of the group is still alive.
    """
    if not _HAS_WAITID_NOWAIT:
        return proc.poll() is not None
    try:
        return os.waitid(os.P_PID, proc.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None
    except ChildProcessError:
        return True


def _live_group_members(pgid: int, leader: int) -> List[int]:
    """Live, non-zombie members of ``pgid`` other than ``leader``."""
    try:
        os.killpg(pgid, 0)
    except OSError:
        # ESRCH: group gone. EPERM (macOS): only the unreaped zombie leader remains.
        return []
    try:
        out = subprocess.run(
            ["ps", "-A", "-o", "pid=,pgid=,stat="], capture_output=True, text=True, timeout=2.0, check=False
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    members: List[int] = []
    for ln in out.splitlines():
        parts = ln.split()
        if len(parts) >= 3 and parts[1] == str(pgid) and parts[0].isdigit():
            pid = int(parts[0])
            if pid != leader and not parts[2].startswith("Z"):
                members.append(pid)
    return members


def _reap_group_after_leader_exit(pgid: int, grace_s: float = 0.5) -> int:
    """Terminate what is left of the suite's own process group after its leader exited.

    The group is re-checked for live members immediately before each signal, so ``killpg``
    only runs while the pgid is held by the leader's zombie or by a live member of the suite's
    own session. Returns the number of surviving members that were signalled.
    """
    members = _live_group_members(pgid, pgid)
    if not members:
        return 0
    for sig, wait_s in ((signal.SIGTERM, grace_s), (signal.SIGKILL, 0.5)):
        if not _live_group_members(pgid, pgid):
            return len(members)
        try:
            os.killpg(pgid, sig)
        except OSError:
            break
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline:
            if not _live_group_members(pgid, pgid):
                return len(members)
            time.sleep(0.02)
    return len(members)


def can_apply_seatbelt() -> bool:
    sb_bin = shutil.which("sandbox-exec")
    if not sb_bin:
        return False
    try:
        probe = subprocess.run(
            [
                sb_bin,
                "-p",
                '(version 1)(allow default)(deny file-write* (literal "/__tb_sb_probe_never__"))',
                "/usr/bin/true",
            ],
            capture_output=True,
            timeout=2.0,
            check=False,
        )
        return probe.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _sb_quote(path: Path) -> str:
    raw = str(path).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{raw}"'


def _sweep_sandbox_survivors(sandbox_dir: Path) -> Tuple[int, int]:
    """Reap any daemonized/setsid descendants still holding files/cwd/argv inside sandbox_dir."""
    if not sandbox_dir.exists():
        return 0, 0
    sb_str = str(sandbox_dir.resolve())
    me = os.getpid()
    pids: set[int] = set()
    lsof_bin = "/usr/sbin/lsof" if os.path.exists("/usr/sbin/lsof") else shutil.which("lsof")
    if lsof_bin:
        try:
            proc = subprocess.run(
                [lsof_bin, "-t", "+D", sb_str],
                capture_output=True,
                text=True,
                timeout=3.0,
                check=False,
            )
            for ln in (proc.stdout or "").splitlines():
                ln = ln.strip()
                if ln.isdigit():
                    p = int(ln)
                    if p > 1 and p != me:
                        pids.add(p)
        except (OSError, subprocess.SubprocessError):
            pass
    try:
        ps_proc = subprocess.run(
            ["/bin/ps", "-ax", "-o", "pid=,command="],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
        for ln in (ps_proc.stdout or "").splitlines():
            ln = ln.strip()
            if not ln or sb_str not in ln:
                continue
            parts = ln.split(None, 1)
            if parts and parts[0].isdigit():
                p = int(parts[0])
                if p > 1 and p != me:
                    pids.add(p)
    except (OSError, subprocess.SubprocessError):
        pass

    if not pids:
        return 0, 0

    for p in pids:
        try:
            os.kill(p, signal.SIGTERM)
        except OSError:
            pass
    deadline = time.monotonic() + 0.25
    while time.monotonic() < deadline:
        if not any(_pid_alive(p) for p in pids):
            break
        time.sleep(0.02)
    for p in pids:
        if _pid_alive(p):
            try:
                os.kill(p, signal.SIGKILL)
            except OSError:
                pass
    time.sleep(0.05)
    leaked = sum(1 for p in pids if _pid_alive(p))
    return len(pids) - leaked, leaked


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except OSError:
        return True


def build_seatbelt_profile(
    *,
    sandbox_dir: Path,
    cwd: Path,
    extra_writable_dirs: Tuple[Path, ...] = (),
    allow_network: bool = False,
    allow_repo_write: bool = False,
    allow_uv_cache_write: bool = False,
) -> str:
    u_home = real_user_home()
    hermes_dir = u_home / ".hermes"
    hxmail_support = u_home / "Library" / "Application Support" / "hxmail"
    hxlab_dir = (hermes_dir / "cache" / "scratch" / "hxlab").resolve()
    resolved_sb = sandbox_dir.resolve()
    resolved_cwd = cwd.resolve()

    uid = os.getuid()
    rules = [
        "(version 1)",
        "(allow default)",
        '(allow process-exec (literal "/bin/ps") (with no-sandbox))',
        "(deny file-write*)",
        f'(deny file-read* (subpath "/private/tmp/tmux-{uid}") (subpath "/private/var/tmp/tmux-{uid}"))',
    ]
    if not allow_network:
        rules.append("(deny network*)")
        rules.append("(allow network* (local unix))")
        rules.append(
            f'(deny network-outbound (remote unix-socket (literal "/private/var/run/mDNSResponder")) '
            f'(remote unix-socket (subpath "/private/tmp/tmux-{uid}")) '
            f'(remote unix-socket (subpath "/private/var/run/docker.sock")))'
        )
        rules.append('(allow network-bind (local ip "localhost:*"))')
        rules.append('(allow network-inbound (local ip "localhost:*"))')
        rules.append('(allow network-outbound (remote ip "localhost:*"))')
        rules.append(
            '(deny network-outbound (remote ip "localhost:9119") '
            '(remote ip "localhost:9222") (remote ip "localhost:11434"))'
        )

    rules.append(
        "(deny file-read-data "
        f"(subpath {_sb_quote(u_home / '.ssh')}) "
        f"(subpath {_sb_quote(u_home / '.aws')}) "
        f"(subpath {_sb_quote(u_home / '.gnupg')}) "
        f"(subpath {_sb_quote(u_home / '.config' / 'gh')}) "
        f"(subpath {_sb_quote(u_home / '.docker')}) "
        f"(literal {_sb_quote(u_home / '.netrc')}) "
        f"(literal {_sb_quote(u_home / '.npmrc')}) "
        f"(subpath {_sb_quote(u_home / 'Library' / 'Keychains')}) "
        f"(literal {_sb_quote(u_home / '.cargo' / 'credentials.toml')}) "
        f"(literal {_sb_quote(u_home / '.cargo' / 'credentials')}) "
        f"(literal {_sb_quote(hermes_dir / 'config.yaml')}) "
        f"(literal {_sb_quote(hermes_dir / '.env')}) "
        f"(literal {_sb_quote(hermes_dir / 'auth.json')}) "
        f"(literal {_sb_quote(hermes_dir / 'state.db')}) "
        f"(literal {_sb_quote(hermes_dir / 'state.db-wal')}) "
        f"(literal {_sb_quote(hermes_dir / 'state.db-shm')}) "
        f"(subpath {_sb_quote(hermes_dir / 'sessions')}) "
        f"(subpath {_sb_quote(hermes_dir / 'profiles')}) "
        f"(subpath {_sb_quote(hermes_dir / 'memories')}) "
        f"(subpath {_sb_quote(hermes_dir / 'logs')}) "
        f"(subpath {_sb_quote(hermes_dir / 'mcp-tokens')}) "
        f"(subpath {_sb_quote(hxmail_support)}))"
    )

    write_targets = [resolved_sb, hxlab_dir, *(p.resolve() for p in extra_writable_dirs)]
    if allow_repo_write:
        write_targets.append(resolved_cwd)
    if allow_uv_cache_write and (u_home / ".cache" / "uv").is_dir():
        write_targets.append((u_home / ".cache" / "uv").resolve())

    # Preserve comment reference to .cargo for auditability while deny-by-default blocks ~/.cargo writes
    rules.append(f"; deny-by-default protects {u_home / '.cargo'} and {hermes_dir}")
    allow_subpaths = " ".join(f"(subpath {_sb_quote(p)})" for p in write_targets)
    dev_allows = (
        '(literal "/dev/null") (literal "/dev/zero") (literal "/dev/tty") '
        '(literal "/dev/ptmx") (literal "/dev/urandom") (literal "/dev/random") '
        '(literal "/dev/dtracehelper") (regex #"^/dev/ttys[0-9]+$") (regex #"^/dev/fd/[0-9]+$")'
    )
    rules.append(f"(allow file-write* {allow_subpaths} {dev_allows})")
    return "\n".join(rules)


def build_isolated_env(
    sandbox_dir: Path,
    *,
    run_id: str,
    nonce: str,
    needs_tmux: bool = False,
    tmux_socket: str = "",
    tmux_tmpdir: Optional[Path] = None,
    env_overrides: Optional[Dict[str, str]] = None,
    base_env: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    sandbox_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(str(sandbox_dir), 0o700)
    subdirs = {}
    dir_names = {
        "home": "home",
        "xdg_config": "xc",
        "xdg_data": "xd",
        "xdg_cache": "xk",
        "tmp": "tmp",
        "tmux_tmp": "tx",
        "cargo_home": "ch",
        "cargo_target": "ct",
        "uv_cache": "uv",
    }
    for sub, short in dir_names.items():
        d = sandbox_dir / short
        d.mkdir(parents=True, exist_ok=True)
        os.chmod(str(d), 0o700)
        subdirs[sub] = d
    u_home = real_user_home()
    for cargo_sub in ("registry", "git", "bin"):
        real_csub = u_home / ".cargo" / cargo_sub
        dst_link = subdirs["cargo_home"] / cargo_sub
        if real_csub.exists() and not dst_link.exists():
            try:
                dst_link.symlink_to(real_csub)
            except OSError:
                pass
    real_uv = u_home / ".cache" / "uv"
    if real_uv.is_dir():
        try:
            for item in real_uv.iterdir():
                if not item.name.startswith(("wheels-", "archive-", "simple-", "interpreter-", "builds-", "git-")):
                    continue
                dst_uv = subdirs["uv_cache"] / item.name
                if not dst_uv.exists():
                    dst_uv.symlink_to(item)
        except OSError:
            pass
    if tmux_tmpdir is not None:
        tmux_tmpdir.mkdir(parents=True, exist_ok=True)
        os.chmod(str(tmux_tmpdir), 0o700)
        subdirs["tmux_tmp"] = tmux_tmpdir
    fake_hermes = subdirs["home"] / ".hermes"
    fake_hermes.mkdir(parents=True, exist_ok=True)
    os.chmod(str(fake_hermes), 0o700)

    source = os.environ if base_env is None else base_env
    clean: Dict[str, str] = {}
    for k, v in source.items():
        if "\x00" in k or "\x00" in v:
            continue
        if k in _SCRUB_EXACT or _SCRUB_KEY_RE.search(k):
            continue
        clean[k] = v

    clean["HOME"] = str(subdirs["home"])
    clean["XDG_CONFIG_HOME"] = str(subdirs["xdg_config"])
    clean["XDG_DATA_HOME"] = str(subdirs["xdg_data"])
    clean["XDG_CACHE_HOME"] = str(subdirs["xdg_cache"])
    clean["TMPDIR"] = str(subdirs["tmp"])
    clean["TESTBENCH_SCRATCH_DIR"] = str(subdirs["tmp"])
    clean["TESTBENCH_HOME"] = str(subdirs["xdg_data"] / "testbench")
    clean["CARGO_HOME"] = str(subdirs["cargo_home"])
    clean["CARGO_TARGET_DIR"] = str(subdirs["cargo_target"])
    if (u_home / ".rustup").is_dir():
        clean["RUSTUP_HOME"] = str(u_home / ".rustup")
    if (u_home / ".cache" / "uv").is_dir():
        clean["UV_CACHE_DIR"] = str(u_home / ".cache" / "uv")
    if (u_home / ".local" / "share" / "uv" / "python").is_dir():
        clean["UV_PYTHON_INSTALL_DIR"] = str(u_home / ".local" / "share" / "uv" / "python")
    # Ensure standard toolchains (user bin dir, Homebrew prefixes, /usr/sbin) are reachable
    cur_path = clean.get("PATH", "/usr/bin:/bin")
    for extra_bin in (*extra_bin_dirs(), "/usr/sbin"):
        if extra_bin not in cur_path.split(":"):
            cur_path = f"{extra_bin}:{cur_path}"
    clean["PATH"] = cur_path
    clean["HERMES_HOME"] = str(fake_hermes)
    clean["HERMES_TMUX"] = "0"
    clean["HXMAIL_DB"] = str(subdirs["tmp"] / "mail.db")
    clean["HXMAIL_LOCAL_DIR"] = str(subdirs["tmp"] / "local")
    clean["PYTHONDONTWRITEBYTECODE"] = "1"
    clean["PYTHONHASHSEED"] = "0"
    clean["TZ"] = "UTC"
    clean.setdefault("LANG", "en_US.UTF-8")
    clean.setdefault("LC_ALL", "en_US.UTF-8")
    clean.setdefault("TERM", "xterm-256color")
    clean["TESTBENCH_SANDBOX"] = "1"
    clean["TESTBENCH_RUN_ID"] = run_id
    clean["TESTBENCH_RUN_NONCE"] = nonce

    if needs_tmux:
        clean["TMUX_TMPDIR"] = str(subdirs["tmux_tmp"])
        if tmux_socket:
            clean["TESTBENCH_TMUX_SOCKET"] = tmux_socket

    if env_overrides:
        for k, v in env_overrides.items():
            if "\x00" in k or "\x00" in str(v):
                raise ValueError(f"Invalid null byte in env_overrides key {k!r}")
            clean[str(k)] = str(v)

    return clean


def _safe_step_slug(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    return slug[:64] or "step"


def _open_0600_write(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(
        str(path),
        os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    os.fchmod(fd, 0o600)
    return os.fdopen(fd, "w", encoding="utf-8")


def _kill_disposable_tmux(tmux_socket: str, tmux_tmpdir: Path) -> None:
    tmux_bin = shutil.which("tmux")
    if not tmux_bin or not tmux_socket:
        return
    env = {"PATH": os.environ.get("PATH", "/opt/homebrew/bin:/usr/bin:/bin"), "TMUX_TMPDIR": str(tmux_tmpdir)}
    try:
        subprocess.run(
            [tmux_bin, "-L", tmux_socket, "kill-server"],
            env=env,
            capture_output=True,
            timeout=3.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        pass


def run_isolated_command(
    command: List[str],
    *,
    cwd: Path,
    evidence_dir: Path,
    step_name: str,
    run_id: str,
    timeout_s: float = 60.0,
    needs_tmux: bool = False,
    needs_pty: bool = False,
    pty_size: Tuple[int, int] = (120, 36),
    allow_network: bool = False,
    allow_repo_write: bool = False,
    use_seatbelt: bool = True,
    # Internal only: TestBench-owned staging dirs (e.g. a disposable mutation copy)
    # that the sandboxed tool must write inside. Never populated from manifest input.
    extra_writable_dirs: Tuple[Path, ...] = (),
    env_overrides: Optional[Dict[str, str]] = None,
    scratch_root: Optional[Path] = None,
    cancel_flag_path: Optional[Path] = None,
    on_spawn: Optional[Callable[[int, int, Optional[str]], None]] = None,
) -> IsolatedExecResult:
    if not command or any(not isinstance(a, str) or "\x00" in a for a in command):
        raise ValueError(f"Invalid command argv: {command!r}")

    resolved_cwd = cwd.expanduser().resolve(strict=True)
    if not resolved_cwd.is_dir():
        raise ValueError(f"cwd is not a directory: {resolved_cwd}")

    evidence_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(str(evidence_dir), 0o700)
    slug = _safe_step_slug(step_name)
    stdout_path = evidence_dir / f"{slug}.stdout.log"
    stderr_path = evidence_dir / f"{slug}.stderr.log"
    sanitized_path = evidence_dir / f"{slug}.sanitized.log"

    sroot = (scratch_root.expanduser().resolve() if scratch_root else default_scratch_root())
    sroot.mkdir(parents=True, exist_ok=True)
    nonce = os.urandom(6).hex()
    sandbox_dir = sroot / f"tbw-{nonce[:4]}"
    # Keep Unix domain socket paths well under macOS sockaddr_un.sun_path (104 bytes)
    tmux_socket = f"s{nonce[:8]}" if needs_tmux else ""
    short_tmux_base = short_writable_tmux_base(sroot) if needs_tmux else sroot
    tmux_tmpdir: Optional[Path] = (short_tmux_base / f"tx{nonce[:6]}") if needs_tmux else None

    env = build_isolated_env(
        sandbox_dir,
        run_id=run_id,
        nonce=nonce,
        needs_tmux=needs_tmux,
        tmux_socket=tmux_socket,
        tmux_tmpdir=tmux_tmpdir,
        env_overrides=env_overrides,
    )

    seatbelt_state = "disabled"
    effective_argv = list(command)
    if use_seatbelt:
        sb_bin = shutil.which("sandbox-exec")
        if sb_bin is None:
            seatbelt_state = "unavailable"
        elif can_apply_seatbelt():
            extra_dirs = tuple(d for d in extra_writable_dirs if d is not None)
            if tmux_tmpdir is not None:
                extra_dirs = (*extra_dirs, tmux_tmpdir)
            uses_uv = Path(command[0]).name == "uv"
            profile = build_seatbelt_profile(
                sandbox_dir=sandbox_dir,
                cwd=resolved_cwd,
                extra_writable_dirs=extra_dirs,
                allow_network=allow_network,
                allow_repo_write=allow_repo_write,
                allow_uv_cache_write=uses_uv,
            )
            effective_argv = [sb_bin, "-p", profile, *effective_argv]
            env["TESTBENCH_SEATBELT_ACTIVE"] = "1"
            seatbelt_state = "active"
        else:
            env["TESTBENCH_SEATBELT_ACTIVE"] = "1"
            seatbelt_state = "inherited_outer"

    stdout_redactor = StreamRedactor()
    stderr_redactor = StreamRedactor()
    stdout_dec = codecs.getincrementaldecoder("utf-8")(errors="replace")
    stderr_dec = codecs.getincrementaldecoder("utf-8")(errors="replace")
    stdout_mem: List[str] = []
    stderr_mem: List[str] = []
    stdout_mem_len = 0
    stderr_mem_len = 0
    disk_bytes_written = 0

    timed_out = False
    cancelled = False
    exit_code: Optional[int] = None
    signal_num: Optional[int] = None
    pgid = -1
    start_fp: Optional[str] = None
    pty_master_fd = -1
    t0 = time.monotonic()

    reaped_survivors = 0
    leaked_children = 0
    try:
        with (
            _open_0600_write(stdout_path) as out_f,
            _open_0600_write(stderr_path) as err_f,
            _open_0600_write(sanitized_path) as san_f,
        ):
            def _emit_out(text_piece: str) -> None:
                nonlocal stdout_mem_len, disk_bytes_written
                if not text_piece:
                    return
                if stdout_mem_len < MAX_MEM_CAPTURE_CHARS:
                    keep = text_piece[: MAX_MEM_CAPTURE_CHARS - stdout_mem_len]
                    stdout_mem.append(keep)
                    stdout_mem_len += len(keep)
                if disk_bytes_written < MAX_DISK_LOG_BYTES:
                    out_f.write(text_piece)
                    san_f.write(sanitize_terminal_text(text_piece))
                    disk_bytes_written += len(text_piece)
                    # Flush per chunk so a live run's log is readable while it
                    # runs; buffered writes only reach disk at completion.
                    out_f.flush()
                    san_f.flush()

            def _emit_err(text_piece: str) -> None:
                nonlocal stderr_mem_len, disk_bytes_written
                if not text_piece:
                    return
                if stderr_mem_len < MAX_MEM_CAPTURE_CHARS:
                    keep = text_piece[: MAX_MEM_CAPTURE_CHARS - stderr_mem_len]
                    stderr_mem.append(keep)
                    stderr_mem_len += len(keep)
                if disk_bytes_written < MAX_DISK_LOG_BYTES:
                    err_f.write(text_piece)
                    san_f.write(sanitize_terminal_text(text_piece))
                    disk_bytes_written += len(text_piece)
                    err_f.flush()
                    san_f.flush()

            if needs_pty:
                pty_master_fd, slave_fd = pty.openpty()
                cols, rows = pty_size
                fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
                os.set_blocking(pty_master_fd, False)
                proc = subprocess.Popen(
                    effective_argv,
                    cwd=str(resolved_cwd),
                    env=env,
                    stdin=slave_fd,
                    stdout=slave_fd,
                    stderr=subprocess.PIPE,
                    start_new_session=True,
                    close_fds=True,
                )
                os.close(slave_fd)
            else:
                proc = subprocess.Popen(
                    effective_argv,
                    cwd=str(resolved_cwd),
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    start_new_session=True,
                    close_fds=True,
                )

            pgid = proc.pid
            start_fp = _process_start_fingerprint(proc.pid)
            if on_spawn is not None:
                on_spawn(proc.pid, pgid, start_fp)

            sel = selectors.DefaultSelector()
            try:
                if needs_pty:
                    sel.register(pty_master_fd, selectors.EVENT_READ, data="stdout")
                elif proc.stdout is not None:
                    os.set_blocking(proc.stdout.fileno(), False)
                    sel.register(proc.stdout.fileno(), selectors.EVENT_READ, data="stdout")
                if proc.stderr is not None:
                    os.set_blocking(proc.stderr.fileno(), False)
                    sel.register(proc.stderr.fileno(), selectors.EVENT_READ, data="stderr")

                deadline = t0 + max(0.05, float(timeout_s))
                while sel.get_map() or not _leader_exited(proc):
                    now = time.monotonic()
                    if cancel_flag_path is not None and cancel_flag_path.exists():
                        cancelled = True
                        terminate_process_group(pgid, expected_start_time=start_fp, grace_s=0.5)
                        break
                    if now >= deadline:
                        timed_out = True
                        terminate_process_group(pgid, expected_start_time=start_fp, grace_s=0.5)
                        break

                    wait_s = min(0.05, max(0.005, deadline - now))
                    events = sel.select(timeout=wait_s)
                    if not events and _leader_exited(proc):
                        # Process exited; do one non-blocking drain and break
                        events = sel.select(timeout=0.0)
                        if not events:
                            break

                    for key, _mask in events:
                        fd = key.fd
                        stream_kind = key.data
                        try:
                            chunk = os.read(fd, 8192)
                        except OSError as exc:
                            if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                                continue
                            # EIO on PTY master EOF on Linux/macOS
                            sel.unregister(fd)
                            continue
                        if not chunk:
                            sel.unregister(fd)
                            continue
                        if stream_kind == "stdout":
                            decoded = stdout_dec.decode(chunk, final=False)
                            if stdout_mem_len < MAX_MEM_CAPTURE_CHARS or disk_bytes_written < MAX_DISK_LOG_BYTES:
                                _emit_out(stdout_redactor.feed(decoded))
                        else:
                            decoded = stderr_dec.decode(chunk, final=False)
                            if stderr_mem_len < MAX_MEM_CAPTURE_CHARS or disk_bytes_written < MAX_DISK_LOG_BYTES:
                                _emit_err(stderr_redactor.feed(decoded))
            finally:
                sel.close()
                if pty_master_fd >= 0:
                    try:
                        os.close(pty_master_fd)
                    except OSError:
                        pass
                    pty_master_fd = -1
                if proc.stdout is not None:
                    proc.stdout.close()
                if proc.stderr is not None:
                    proc.stderr.close()

            if not (timed_out or cancelled) and _leader_exited(proc):
                # Leader exited on its own (any exit status): reap its leftover group members
                # while the zombie still pins the pgid, then reap the leader.
                reaped_survivors += _reap_group_after_leader_exit(pgid)
            try:
                rc = proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                terminate_process_group(pgid, expected_start_time=start_fp, grace_s=0.2)
                rc = proc.wait(timeout=1.0)

            _emit_out(stdout_redactor.feed(stdout_dec.decode(b"", final=True)))
            _emit_out(stdout_redactor.flush())
            _emit_err(stderr_redactor.feed(stderr_dec.decode(b"", final=True)))
            _emit_err(stderr_redactor.flush())
            out_f.flush()
            err_f.flush()
            san_f.flush()

            if rc < 0:
                signal_num = -rc
                exit_code = 128 + signal_num
            else:
                exit_code = rc
    finally:
        if pgid > 1:
            terminate_process_group(pgid, expected_start_time=start_fp, grace_s=0.2)
        if needs_tmux and tmux_socket and tmux_tmpdir is not None:
            _kill_disposable_tmux(tmux_socket, tmux_tmpdir)
            _safe_rmtree_no_symlinks(tmux_tmpdir)
        if timed_out or cancelled or (exit_code is not None and exit_code != 0):
            swept, leaked_children = _sweep_sandbox_survivors(sandbox_dir)
            reaped_survivors += swept
        if leaked_children == 0:
            _safe_rmtree_no_symlinks(sandbox_dir)

    duration_ms = (time.monotonic() - t0) * 1000.0
    return IsolatedExecResult(
        command=list(command),
        cwd=str(resolved_cwd),
        exit_code=exit_code,
        signal_num=signal_num,
        duration_ms=duration_ms,
        timed_out=timed_out,
        cancelled=cancelled,
        stdout_text="".join(stdout_mem),
        stderr_text="".join(stderr_mem),
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        sanitized_path=str(sanitized_path),
        isolation_meta={
            "seatbelt": seatbelt_state,
            "allow_network": allow_network,
            "allow_repo_write": allow_repo_write,
            "needs_tmux": needs_tmux,
            "needs_pty": needs_pty,
            "tmux_socket": tmux_socket or None,
            "reaped_survivors": reaped_survivors,
            "leaked_children": leaked_children,
        },
    )
