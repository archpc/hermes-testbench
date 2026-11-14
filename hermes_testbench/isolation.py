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
        "SSH_AUTH_SOCK",
        "SSH_AGENT_PID",
        "GPG_AGENT_INFO",
        "SECURITYSESSIONID",
    }
)

_SCRUB_KEY_RE = re.compile(
    r"(?i)(?:"
    r"API_KEY|SECRET|TOKEN|PASSWORD|PASSWD|CREDENTIAL|PRIVATE_KEY|COOKIE|SESSION_KEY|AUTH_KEY"
    r"|^(?:OPENAI|ANTHROPIC|OPENROUTER|NOUS|DEEPSEEK|GITHUB|GH|AWS|FASTMAIL|SLACK|DISCORD|TELEGRAM)_"
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


def default_scratch_root() -> Path:
    for var in ("TESTBENCH_SCRATCH_DIR", "TMPDIR"):
        val = os.environ.get(var, "").strip()
        if val:
            p = Path(val).expanduser()
            p.mkdir(parents=True, exist_ok=True)
            return p.resolve()
    fallback = real_user_home() / ".hermes" / "cache" / "scratch"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback.resolve()


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
        # If the leader is still alive with a different start fingerprint, PGID was recycled.
        if live_fp is not None and live_fp != expected_start_time:
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


def build_seatbelt_profile(
    *,
    sandbox_dir: Path,
    cwd: Path,
    extra_writable_dirs: Tuple[Path, ...] = (),
    allow_network: bool = False,
    allow_repo_write: bool = False,
) -> str:
    u_home = real_user_home()
    hermes_dir = u_home / ".hermes"
    hxmail_support = u_home / "Library" / "Application Support" / "hxmail"
    resolved_sb = sandbox_dir.resolve()
    resolved_cwd = cwd.resolve()

    rules = [
        "(version 1)",
        "(allow default)",
    ]
    if not allow_network:
        rules.append("(deny network*)")
        rules.append("(allow network* (local unix))")

    rules.append(
        "(deny file-read-data "
        f"(subpath {_sb_quote(u_home / '.ssh')}) "
        f"(subpath {_sb_quote(u_home / '.aws')}) "
        f"(subpath {_sb_quote(u_home / '.gnupg')}) "
        f"(literal {_sb_quote(hermes_dir / '.env')}) "
        f"(literal {_sb_quote(hermes_dir / 'auth.json')}) "
        f"(subpath {_sb_quote(hermes_dir / 'mcp-tokens')}) "
        f"(subpath {_sb_quote(hxmail_support)}))"
    )
    rules.append(
        "(deny file-write* "
        f"(subpath {_sb_quote(hermes_dir)}) "
        f"(subpath {_sb_quote(hxmail_support)}))"
    )
    if not allow_repo_write:
        rules.append(f"(deny file-write* (subpath {_sb_quote(resolved_cwd)}))")

    extra_allows = " ".join(f"(subpath {_sb_quote(p.resolve())})" for p in extra_writable_dirs)
    rules.append(
        f"(allow file-write* (subpath {_sb_quote(resolved_sb)}) {extra_allows} (subpath \"/dev\"))"
    )
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
    for sub in ("home", "xdg_config", "xdg_data", "xdg_cache", "tmp", "tmux_tmp"):
        d = sandbox_dir / sub
        d.mkdir(parents=True, exist_ok=True)
        os.chmod(str(d), 0o700)
        subdirs[sub] = d
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
    sandbox_dir = sroot / f"tbw-{nonce[:8]}"
    # Keep Unix domain socket paths well under macOS sockaddr_un.sun_path (104 bytes)
    tmux_socket = f"s{nonce[:5]}" if needs_tmux else ""
    short_tmux_base = sroot
    if needs_tmux and len(str(sroot)) > 52:
        cand_short = real_user_home() / ".hermes" / "cache" / "scratch"
        if cand_short.is_dir() and os.access(str(cand_short), os.W_OK):
            short_tmux_base = cand_short
    tmux_tmpdir: Optional[Path] = (short_tmux_base / f"tbt-{nonce[:6]}") if needs_tmux else None

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
            extra_dirs = (tmux_tmpdir,) if tmux_tmpdir is not None else ()
            profile = build_seatbelt_profile(
                sandbox_dir=sandbox_dir,
                cwd=resolved_cwd,
                extra_writable_dirs=extra_dirs,
                allow_network=allow_network,
                allow_repo_write=allow_repo_write,
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
                while sel.get_map() or proc.poll() is None:
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
                    if not events and proc.poll() is not None:
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
                            _emit_out(stdout_redactor.feed(decoded))
                        else:
                            decoded = stderr_dec.decode(chunk, final=False)
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
        },
    )
