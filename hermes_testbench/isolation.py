"""Execution isolation: sandboxes, Seatbelt, process-group lifecycle, teardown.

Every suite process in TestBench is spawned here, with two documented exceptions:
the messaging suites (``messaging_suite``) and ``PtySession`` spawn their own
children outside this path. Three layers, in decreasing strength:

1. **Environment** — a disposable ``tbw-<nonce>`` tree with ``0700`` ``HOME``, ``XDG_*``,
   ``TMPDIR``, ``HERMES_HOME``, and credential variables scrubbed by *name*. This is
   redirection, not confinement: a suite that opens an absolute path still reaches the
   real file.
2. **Kernel confinement (macOS only)** — a deny-by-default ``sandbox-exec`` profile built
   by :func:`build_seatbelt_profile`. ``allow_network`` / ``allow_repo_write`` are
   *recorded* on every platform but only *enforced* where Seatbelt exists.
3. **Process ownership** — one ``setsid`` session per suite, so the leader's pid is also
   the process-group id. Every signal goes to that group, or to a survivor that provably
   holds this run's sandbox directory. ``terminate_process_group`` gates its group signal
   on a ``ps -o lstart=`` start-time fingerprint so a recycled pid is never hit; the
   post-reap member sweep (``_reap_group_after_leader_exit``) and ``PtySession.close``
   signal the group without that fingerprint, within a bounded TOCTOU window.

Invariant the rest of the package leans on: **containment is not lifecycle correctness.**
``isolation_meta['seatbelt'] == 'active'`` says this process was confined;
``leaked_children == 0`` says *this* process found nothing left to kill. Neither says the
suite cleaned up after itself. A descendant that called ``setsid()``/``setpgid()`` into a
new group is outside the owned group, and on a clean exit nothing hunts it — the survivor
sweep only runs for timeout, cancel and non-zero exit. So a green run proves the
environment was protected, not that the suite is well-behaved. Read
``reaped_survivors``/``leaked_children`` as observations of what this run had to clean up,
never as a verdict on the suite.

Teardown (group signal, disposable tmux ``kill-server``, sandbox deletion) runs from a
``finally`` block, so it also executes on ``KeyboardInterrupt`` and on adapter exceptions.
It must never follow a symlink the sandboxed child planted: after spawn the child owns its
``TMPDIR``/``TMUX_TMPDIR`` and may replace them with links to ambient directories.
"""

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
import stat
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
        # Shell startup/config leakage: a suite must not inherit the caller's rc
        # plumbing, option bundles, or directory-walk behaviour. ``HOME`` is
        # already a disposable overlay, but ZDOTDIR/BASH_ENV/ENV can point past
        # it at the real configuration, and inherited SHELLOPTS can hand
        # ``errexit``/``noclobber`` to a script that never asked for them.
        "ZDOTDIR",
        "BASH_ENV",
        "ENV",
        "SHELLOPTS",
        "BASHOPTS",
        "CDPATH",
        "GLOBIGNORE",
    }
)

_SCRUB_KEY_RE = re.compile(
    r"(?i)(?:"
    r"API_KEY|SECRET|TOKEN|PASSWORD|PASSWD|CREDENTIAL|PRIVATE_KEY|COOKIE|SESSION_KEY|AUTH_KEY|_KEY$"
    r"|^(?:OPENAI|ANTHROPIC|OPENROUTER|NOUS|DEEPSEEK|GITHUB|GH|AWS|FASTMAIL|SLACK|DISCORD|TELEGRAM|HERMES_PEER)_"
    r")"
)

MAX_MEM_CAPTURE_CHARS = 4 * 1024 * 1024
# Both budgets are counted in *decoded characters*, not bytes, and MAX_DISK_LOG_BYTES is a
# single budget shared by stdout and stderr (whichever streams harder starves the other).
# UTF-8 output can therefore reach up to ~4x this many bytes on disk. The cap is a runaway
# guard, not a quota; the name is historical.
MAX_DISK_LOG_BYTES = 64 * 1024 * 1024


@dataclass
class IsolatedExecResult:
    """Everything the engine learns about one suite execution.

    ``timed_out`` and ``cancelled`` are mutually exclusive (the reader loop breaks on the
    first one it sees) but neither suppresses ``exit_code``: a killed leader yields
    ``128 + signal`` — a timeout normally reports 143, not ``None``. Classify on the flags,
    never on the code. ``stdout_text``/``stderr_text`` are the redacted in-memory capture
    (each capped at ``MAX_MEM_CAPTURE_CHARS``); the ``*_path`` fields point at the ``0600``
    logs, which are capped separately by ``MAX_DISK_LOG_BYTES``. ``isolation_meta`` is the
    provenance record persisted with the run: it describes what isolation *was applied*, not
    what the suite did.
    """

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
    """Return a writable directory short enough for macOS ``sockaddr_un.sun_path``.

    ``sun_path`` is 104 bytes on macOS (108 on Linux) *including* the terminating NUL, and
    tmux appends ``/tmux-<uid>/<socket-name>`` to ``TMUX_TMPDIR``. A base of at most 68
    characters leaves room for ``tx``+6 hex, ``tmux-501`` and ``default`` (68+9+9+8+NUL ≈ 95
    < 104). A longer base does not fail here — it fails later as ``bind: File name too long``
    from tmux, inside the suite, which reads as a suite bug.

    Candidates are probed for writability in order, then ancestors of ``preferred`` are tried.
    The walk stops before ``$HOME`` and ``~/.hermes`` themselves: a tmux socket directory
    directly in a protected root is one the sandbox and the teardown sweeps may not legally
    touch. If nothing qualifies the unresolved preference is returned anyway — a too-long path
    is a visible failure, a redirected one is not.
    """

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
    """SIGTERM a process group, then SIGKILL it after ``grace_s``.

    Returns True when the group is gone (or never existed), False when the fingerprint
    guard refused to signal. ``expected_start_time`` is the ``ps -o lstart=`` value sampled
    from the leader at spawn time; if that group id now belongs to another process the
    values differ and nothing is sent. Callers pass the sampled value unconditionally, which
    is also why this function cannot finish a job whose leader has already been reaped: the
    pgid then fingerprints as ``None``, the comparison fails, and the call returns False.
    :func:`_reap_group_after_leader_exit` exists for exactly that state — it identifies the
    group's members by ``ps`` instead of by the leader's identity.
    """
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
# macOS has no waitid(2) at the libc level before CPython 3.13, so on 3.11/3.12 there the
# leader must be reaped by poll() and its pgid stops being reserved. Everything downstream of
# this flag is written to survive both worlds: no code may assume the pgid is still pinned.


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
    """Live, non-zombie members of ``pgid`` other than ``leader``.

    The ``killpg(pgid, 0)`` probe is a cheap group-exists test and its error codes are
    load-bearing: ``ESRCH`` means the group is gone, while ``EPERM`` on macOS is what an
    all-zombie group looks like (the unreaped leader still holds the id), so both are
    reported as "no live members" rather than as a signal failure. The members themselves
    come from ``ps`` because there is no portable way to enumerate a process group; a
    member in state ``Z`` is excluded because killing it is a no-op that would otherwise
    keep the caller waiting out its grace period.
    """
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

    This path is member-based, not fingerprint-based, on purpose: when the leader has been
    reaped (always on macOS < 3.13, and after ``proc.wait()`` everywhere) its pid may already
    be recycled, so ``terminate_process_group`` would refuse. That leaves a TOCTOU window
    between the last ``ps`` sample and ``killpg``; it is bounded by the re-check immediately
    before the signal and by the fact that a recycled pgid must also be spawned into a group
    whose id equals the dead leader's pid inside those few milliseconds.
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
    """True when ``sandbox-exec`` exists *and* accepts a trivial profile.

    The probe really executes ``/usr/bin/true`` under a one-rule deny profile, so it tests
    the mechanism, not just the binary. A missing binary means "no Seatbelt on this host";
    a present binary that fails the probe is read by the caller as "we are already inside an
    outer Seatbelt" (TestBench nested in TestBench). That second inference is a guess: a
    future macOS that rejects ``sandbox-exec`` for an unrelated reason is then reported as
    ``inherited_outer`` — a claim about confinement this process cannot actually verify.
    """
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
    """Kill processes that belong to this run but escaped its process group.

    Two detectors because neither is complete: ``lsof +D`` finds holders of any file under
    the directory, ``ps -ax`` finds the path in a command line or inherited cwd. Both are
    *mentions*, not proofs of ownership — a process that merely quotes the path in argv is
    killed too. That is acceptable only because the path carries a per-run nonce, so a false
    positive has to be another process quoting this exact run's sandbox.

    Never called on a clean exit (see :func:`run_isolated_command`): it costs tens of
    milliseconds and would turn every fast, well-behaved suite into a full ``ps`` scan.
    Returns ``(signalled, still_alive)``; a non-zero second element is what keeps the
    sandbox tree on disk for inspection.
    """

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
    """Render the SBPL profile for one suite.

    SBPL is *last match wins*, so the order below is load-bearing rather than cosmetic:
    ``(allow default)``, then ``(deny file-write*)``, then one ``(allow file-write* ...)``
    for the sandbox, TestBench-owned staging dirs and the specific ``/dev`` nodes. Writes are
    deny-by-default because the set of places a suite may legitimately write cannot be
    enumerated (LaunchAgents, dotfiles, cron, sibling repos, ``state.db``). Reads are a
    *denylist*, and that asymmetry is deliberate: denying a read breaks tools that probe for
    optional config, while allowing one leaks a credential. A credential file not named in
    the ``deny file-read-data`` clause stays readable.

    The tmux rules close a channel rather than a path: XNU resolves the *vnode* for
    ``AF_UNIX``, so ``(deny file-read* (subpath /private/tmp/tmux-<uid>))`` defeats tmux's
    ``chdir``+relative-``sun_path`` ``connect()`` where ``(remote unix-socket (subpath ...))``
    would not, and the explicit ``network-outbound`` denies cover the same socket and
    ``mDNSResponder`` by address (ADR-0005). Loopback is re-allowed for ``localhost:*`` after
    the blanket ``(deny network*)`` so suites can host mock servers, with the Hermes
    dashboard/dev ports (9119, 9222, 11434) denied again afterwards.

    ``(allow process-exec (literal "/bin/ps") (with no-sandbox))`` exists because suites (and
    tmux itself) shell out to ``ps`` to inspect processes, and a sandboxed ``ps`` cannot read
    other processes' information. It is one literal path on purpose — not a prefix, not
    ``/bin/*``.

    ``allow_repo_write=False`` leaves the checkout read-only but still *readable*; the profile
    never denies reads of the project tree, because a test suite that cannot read its own
    sources is useless.
    """
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

    # Emitted as an SBPL ``;`` comment, not as a rule: it records inside the profile which
    # roots are protected only by the *absence* of an allow rule. An audit reading the log
    # should not expect a matching ``(deny file-write* ...)`` for these — deny-by-default is
    # the mechanism. Keep the payload a syntactically valid SBPL comment; anything else here
    # would be parsed as a rule and, being last, would win.
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
    """Build one suite's environment: scrubbed, redirected, pinned.

    Not a security boundary by itself. Scrubbing matches variable *names*
    (``_SCRUB_EXACT`` plus ``_SCRUB_KEY_RE``), so a secret held in a variable with an
    unrecognised name passes through untouched. The ``HOME``/``TMPDIR``/``XDG_*`` redirect
    only moves tools that honour those variables; anything opening an absolute path reaches
    the real file on Linux, and is stopped (if at all) by Seatbelt on macOS.

    Choices worth not "cleaning up" later:

    * ``TESTBENCH_SCRATCH_DIR`` points at the sandbox ``tmp``, so a suite that invokes
      ``testbench`` (nested run, e.g. the terminal-tape or mutation adapters) puts its own
      sandboxes inside this one and cannot leak them into the real scratch root.
    * ``HERMES_TMUX=0`` keeps a suite from opening permanent ``hx:*`` windows on the
      developer's live tmux server; ``HXMAIL_DB``/``HXMAIL_LOCAL_DIR`` keep the mail fixture
      off the real mailbox.
    * ``PYTHONHASHSEED=0``, ``TZ=UTC``, ``LANG``/``LC_ALL``/``TERM`` are pinned so two runs
      of the same suite produce comparable evidence on different hosts.
    * ``CARGO_HOME`` is a fresh directory containing *symlinks* to ``registry``/``git``/``bin``
      only — ``credentials.toml`` sits at the ``CARGO_HOME`` root and is therefore absent,
      which is what makes the read-only crate cache safe to share. ``uv`` caches are linked
      per-item for the same reason.
    * ``env_overrides`` is applied last and is *not* scrubbed: a declared name wins over the
      scrubber. That is the point for build settings, and it is also a live bypass — a
      manifest ``[suite.env]`` entry can put ``TMUX``, ``SSH_AUTH_SOCK`` or an
      ``*_API_KEY`` back into the child (ADR-0005's trust gate covers ``allow_network`` and
      ``allow_repo_write``, not ``env``). Do not "tidy" the ordering into scrub-after-
      override without adding that gate; the mutation adapter already had one env-var
      confinement bypass removed.

    Subdirectory names are abbreviated (``xc``, ``xd``, ``xk``, ``tx``) purely to keep
    Unix-socket and ``TMPDIR`` paths inside ``sun_path``; see
    :func:`short_writable_tmux_base`.
    """
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


def _tmux_tmpdir_is_trusted(tmux_tmpdir: Path, expected_parent: Optional[Path] = None) -> bool:
    """True only when ``tmux_tmpdir`` is a real directory directly under ``expected_parent``.

    The sandboxed suite *owns* ``tmux_tmpdir`` and may delete it and plant a symlink to an
    ambient directory (e.g. ``/private/tmp``) before exiting. Teardown must never follow such
    a link when walking sockets, signalling servers, or sweeping survivors.
    """
    try:
        st = os.lstat(tmux_tmpdir)
    except OSError:
        return False
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
        return False
    if expected_parent is None:
        return True
    try:
        return Path(os.path.realpath(tmux_tmpdir)).parent == Path(os.path.realpath(expected_parent))
    except OSError:
        return False


def _tmux_socket_files(tmux_tmpdir: Path, expected_parent: Optional[Path] = None) -> List[Path]:
    """Socket paths a tmux server may have created inside our own ``TMUX_TMPDIR``.

    tmux names its default socket ``$TMUX_TMPDIR/tmux-<uid>/default``; ``-L name`` servers land
    in the same directory under ``name``. Only this directory is ever listed — nothing outside
    ``tmux_tmpdir`` is inspected or signalled.

    ``tmux_tmpdir``, ``tmux-<uid>`` and every entry are checked with ``lstat`` so a
    child-planted symlink (or a symlinked socket) is never traversed.
    """
    if not _tmux_tmpdir_is_trusted(tmux_tmpdir, expected_parent):
        return []
    base = tmux_tmpdir / f"tmux-{os.getuid()}"
    try:
        base_st = os.lstat(base)
        if stat.S_ISLNK(base_st.st_mode) or not stat.S_ISDIR(base_st.st_mode):
            return []
        if Path(os.path.realpath(base)).parent != Path(os.path.realpath(tmux_tmpdir)):
            return []
        uid_real = Path(os.path.realpath(base))
        found: List[Path] = []
        for entry in base.iterdir():
            try:
                if entry.is_symlink():
                    continue
                if not stat.S_ISSOCK(os.lstat(entry).st_mode):
                    continue
                if Path(os.path.realpath(entry)).parent != uid_real:
                    continue
            except OSError:
                continue
            found.append(entry)
        return found
    except OSError:
        return []


def _kill_disposable_tmux(
    tmux_socket: str,
    tmux_tmpdir: Path,
    expected_parent: Optional[Path] = None,
) -> None:
    """Kill every tmux server reachable through this disposable ``TMUX_TMPDIR``.

    ``-L <socket>`` alone is not enough: suites that call plain ``tmux new-session`` get the
    default server at ``$TMUX_TMPDIR/tmux-<uid>/default``, which survives the ``-L`` kill and
    outlives the unlinked ``tmux_tmpdir`` as an orphan (reproduced: 83 leaked ``deepcheck``
    servers). Each socket inside ``tmux_tmpdir`` is therefore killed by path with ``-S``.

    If the child replaced ``tmux_tmpdir`` with a symlink (escaping ``expected_parent``) no
    server is signalled at all — only the planted link itself is unlinked.
    """
    if not _tmux_tmpdir_is_trusted(tmux_tmpdir, expected_parent):
        _safe_rmtree_no_symlinks(tmux_tmpdir)
        return
    tmux_bin = shutil.which("tmux")
    # Resolved from the *parent's* PATH, not the suite's: if the host has no tmux, no server
    # can be killed here and the TMUX_TMPDIR survivor sweep below is the only cleanup left.
    if not tmux_bin:
        return
    env = {"PATH": os.environ.get("PATH", "/opt/homebrew/bin:/usr/bin:/bin"), "TMUX_TMPDIR": str(tmux_tmpdir)}
    targets: List[List[str]] = []
    if tmux_socket:
        targets.append([tmux_bin, "-L", tmux_socket, "kill-server"])
    targets.extend(
        [tmux_bin, "-S", str(sock), "kill-server"]
        for sock in _tmux_socket_files(tmux_tmpdir, expected_parent)
    )
    for argv in targets:
        try:
            subprocess.run(argv, env=env, capture_output=True, timeout=3.0, check=False)
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
    on_output: Optional[Callable[[str], None]] = None,
) -> IsolatedExecResult:
    """Run one suite argv under isolation and return the full record of how it ended.

    Arguments that change behavior, not just labeling:

    * ``cwd`` must already exist (resolved strict). It is the *only* directory the suite is
      told to work in; it is not made writable unless ``allow_repo_write``.
    * ``evidence_dir`` gets ``<slug>.stdout.log`` / ``.stderr.log`` / ``.sanitized.log`` at
      ``0600``; the slug is derived from ``step_name``, so two suites with the same name in
      one run overwrite each other's logs.
    * ``cancel_flag_path`` is polled from inside the read loop; its presence counts as
      cancellation and terminates the group. The flag is never removed here — its existence
      is also what fences the run's final status.
    * ``on_spawn(pid, pgid, start_fp)`` is called once, synchronously, before the loop, so a
      *different* process can cancel this run without guessing. ``on_output`` receives
      sanitized text only (never raw bytes), which is why foreground ``--stream`` cannot
      print what the disk log withheld.
    * ``use_seatbelt=False`` is for tests and for suites that must run unconfined; the
      resulting ``isolation_meta['seatbelt'] == 'disabled'`` is recorded, not hidden.

    Completion is not simply "the leader exited": the loop also needs both pipes at EOF.
    A descendant that escapes the session but keeps an inherited pipe open keeps the loop
    running until ``timeout_s``, and the suite is then reported TIMEOUT even though the
    leader exited 0. That is deliberate — an open pipe is indistinguishable from output
    still arriving, and cutting it off by pid would be a guess.

    Teardown, in ``finally``, in this order: signal our own process group → kill disposable
    tmux servers and sweep their ``TMUX_TMPDIR`` → sweep sandbox survivors (timeout, cancel
    or non-zero exit only) → delete the sandbox tree unless a survivor is still holding it.
    Consequences worth knowing before editing: a *clean* exit deliberately runs no sandbox
    sweep, so a ``setsid()`` daemon survives it and its sandbox directory is then deleted
    from under it; and a tree with a live survivor is left in place on purpose, because the
    survivor's cwd/argv point into it and a human needs it more than the disk.

    Raises ``ValueError`` for a malformed argv or a missing cwd — before spawning anything,
    so an adapter's error path can report ``ERROR`` without having started a process.
    """
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
    # Captured BEFORE the child is spawned: the child owns its tmpdir after that and may
    # replace it with a symlink, so teardown must validate against this trusted parent.
    expected_tmux_parent: Optional[Path] = short_tmux_base.resolve() if needs_tmux else None
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
                safe = sanitize_terminal_text(text_piece)
                if on_output is not None:
                    on_output(safe)
                if disk_bytes_written < MAX_DISK_LOG_BYTES:
                    out_f.write(text_piece)
                    san_f.write(safe)
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
                safe = sanitize_terminal_text(text_piece)
                if on_output is not None:
                    on_output(safe)
                if disk_bytes_written < MAX_DISK_LOG_BYTES:
                    err_f.write(text_piece)
                    san_f.write(safe)
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
            # Safety net for exits that never reach proc.wait(): an adapter exception or a
            # ^C raised inside the loop leaves the leader unreaped, so the fingerprint still
            # matches and the group is really signalled. On the normal path the leader has
            # been waited on by now, _process_start_fingerprint returns None, and this call
            # refuses — by design, since the pgid may already belong to someone else.
            terminate_process_group(pgid, expected_start_time=start_fp, grace_s=0.2)
        if needs_tmux and tmux_socket and tmux_tmpdir is not None:
            _kill_disposable_tmux(tmux_socket, tmux_tmpdir, expected_parent=expected_tmux_parent)
            # A default-named server (plain `tmux new-session`) is only reachable by the
            # socket sweep above; reap anything else still holding this tmpdir open. Both the
            # kill and this sweep are skipped when the child planted a symlink in place of
            # tmux_tmpdir, so an ambient directory (e.g. /private/tmp) is never touched.
            if _tmux_tmpdir_is_trusted(tmux_tmpdir, expected_tmux_parent):
                _sweep_sandbox_survivors(tmux_tmpdir)
            _safe_rmtree_no_symlinks(tmux_tmpdir)
        if timed_out or cancelled or (exit_code is not None and exit_code != 0):
            swept, leaked_children = _sweep_sandbox_survivors(sandbox_dir)
            reaped_survivors += swept
        if leaked_children == 0:
            # Left on disk when a survivor is still alive: its cwd/argv point into the tree, so
            # it is the only evidence of what escaped this run.
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
            # Provenance, not a verdict: what isolation was applied and what this run had to
            # clean up. A green run with rounded-down counters says nothing about whether the
            # suite would clean up after itself on a clean exit (see module docstring).
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
