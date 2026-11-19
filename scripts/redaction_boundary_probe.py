"""Redaction boundary probe: deterministic, bounded fuzz of the sanitize surface.

Declared as a TestBench suite (profile security). Complements the example-based
tests in tests/test_sanitize.py with property checks that found real defects
(campaign F8: secret continuations leaked by the over-cap stream flush).

Invariants, checked with an independent oracle (not the module's own filter):
- sanitize_terminal_text / strip_terminal_escapes: no ESC or C1 introducers,
  no forbidden codepoints, idempotent, never raises, benign text preserved.
- StreamRedactor: planted secrets (assignments, known tokens, bearer, URL
  userinfo, PEM) never appear in emitted output under arbitrary chunk splits,
  including values longer than MAX_BUFFER_CHARS on whitespace-free lines.

Stdlib only, seeded, no writes, no network. Exit 0 = all invariants hold;
exit 1 = violation (printed with a minimal repro).
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hermes_testbench.sanitize import (  # noqa: E402
    StreamRedactor,
    sanitize_terminal_text,
    strip_terminal_escapes,
)

ESCAPE_CASES = 20000
ALNUM = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
CSI_FINALS = "mHJKABCDfsuhlrn"
TEXT = "abcXYZ019 \t\n.,;:!?-_=+/\\\"'()[]{}<>|@#$%^&*~`"
BIDI = "\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069"
ZW = "\u200b\u200c\u200d\u200e\u200f\u00ad\u034f\u061c\u180e\ufeff"
C1 = "\x80\x81\x90\x98\x9b\x9d\x9e\x9f"
C1_INTRODUCERS = (0x9B, 0x90, 0x98, 0x9D, 0x9E, 0x9F)


def forbidden(cp: int) -> bool:
    if cp in (0x09, 0x0A):
        return False
    if cp < 0x20 or cp == 0x7F:
        return True
    if 0x80 <= cp <= 0x9F:
        return True
    if cp in (0x00AD, 0x034F, 0x061C, 0x180E, 0xFEFF):
        return True
    if 0x200B <= cp <= 0x200F:
        return True
    if 0x202A <= cp <= 0x202E:
        return True
    if 0x2060 <= cp <= 0x206F:
        return True
    if 0xFFF9 <= cp <= 0xFFFB:
        return True
    if 0xE0000 <= cp <= 0xE007F:
        return True
    return False


def token(rng: random.Random) -> str:
    kind = rng.randrange(12)
    if kind == 0:
        params = "".join(rng.choice("0123456789;?") for _ in range(rng.randrange(0, 8)))
        body = params + rng.choice(CSI_FINALS)
        return "\x1b[" + body if rng.random() < 0.8 else "\x1b[" + params
    if kind == 1:
        payload = "".join(rng.choice(TEXT) for _ in range(rng.randrange(0, 20)))
        return "\x1b]" + payload + rng.choice(["\x07", "\x1b\\", "\n", ""])
    if kind == 2:
        return "\x1b]52;c;aGVsbG8=" + rng.choice(["\x07", "\x1b\\", ""])
    if kind == 3:
        return rng.choice(["\x1bP", "\x1bX", "\x1b^", "\x1b_"]) + "payload" + rng.choice(["\x1b\\", "\x07", ""])
    if kind == 4:
        return rng.choice(C1) + "31m"
    if kind == 5:
        return "\x1b]0;\x1b[31mtitle" + rng.choice(["\x07", ""])
    if kind == 6:
        return rng.choice(BIDI + ZW)
    if kind == 7:
        return "\r"
    if kind == 8:
        return rng.choice(["\ud800", "\udfff", "\ufff9", "\ufffa", "\U000e0001", "\x7f"])
    if kind == 9:
        return "\x1b" + rng.choice("c78()B=>")
    if kind == 10:
        return "\x1b"
    return "".join(rng.choice(TEXT) for _ in range(rng.randrange(1, 24)))


def escape_phase(rng: random.Random) -> list[str]:
    problems: list[str] = []
    for _ in range(ESCAPE_CASES):
        s = "".join(token(rng) for _ in range(rng.randrange(1, 10)))
        try:
            out = sanitize_terminal_text(s)
            stripped = strip_terminal_escapes(s)
        except Exception as exc:  # noqa: BLE001 - report any raise as a violation
            problems.append(f"raised {type(exc).__name__} on {s[:80]!r}")
            continue
        for label, text in (("sanitize", out), ("strip", stripped)):
            if "\x1b" in text or any(ord(c) in C1_INTRODUCERS for c in text):
                problems.append(f"{label} escape leak on {s[:80]!r}")
        bad = [hex(ord(c)) for c in out if forbidden(ord(c))]
        if bad:
            problems.append(f"forbidden codepoints {bad[:4]} on {s[:80]!r}")
        if sanitize_terminal_text(out) != out:
            problems.append(f"not idempotent on {s[:80]!r}")
    for benign in ("hello world", "plain\ttext\nline2", "100% done: ok (yes)"):
        if sanitize_terminal_text(benign) != benign:
            problems.append(f"benign text changed: {benign!r}")
    return problems


def feed_split(rng: random.Random, text: str, nchunks: int) -> str:
    r = StreamRedactor()
    out = []
    cuts = (
        sorted(rng.sample(range(1, len(text)), min(nchunks - 1, len(text) - 1)))
        if len(text) > 1 and nchunks > 1
        else []
    )
    prev = 0
    for c in cuts + [len(text)]:
        out.append(r.feed(text[prev:c]))
        prev = c
    out.append(r.flush())
    return "".join(out)


def stream_phase(rng: random.Random) -> list[str]:
    problems: list[str] = []
    cases: list[tuple[str, str, str]] = []
    for length in (8, 64, 600):
        val = "".join(rng.choice(ALNUM) for _ in range(length))
        cases.append((f"API_KEY={val}", val, f"api_key len {length}"))
        cases.append((f"Authorization: Bearer {val}", val, f"bearer len {length}"))
    val = "sk-" + "".join(rng.choice(ALNUM) for _ in range(60))
    cases.append((f"token={val}", val, "sk- token"))
    # Over-cap values on whitespace-free lines: the F8 leak class.
    long_val = "".join(rng.choice(ALNUM) for _ in range(70000))
    cases.append(("PASSWORD=" + long_val, long_val, "70K password, no newline"))
    pw = "".join(rng.choice(ALNUM) for _ in range(69000))
    cases.append(("https://user:" + pw + "@api.example.com/v1", pw, "70K URL userinfo"))
    pem = "-----BEGIN RSA PRIVATE KEY-----\n" + "\n".join(
        "".join(rng.choice(ALNUM) for _ in range(64)) for _ in range(30)
    ) + "\n-----END RSA PRIVATE KEY-----"
    cases.append((pem, pem.split("\n")[1], "pem body line"))

    for text, secret, label in cases:
        for nchunks in (1, 3, 20):
            for _ in range(3):
                try:
                    out = feed_split(rng, text, nchunks)
                except Exception as exc:  # noqa: BLE001
                    problems.append(f"{label}: raised {type(exc).__name__} ({nchunks} chunks)")
                    continue
                if secret in out:
                    problems.append(f"{label}: FULL VALUE LEAK ({nchunks} chunks)")
                elif len(secret) >= 32:
                    for i in range(0, len(secret) - 32, 4096):
                        if secret[i:i + 32] in out:
                            problems.append(f"{label}: FRAGMENT LEAK @{i} ({nchunks} chunks)")
                            break
    # Benign long lines must survive byte-for-byte.
    for text in ("word " * 15000, "x" * 70000):
        if feed_split(rng, text, 7) != text:
            problems.append("benign long line altered")
    return problems


def main() -> int:
    rng = random.Random(0x5EED2026)
    problems = escape_phase(rng) + stream_phase(rng)
    if problems:
        print(f"redaction-boundary probe: {len(problems)} VIOLATIONS")
        for p in problems[:10]:
            print("  " + p)
        return 1
    print(
        f"redaction-boundary probe: OK ({ESCAPE_CASES} escape cases, "
        "stream chunk-split trials incl. 70K over-cap values; 0 violations)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())