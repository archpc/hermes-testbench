#!/usr/bin/env python3
"""Haunted Terminal probe: executable ``pty_tmux`` suite entrypoint (Phase 1C).

Run under TestBench's ``terminal`` profile::

    testbench run --profile terminal

Command declared in ``.testbench.toml`` (needs_pty=True, needs_tmux=True). The probe
measures eight observable terminal-integrity areas over a real PTY and a disposable
``-L`` tmux server, then prints one ``PASS``/``FAIL``/``SKIP`` marker per invariant so
``parse_pty_tmux`` can count them.

``--fault-injection`` routes the same invariant checker through a deliberately broken
renderer and a geometry-caching PTY child. The probe must then exit non-zero with
actionable per-case failures: that is the control proving the assertions bite.

Never touches the ambient environment: no live Ghostty, no ambient tmux server, and
every tmux invocation carries ``-L <disposable-socket>`` plus an isolated TMUX_TMPDIR.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hermes_testbench.fixtures import (  # noqa: E402
    HAUNTED_TERMINAL_AREAS,
    run_haunted_terminal_probe,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fault-injection",
        action="store_true",
        help="Exercise the assertion checker against deliberately broken behaviour",
    )
    parser.add_argument("--no-tmux", action="store_true", help="Skip the disposable tmux transport")
    parser.add_argument("--scratch-root", default="", help="Isolated scratch directory")
    parser.add_argument("--json", action="store_true", help="Also print the full report as JSON")
    args = parser.parse_args(argv)

    report = run_haunted_terminal_probe(
        fault_injection=bool(args.fault_injection),
        scratch_root=Path(args.scratch_root) if args.scratch_root else None,
        use_tmux=not args.no_tmux,
    )

    for case in report["cases"]:
        if case["status"] == "PASS":
            print("PASS " + case["name"])
        elif case["status"] == "SKIP":
            print("SKIP " + case["name"] + ": " + (case["message"] or "not applicable"))
        else:
            print("FAIL " + case["name"] + ": " + (case["message"] or "invariant violated"))

    for area in HAUNTED_TERMINAL_AREAS:
        counts = report["areas"].get(area, {"total": 0, "pass": 0, "fail": 0, "skip": 0})
        print(
            "AREA %s cases=%d pass=%d fail=%d skip=%d"
            % (area, counts["total"], counts["pass"], counts["fail"], counts["skip"])
        )

    mode = "fault-injection" if report["fault_injection"] else "faithful"
    print(
        "HAUNTED_SUMMARY mode=%s areas=%d total=%d pass=%d fail=%d skip=%d tmux=%s pty=%s"
        % (
            mode,
            len(HAUNTED_TERMINAL_AREAS),
            report["total"],
            report["passed"],
            report["failed"],
            report["skipped"],
            "on" if report["tmux"]["available"] else "off",
            "on" if report["pty"]["available"] else "off",
        )
    )
    if args.json:
        print("HAUNTED_JSON=" + json.dumps(report, sort_keys=True))
    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
