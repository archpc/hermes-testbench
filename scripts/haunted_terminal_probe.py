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

This file is only the CLI shell around ``run_haunted_terminal_probe``: it owns argument
parsing, the one-line-per-case rendering, and the process exit code. Every measurement,
invariant, and teardown lives in the fixture, so a change in what is *checked* belongs
there and a change in how it is *printed* belongs here.

Output contract (consumed by ``parse_pty_tmux`` and by humans):

* ``PASS <case>`` / ``FAIL <case>: <why>`` / ``SKIP <case>: <why>`` -- one line per case.
  These are the count that matters: ``parse_pty_tmux`` applies its marker grammar at
  line start, so total/pass/fail/skip are derived from exactly these lines. The text
  after ``FAIL <case>:`` is kept as the failure label, which is why the reason is on the
  marker line rather than on a following line;
* ``AREA <area> cases=N pass=N fail=N skip=N`` -- one line per area, always in
  ``HAUNTED_TERMINAL_AREAS`` order and always present even when an area produced
  nothing, so an empty area reads as ``cases=0`` rather than a vanished line;
* ``HAUNTED_SUMMARY mode=... areas=8 total=... tmux=on|off pty=on|off`` -- the single
  aggregate line. It uses ``pass=``/``fail=`` tokens, not ``"N passed"``, so the parser's
  summary heuristic deliberately ignores it: counts come from the markers, and this line
  exists for logs and humans;
* ``HAUNTED_JSON=<json>`` -- only with ``--json``, the full report including per-area
  buckets and the transport sub-reports.

Exit code is ``0`` iff ``report["failed"] == 0``. That is *not* the same as "passed":
an all-SKIP run exits 0 by this rule, which is why the summary also carries ``tmux=`` /
``pty=`` availability and the fixture's own ``all_passed`` (``failed == 0 and passed >
0``) is the stricter signal. The script deliberately does not substitute its exit code
for that distinction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Run as a bare script (``.testbench.toml`` invokes it by path), so the package beside
# it is not importable yet: prepend the repo root rather than requiring an install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hermes_testbench.fixtures import (  # noqa: E402
    HAUNTED_TERMINAL_AREAS,
    run_haunted_terminal_probe,
)


def main(argv=None) -> int:
    """Parse the CLI, run the probe, render markers, and return the exit code.

    Arguments map 1:1 onto ``run_haunted_terminal_probe``:

    ``--fault-injection``
        route the checker through the defective renderer/geometry-caching child; the run
        is *expected* to fail, and is used as the control that the assertions bite.
    ``--no-tmux``
        skip the disposable tmux transport; those cases become ``SKIP`` with
        "tmux disabled by caller", not failures.
    ``--scratch-root``
        parent directory for the probe's temp tree (default: the isolation scratch
        root). The tree is created by the fixture and removed in its ``finally``.
    ``--json``
        append the whole report as one ``HAUNTED_JSON=`` line after the summary.

    ``argv`` defaults to ``sys.argv[1:]`` so the function is importable and callable from
    a test with an explicit list. Side effects are confined to the scratch tree: the
    probe spawns a PTY child and, unless ``--no-tmux``, a short-lived ``-L`` tmux server
    which it kills during teardown.
    """
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
