"""Hermes TestBench: reusable testing, benchmarking, and verification core.

The package is deliberately import-light: this module pulls in nothing, so
``import hermes_testbench`` (and reading ``__version__`` from packaging or a
build hook) cannot drag the PTY, sandbox, or SQLite machinery along with it.
Submodules are imported explicitly by whoever needs them:

* ``cli`` / ``__main__`` — command-line entry point.
* ``engine``, ``adapters``, ``store``, ``schema`` — suite execution, results,
  persistence.
* ``isolation`` — sandboxes, process-group ownership, teardown.
* ``fixtures`` — deterministic corpora and fault probes (measured results plus
  derived booleans).
* ``tui_render``, ``sanitize`` — the rendering and escaping paths the terminal
  fixtures exercise.

Nothing in the package starts a process, opens a socket, or writes a file at
import time; every side effect belongs to a function call.
"""

__version__ = "0.1.0"
