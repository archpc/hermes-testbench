"""Module entrypoint: python3.14 -m hermes_testbench.

``main`` returns an exit code rather than raising, so it is passed straight to
``sys.exit``; an exception escaping it would be reported as a traceback and exit
1, indistinguishable from a suite that legitimately failed. Keeping the call in
this module (and only under ``__main__``) means importing the package from
``python -m``'s machinery never runs the CLI.
"""

from __future__ import annotations

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
