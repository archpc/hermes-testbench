"""Module entrypoint: python3.14 -m hermes_testbench."""

from __future__ import annotations

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
