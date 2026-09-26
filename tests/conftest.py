"""Shared pytest fixtures. Kept minimal on purpose - the test suite is
opinionated about touching real disk only via `tmp_path`, so most tests
build up whatever they need inline and don't share state."""
from __future__ import annotations

import os
import sys

# Make the repo root importable no matter where pytest is invoked from.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
