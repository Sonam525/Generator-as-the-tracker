"""Make `core/` (the gentrack package) importable for pytest without requiring an editable install.

A real install (`pip install -e .`) also works; this just keeps local/CI runs
zero-setup. See "Repository layout" in README.md.
"""
import sys
from pathlib import Path

_SRC = Path(__file__).parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
