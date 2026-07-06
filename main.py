"""Root entry point.

Adds ``src/`` to the import path so the project runs with a plain
``python main.py`` after ``pip install -r requirements.txt`` — no editable
install required — then delegates to the package CLI.
"""

from __future__ import annotations
import os
import sys

# Add the src/ directory to the import path so that the package can be imported
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from tracker_system.cli import main  # noqa: E402  (import after sys.path setup)

if __name__ == "__main__":
    raise SystemExit(main())
