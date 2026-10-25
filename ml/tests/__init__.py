"""Test package for the GSMS ML laboratory (stdlib unittest only).

Imports here set up sys.path so that every test module can do a plain
``from gsms_ml... import ...`` regardless of the caller's working directory:
the ``ml`` directory (which contains the ``gsms_ml`` package) is prepended to
sys.path. Run with, from the repository root:

    python3.10 -m unittest discover -s ml/tests -t .
"""

import sys
from pathlib import Path

_ML_DIR = Path(__file__).resolve().parent.parent
for _path in (str(_ML_DIR), str(_ML_DIR.parent)):
    if _path not in sys.path:
        sys.path.insert(0, _path)
