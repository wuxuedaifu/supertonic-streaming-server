"""Pytest configuration for the service tests.

The production modules under ``app/src/`` import each other flat
(``from audio_format import ...``, ``from text_preprocess import ...``) and
are run in the container as ``python src/server.py``, which puts ``src/`` on
``sys.path`` automatically. Under pytest nothing does that, so every test
module used to carry its own ``sys.path.insert(...)`` preamble. That lives
here once instead.

``tests/`` goes on the path too: test_chat_shape_compat, test_request_logging
and test_legacy_api_compat all import the stub helpers
(``FakeSynth``, ``_install_stubs``) out of test_speech_endpoint.

Run from anywhere:

    pytest app/tests -q
"""

from __future__ import annotations

import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
APP_DIR = TESTS_DIR.parent
SRC_DIR = APP_DIR / "src"

for _p in (SRC_DIR, TESTS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# test_tts.py is not a pytest module — it is the standalone Auralis-derived
# pressure-test CLI, superseded by benchmarks/pressure_test.py. Its name
# matches pytest's default `test_*.py` discovery pattern, so collection
# imports it, and it pulls in aiohttp (a `bench` dependency, not `dev`) at
# module level. That ImportError aborts the whole run — one unrunnable file
# takes the other 115 tests down with it.
collect_ignore = ["test_tts.py"]
