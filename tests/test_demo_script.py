"""Runs scripts/live_demo.py as a separate process."""
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "live_demo.py"


def _run(*args, timeout=240):
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True, timeout=timeout)


def test_synthetic_demo_runs_without_data():
    pytest.importorskip("pylsl")
    r = _run("--synthetic", "--minutes", "16", "--speed", "400")
    if r.returncode == 3:
        pytest.skip("LSL could not find a local stream on this machine")
    assert r.returncode == 0, r.stderr[-500:]
    assert "SYNTHETIC SIGNAL" in r.stdout                                      # the banner says the stages mean nothing
    # 16 minutes = 960 s = 32 epochs of 30 s; every sample arrives (replay starts after a 2 s lead)
    assert "received 61440 of 61440 samples | decoded 32 epochs" in r.stdout


def test_synthetic_demo_refuses_a_short_run():
    r = _run("--synthetic", "--minutes", "5")
    assert r.returncode != 0
    assert "--minutes 15 or more" in (r.stdout + r.stderr)                       # 10 epochs of 30 s = 5 min needed for scaling
