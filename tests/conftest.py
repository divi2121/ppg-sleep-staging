"""Puts src/ and qc/ on the import path for every test."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "qc"))


# ---- shared helpers for the live tests (added 2026-10-24) ----
import numpy as np
import pandas as pd


def simple_detector(seg):
    """Fast stand-in for msptdfast in tests: local maxima above half the buffer's maximum."""
    seg = np.asarray(seg, "float64")
    if seg.size < 3 or seg.max() <= 0:
        return np.array([], int)
    i = np.flatnonzero((seg[1:-1] > seg[:-2]) & (seg[1:-1] >= seg[2:]) & (seg[1:-1] > 0.5 * seg.max())) + 1
    return i


from synthetic import synthetic_recording  # noqa: E402,F401  (moved to src/synthetic.py, 2026-10-26)


def tiny_model(directory, seed=0):
    """Saves a small forward-only model with the live feature layout and returns its path.

    ASSUMPTIONS: trained for 1 pass on random data, so its predictions mean nothing; it
    only has to be a real saved model with 22 z columns (+ 22 missing flags) as input.
    """
    from live_features import RAW_COLS
    from sequence import BiGRUSequence
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    nights = [{"subject": f"s{i}", "x": rng.normal(size=(50, 44)).astype("float32"),
               "y": rng.integers(0, 3, 50), "row": np.arange(50)} for i in range(4)]
    m = BiGRUSequence(n_classes=3, max_epochs=1, causal=True, conv_ch=4, hidden=4).fit(nights[:3], nights[3:])
    p = directory / "tiny.pt"
    m.save(p, extra={"cols": [c + "_z" for c in RAW_COLS], "medians": [0.0] * len(RAW_COLS)})
    return p
