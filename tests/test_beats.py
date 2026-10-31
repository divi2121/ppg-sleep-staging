"""Known-answer tests for the clock offset and device shift."""
import sys
from pathlib import Path

import numpy as np
import pytest


from beats import compare_to_device_ibi, device_fiducial_shift, find_clock_offset, match_beats


def _beats(n=6000, seed=0):
    """Beat times with realistic, non-repeating variability (about 70 bpm)."""
    rng = np.random.default_rng(seed)
    slow = 0.08 * np.sin(np.arange(n) / 40.0) + 0.05 * np.sin(np.arange(n) / 9.0)
    ibi = 0.85 + slow + rng.normal(0, 0.02, n)
    return np.cumsum(ibi)


def _pulse(beat_times, t):
    """A smooth pulse wave with one bump per beat."""
    x = np.zeros_like(t)
    for b in beat_times:
        m = (t > b - 0.4) & (t < b + 0.4)
        x[m] += np.exp(-((t[m] - b) / 0.08) ** 2)
    return x


def test_match_beats_one_to_one():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.01, 1.02, 2.2, 3.04])
    ia, ib = match_beats(a, b, tol=0.05)
    assert list(ia) == [0, 2]
    assert list(ib) == [0, 3]


def test_device_shift_recovered():
    ours = _beats()
    dev = ours[::2] - 0.236
    res = device_fiducial_shift(ours, dev)
    assert res["median_s"] == pytest.approx(-0.236, abs=1e-9)
    assert res["iqr_s"] == pytest.approx(0.0, abs=1e-9)


def test_device_match_after_shift():
    ours = _beats()
    dev = ours - 0.236
    assert compare_to_device_ibi(ours, dev)["pct_device_matched"] == 0.0
    assert compare_to_device_ibi(ours, dev, shift=-0.236)["pct_device_matched"] == 100.0


def test_clock_offset_recovered():
    true_offset = 1234.56          # t64 = t100 + offset
    beats64 = _beats(n=4000) + 50.0
    t64 = np.arange(0, beats64[-1] + 5, 1 / 64)
    bvp64 = _pulse(beats64, t64)
    # the 100 Hz recording starts later and runs on its own clock
    keep = (beats64 > true_offset + 30) & (beats64 < beats64[-1] - 30)
    beats100 = beats64[keep] - true_offset
    t100 = np.arange(beats100[0] - 10, beats100[-1] + 10, 0.01)
    bvp100 = np.interp(t100 + true_offset, t64, bvp64)
    res = find_clock_offset(beats64, beats100, bvp64, t64, bvp100, t100,
                            n_windows=3, win_s=120)
    assert res["offset_s"] == pytest.approx(true_offset, abs=0.011)
    assert res["spread_ms"] <= 10


def test_clock_offset_refuses_unrelated_signal():
    beats64 = _beats(n=4000, seed=1) + 50.0
    t64 = np.arange(0, beats64[-1] + 5, 1 / 64)
    bvp64 = _pulse(beats64, t64)
    other = _beats(n=3000, seed=2)
    t100 = np.arange(other[0] - 10, other[-1] + 10, 0.01)
    bvp100 = _pulse(other, t100)
    with pytest.raises(ValueError):
        find_clock_offset(beats64, other, bvp64, t64, bvp100, t100, n_windows=3, win_s=120)


def test_device_shift_empty_is_nan_not_crash():
    ours = _beats(n=100)
    res = device_fiducial_shift(ours, ours[:5] + 10_000.0)     # nothing within 0.5 s
    assert res["n"] == 0 and np.isnan(res["median_s"])
    out = compare_to_device_ibi(ours, ours[:5], shift=res["median_s"])
    assert out["pct_ours_matched"] == 0.0 and np.isnan(out["median_abs_resid_ms"])
