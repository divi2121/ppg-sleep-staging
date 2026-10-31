"""Tests for quality.py."""
import numpy as np
import pandas as pd

from config import Config
from quality import epoch_quality, quality_gate, template_r
from beats import clean_intervals


def _clean(n=1920, fs=64, ibi=0.9):
    t = np.arange(n) / fs
    beats = np.arange(0.5, t[-1] - 0.5, ibi)
    x = np.zeros(n)
    for b in beats:
        m = np.abs(t - b) < 0.3
        x[m] += np.exp(-((t[m] - b) / 0.08) ** 2)
    return x, np.round(beats * fs).astype(int), beats


def test_template_r_clean_vs_noise():
    x, idx, _ = _clean()
    assert template_r(x, idx) > 0.99
    noise = np.random.default_rng(0).normal(size=x.size)
    assert template_r(noise, idx) < 0.5
    assert np.isnan(template_r(x, idx[:2]))


def test_epoch_quality_counts():
    x, idx, beats = _clean()
    q = epoch_quality(x, idx, clean_intervals(beats), [0])
    assert q.n_beats[0] == idx.size and q.plausible[0] and q.pct_kept[0] == 1.0


def test_quality_gate():
    q = pd.DataFrame({"plausible": [True, True, False], "pct_kept": [0.9, 0.5, 1.0], "template_r": [0.9, 0.9, 0.9]})
    assert list(quality_gate(q, [0.1, 0.1, 0.1], Config())) == [True, False, False]
