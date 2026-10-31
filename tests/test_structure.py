"""Tests for Config, Recording and the earlier functions."""
import dataclasses
import json

import numpy as np
import pandas as pd
import pytest

from config import Config
from preprocess import acc_motion, find_scoring_offset, make_epochs
from beats import (clean_intervals, compare_to_ecg, device_beats_from_ibi, refine_peaks,
                   sample_check_windows)


def test_config_frozen_and_roundtrip():
    c = Config()
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.fs = 100
    c2 = c.with_(causal=True)
    assert c2.causal and not c.causal
    assert json.loads(c2.to_json())["causal"] is True


def test_refine_peaks_parabola():
    t = np.arange(20, dtype=float)
    x = -(t - 10.3) ** 2
    assert refine_peaks(x, [10], fs=1)[0] == pytest.approx(10.3)
    assert refine_peaks(x, [0], fs=1)[0] == 0.0


def test_clean_intervals_flags():
    beats = np.cumsum([0.0] + [1.0] * 10 + [0.2] + [1.0] * 10)
    ci = clean_intervals(beats)
    assert (~ci.keep).sum() >= 1
    assert not ci.keep[np.isclose(ci.ibi, 0.2)].any()


def test_device_beats_from_ibi():
    ibi = np.array([np.nan, 0.8, 0.8, 0.8, 0.9, 0.9, 0.8])
    ts = np.arange(7, dtype=float)
    assert list(device_beats_from_ibi(ibi, ts)) == [1.0, 4.0, 6.0]


def test_compare_to_ecg_finds_delay():
    e = np.cumsum(np.full(500, 0.9))
    tab, best = compare_to_ecg(e + 0.25, e, delays=np.arange(0.0, 0.5, 0.01))
    assert best["delay_s"] == pytest.approx(0.25, abs=0.011) and best["f1"] > 0.99


def test_sample_check_windows():
    s = sample_check_windows(0, 30000, n=20, length=300, seed=1)
    assert len(s) == 20 and np.all(np.diff(s) >= 300)
    with pytest.raises(ValueError):
        sample_check_windows(0, 1000, n=20)


def test_scoring_offset():
    st = np.array(["P"] * 1000 + ["W"] * 1920 + ["N1"] * 1920 + ["N2"] * 3840, dtype=object)
    res = find_scoring_offset(st)
    assert res["anchor"] == 1000 and res["offset"] == 1000 and res["n_off_grid_scored"] == 0


def test_make_epochs():
    st = np.array(["P"] * 1000 + ["W"] * 1920 + ["N1"] * 1920, dtype=object)
    e = make_epochs(st, offset=1000)
    assert list(e.raw) == ["W", "N1"] and e.keep.all()
    e0 = make_epochs(st, offset=0)
    assert "mixed" in set(e0.raw)


def test_acc_motion():
    z = np.zeros(1920 * 2); z[1920:] = np.tile([1.0, -1.0], 960)
    m = acc_motion(np.zeros_like(z), np.zeros_like(z), 64 + z, [0, 1920])
    assert m[0] == 0.0 and m[1] == pytest.approx(1.0)


def test_recording_on_synthetic_file(tmp_path):
    from recording import Recording
    fs = 64
    n = fs * 30 * 40
    t = np.arange(n) / fs
    beats = np.arange(0.5, t[-1], 0.9)
    bvp = np.zeros(n)
    for b in beats:
        m = np.abs(t - b) < 0.45      # realistic width: msptdfast double-counts
        bvp[m] += np.exp(-((t[m] - b) / 0.15) ** 2)   # pulses narrower than ~0.1 s
    stage = np.array(["P"] * (fs * 100) + ["N2"] * (n - fs * 100), dtype=object)
    df = pd.DataFrame({"TIMESTAMP": t, "BVP": bvp * 50, "ACC_X": 0.0, "ACC_Y": 0.0, "ACC_Z": 64.0,
                       "TEMP": 33.0, "EDA": 0.1, "HR": 66.0, "IBI": 0.9, "Sleep_Stage": stage})
    p = tmp_path / "S999_whole_df.csv"; df.to_csv(p, index=False)
    rec = Recording("S999", p)
    assert not rec.has_ecg
    assert rec.scoring["offset"] == (fs * 100) % 1920
    assert rec.epochs.keep.sum() == 36
    assert abs(len(rec.beat_times) - beats.size) <= 2
    with pytest.raises(ValueError):
        Recording("S999", p, config=Config(label_timing="corrected")).epochs


def test_recording_in_scored(tmp_path):
    from recording import Recording
    fs = 64; n = fs * 30 * 6
    stage = np.array(["P"] * (fs * 60) + ["N2"] * (n - fs * 60), dtype=object)
    df = pd.DataFrame({"TIMESTAMP": np.arange(n) / fs, "BVP": 0.0, "ACC_X": 0.0, "ACC_Y": 0.0, "ACC_Z": 64.0,
                       "TEMP": 33.0, "EDA": 0.1, "HR": 60.0, "IBI": 1.0, "Sleep_Stage": stage})
    p = tmp_path / "S998_whole_df.csv"; df.to_csv(p, index=False)
    rec = Recording("S998", p)
    assert list(rec.in_scored([10.0, 59.0, 61.0, 179.0, 200.0])) == [False, False, True, True, False]


def _pulses(seconds=1800, fs=64, ibi=0.9):
    t = np.arange(seconds * fs) / fs
    beats = np.arange(0.5, t[-1] - 0.5, ibi)
    x = np.zeros(t.size)
    for b in beats:
        m = np.abs(t - b) < 0.45
        x[m] += np.exp(-((t[m] - b) / 0.15) ** 2)
    return x * 50, beats


def test_detect_beats_chunked_matches_whole():
    from beats import detect_beats, detect_beats_chunked
    from preprocess import filter_bvp
    x, beats = _pulses()
    f = filter_bvp(x)
    _, t_whole = detect_beats(f)
    _, t_chunk, failed = detect_beats_chunked(f, chunk_s=300)
    assert failed == [] and len(t_chunk) == len(t_whole) == beats.size
    assert np.allclose(t_chunk, t_whole)


def test_detect_beats_chunked_records_failure():
    from beats import detect_beats_chunked
    x = np.zeros(64 * 1800); x[64 * 700] = 99.0          # marker inside chunk 2 (600-1200 s)
    def det(seg):
        if (seg == 99.0).any():
            raise ValueError("boom")
        return np.array([10, 20])
    idx, _, failed = detect_beats_chunked(x, chunk_s=600, overlap_s=0, detector=det)
    assert failed == [(600.0, 1200.0)]
    assert list(idx) == [10, 20, 64 * 1200 + 10, 64 * 1200 + 20]


def test_clean_intervals_causal_uses_past_only():
    # live plan Task 1 (2026-10-24): the causal reference is the median of the previous
    # win intervals, so changing the future cannot change an earlier keep flag
    t = np.cumsum(np.r_[0.0, np.full(30, 1.0)])
    a = clean_intervals(t, causal=True)
    t2 = t.copy(); t2[25:] += np.arange(1, 7) * 0.5            # changes intervals 24 onwards
    b = clean_intervals(t2, causal=True)
    assert a.keep[:20].tolist() == b.keep[:20].tolist()
    t3 = np.r_[t[:12], t[11] + 1.5]                            # 1.5 s after eleven 1.0 s intervals
    assert not clean_intervals(t3, causal=True).keep.iloc[-1]
    assert clean_intervals(t, causal=False).equals(clean_intervals(t))
