"""Tests for families.py."""
import numpy as np
import pandas as pd
import pytest

from families import _dfa_alpha1, _sampen, eda_features, hr_minutes, nonlinear_hrv, stillness


def _iv(t, ibi, keep=True):
    return pd.DataFrame({"t": np.asarray(t, float), "ibi": np.asarray(ibi, float), "keep": keep})


def test_hr_minutes_known():
    t = np.arange(1.0, 601.0)                                   # one beat per second
    r = hr_minutes(_iv(t, np.ones_like(t)), [300.0])            # constant 60 bpm
    assert r.hr1_median[0] == pytest.approx(60) and r.hr1_range[0] == pytest.approx(0)
    assert r.hr5_slope[0] == pytest.approx(0)
    ibi = 60.0 / (60.0 + 0.1 * t)                                # HR rises 0.1 bpm per s = 6 per min
    r = hr_minutes(_iv(t, ibi), [300.0])
    assert r.hr5_slope[0] == pytest.approx(6.0, rel=1e-3)
    assert r.hr1_min[0] == pytest.approx(60 + 0.1 * 300) and r.hr1_max[0] == pytest.approx(60 + 0.1 * 329)
    k = np.ones_like(t, bool); k[(t > 290) & (t < 340)] = False  # a 50 s hole of rejected beats
    r = hr_minutes(_iv(t, np.ones_like(t), k), [300.0])
    assert r[["hr1_median", "hr1_std"]].isna().all(axis=None)    # the epoch is inside the hole


def test_nonlinear_known():
    x = np.array([800.0, 820.0, 800.0, 820.0])
    d = np.diff(x)                                               # 20, -20, 20
    iv = _iv(np.arange(1, 5) * 0.8, x / 1000)
    r = nonlinear_hrv(iv, [0.0], epoch_s=4, win_s=100, min_n=4).iloc[0]
    assert r.nl_sd1 == pytest.approx(np.sqrt(0.5) * d.std(ddof=1))
    assert r.nl_sd2 == pytest.approx(np.sqrt(max(2 * x.var(ddof=1) - 0.5 * d.var(ddof=1), 0)))
    assert nonlinear_hrv(iv, [0.0], epoch_s=4, win_s=100, min_n=5).isna().all(axis=None)
    rng = np.random.default_rng(0)
    assert _sampen(np.tile([1.0, 2.0], 100)) == pytest.approx(0.0)   # perfectly regular
    assert _sampen(rng.normal(size=300)) > 1.5                       # white noise is unpredictable
    assert _dfa_alpha1(rng.normal(size=2000)) == pytest.approx(0.5, abs=0.12)
    assert _dfa_alpha1(np.cumsum(rng.normal(size=2000))) == pytest.approx(1.5, abs=0.15)


def test_stillness_known():
    fs = 4
    mag_z = np.ones(fs * 100)                                    # 100 s, still
    mag_z[fs * 50: fs * 52] = np.tile([1, 5], fs)                # seconds 50 and 51 move
    z0 = np.zeros_like(mag_z)
    r = stillness(z0, z0, mag_z, [fs * 40, fs * 50, fs * 60], fs=fs, epoch_s=10)
    assert list(r.move_secs) == [0, 2, 0]
    assert r.still_frac[1] == pytest.approx(0.8)
    assert r.secs_since_move[2] == pytest.approx(69 - 51)        # end of epoch (s 69) minus last move (s 51)
    assert r.secs_since_move[0] == 1800.0                        # nothing moved yet


def test_eda_known():
    fs = 64; n = fs * 600                                        # 10 min
    t = np.arange(n) / fs
    eda = 2.0 + 0.001 * t                                        # slow rise, 0.06 microsiemens per min
    for c in [100.0, 110.0, 300.0]:                              # three sharp bursts
        eda = eda + 0.2 * np.exp(-((t - c) ** 2) / (2 * 0.5 ** 2))
    r = eda_features(eda, [fs * 90, fs * 285, fs * 400])
    assert list(r.eda_bursts) == [2, 1, 0]
    assert r.eda_tonic_slope[2] == pytest.approx(0.06, rel=0.15)
    assert r.eda_bursts5[0] == 2      # regression: a filter ripple at 123.8 s was counted as a burst


def test_hr_minutes_causal_window_ends_at_epoch():
    # live plan Task 1 (2026-10-24): window [t0 + 30 - 300, t0 + 30), nothing after it
    t = np.arange(1.0, 601.0)
    ibi = 60.0 / (60.0 + 0.1 * t)                                # HR rises 6 bpm per min
    r = hr_minutes(_iv(t, ibi), [300.0], causal=True)
    assert r.hr5_slope[0] == pytest.approx(6.0, rel=1e-3)        # window is [30, 330)
    ibi2 = ibi.copy(); ibi2[t > 330] = 0.5                       # the future changes wildly
    assert hr_minutes(_iv(t, ibi2), [300.0], causal=True).equals(r)


def test_nonlinear_causal_ignores_future():
    rng = np.random.default_rng(0)
    t = np.cumsum(rng.uniform(0.8, 1.0, 900)); x = np.diff(np.r_[0, t])
    a = nonlinear_hrv(_iv(t, x), [400.0], causal=True)           # window [130, 430)
    x2 = x.copy(); x2[t > 430] = 0.6
    assert nonlinear_hrv(_iv(t, x2), [400.0], causal=True).equals(a)
