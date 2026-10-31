"""Tests for features.py and rmssd."""
import numpy as np
import pandas as pd
import pytest

from beats import rmssd
from features import (breathing_features, context_features, epoch_hrv, freq_features,
                      movement_features, normalise_per_person, pulse_amplitude, signal_features,
                      temp_features)


def _iv(ibi):
    t = np.cumsum(ibi)
    return pd.DataFrame({"t": t, "ibi": ibi, "keep": True})


def test_rmssd_known():
    iv = _iv(np.array([1.0, 1.1, 1.0, 1.1, 1.0]))
    assert rmssd(iv) == pytest.approx(100.0)


def test_epoch_hrv_known():
    iv = _iv(np.array([1.0, 1.1] * 20))
    f = epoch_hrv(iv, [0.0], epoch_s=30)
    assert f.rmssd_ms[0] == pytest.approx(100.0)
    assert f.pnn50[0] == 1.0
    assert f.hr_mean[0] == pytest.approx(60 / 1.05, rel=0.01)


def test_freq_features_find_hf_rhythm():
    t = np.cumsum(np.full(600, 0.5))
    ibi = 1.0 + 0.05 * np.sin(2 * np.pi * 0.25 * t)       # 0.25 Hz = HF band
    iv = pd.DataFrame({"t": t, "ibi": ibi, "keep": True})
    f = freq_features(iv, [150.0])
    assert f.hf[0] > 5 * f.lf[0]
    fc = freq_features(iv, [-20.0], causal=True)
    assert np.isnan(fc.lf[0])                                 # not enough past data


def test_signal_features_means():
    w = pd.DataFrame({"TEMP": np.r_[np.full(1920, 33.0), np.full(1920, 34.0)], "EDA": 0.2})
    f = signal_features(w, [0, 1920])
    assert list(f.temp_mean) == [33.0, 34.0]


def test_context_causal_uses_past_only():
    df = pd.DataFrame({"subject": ["A"] * 6 + ["B"] * 3, "x": [1, 2, 3, 4, 5, 6, 10, 20, 30.0]})
    c = context_features(df, ["x"], k=2, causal=True)
    assert np.isnan(c["x_past2"].iloc[0]) and c["x_past2"].iloc[3] == 2.5
    assert np.isnan(c["x_past2"].iloc[6])                    # no leakage from A into B
    o = context_features(df, ["x"], k=2, causal=False)
    assert o["x_next2"].iloc[0] == 2.5 and np.isnan(o["x_next2"].iloc[5])


def test_normalise_causal_matches_expanding():
    df = pd.DataFrame({"subject": "A", "x": np.arange(20, dtype=float)})
    z = normalise_per_person(df, ["x"], causal=True, min_periods=5)
    exp = (df.x - df.x.expanding(5).mean()) / df.x.expanding(5).std()
    assert np.allclose(z.x.dropna(), exp.dropna())
    zo = normalise_per_person(df, ["x"], causal=False)
    assert zo.x.mean() == pytest.approx(0.0)


def test_pulse_amplitude_known():
    fs = 64
    x = np.zeros(4 * fs)
    peaks = np.array([32, 96, 160, 224])              # one beat per second
    x[peaks] = [1.0, 2.0, 4.0, 6.0]                    # troughs stay at 0 -> amplitudes 2, 4, 6 after the first
    iv = pd.DataFrame({"t": peaks[1:] / fs, "ibi": 1.0, "keep": True})
    r = pulse_amplitude(x, peaks, iv, [0], fs=fs, epoch_s=4)
    assert r.amp_mean[0] == pytest.approx(4.0)
    assert r.amp_std[0] == pytest.approx(2.0)
    assert r.amp_cv[0] == pytest.approx(0.5)

    iv2 = iv.copy(); iv2.loc[1, "keep"] = False        # beat at 160 dropped -> 2 used beats -> NaN
    r2 = pulse_amplitude(x, peaks, iv2, [0], fs=fs, epoch_s=4)
    assert r2[["amp_mean", "amp_std", "amp_cv"]].isna().all(axis=None)

    with pytest.raises(ValueError):                    # intervals from a different beat list
        pulse_amplitude(x, peaks, iv.iloc[:2], [0], fs=fs, epoch_s=4)


def test_breathing_known_rates():
    t = np.arange(1.0, 400.0)                                  # one beat per second
    iv = pd.DataFrame({"t": t, "ibi": 1 + 0.05 * np.sin(2 * np.pi * 0.25 * t), "keep": True})
    amp = np.r_[np.nan, 1 + 0.1 * np.sin(2 * np.pi * 0.20 * t)]  # one per beat, first beat has none
    r = breathing_features(iv, amp, [185.0])                   # window 50 to 350 s
    assert r.br_rate_ibi[0] == pytest.approx(15.0)             # 0.25 Hz = 15 per minute
    assert r.br_rate_amp[0] == pytest.approx(12.0)             # 0.20 Hz = 12 per minute
    assert r.br_reg_ibi[0] > 0.9 and r.br_reg_amp[0] > 0.9     # a pure rhythm is regular
    assert breathing_features(iv, amp, [185.0], min_n=400).isna().all(axis=None)


def test_temp_features_known():
    x = np.arange(100) * 0.5                                    # 0.5 degrees per sample at fs = 1
    r = temp_features(x, [50], fs=1, epoch_s=2, win_s=10)       # centre 51, samples 46..55 = 23..27.5
    assert r.temp_min[0] == 23.0 and r.temp_max[0] == 27.5
    assert r.temp_slope[0] == pytest.approx(30.0)               # 0.5 per second = 30 per minute
    assert r.temp_std[0] == pytest.approx(0.5 * np.std(np.arange(10), ddof=1))
    assert temp_features(x, [99], fs=1, epoch_s=2, win_s=2).isna().all(axis=None)   # 1 sample left


def test_movement_features_known():
    x = np.r_[np.zeros(9), 100.0]; y = np.full(10, 3.0); z = np.arange(10.0)
    r = movement_features(x, y, z, [0], fs=5, epoch_s=2).iloc[0]
    assert r.acc_x_tmean == 0 and r.acc_x_iqr == 0 and r.acc_x_mad == 0   # one spike is trimmed
    assert r.acc_y_tmean == 3 and r.acc_y_iqr == 0
    assert r.acc_z_tmean == pytest.approx(4.5) and r.acc_z_iqr == pytest.approx(4.5) and r.acc_z_mad == pytest.approx(2.5)
    z2 = np.tile([0, 2, 0, 2, 0], 2).astype(float)              # magnitude = z when x = y = 0
    r2 = movement_features(np.zeros(10), np.zeros(10), z2, [0], fs=5, epoch_s=2).iloc[0]
    assert r2.act_index == pytest.approx(np.sqrt(0.96))         # std of 0,2,0,2,0 in each 1 s block
