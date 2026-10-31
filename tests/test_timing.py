"""Known-answer tests for timing.py."""
import numpy as np
import pandas as pd
import pytest

from timing import (ClockModel, correct_epoch_times, fit_clock_model, label_offset,
                    local_lock_offset, misalignment_line, pool_misalignment)


def test_clock_model_roundtrip():
    c = ClockModel(a=8594.0, s=0.0013)
    t = np.array([0.0, 1000.0, 20000.0])
    assert np.allclose(c.to100(c.to64(t)), t)
    assert c.to64(1000.0) == pytest.approx(1000 + 8594 + 1.3)


def test_fit_clock_model_ignores_outlier():
    x = np.linspace(600, 22000, 12)
    y = 8594.0 + 0.0013 * x
    y[7] -= 7.0                      # one bad window, like S022 at 6 h
    c = fit_clock_model(x, y)
    assert c.s == pytest.approx(0.0013, rel=1e-6)
    assert c.a == pytest.approx(8594.0, abs=1e-6)
    assert c.max_resid_s == pytest.approx(7.0, abs=1e-6)


def test_fit_clock_model_needs_three():
    with pytest.raises(ValueError):
        fit_clock_model([1, 2], [3, 4])


def test_misalignment_line_matches_definition():
    c = ClockModel(a=8594.0, s=0.0013)
    L = 8580.0
    m0, m1 = misalignment_line(c, L)
    for t64 in [9000.0, 20000.0, 30000.0]:
        direct = c.to64(t64 - L) - t64
        assert m0 + m1 * t64 == pytest.approx(direct)


def test_pool_misalignment_mean():
    assert pool_misalignment([(2.0, 0.001), (4.0, 0.003)]) == (3.0, 0.002)
    with pytest.raises(ValueError):
        pool_misalignment([])


def test_correct_epoch_times_shift():
    t64 = np.arange(0, 3600, 1 / 64)
    ep = pd.DataFrame({"start_sample": np.arange(0, 3000 * 64, 1920), "keep": True})
    out = correct_epoch_times(ep, t64, m0=10.0, m1=0.0)
    assert np.allclose(out.t_start_wrist - t64[ep.start_sample], 10.0)
    assert (out.start_sample_wrist - ep.start_sample == 640).all()
    late = correct_epoch_times(ep, t64, m0=700.0, m1=0.0)
    assert not late.keep.iloc[-1] and late.keep.iloc[0]


def test_label_offset_recovered():
    st = np.array(["W"] * 30 + ["N1"] * 60 + ["N2"] * 90 + ["W"] * 30 + ["R"] * 60, dtype=object)
    t100 = np.arange(st.size, dtype=float)
    st64 = np.concatenate([np.array(["P"] * 500, dtype=object), st])
    t64 = np.arange(st64.size, dtype=float)
    off, share = label_offset(st, t100, st64, t64)
    assert off == 500.0 and share == 1.0


def test_local_lock_offset_recovered():
    rng = np.random.default_rng(0)
    r = np.cumsum(0.9 + rng.normal(0, 0.05, 400))
    p = r + 1234.5 + 0.30            # clock offset 1234.5, constant delay 0.30
    res = local_lock_offset(r, p, np.arange(1230, 1240, 0.01), t_start=r[10], win_s=300)
    assert abs(((res["offset_s"] + res["delay_s"]) - 1234.8)) < 0.02
    assert res["lock_share"] > 0.95
