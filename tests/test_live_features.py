"""Tests for live_features.py."""
import numpy as np
import pandas as pd

from conftest import simple_detector, synthetic_recording
from config import Config
from features import normalise_per_person
from live_features import RAW_COLS, causal_features_one

CFG = Config(causal=True)


def test_causal_features_ignore_the_future(tmp_path):
    a = causal_features_one(synthetic_recording(tmp_path / "a", 40, 0), CFG, detector=simple_detector)
    b = causal_features_one(synthetic_recording(tmp_path / "b", 40, 0, scramble_after_min=20), CFG,
                            detector=simple_detector)
    assert list(a.columns[:5]) == ["subject", "epoch", "t_start", "pass", "stage3"]
    assert set(RAW_COLS) <= set(a.columns) and len(RAW_COLS) == 22
    # an epoch's stage is decided live_lag_s + 1 s after it ends; decided before minute 20 = unchanged
    early = a.t_start + 30 + CFG.live_lag_s + 1 <= 20 * 60
    assert early.sum() >= 15
    pd.testing.assert_frame_equal(a[early], b[early])
    late = a.t_start >= 22 * 60
    assert not a.loc[late, RAW_COLS].equals(b.loc[late, RAW_COLS])   # the scramble did reach the features


def test_heart_features_blank_when_gate_fails(tmp_path):
    f = causal_features_one(synthetic_recording(tmp_path / "c", 40, 0, scramble_after_min=20), CFG,
                            detector=simple_detector)
    from live_features import HEART_COLS
    assert f.loc[~f["pass"], HEART_COLS].isna().all(axis=None)
    assert (~f["pass"]).sum() > 0


def test_first_nine_epochs_have_no_z():
    F = pd.DataFrame({"subject": "A", "x": np.arange(20.0)})
    Z = normalise_per_person(F, ["x"], causal=True, min_periods=10)
    assert Z.x.iloc[:9].isna().all() and Z.x.iloc[9:].notna().all()
