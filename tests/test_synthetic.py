"""Tests for synthetic.py."""
import numpy as np

from synthetic import STREAM_COLS, synthetic_stream, synthetic_wrist


def test_synthetic_stream_shape_and_determinism():
    x = synthetic_stream(minutes=2, seed=5)
    assert x.shape == (2 * 60 * 64, 6)                                  # 7,680 samples, 6 channels
    assert STREAM_COLS == ["BVP", "ACC_X", "ACC_Y", "ACC_Z", "TEMP", "EDA"]
    assert abs(x[:, 3].mean() - 64.0) < 0.05                             # gravity on z; noise sd 0.5 over 7,680 samples
    assert x[0, 4] == 33.0 and x[-1, 4] > 33.0                           # TEMP starts at 33 and rises
    assert np.array_equal(x, synthetic_stream(minutes=2, seed=5))        # same seed, same signal
    assert not np.array_equal(x, synthetic_stream(minutes=2, seed=6))    # other seed, other signal
    w = synthetic_wrist(minutes=3)
    assert (w.Sleep_Stage == "P").sum() == 100 * 64 and len(w) == 3 * 60 * 64


def test_synthetic_stream_accepts_fractional_minutes():
    # regression: the demo passes --minutes as a float; 2.5 min at 64 Hz = 150 s x 64 = 9,600 samples
    assert synthetic_stream(minutes=2.5).shape == (9600, 6)
