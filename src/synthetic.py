"""Made-up wrist recordings for tests and the no-data demo.

Nothing here comes from DREAMT.

ASSUMPTIONS: one Gaussian pulse per beat, beat intervals uniform between 0.855 and 0.945 s
(seeded); ACC is gravity on z (64 raw units) plus small seeded noise; TEMP and EDA rise slowly.
A model trained on real sleep decodes such a signal into stages that mean nothing.
"""
import numpy as np
import pandas as pd

STREAM_COLS = ["BVP", "ACC_X", "ACC_Y", "ACC_Z", "TEMP", "EDA"]


# LOGIC: a 64 Hz wrist table with the DREAMT column names. The first 100 s are labelled P and
#   the rest N2. With scramble_after_min, BVP and ACC after that minute are replaced by seeded
#   noise, to check that nothing earlier changes. extra_beat_s adds one pulse at that time.
# EDGE CASES: the same seed gives the same table; minutes may be fractional (bug found by running
#   the demo with --minutes 16.0, regression test in test_synthetic.py).
# NOT HANDLED: realistic physiology of any kind.
# CHECKED BY: tests/test_synthetic.py::test_synthetic_stream_shape_and_determinism and the
#   tests that build recordings from it (test_live_features.py, test_live.py)
def synthetic_wrist(minutes=40, seed=0, scramble_after_min=None, extra_beat_s=None):
    rng = np.random.default_rng(seed)
    fs = 64; n = int(round(fs * 60 * minutes)); t = np.arange(n) / fs     # minutes may be a float (--minutes 16.0)
    beats = np.cumsum(rng.uniform(0.855, 0.945, int(minutes * 60 / 0.85)))
    beats = beats[beats < t[-1] - 1]
    if extra_beat_s is not None:
        beats = np.sort(np.r_[beats, extra_beat_s])
    bvp = np.zeros(n)
    for b in beats:
        m = np.abs(t - b) < 0.45
        bvp[m] += np.exp(-((t[m] - b) / 0.15) ** 2)
    acc = rng.normal(0, 0.5, (n, 3)); acc[:, 2] += 64.0
    if scramble_after_min is not None:
        k = t >= scramble_after_min * 60
        noise = np.random.default_rng(seed + 99)
        bvp[k] = noise.normal(0, 1, k.sum()); acc[k] = noise.normal(0, 5, (k.sum(), 3))
    stage = np.array(["P"] * (fs * 100) + ["N2"] * (n - fs * 100), dtype=object)
    return pd.DataFrame({"TIMESTAMP": t, "BVP": bvp * 50, "ACC_X": acc[:, 0], "ACC_Y": acc[:, 1],
                         "ACC_Z": acc[:, 2], "TEMP": 33.0 + t / 3600, "EDA": 0.1 + t / 36000,
                         "HR": 66.0, "IBI": 0.9, "Sleep_Stage": stage})


# LOGIC: the six channels a live decoder reads (STREAM_COLS), as an (n, 6) float array.
# EDGE CASES: none.
# NOT HANDLED: none.
# CHECKED BY: tests/test_synthetic.py::test_synthetic_stream_shape_and_determinism
def synthetic_stream(minutes=20, seed=0):
    return synthetic_wrist(minutes, seed)[STREAM_COLS].to_numpy("float64")


# LOGIC: write the table to disk as a DREAMT-style file and open it as a Recording.
# EDGE CASES: the Recording uses Config(causal=True) unless another config is given.
# NOT HANDLED: none.
# CHECKED BY: tests/test_live_features.py, tests/test_live.py
def synthetic_recording(directory, minutes=40, seed=0, scramble_after_min=None, extra_beat_s=None, config=None):
    from config import Config
    from recording import Recording
    directory.mkdir(parents=True, exist_ok=True)
    df = synthetic_wrist(minutes, seed, scramble_after_min, extra_beat_s)
    p = directory / "S999_whole_df.csv"; df.to_csv(p, index=False)
    return Recording("S999", p, config=config if config is not None else Config(causal=True))
