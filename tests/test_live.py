"""Tests for live.py, the streaming decoder."""
import numpy as np
import pandas as pd

from conftest import simple_detector, synthetic_recording, tiny_model
from config import Config
from evaluation import make_xy
from features import normalise_per_person
from live import STREAM_COLS, LiveDecoder
from live_features import RAW_COLS, causal_features_one
from sequence import BiGRUSequence, make_sequences

CFG = Config(causal=True)


def _pulses(seconds, beat_times, fs=64):
    t = np.arange(seconds * fs) / fs
    x = np.zeros((t.size, 6)); x[:, 3] = 64.0; x[:, 4] = 33.0; x[:, 5] = 0.1
    for b in beat_times:
        m = np.abs(t - b) < 0.3
        x[m, 0] += 50 * np.exp(-((t[m] - b) / 0.08) ** 2)
    return x


def test_incomplete_epoch_gives_no_output(tmp_path):
    # Review Focus 4: 45 s of signal holds one complete epoch; the half epoch gives nothing
    dec = LiveDecoder(tiny_model(tmp_path), CFG, epoch_starts=[0, 1920], beat_start=0, detector=simple_detector)
    out = dec.push(_pulses(45, np.arange(0.25, 45, 1.0))) + dec.finish()
    assert [r.epoch for r in out] == [0]


def test_beat_at_epoch_edge(tmp_path):
    # Review Focus 1: an extra beat 0.25 s before the end of epoch 0 counts in epoch 0;
    # 0.75 s after the end it counts in epoch 1
    base = np.arange(0.25, 60, 1.0)
    res = {}
    for name, extra in [("before", 29.75), ("after", 30.75)]:
        dec = LiveDecoder(tiny_model(tmp_path), CFG, epoch_starts=[0, 1920], beat_start=0, detector=simple_detector)
        out = dec.push(_pulses(60, np.sort(np.r_[base, extra]))) + dec.finish()
        res[name] = [r.features["n_beats"] for r in out]
    assert res["before"][0] == res["after"][0] + 1
    assert res["after"][1] == res["before"][1] + 1


def test_live_matches_batch_causal(tmp_path):
    # the streaming decoder reproduces the batch-causal features and stages epoch by epoch
    rec = synthetic_recording(tmp_path / "r", minutes=40, seed=1)
    path = tiny_model(tmp_path)
    batch = causal_features_one(rec, CFG, detector=simple_detector)
    Z = normalise_per_person(batch, RAW_COLS, causal=True, min_periods=10); Z.columns = [c + "_z" for c in RAW_COLS]
    D = pd.concat([batch, Z], axis=1)
    m = BiGRUSequence.load(path)
    X, y, s, idx = make_xy(D, m.extra["cols"])
    seq = make_sequences(X, y, s, idx, D.loc[idx, "epoch"].to_numpy(), np.asarray(m.extra["medians"]))[0]
    stages_batch = m.predict([seq])[0]

    ep = rec.epochs; k = ep.keep.to_numpy(bool); ss = ep.start_sample_wrist.to_numpy()[k]
    start = max(0, rec.scoring["anchor"] - CFG.detect_pre_anchor_s * CFG.fs)
    dec = LiveDecoder(path, CFG, epoch_starts=ss, beat_start=start, detector=simple_detector)
    raw = rec.wrist[STREAM_COLS].to_numpy("float64")
    out = []
    for i in range(0, len(raw), 1000):                           # chunks that do not line up with seconds
        out += dec.push(raw[i:i + 1000])
    out += dec.finish()
    assert [r.epoch for r in out] == list(range(len(ss)))
    live = pd.DataFrame([r.features for r in out])
    pd.testing.assert_frame_equal(live[RAW_COLS], batch[RAW_COLS].reset_index(drop=True), check_dtype=False)
    codes = {"W": 0, "NREM": 1, "REM": 2}
    assert [codes[r.stage] for r in out] == list(stages_batch)
    assert [r.abstain for r in out] == list(~batch["pass"].to_numpy())


# ---- demo comparison with saved batch predictions (2026-10-26, her choice: option 1 only) ----

def test_agreement_known():
    from live import agreement
    share, n = agreement(np.array([0, 1, 2, 1]), np.array([0, 1, 1, 1, 9]))   # compared on the first 4
    assert (share, n) == (0.75, 4)
    share, n = agreement(np.array([], int), np.array([1, 2]))
    assert np.isnan(share) and n == 0


def test_saved_batch_stages(tmp_path):
    from live import saved_batch_stages
    pd.DataFrame({"subject": ["A", "A", "B", "B", "B"]}).to_csv(tmp_path / "features_all35.csv", index=False)
    assert saved_batch_stages("B", tmp_path) is None                         # predictions file missing
    pd.DataFrame({"subject": ["B", "A", "B", "A", "B"], "row": [3, 0, 4, 1, 2],   # shuffled on purpose
                  "y": 0, "yhat": [2, 0, 0, 0, 1], "fold": 0}).to_csv(tmp_path / "preds_live_5fold.csv", index=False)
    assert list(saved_batch_stages("B", tmp_path)) == [1, 2, 0]              # rows 2, 3, 4 in epoch order
    assert saved_batch_stages("C", tmp_path) is None                         # person not in the files


# ---- section H item 4 (2026-10-26): fixed 30 s clock when no epoch list is given ----

def test_fixed_clock_equals_explicit_grid(tmp_path):
    rec = synthetic_recording(tmp_path / "r", minutes=12, seed=2)
    raw = rec.wrist[STREAM_COLS].to_numpy("float64")
    path = tiny_model(tmp_path)
    a = LiveDecoder(path, CFG, detector=simple_detector)                       # fixed clock from sample 0
    b = LiveDecoder(path, CFG, epoch_starts=np.arange(24) * 1920, beat_start=0, detector=simple_detector)
    ra, rb = [], []
    for i in range(0, len(raw), 1000):
        ra += a.push(raw[i:i + 1000]); rb += b.push(raw[i:i + 1000])
    ra += a.finish(); rb += b.finish()
    assert len(ra) == len(rb) == 24                                             # 12 minutes = 24 epochs of 30 s
    assert [(r.epoch, r.stage, r.abstain) for r in ra] == [(r.epoch, r.stage, r.abstain) for r in rb]
    pd.testing.assert_frame_equal(pd.DataFrame([r.features for r in ra]), pd.DataFrame([r.features for r in rb]))


def test_fixed_clock_counts_epochs_and_beats(tmp_path):
    # one beat per second from 0.25 s: 95 s hold 3 complete epochs (0 to 90 s) and 5 s more.
    # Epoch k covers [30k, 30k + 30) s, so epochs 1 and 2 each hold exactly 30 beats.
    dec = LiveDecoder(tiny_model(tmp_path), CFG, detector=simple_detector)
    out = dec.push(_pulses(95, np.arange(0.25, 95, 1.0))) + dec.finish()
    assert [r.epoch for r in out] == [0, 1, 2]
    assert [r.features["n_beats"] for r in out][1:] == [30, 30]
