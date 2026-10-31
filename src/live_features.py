"""Per-epoch features exactly as a live decoder sees them (live plan Task 3).

Design: docs/live_decoding_design.md section 7.

ASSUMPTIONS, all stated here once:
- The pulse is filtered forward only (Config(causal=True) -> filter_bvp causal), from the
  first sample of the file, as a live stream would be.
- Beats come from StreamingBeats fed 1 s at a time from detect_pre_anchor_s (300 s) before
  the scorer's first epoch, the same start as the offline detector. A beat counts once it
  is Config.live_lag_s old.
- Intervals are cleaned against the previous 11 intervals (clean_intervals causal=True).
- Every window ends at the epoch end (freq_features, hr_minutes, nonlinear_hrv causal=True).
- The quality gate and its thresholds are the offline ones (Config); heart features are
  blank (NaN) where it fails, as in features_all35.csv.
- Times are seconds from the first sample of the file, sample n at n / fs. The offline
  table used the file's TIMESTAMP column; for DREAMT these agree (checked in notebook 10).
- Scaling (z-scores) is NOT done here; the caller scales with the night so far.
"""
import numpy as np
import pandas as pd

from beats import clean_intervals
from families import hr_minutes, nonlinear_hrv
from features import epoch_hrv, freq_features, signal_features
from preprocess import acc_motion
from quality import epoch_quality, quality_gate
from stream import StreamingBeats

HRV = ["hr_mean", "sdnn_ms", "rmssd_ms", "pnn50", "lf", "hf", "lf_hf"]
OTHER = ["motion", "temp_mean", "eda_mean"]
HRMIN = ["hr1_median", "hr1_min", "hr1_max", "hr1_range", "hr1_std", "hr5_std", "hr5_slope"]
NONLIN = ["nl_sd1", "nl_sd2", "nl_sd1_sd2", "nl_sampen", "nl_dfa_a1"]
RAW_COLS = HRV + OTHER + HRMIN + NONLIN            # 22 columns, the order the live model uses
HEART_COLS = HRV + HRMIN + NONLIN                  # blanked where the quality gate fails


# LOGIC: run StreamingBeats over a filtered pulse as if it arrived live: 1 s pushes from
#   sample `start` to the end. Returns beat sample indices and times (s from sample 0).
# EDGE CASES: the last push may be shorter than 1 s; beats in the final lag_s are never
#   confirmed and are missing, as they would be live.
# NOT HANDLED: none.
# CHECKED BY: tests/test_stream.py (StreamingBeats); notebook 10 section 1 (real nights)
def stream_beats(bvp_filt, start, cfg, detector=None):
    fs = cfg.fs
    sb = StreamingBeats(fs=fs, buffer_s=cfg.live_buffer_s, lag_s=cfg.live_lag_s, detector=detector)
    idx, t = [], []
    for i in range(int(start), len(bvp_filt), fs):
        a, b = sb.push(bvp_filt[i:i + fs])
        idx.append(a); t.append(b)
    idx = np.concatenate(idx) + int(start) if idx else np.array([], int)
    t = np.concatenate(t) + start / fs if t else np.array([])
    return idx, t


# LOGIC: the 22 raw features and the gate for epochs starting at samples ss, from signals
#   that share one frame (sample 0 = time 0 s). Same steps as the notebook 05 build cell,
#   with the causal windows, plus the two heart families. Shared by the batch builder
#   (whole file) and LiveDecoder (a recent slice, frame shifted).
# EDGE CASES: heart features are NaN where the gate fails; n_beats is returned for checks.
# NOT HANDLED: scaling (caller).
# CHECKED BY: tests/test_live_features.py, tests/test_live.py::test_live_matches_batch_causal
def features_for_epochs(sig, beat_idx, intervals, ss, cfg):
    """sig: dict with bvp_filt, acc_x, acc_y, acc_z, temp, eda arrays. Returns a DataFrame."""
    fs, es = cfg.fs, cfg.epoch_s
    ss = np.asarray(ss, int); ts = ss / fs
    q = epoch_quality(sig["bvp_filt"], beat_idx, intervals, ss, 0.0, fs, es)
    motion = acc_motion(sig["acc_x"], sig["acc_y"], sig["acc_z"], ss, fs, es)
    wrist = pd.DataFrame({"TEMP": sig["temp"], "EDA": sig["eda"]})
    f = pd.concat([epoch_hrv(intervals, ts, es),
                   freq_features(intervals, ts, es, cfg.freq_win_s, causal=True),
                   signal_features(wrist, ss, fs, es),
                   hr_minutes(intervals, ts, es, causal=True),
                   nonlinear_hrv(intervals, ts, es, causal=True)], axis=1)
    f["motion"] = motion
    f["pass"] = quality_gate(q, motion, cfg)
    f.loc[~f["pass"], HEART_COLS] = np.nan
    f["n_beats"] = q["n_beats"].to_numpy()
    return f


# LOGIC: the batch version for one person: forward-filtered pulse, streamed beats, causal
#   intervals, then features_for_epochs on the kept (scored) epochs.
# EDGE CASES: cfg.causal must be True (otherwise the pulse filter would be zero-phase).
# NOT HANDLED: epochs the scorer did not score (P, missing) are not decoded.
# CHECKED BY: tests/test_live_features.py::test_causal_features_ignore_the_future
def causal_features_one(rec, cfg, detector=None):
    if not cfg.causal:
        raise ValueError("causal_features_one needs Config(causal=True)")
    if rec.config != cfg:
        raise ValueError("the Recording must be loaded with the same Config")
    fs = cfg.fs
    w = rec.wrist
    x = rec.bvp_filt
    start = max(0, rec.scoring["anchor"] - cfg.detect_pre_anchor_s * fs)
    bi, bt = stream_beats(x, start, cfg, detector)
    iv = clean_intervals(bt, cfg.ibi_lo, cfg.ibi_hi, cfg.ibi_rel, cfg.ibi_win, causal=True)
    ep = rec.epochs; k = ep.keep.to_numpy(bool)
    ss = ep.start_sample_wrist.to_numpy()[k]
    sig = {"bvp_filt": x, "acc_x": w.ACC_X.to_numpy("float64"), "acc_y": w.ACC_Y.to_numpy("float64"),
           "acc_z": w.ACC_Z.to_numpy("float64"), "temp": w.TEMP.to_numpy("float64"),
           "eda": w.EDA.to_numpy("float64")}
    f = features_for_epochs(sig, bi, iv, ss, cfg)
    head = pd.DataFrame({"subject": rec.subject, "epoch": np.flatnonzero(k), "t_start": ss / fs,
                         "pass": f["pass"].to_numpy(), "stage3": ep.label3.to_numpy()[k]})
    return pd.concat([head, f[RAW_COLS + ["n_beats"]]], axis=1)


# LOGIC: build one person for the all-35 table (worker for a process pool): load with
#   Config(causal=True), compute, free memory.
# EDGE CASES: none.
# NOT HANDLED: none.
# CHECKED BY: notebook 10 section 2 (row keys equal features_all35.csv)
def build_person(subject, data_root, cfg):
    import gc
    from recording import Recording
    rec = Recording.load(subject, data_root, cfg)
    out = causal_features_one(rec, cfg)
    del rec; gc.collect()
    return out
