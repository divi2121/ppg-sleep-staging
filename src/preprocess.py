"""Preprocessing functions for DREAMT 64 Hz wristband data.

Logic agreed in chat before writing.
"""
from pathlib import Path

import numpy as np
import pandas as pd

STAGES_5 = ["W", "N1", "N2", "N3", "R"]
TO_3 = {"W": "W", "N1": "NREM", "N2": "NREM", "N3": "NREM", "R": "REM"}


# LOGIC: search the data root for every 64 Hz wristband CSV (S???_whole_df.csv, plus
#   the renamed S006_test.csv) and return {subject: path}, sorted by subject.
# EDGE CASES: raises if a subject is found twice, so a duplicate copy cannot be
#   silently used.
# NOT HANDLED: files with other names are ignored.
# CHECKED BY: notebook 01 (35 subjects found).
def find_wristband_files(root):
    """Return {subject: path} for every 64 Hz wristband CSV under root.

    Matches S???_whole_df.csv and the renamed S006_test.csv.
    Raises if a subject appears twice.
    """
    root = Path(root)
    files = list(root.rglob("S???_whole_df.csv")) + list(root.rglob("S006_test.csv"))
    out = {}
    for f in files:
        subject = f.name[:4]
        if subject in out:
            raise ValueError(f"{subject} found twice: {out[subject]} and {f}")
        out[subject] = f
    return dict(sorted(out.items()))


# LOGIC: find where the scorer's 30 s grid starts. The anchor is the first scored
#   sample (W/N1/N2/N3/R) after the last P sample; offset = anchor modulo 1,920 samples.
#   Then check every later label change sits on that grid, counting changes between
#   scored stages and changes involving Missing separately.
# EDGE CASES: no P at all means the search starts at sample 0; raises if nothing is
#   scored after P.
# NOT HANDLED: a grid that changes mid-night (none found: 0 of 5,157 changes off-grid).
# CHECKED BY: notebook 01 step 2; tests/test_structure.py::test_scoring_offset
def find_scoring_offset(stage, fs=64, epoch_s=30):
    """Find where the scorer's 30 s grid starts, and check stage changes sit on it.

    The grid anchor is the first sample after the last P sample that carries
    one of W/N1/N2/N3/R. offset = anchor modulo the epoch length in samples.

    Every later sample where the label changes is checked against that grid.
    Changes between two scored stages and changes involving Missing are
    counted separately, since Missing may not follow the scorer's pages.

    Returns a dict: anchor, offset, n_changes_scored, n_off_grid_scored,
    n_changes_missing, n_off_grid_missing.
    """
    stage = np.asarray(stage, dtype=object)
    n = int(fs * epoch_s)
    is_p = stage == "P"
    after_p = (np.flatnonzero(is_p)[-1] + 1) if is_p.any() else 0
    scored = np.isin(stage, STAGES_5)
    cand = np.flatnonzero(scored[after_p:])
    if cand.size == 0:
        raise ValueError("no scored label after P")
    anchor = int(after_p + cand[0])

    k = np.flatnonzero(stage[anchor + 1:] != stage[anchor:-1]) + anchor + 1
    on_grid = (k - anchor) % n == 0
    both_scored = scored[k] & scored[k - 1]
    return {
        "anchor": anchor,
        "offset": anchor % n,
        "n_changes_scored": int(both_scored.sum()),
        "n_off_grid_scored": int((both_scored & ~on_grid).sum()),
        "n_changes_missing": int((~both_scored).sum()),
        "n_off_grid_missing": int((~both_scored & ~on_grid).sum()),
    }


# LOGIC: cut the per-sample stage column into 30 s epochs starting at offset (the
#   scorer's grid). An epoch takes a label only if all its samples agree, else
#   "mixed". Adds 5-class and 3-class labels; keep=True only for W/N1/N2/N3/R.
# EDGE CASES: samples before offset and after the last whole epoch are ignored;
#   P, Missing, mixed and unknown labels get keep=False.
# NOT HANDLED: label timing drift (see timing.correct_epoch_times).
# CHECKED BY: notebook 01 step 3 (0 mixed in all 35); tests/test_structure.py::test_make_epochs
def make_epochs(stage, fs=64, epoch_s=30, offset=0):
    """Cut a per-sample stage column into fixed epochs starting at `offset`.

    `offset` is the sample where the scorer's grid starts, from
    find_scoring_offset()["offset"]. Samples before it, and samples left
    over after the last whole epoch, are ignored.

    An epoch takes a label only if all its samples share one label,
    otherwise it is "mixed".

    Returns a DataFrame, one row per epoch:
      epoch, start_sample, raw (label or "mixed"),
      label5 (W/N1/N2/N3/R or None), label3 (W/NREM/REM or None),
      keep (True only for one of the 5 sleep-scored stages).
    P, Missing, mixed and any unexpected label get keep=False.
    """
    stage = np.asarray(stage, dtype=object)
    n = int(fs * epoch_s)
    n_ep = (stage.size - offset) // n
    blocks = stage[offset: offset + n_ep * n].reshape(n_ep, n)
    first = blocks[:, 0]
    pure = (blocks == first[:, None]).all(axis=1)
    raw = np.where(pure, first, "mixed").astype(object)
    keep = np.isin(raw, STAGES_5)
    label5 = np.where(keep, raw, None)
    label3 = np.array([TO_3.get(x) if k else None for x, k in zip(raw, keep)], dtype=object)
    return pd.DataFrame({
        "epoch": np.arange(n_ep),
        "start_sample": offset + np.arange(n_ep) * n,
        "raw": raw,
        "label5": label5,
        "label3": label3,
        "keep": keep,
    })


# LOGIC: Butterworth band-pass in second-order sections, 0.5 to 8 Hz by default.
#   Offline runs it forward and backward (no peak shift, doubled attenuation);
#   causal=True runs forward only, as live, and delays peaks.
# EDGE CASES: scipy's butter(order, band) is order 2*order; stated so nobody misreads it.
# NOT HANDLED: NaN in BVP raises (the 64 Hz BVP has none).
# CHECKED BY: notebook 01 step 4 figure; tests/test_stream.py::test_causal_filter_chunks_equal_whole
def filter_bvp(bvp, fs=64, low=0.5, high=8.0, order=2, causal=False):
    """Band-pass a BVP signal with a Butterworth filter (second-order sections).

    Note: scipy's butter(order, [low, high], "band") gives a filter of
    order 2 * order. Offline (causal=False) runs it forward and backward,
    so peaks do not shift and the attenuation is doubled. causal=True runs
    forward only, as a live system would, and delays the peaks.

    NaNs are not handled: the 64 Hz BVP has none (checked 2026-09-23).
    """
    from scipy.signal import butter, sosfilt, sosfiltfilt
    x = np.asarray(bvp, dtype="float64")
    if np.isnan(x).any():
        raise ValueError("BVP contains NaN")
    sos = butter(order, [low, high], btype="band", fs=fs, output="sos")
    return sosfilt(sos, x) if causal else sosfiltfilt(sos, x)


# LOGIC: acceleration magnitude sqrt(x^2+y^2+z^2), then per epoch the mean absolute
#   deviation from that epoch's mean. Higher = more movement.
# EDGE CASES: epochs reaching past the end use the samples that exist.
# NOT HANDLED: units are E4 raw (1/64 g); no gravity removal beyond the per-epoch mean.
# CHECKED BY: notebook 01 step 5 (N3 stillest, P most active); tests/test_structure.py::test_acc_motion
def acc_motion(acc_x, acc_y, acc_z, start_samples, fs=64, epoch_s=30):
    """Motion score per epoch: mean absolute deviation of acceleration magnitude.

    start_samples: epoch start indices, from make_epochs()["start_sample"].
    Units follow the input (E4 raw units). Higher = more movement.
    """
    mag = np.sqrt(np.asarray(acc_x, "float64") ** 2
                  + np.asarray(acc_y, "float64") ** 2
                  + np.asarray(acc_z, "float64") ** 2)
    n = int(fs * epoch_s)
    out = np.empty(len(start_samples))
    for i, s in enumerate(start_samples):
        seg = mag[s: s + n]
        out[i] = np.mean(np.abs(seg - seg.mean()))
    return out
