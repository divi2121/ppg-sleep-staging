"""Data quality functions for the DREAMT 64 Hz wristband files.

Written by Claude, 2026-10-10. Logic agreed in chat before writing.
"""
from pathlib import Path

import numpy as np
import pandas as pd


# LOGIC: read one 64 Hz CSV; TIMESTAMP stays float64, other numeric columns become
#   float32, Sleep_Stage becomes a category. Subject ID = first 4 characters of the
#   filename, so S006_test.csv still gives S006.
# EDGE CASES: non-float columns are left as read.
# NOT HANDLED: files over memory; one file is about 2 million rows, which fits.
# CHECKED BY: run_step2 (35 files, 14 columns each).
def load_wristband(path):
    """Read one DREAMT 64 Hz wristband CSV.

    TIMESTAMP stays float64 so sub-second steps keep full precision.
    Other numeric columns become float32. Sleep_Stage becomes a category.
    The subject ID is the first 4 characters of the filename, so the
    renamed S006_test.csv still gives "S006".

    Returns (subject_id, DataFrame).
    """
    path = Path(path)
    subject = path.name[:4]
    df = pd.read_csv(path)
    for col in df.columns:
        if col == "TIMESTAMP":
            df[col] = df[col].astype("float64")
        elif col == "Sleep_Stage":
            df[col] = df[col].astype("category")
        elif pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].astype("float32")
    return subject, df


# LOGIC: from the timestamp column: hours recorded, whether time always moves forward,
#   steps more than tol away from 1/fs, gaps longer than gap_s, longest step.
# EDGE CASES: float timestamps are compared with a tolerance, never exact equality.
# NOT HANDLED: a clock that runs at the wrong rate but evenly looks perfect here
#   (this is how the 0.13% drift stayed invisible to QC).
# CHECKED BY: run_step2 (0 of 35 with any off-grid step or gap).
def check_timeline(ts, fs=64, tol=0.001, gap_s=1.0):
    """Check that a timestamp column sits on a regular 1/fs grid.

    ts     : timestamps in seconds
    tol    : a step counts as off-grid if it differs from 1/fs by more
             than this many seconds (floats, so no exact equality)
    gap_s  : a step longer than this counts as a gap

    Returns a dict: n_samples, hours, monotonic, n_off_grid,
    pct_off_grid, n_gaps, longest_step_s.
    """
    ts = np.asarray(ts, dtype="float64")
    d = np.diff(ts)
    off = np.abs(d - 1.0 / fs) > tol
    return {
        "n_samples": int(ts.size),
        "hours": float((ts[-1] - ts[0] + 1.0 / fs) / 3600.0),
        "monotonic": bool(np.all(d > 0)),
        "n_off_grid": int(off.sum()),
        "pct_off_grid": float(100.0 * off.mean()),
        "n_gaps": int((d > gap_s).sum()),
        "longest_step_s": float(d.max()),
    }


# LOGIC: count sample-to-sample value changes over pairs where both samples exist,
#   divided by the time those pairs cover: changes per second.
# EDGE CASES: NaN if no valid pairs.
# NOT HANDLED: a value that truly repeats counts as no change, so this is a LOWER BOUND.
# CHECKED BY: run_step3 (BVP 64/s, as expected).
def native_rate(x, fs=64):
    """Estimate how often a column really updates, in changes per second.

    Counts sample-to-sample value changes, using only pairs where both
    samples are present. A value that truly repeats is counted as no
    change, so this is a LOWER BOUND on the real update rate.

    Returns changes per second over the valid pairs (NaN if none).
    """
    x = np.asarray(x, dtype="float64")
    a, b = x[:-1], x[1:]
    valid = ~np.isnan(a) & ~np.isnan(b)
    n_valid = int(valid.sum())
    if n_valid == 0:
        return float("nan")
    changes = int((a[valid] != b[valid]).sum())
    return changes / (n_valid / fs)
