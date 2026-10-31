"""Beat detection, interval cleaning and beat comparison.

Logic approved in chat and in CODE_PLAN.md group B.
Thresholds are first guesses, to be tuned on S042 only (VALIDATION_PLAN section 8).
Timing functions moved to timing.py on 2026-10-11; re-exported here so older
notebooks and tests keep working.
"""
import numpy as np
import pandas as pd

from timing import (find_clock_offset, label_offset, local_lock_offset,  # noqa: F401
                    _period_series)


# LOGIC: for each detected peak index i, fit a parabola through samples i-1, i, i+1
#   and move the peak to the parabola's top. This gives beat times between samples,
#   needed because 64 Hz gives only 15.6 ms steps (SCOPE section 2.2).
# EDGE CASES: peaks on the first or last sample, or where the 3 points do not curve
#   downwards, keep their sample time. The shift is clipped to +-0.5 samples.
# NOT HANDLED: flat-topped (clipped) peaks; their top is ambiguous.
# CHECKED BY: tests/test_structure.py::test_refine_peaks_parabola
def refine_peaks(x, idx, fs):
    """Peak times in seconds from sample 0."""
    x = np.asarray(x, dtype="float64")
    idx = np.asarray(idx, dtype=int)
    t = idx.astype("float64")
    ok = (idx > 0) & (idx < x.size - 1)
    i = idx[ok]
    y0, y1, y2 = x[i - 1], x[i], x[i + 1]
    denom = y0 - 2 * y1 + y2
    shift = np.zeros_like(y1)
    curved = denom < 0
    shift[curved] = 0.5 * (y0[curved] - y2[curved]) / denom[curved]
    t[ok] = i + np.clip(shift, -0.5, 0.5)
    return t / fs


# LOGIC: find pulse peaks in filtered BVP with a published open detector from
#   neurokit2, then refine each to sub-sample time. Default msptdfast (Charlton 2025),
#   the faster variant of MSPTD, the best detector in Charlton 2022.
# EDGE CASES: method is a parameter so detectors can be compared on S042.
# NOT HANDLED: arrhythmia-specific detection; known to degrade with bigeminy (CinC 2023).
# CHECKED BY: notebook 02 figure (peaks on the waveform); F1 vs ECG once timing is fixed.
def detect_beats(bvp_filt, fs=64, method="msptdfast"):
    """Returns (peak_indices, peak_times_s)."""
    import neurokit2 as nk
    peaks = nk.ppg_findpeaks(np.asarray(bvp_filt, "float64"),
                             sampling_rate=fs, method=method)["PPG_Peaks"]
    peaks = np.asarray(peaks, dtype=int)
    return peaks, refine_peaks(bvp_filt, peaks, fs)


# LOGIC: run the beat detector in chunks, so one bad stretch cannot stop a whole run
#   (added 2026-10-11 after msptdfast crashed on S080 and S102 inside P). Starting at
#   sample `start`, the signal is cut into chunks of chunk_s; each chunk is detected
#   with overlap_s of extra signal on both sides for context, and only peaks inside
#   the chunk's own span are kept, so overlaps never double-count. If the detector
#   raises on a chunk, that chunk's span is recorded as failed and the run continues.
#   Peaks are refined to sub-sample time on the whole signal.
# EDGE CASES: `detector` is a parameter so the failure path can be tested; the last
#   chunk may be short and may fail, which is then recorded like any other.
# NOT HANDLED: nothing is filled in for failed spans; callers report failed minutes.
# CHECKED BY: tests/test_structure.py::test_detect_beats_chunked_matches_whole and
#   ::test_detect_beats_chunked_records_failure
def detect_beats_chunked(bvp_filt, fs=64, method="msptdfast", start=0, chunk_s=600,
                         overlap_s=10, detector=None):
    """Returns (peak_indices, peak_times_s, failed_spans_s as a list of (t0, t1))."""
    x = np.asarray(bvp_filt, "float64")
    if detector is None:
        def detector(seg):
            import neurokit2 as nk
            return np.asarray(nk.ppg_findpeaks(seg, sampling_rate=fs, method=method)["PPG_Peaks"], int)
    n, ov = int(chunk_s * fs), int(overlap_s * fs)
    peaks, failed = [], []
    for a in range(int(start), x.size, n):
        b = min(a + n, x.size)
        lo, hi = max(int(start), a - ov), min(x.size, b + ov)
        try:
            pk = np.asarray(detector(x[lo:hi]), int) + lo
        except Exception:
            failed.append((a / fs, b / fs))
            continue
        peaks.append(pk[(pk >= a) & (pk < b)])
    idx = np.unique(np.concatenate(peaks)) if peaks else np.array([], int)
    return idx, refine_peaks(x, idx, fs), failed


# LOGIC: R-peaks from ECG with neurokit2 defaults (ecg_clean then ecg_peaks), then
#   sub-sample refinement on the cleaned signal.
# EDGE CASES: none beyond neurokit2's own.
# NOT HANDLED: this is the comparison reference, not ground truth; VALIDATION_PLAN
#   section 8 requires a hand-checked sample (sample_check_windows) before trusting it.
# CHECKED BY: notebook 02 window figure; median ECG rate matched the device HR on S022.
def ecg_rpeaks(ecg, fs=100):
    """Returns (r_indices, r_times_s)."""
    import neurokit2 as nk
    clean = nk.ecg_clean(np.asarray(ecg, "float64"), sampling_rate=fs)
    _, info = nk.ecg_peaks(clean, sampling_rate=fs)
    idx = np.asarray(info["ECG_R_Peaks"], dtype=int)
    return idx, refine_peaks(clean, idx, fs)


# LOGIC: intervals between consecutive beats, each flagged keep or reject. Reject if
#   outside [lo, hi] s (30 to 200 bpm), or more than rel away from the median of the
#   surrounding win intervals. Nothing is filled in. causal=True (live, 2026-10-24): the
#   reference is the median of the previous win intervals instead of a centred one, so
#   nothing after interval i is used.
# EDGE CASES: the rolling median uses whatever neighbours exist at the ends; causal, the
#   first interval has no reference and is judged on [lo, hi] only.
# NOT HANDLED: a long run of wrong beats can shift the local median; thresholds are
#   first guesses to be tuned on S042.
# CHECKED BY: tests/test_structure.py::test_clean_intervals_flags,
#   test_clean_intervals_causal_uses_past_only
def clean_intervals(beat_times, lo=0.3, hi=2.0, rel=0.30, win=11, causal=False):
    """Returns a DataFrame: t (beat ending the interval), ibi, keep."""
    t = np.asarray(beat_times, dtype="float64")
    ibi = np.diff(t)
    if causal:   # median of the previous win intervals (all of them, so a run of bad beats shows)
        med = pd.Series(ibi).shift(1).rolling(win, min_periods=1).median().to_numpy()
    else:
        med = pd.Series(ibi).rolling(win, center=True, min_periods=1).median().to_numpy()
    near = np.where(np.isnan(med), True, np.abs(ibi - med) <= rel * med)   # first interval: lo/hi only
    keep = (ibi >= lo) & (ibi <= hi) & near
    return pd.DataFrame({"t": t[1:], "ibi": ibi, "keep": keep})


# LOGIC: RMSSD in ms, using only successive differences where both intervals are
#   kept and adjacent in the series.
# EDGE CASES: NaN if fewer than 2 such pairs.
# NOT HANDLED: gaps in time between adjacent rows are not checked (a rejected
#   interval between two kept ones breaks the pair, which covers most cases).
# CHECKED BY: tests/test_features.py::test_rmssd_known
def rmssd(intervals):
    ibi = intervals["ibi"].to_numpy()
    keep = intervals["keep"].to_numpy()
    pair = keep[1:] & keep[:-1]
    if pair.sum() < 2:
        return float("nan")
    d = np.diff(ibi)[pair]
    return float(np.sqrt(np.mean(d ** 2)) * 1000)


# LOGIC: rebuild the device's beat times from its held-forward IBI column: a beat at
#   each sample where the value changes to a new non-missing value.
# EDGE CASES: NaN samples are skipped; the first non-missing value counts as a beat.
# NOT HANDLED: two beats in a row with the same IBI look like no change, so this is a
#   LOWER BOUND on the device's beats.
# CHECKED BY: tests/test_structure.py::test_device_beats_from_ibi
def device_beats_from_ibi(ibi_col, ts):
    v = np.asarray(ibi_col, dtype="float64")
    ts = np.asarray(ts, dtype="float64")
    prev = np.concatenate([[np.nan], v[:-1]])
    change = ~np.isnan(v) & ((v != prev) | np.isnan(prev))
    return ts[change]


# LOGIC: pair two beat trains 1:1. A pair is kept when each beat is the other's
#   nearest neighbour and they are within tol seconds.
# EDGE CASES: empty input returns empty pairs.
# NOT HANDLED: no global optimisation; mutual-nearest is enough when beats are
#   about 1 s apart and tol is 50 ms.
# CHECKED BY: tests/test_beats.py::test_match_beats_one_to_one
def match_beats(a, b, tol=0.05):
    """Returns (ia, ib) index arrays of matched pairs."""
    a = np.asarray(a, dtype="float64")
    b = np.asarray(b, dtype="float64")
    if a.size == 0 or b.size == 0:
        return np.array([], int), np.array([], int)

    def nearest(src, dst):
        j = np.clip(np.searchsorted(dst, src), 1, dst.size - 1)
        left, right = dst[j - 1], dst[j]
        return np.where(np.abs(src - left) <= np.abs(src - right), j - 1, j)

    a2b = nearest(a, b)
    b2a = nearest(b, a)
    ia = np.arange(a.size)
    mutual = b2a[a2b] == ia
    close = np.abs(a - b[a2b]) <= tol
    sel = mutual & close
    return ia[sel], a2b[sel]


# LOGIC: the fixed difference between where the device marks a beat and where we do.
#   For each device beat, the signed distance to our nearest beat (device minus
#   ours); report the median, the IQR, and the median per hour to check stability.
# EDGE CASES: distances beyond max_dist are ignored (unmatched beats). If no device
#   beat lies within max_dist of ours (or either list is empty), every value is NaN
#   and n is 0, so callers see it instead of crashing (added 2026-10-11).
# NOT HANDLED: does not say which point of the pulse the device uses; the ECG
#   comparison (both distances from the R-peak) answers that.
# CHECKED BY: tests/test_beats.py::test_device_shift_recovered
def device_fiducial_shift(our_times, device_times, max_dist=0.5):
    """Returns a dict: median_s, iqr_s, n, hourly_median_s."""
    ours = np.asarray(our_times, "float64")
    dev = np.asarray(device_times, "float64")
    empty = {"median_s": float("nan"), "iqr_s": float("nan"), "n": 0, "hourly_median_s": np.array([])}
    if ours.size < 2 or dev.size == 0:
        return empty
    j = np.clip(np.searchsorted(ours, dev), 1, ours.size - 1)
    left, right = ours[j - 1], ours[j]
    near = np.where(np.abs(dev - left) <= np.abs(dev - right), left, right)
    d = dev - near
    ok = np.abs(d) <= max_dist
    d, t = d[ok], dev[ok]
    if d.size == 0:
        return empty
    hours = np.floor((t - t.min()) / 3600).astype(int)
    hourly = pd.Series(d).groupby(hours).median().to_numpy()
    q1, q3 = np.percentile(d, [25, 75])
    return {"median_s": float(np.median(d)), "iqr_s": float(q3 - q1),
            "n": int(d.size), "hourly_median_s": hourly}


# LOGIC: agreement (not accuracy) between our beats and the device's. Move device
#   beats by -shift (their fixed fiducial difference), pair 1:1 within tol, then
#   report the share matched each way, the leftover time difference, and the
#   interval difference over consecutive matched pairs.
# EDGE CASES: NaN for the residual and interval difference if nothing matches. A
#   NaN shift (no usable device beats) returns 0% matched and NaN differences.
# NOT HANDLED: the device skips beats, so pct_ours_matched is low by design.
# CHECKED BY: tests/test_beats.py::test_device_match_after_shift
def compare_to_device_ibi(our_times, device_times, tol=0.05, shift=0.0):
    """Returns a dict of agreement measures."""
    ours = np.asarray(our_times, "float64")
    if not np.isfinite(shift):
        dev = np.array([])
    else:
        dev = np.asarray(device_times, "float64") - shift
    ia, ib = match_beats(ours, dev, tol)
    out = {
        "n_ours": int(ours.size),
        "n_device": int(dev.size),
        "pct_device_matched": 100.0 * ia.size / max(dev.size, 1),
        "pct_ours_matched": 100.0 * ia.size / max(ours.size, 1),
        "median_abs_resid_ms": float(np.median(np.abs(ours[ia] - dev[ib])) * 1000) if ia.size else float("nan"),
    }
    consec = (np.diff(ia) == 1) & (np.diff(ib) == 1)
    if consec.any():
        a = np.diff(ours[ia])[consec]
        b = np.diff(dev[ib])[consec]
        out["median_abs_ibi_diff_ms"] = float(np.median(np.abs(a - b)) * 1000)
    else:
        out["median_abs_ibi_diff_ms"] = float("nan")
    return out


# LOGIC: score PPG beats against ECG beats. Both must already be on ONE clock (ECG
#   moved with ClockModel.to64). For each candidate pulse arrival delay, shift PPG
#   back by it, pair 1:1 within tol, and compute sensitivity, PPV and F1 over the
#   span both cover. Keep the delay with the best F1.
# EDGE CASES: delays has NO default on purpose: the range must come from a verified
#   source (VALIDATION_PLAN section 17 item 5).
# NOT HANDLED: a delay that changes through the night (one delay per call; call per
#   window if needed).
# CHECKED BY: tests/test_structure.py::test_compare_to_ecg_finds_delay
def compare_to_ecg(ppg_times, ecg_times, delays, tol=0.05):
    """Returns (table with one row per delay, best row as a dict)."""
    p = np.asarray(ppg_times, "float64")
    e = np.asarray(ecg_times, "float64")
    rows = []
    for d in delays:
        ps = p - d
        lo, hi = max(ps.min(), e.min()), min(ps.max(), e.max())
        ps_in = ps[(ps >= lo) & (ps <= hi)]
        e_in = e[(e >= lo) & (e <= hi)]
        ia, _ = match_beats(ps_in, e_in, tol)
        tp = ia.size
        sens = tp / max(e_in.size, 1)
        ppv = tp / max(ps_in.size, 1)
        f1 = 2 * sens * ppv / (sens + ppv) if sens + ppv > 0 else 0.0
        rows.append({"delay_s": d, "tp": tp, "n_ecg": e_in.size, "n_ppg": ps_in.size,
                     "sensitivity": sens, "ppv": ppv, "f1": f1})
    tab = pd.DataFrame(rows)
    return tab, tab.loc[tab.f1.idxmax()].to_dict()


# LOGIC: draw the plan's random five-minute stretches for checking beats by eye. Pick
#   n start times uniformly at random (fixed seed) from [t_min, t_max - length],
#   without overlap, sorted.
# EDGE CASES: raises if the span cannot hold n non-overlapping stretches.
# NOT HANDLED: does not restrict to scored sleep; pass t_min/t_max for that.
# CHECKED BY: tests/test_structure.py::test_sample_check_windows
def sample_check_windows(t_min, t_max, n=20, length=300.0, seed=20261011):
    """Returns sorted start times (s)."""
    span = t_max - t_min
    slots = int(span // length)
    if slots < n:
        raise ValueError(f"span holds only {slots} stretches of {length} s, need {n}")
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(slots, size=n, replace=False))
    return t_min + chosen * length
