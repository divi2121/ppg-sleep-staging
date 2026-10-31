"""Features per 30 s epoch.

Logic approved in CODE_PLAN.md group D.
Every function takes epoch start TIMES on the 64 Hz clock (t_start_wrist), so the
label-timing choice is made once, upstream, in Recording.epochs.
"""
import numpy as np
import pandas as pd


# LOGIC: time-domain heart rate variability per epoch, from kept intervals ending
#   inside [start, start + epoch_s): count, mean heart rate (60 / mean interval),
#   SDNN (spread of intervals), RMSSD and pNN50 (from successive differences where
#   both intervals are kept and adjacent).
# EDGE CASES: fewer than 3 kept intervals gives NaN for everything except n_ibi.
# NOT HANDLED: 30 s is short for SDNN; it is reported as is and labelled per epoch.
# CHECKED BY: tests/test_features.py::test_epoch_hrv_known
def epoch_hrv(intervals, t_starts, epoch_s=30):
    t = intervals["t"].to_numpy(); ibi = intervals["ibi"].to_numpy(); keep = intervals["keep"].to_numpy()
    rows = []
    for t0 in np.asarray(t_starts, "float64"):
        m = (t >= t0) & (t < t0 + epoch_s)
        x, k = ibi[m], keep[m]
        xk = x[k]
        row = {"n_ibi": int(xk.size)}
        if xk.size >= 3:
            pair = k[1:] & k[:-1]
            d = np.diff(x)[pair]
            row.update(hr_mean=60.0 / xk.mean(), sdnn_ms=1000 * xk.std(ddof=1),
                       rmssd_ms=1000 * np.sqrt(np.mean(d ** 2)) if d.size else np.nan,
                       pnn50=float(np.mean(np.abs(d) > 0.05)) if d.size else np.nan)
        else:
            row.update(hr_mean=np.nan, sdnn_ms=np.nan, rmssd_ms=np.nan, pnn50=np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


# LOGIC: frequency-domain HRV per epoch over a longer window (win_s, default 5 min),
#   since LF (0.04-0.15 Hz) needs minutes. Offline the window is centred on the
#   epoch; causal it is the win_s ending at the epoch's end. Lomb-Scargle
#   periodogram of kept intervals (uneven times, mean removed); power summed in LF
#   and HF (0.15-0.40 Hz); also LF/HF.
# EDGE CASES: fewer than min_n kept intervals in the window gives NaN.
# NOT HANDLED: absolute power units depend on scipy's normalisation; use within this
#   project only, or the LF/HF ratio.
# CHECKED BY: tests/test_features.py::test_freq_features_find_hf_rhythm
def freq_features(intervals, t_starts, epoch_s=30, win_s=300, causal=False, min_n=60):
    from scipy.signal import lombscargle
    t = intervals["t"].to_numpy(); ibi = intervals["ibi"].to_numpy(); keep = intervals["keep"].to_numpy()
    tk, xk = t[keep], ibi[keep]
    f = np.linspace(0.04, 0.40, 73)
    w = 2 * np.pi * f
    lf, hf = (f >= 0.04) & (f < 0.15), (f >= 0.15) & (f <= 0.40)
    rows = []
    for t0 in np.asarray(t_starts, "float64"):
        if causal:
            a, b = t0 + epoch_s - win_s, t0 + epoch_s
        else:
            c = t0 + epoch_s / 2
            a, b = c - win_s / 2, c + win_s / 2
        m = (tk >= a) & (tk < b)
        if m.sum() < min_n:
            rows.append({"lf": np.nan, "hf": np.nan, "lf_hf": np.nan}); continue
        y = xk[m] - xk[m].mean()
        p = lombscargle(tk[m], y, w)
        L, H = p[lf].sum(), p[hf].sum()
        rows.append({"lf": L, "hf": H, "lf_hf": L / H if H > 0 else np.nan})
    return pd.DataFrame(rows)


# LOGIC: per-beat peak-to-trough amplitude of the filtered BVP (split out of
#   pulse_amplitude on 2026-10-21 so the breathing group can reuse it). Trough = minimum
#   between the previous peak and this peak. NaN for the first beat and for beats whose
#   ending interval is rejected.
# EDGE CASES: intervals not matching the beats (one row per beat after the first) raise.
# NOT HANDLED: see pulse_amplitude.
# CHECKED BY: tests/test_features.py::test_pulse_amplitude_known (through pulse_amplitude)
def beat_amplitudes(bvp_filt, beat_idx, intervals):
    """Returns an array with one amplitude per beat (NaN where not used)."""
    x = np.asarray(bvp_filt, "float64"); idx = np.asarray(beat_idx, int)
    keep = intervals["keep"].to_numpy(bool)
    if keep.size != max(idx.size - 1, 0):
        raise ValueError("intervals must have one row per beat after the first")
    amp = np.full(idx.size, np.nan)
    for i in range(1, idx.size):
        if keep[i - 1]:
            amp[i] = x[idx[i]] - x[idx[i - 1]:idx[i]].min()
    return amp


# LOGIC: per-beat peak-to-trough amplitude of the filtered BVP, then mean, std and CV
#   per epoch (added 2026-10-20, group 1). Trough = minimum of bvp_filt between the
#   previous peak and this peak. A beat is used only if the interval ending at it is
#   kept. Beats are assigned to an epoch by peak sample in [start, start + epoch).
# EDGE CASES: first beat skipped (no previous peak); amplitude <= 0 dropped;
#   fewer than 3 used beats in an epoch gives NaN for all three.
# NOT HANDLED: arbitrary units and within-night strap/contact drift; motion spikes
#   that pass the gate; peak value read at the integer sample.
# CHECKED BY: tests/test_features.py::test_pulse_amplitude_known
def pulse_amplitude(bvp_filt, beat_idx, intervals, start_samples, fs=64, epoch_s=30):
    """Pulse amplitude per epoch.

    bvp_filt: filtered BVP. beat_idx: peak sample indices, sorted, one per beat.
    intervals: clean_intervals output for the same beats (row i ends at beat i+1).
    start_samples: epoch start samples on the wrist signal.
    Returns a DataFrame, one row per epoch: amp_mean, amp_std, amp_cv.
    """
    idx = np.asarray(beat_idx, int)
    amp = beat_amplitudes(bvp_filt, idx, intervals)
    ok = amp > 0
    n = fs * epoch_s
    rows = []
    for s in np.asarray(start_samples, int):
        a = amp[ok & (idx >= s) & (idx < s + n)]
        if a.size >= 3:
            m, sd = a.mean(), a.std(ddof=1)
            rows.append({"amp_mean": m, "amp_std": sd, "amp_cv": sd / m})
        else:
            rows.append({"amp_mean": np.nan, "amp_std": np.nan, "amp_cv": np.nan})
    return pd.DataFrame(rows)


# LOGIC: breathing from the pulse, group 2 (2026-10-21). Two beat-by-beat series carry
#   the breath: the beat interval (respiratory sinus arrhythmia, the heart speeding up on
#   the in-breath) and the pulse amplitude. For each series, over the same win_s window
#   as freq_features (centred offline, ending at the epoch end in causal mode): kept
#   beats only, linear trend removed, Lomb-Scargle spectrum on 0.10 to 0.50 Hz
#   (6 to 30 breaths per minute). Rate = 60 x the peak frequency. Regularity = share of
#   the band's power within +-0.02 Hz of the peak (near 1 = one steady rhythm, low =
#   irregular or no clear breathing).
# EDGE CASES: fewer than min_n usable beats in the window gives NaN for that series.
# NOT HANDLED: the peak can be a slow heart-rate wave instead of breathing; amplitude
#   series inherit strap drift; no check that the two rates agree. Regularity stands in
#   for "rate variability" (one rate per 300 s window has no spread of its own).
# CHECKED BY: tests/test_features.py::test_breathing_known_rates
def breathing_features(intervals, amp, t_starts, epoch_s=30, win_s=300, causal=False, min_n=60):
    """intervals: clean_intervals output (row i ends at beat i+1). amp: beat_amplitudes for
    the same beats (one per beat). Returns br_rate_ibi, br_reg_ibi, br_rate_amp, br_reg_amp."""
    from scipy.signal import lombscargle
    t = intervals["t"].to_numpy("float64"); keep = intervals["keep"].to_numpy(bool)
    a = np.asarray(amp, "float64")[1:]
    series = {"ibi": (t[keep], intervals["ibi"].to_numpy("float64")[keep]),
              "amp": (t[keep & (a > 0)], a[keep & (a > 0)])}
    f = np.arange(0.10, 0.50 + 1e-9, 0.005); w = 2 * np.pi * f
    rows = []
    for t0 in np.asarray(t_starts, "float64"):
        lo, hi = (t0 + epoch_s - win_s, t0 + epoch_s) if causal else (t0 + epoch_s / 2 - win_s / 2, t0 + epoch_s / 2 + win_s / 2)
        row = {}
        for nm, (ts, xs) in series.items():
            m = (ts >= lo) & (ts < hi)
            if m.sum() < min_n:
                row[f"br_rate_{nm}"] = row[f"br_reg_{nm}"] = np.nan; continue
            tt, xx = ts[m], xs[m]
            xx = xx - np.polyval(np.polyfit(tt - tt.mean(), xx, 1), tt - tt.mean())
            p = lombscargle(tt, xx, w)
            k = int(np.argmax(p)); tot = p.sum()
            row[f"br_rate_{nm}"] = 60.0 * f[k]
            row[f"br_reg_{nm}"] = p[np.abs(f - f[k]) <= 0.02].sum() / tot if tot > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows, columns=["br_rate_ibi", "br_reg_ibi", "br_rate_amp", "br_reg_amp"])


# LOGIC: skin temperature over the win_s window centred on the epoch, group 3
#   (2026-10-21): std, min, max, and slope (least-squares line, degrees per minute). The
#   30 s epoch is too short for a slope, so the window is 300 s like freq_features.
# EDGE CASES: the window is clipped at the file ends; fewer than 2 s of samples gives NaN.
# NOT HANDLED: TEMP updates about 4 times a second and is held between updates at 64 Hz,
#   so the statistics are time-weighted; offline only (centred window).
# CHECKED BY: tests/test_features.py::test_temp_features_known
def temp_features(temp, start_samples, fs=64, epoch_s=30, win_s=300):
    x = np.asarray(temp, "float64"); half = int(win_s * fs / 2); mid = int(epoch_s * fs / 2)
    rows = []
    for s in np.asarray(start_samples, int):
        c = s + mid; seg = x[max(c - half, 0):min(c + half, x.size)]
        tm = (np.arange(seg.size) / fs / 60.0)[np.isfinite(seg)]; seg = seg[np.isfinite(seg)]
        if seg.size < 2 * fs:
            rows.append({"temp_std": np.nan, "temp_min": np.nan, "temp_max": np.nan, "temp_slope": np.nan}); continue
        tc = tm - tm.mean()
        rows.append({"temp_std": seg.std(ddof=1), "temp_min": seg.min(), "temp_max": seg.max(),
                     "temp_slope": float((tc * (seg - seg.mean())).sum() / (tc ** 2).sum())})
    return pd.DataFrame(rows)


# LOGIC: movement per axis, group 4 (2026-10-21). For each of ACC_X, ACC_Y, ACC_Z over
#   the epoch: 10% trimmed mean (wrist orientation, i.e. posture), IQR and MAD (spread,
#   i.e. small movements). Plus one activity index: the mean over the epoch's 1 s blocks
#   of the std of the acceleration magnitude (fast movement only; the existing `motion`
#   is the deviation from the 30 s mean, so it also catches slow posture changes).
# EDGE CASES: epochs past the end use the samples that exist; under 1 s of samples gives NaN.
# NOT HANDLED: ACC updates about 8 times a second (up to 13, STATUS 2026-10-10) and is held
#   between updates at 64 Hz, so all statistics are time-weighted over the held values.
#   The brief's "trimmed IQR" is plain IQR: the IQR already ignores the outer quarters.
# CHECKED BY: tests/test_features.py::test_movement_features_known
def movement_features(acc_x, acc_y, acc_z, start_samples, fs=64, epoch_s=30):
    from scipy.stats import trim_mean
    A = {"x": np.asarray(acc_x, "float64"), "y": np.asarray(acc_y, "float64"), "z": np.asarray(acc_z, "float64")}
    mag = np.sqrt(A["x"] ** 2 + A["y"] ** 2 + A["z"] ** 2)
    n = fs * epoch_s; rows = []
    for s in np.asarray(start_samples, int):
        row = {}
        for ax, v in A.items():
            seg = v[s:s + n]
            if seg.size < fs:
                row.update({f"acc_{ax}_tmean": np.nan, f"acc_{ax}_iqr": np.nan, f"acc_{ax}_mad": np.nan}); continue
            q1, q3 = np.percentile(seg, [25, 75])
            row.update({f"acc_{ax}_tmean": trim_mean(seg, 0.1), f"acc_{ax}_iqr": q3 - q1,
                        f"acc_{ax}_mad": np.median(np.abs(seg - np.median(seg)))})
        m = mag[s:s + n]; k = m.size // fs
        row["act_index"] = m[:k * fs].reshape(k, fs).std(axis=1).mean() if k else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# LOGIC: mean skin temperature and mean skin conductance (EDA) over each epoch's
#   wrist window.
# EDGE CASES: windows past the end use the samples that exist.
# NOT HANDLED: both update about 4 times a second and are held between updates; the
#   mean is still fine.
# CHECKED BY: tests/test_features.py::test_signal_features_means
def signal_features(wrist, start_samples, fs=64, epoch_s=30):
    temp = wrist["TEMP"].to_numpy("float64"); eda = wrist["EDA"].to_numpy("float64")
    n = fs * epoch_s
    rows = [{"temp_mean": np.nanmean(temp[s:s + n]), "eda_mean": np.nanmean(eda[s:s + n])}
            for s in np.asarray(start_samples, int)]
    return pd.DataFrame(rows)


# LOGIC: context from neighbouring epochs of the same person. For each column, the
#   mean of the previous k epochs (not including the current one). Offline also adds
#   the mean of the next k epochs. Computed per person, rows in time order.
# EDGE CASES: at the start or end, uses whatever neighbours exist (min 1), else NaN.
# NOT HANDLED: assumes df is sorted by person then time; the caller ensures it.
# CHECKED BY: tests/test_features.py::test_context_causal_uses_past_only
def context_features(df, cols, person_col="subject", k=4, causal=False):
    out = pd.DataFrame(index=df.index)
    g = df.groupby(person_col, sort=False)
    for c in cols:
        out[f"{c}_past{k}"] = g[c].transform(lambda s: s.shift(1).rolling(k, min_periods=1).mean())
        if not causal:
            out[f"{c}_next{k}"] = g[c].transform(lambda s: s[::-1].shift(1).rolling(k, min_periods=1).mean()[::-1])
    return out


# LOGIC: scale each feature within each person. Offline: z-score over the person's
#   whole night. Causal: z-score with the running mean and spread of the night so
#   far (including the current epoch), so no future data is used.
# EDGE CASES: causal needs min_periods epochs before giving a value (else NaN); a
#   zero spread gives NaN.
# NOT HANDLED: the offline version needs a full night first, which matters for
#   deployment (VALIDATION_PLAN section 11 step 3 notes the same).
# CHECKED BY: tests/test_features.py::test_normalise_causal_matches_expanding
def normalise_per_person(df, cols, person_col="subject", causal=False, min_periods=10):
    out = pd.DataFrame(index=df.index)
    g = df.groupby(person_col, sort=False)
    for c in cols:
        if causal:
            mu = g[c].transform(lambda s: s.expanding(min_periods).mean())
            sd = g[c].transform(lambda s: s.expanding(min_periods).std())
        else:
            mu = g[c].transform("mean")
            sd = g[c].transform("std")
        out[c] = (df[c] - mu) / sd.replace(0, np.nan)
    return out
