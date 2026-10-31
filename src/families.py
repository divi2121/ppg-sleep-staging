"""Feature families missing after the comparison with Tallal 2026 and DREAMT_FE (2026-10-23).

Every function returns one row per epoch. Heart functions take
epoch start TIMES on the 64 Hz clock and the kept-interval table, like features.py; signal
functions take epoch start SAMPLES on the 64 Hz wrist file, like signal_features.
"""
import numpy as np
import pandas as pd


def _slope_per_min(t_s, v):
    tc = t_s - t_s.mean()
    d = (tc ** 2).sum()
    return float((tc * (v - v.mean())).sum() / d * 60.0) if d > 0 else np.nan


# LOGIC: heart rate over minutes, from our own kept beats (not the device HR). Each kept
#   interval gives an instantaneous rate 60 / ibi at the beat that ends it. These are put
#   on a 1 s grid by linear interpolation; a grid second whose neighbouring kept beats are
#   more than max_gap apart is missing, not filled. Per epoch, from its 30 grid seconds:
#   median, min, max, range, std. Over the centred win_s window: std, and the slope of a
#   straight-line fit in beats per minute per minute.
# EDGE CASES: under min_epoch valid seconds in the epoch gives NaN for the epoch values;
#   under min_win valid seconds in the window gives NaN for the window values; under 2
#   kept beats in the whole night gives all NaN.
#   causal=True (live, 2026-10-24): the window is the win_s ending at the epoch end, and a
#   grid second counts only if both beats it is interpolated between are at or before it.
# NOT HANDLED: none beyond the edge cases.
# CHECKED BY: tests/test_families.py::test_hr_minutes_known, test_hr_minutes_causal_window_ends_at_epoch
def hr_minutes(intervals, t_starts, epoch_s=30, win_s=300, max_gap=2.0, min_epoch=10, min_win=60, causal=False):
    cols = ["hr1_median", "hr1_min", "hr1_max", "hr1_range", "hr1_std", "hr5_std", "hr5_slope"]
    t_starts = np.asarray(t_starts, "float64")
    keep = intervals["keep"].to_numpy(bool)
    tk = intervals["t"].to_numpy("float64")[keep]
    hr = 60.0 / intervals["ibi"].to_numpy("float64")[keep]
    if tk.size < 2:
        return pd.DataFrame(np.nan, index=range(len(t_starts)), columns=cols)
    g = np.arange(np.ceil(tk[0]), np.floor(tk[-1]) + 1.0)
    j = np.searchsorted(tk, g)
    jl, jr = np.clip(j - 1, 0, tk.size - 1), np.clip(j, 0, tk.size - 1)
    ok = (tk[jr] - tk[jl] <= max_gap) | (tk[jr] == g)
    v = np.where(ok, np.interp(g, tk, hr), np.nan)
    rows = []
    for t0 in t_starts:
        if causal:   # a grid second needs both neighbouring beats at or before the epoch end
            t_end = t0 + epoch_s
            vv = np.where(tk[jr] <= t_end, v, np.nan); lo, hi = t_end - win_s, t_end
        else:
            vv = v; lo, hi = t0 + epoch_s / 2 - win_s / 2, t0 + epoch_s / 2 + win_s / 2
        e = vv[(g >= t0) & (g < t0 + epoch_s)]; e = e[np.isfinite(e)]
        mw = (g >= lo) & (g < hi) & np.isfinite(vv)
        row = dict.fromkeys(cols, np.nan)
        if e.size >= min_epoch:
            row.update(hr1_median=np.median(e), hr1_min=e.min(), hr1_max=e.max(),
                       hr1_range=e.max() - e.min(), hr1_std=e.std(ddof=1))
        if mw.sum() >= min_win:
            row.update(hr5_std=vv[mw].std(ddof=1), hr5_slope=_slope_per_min(g[mw], vv[mw]))
        rows.append(row)
    return pd.DataFrame(rows, columns=cols)


def _sampen(x, m=2, r=None):
    x = np.asarray(x, "float64")
    r = 0.2 * x.std() if r is None else r
    n = x.size

    def count(mm):
        # the same n - m starting points for both lengths (Richman and Moorman 2000)
        t = np.lib.stride_tricks.sliding_window_view(x, mm)[: n - m]
        d = np.abs(t[:, None, :] - t[None, :, :]).max(axis=2)
        return (np.triu(d <= r, k=1)).sum()

    b, a = count(m), count(m + 1)
    return float(-np.log(a / b)) if a > 0 and b > 0 else np.nan


def _dfa_alpha1(x, scales=range(4, 17)):
    y = np.cumsum(np.asarray(x, "float64") - np.mean(x))
    ns, fs = [], []
    for n in scales:
        k = y.size // n
        if k < 2:
            continue
        seg = y[: k * n].reshape(k, n)
        tt = np.arange(n)
        coef = np.polyfit(tt, seg.T, 1)
        trend = np.outer(coef[0], tt) + coef[1][:, None]
        fs.append(np.sqrt(np.mean((seg - trend) ** 2))); ns.append(n)
    if len(ns) < 3 or min(fs) <= 0:
        return np.nan
    return float(np.polyfit(np.log(ns), np.log(fs), 1)[0])


# LOGIC: nonlinear beat-interval measures over the centred win_s window, from kept
#   intervals (ms). Poincare SD1 = spread across the line of identity (short-term,
#   sqrt(0.5) x std of successive differences), SD2 = spread along it (long-term,
#   sqrt(2 var - 0.5 var of differences)), and SD1/SD2. Sample entropy with m = 2 and
#   r = 0.2 x std (how unpredictable the next beat is). DFA alpha1 over boxes of 4 to 16
#   beats (short-range correlation; about 0.5 for white noise, 1.5 for a random walk).
# EDGE CASES: under min_n kept intervals in the window gives NaN for all; sample entropy
#   with no matches gives NaN.
#   causal=True (live, 2026-10-24): the window is the win_s ending at the epoch end.
# NOT HANDLED: kept intervals on either side of a rejected one are treated as neighbours.
# CHECKED BY: tests/test_families.py::test_nonlinear_known, test_nonlinear_causal_ignores_future
def nonlinear_hrv(intervals, t_starts, epoch_s=30, win_s=300, min_n=60, causal=False):
    cols = ["nl_sd1", "nl_sd2", "nl_sd1_sd2", "nl_sampen", "nl_dfa_a1"]
    keep = intervals["keep"].to_numpy(bool)
    tk = intervals["t"].to_numpy("float64")[keep]
    xk = 1000.0 * intervals["ibi"].to_numpy("float64")[keep]
    rows = []
    for t0 in np.asarray(t_starts, "float64"):
        hi = t0 + epoch_s if causal else t0 + epoch_s / 2 + win_s / 2
        x = xk[(tk >= hi - win_s) & (tk < hi)]
        if x.size < min_n:
            rows.append(dict.fromkeys(cols, np.nan)); continue
        d = np.diff(x)
        sd1 = np.sqrt(0.5) * d.std(ddof=1)
        sd2 = np.sqrt(max(2 * x.var(ddof=1) - 0.5 * d.var(ddof=1), 0.0))
        rows.append({"nl_sd1": sd1, "nl_sd2": sd2, "nl_sd1_sd2": sd1 / sd2 if sd2 > 0 else np.nan,
                     "nl_sampen": _sampen(x), "nl_dfa_a1": _dfa_alpha1(x)})
    return pd.DataFrame(rows, columns=cols)


# LOGIC: stillness. Activity per second = std of the acceleration magnitude inside that
#   second. A second counts as moving when its activity is above that person's threshold,
#   median + k x MAD of all their seconds (one night, offline). Per epoch: share of still
#   seconds, number of moving seconds, and seconds since the last moving second at the end
#   of the epoch (capped at cap_s).
# EDGE CASES: no moving second yet in the night gives cap_s; a zero MAD makes any second
#   above the median count as moving.
# NOT HANDLED: ACC updates about 8 times a second and is held at 64 Hz, so activity is
#   computed on held values; frequency content is skipped for the same reason.
# CHECKED BY: tests/test_families.py::test_stillness_known
def stillness(acc_x, acc_y, acc_z, start_samples, fs=64, epoch_s=30, k=3.0, cap_s=1800.0):
    mag = np.sqrt(np.asarray(acc_x, "float64") ** 2 + np.asarray(acc_y, "float64") ** 2
                  + np.asarray(acc_z, "float64") ** 2)
    n_sec = mag.size // fs
    act = mag[: n_sec * fs].reshape(n_sec, fs).std(axis=1)
    med = np.median(act); mad = np.median(np.abs(act - med))
    moving = act > med + k * mad
    last = np.full(n_sec, -np.inf); seen = -np.inf
    for i in range(n_sec):
        if moving[i]:
            seen = i
        last[i] = seen
    rows = []
    for s in np.asarray(start_samples, int):
        a, b = s // fs, min(s // fs + epoch_s, n_sec)
        mv = moving[a:b]
        since = (b - 1) - last[b - 1] if b > a else np.nan
        rows.append({"still_frac": float(1 - mv.mean()) if mv.size else np.nan,
                     "move_secs": int(mv.sum()), "secs_since_move": float(min(since, cap_s))})
    return pd.DataFrame(rows)


# LOGIC: skin conductance split into a slow tonic level and fast phasic bursts (Tallal 2026:
#   tonic = 0.05 Hz low-pass). EDA is first taken at 4 Hz (every 16th 64 Hz sample; it
#   updates about 3 times a second). Tonic = zero-phase 2nd-order Butterworth low-pass at
#   0.05 Hz; phasic = EDA minus tonic. Bursts = phasic peaks with prominence at least
#   min_peak microsiemens and at least 1 s apart. Per epoch: tonic mean, phasic std, burst
#   count. Over the centred win_s window: burst count and tonic slope (microsiemens per min).
# EDGE CASES: epochs or windows past the file ends use the samples that exist; under 4 samples
#   gives NaN.
# NOT HANDLED: min_peak = 0.01 microsiemens is a first guess, not tuned; causal use.
# CHECKED BY: tests/test_families.py::test_eda_known
def eda_features(eda, start_samples, fs=64, epoch_s=30, win_s=300, min_peak=0.01):
    from scipy.signal import butter, find_peaks, sosfiltfilt
    step = fs // 4
    x = np.asarray(eda, "float64")[::step]
    f4 = fs / step
    sos = butter(2, 0.05, btype="low", fs=f4, output="sos")
    tonic = sosfiltfilt(sos, x) if x.size > 30 else np.full_like(x, x.mean())
    phasic = x - tonic
    # height as well as prominence: after a burst the low-pass tonic settles with a small
    # ripple that has prominence but almost no height (bug found by test_eda_known, 2026-10-24)
    pk, _ = find_peaks(phasic, prominence=min_peak, height=min_peak, distance=int(f4))
    t = np.arange(x.size) / f4
    rows = []
    for s in np.asarray(start_samples, int):
        t0 = s / fs
        e = (t >= t0) & (t < t0 + epoch_s)
        c = t0 + epoch_s / 2
        w = (t >= c - win_s / 2) & (t < c + win_s / 2)
        tp = t[pk]
        row = {"eda_tonic": np.nan, "eda_phasic_std": np.nan, "eda_bursts": np.nan,
               "eda_bursts5": np.nan, "eda_tonic_slope": np.nan}
        if e.sum() >= 4:
            row.update(eda_tonic=tonic[e].mean(), eda_phasic_std=phasic[e].std(ddof=1),
                       eda_bursts=int(((tp >= t0) & (tp < t0 + epoch_s)).sum()))
        if w.sum() >= 4:
            row.update(eda_bursts5=int(((tp >= c - win_s / 2) & (tp < c + win_s / 2)).sum()),
                       eda_tonic_slope=_slope_per_min(t[w], tonic[w]))
        rows.append(row)
    return pd.DataFrame(rows)
