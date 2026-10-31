"""Clock timing between the 100 Hz sleep-lab file and the 64 Hz wristband file.

Logic approved in chat and in CODE_PLAN.md group A2.
Convention everywhere: t64 = t100 + offset(t100).
"""
from dataclasses import dataclass

import numpy as np


# LOGIC: holds the straight-line clock model offset(t100) = a + s * t100, so
#   t64 = t100 + a + s * t100. to64 and to100 convert in each direction. Every time
#   conversion goes through this object, so the drift correction lives in one place.
# EDGE CASES: estimated=True marks a model not fitted on this person's own ECG
#   (borrowed slope); code that reports results must say so.
# NOT HANDLED: non-linear drift. If a person's residuals are large, the line is
#   the wrong model and fit_clock_model reports that in max_resid_s.
# CHECKED BY: tests/test_timing.py::test_clock_model_roundtrip
@dataclass(frozen=True)
class ClockModel:
    a: float
    s: float
    estimated: bool = False
    max_resid_s: float = float("nan")

    def to64(self, t100):
        return np.asarray(t100, "float64") * (1.0 + self.s) + self.a

    def to100(self, t64):
        return (np.asarray(t64, "float64") - self.a) / (1.0 + self.s)


# LOGIC: heart period (beat-to-beat interval) series on a regular grid, z-scored,
#   so two beat trains on different clocks can be cross-correlated.
# EDGE CASES: intervals clipped to 0.3-2.0 s so one missed beat cannot dominate.
# NOT HANDLED: gaps are bridged by linear interpolation; long gaps flatten the series.
# CHECKED BY: used only inside find_clock_offset (tests/test_beats.py).
def _period_series(beat_times, t0, t1, fs=4.0):
    t = np.asarray(beat_times, "float64")
    ibi = np.clip(np.diff(t), 0.3, 2.0)
    grid = np.arange(t0, t1, 1.0 / fs)
    y = np.interp(grid, t[1:], ibi)
    return grid, (y - y.mean()) / y.std()


# LOGIC: SUPERSEDED on 2026-10-10 by local_lock_offset + fit_clock_model. Kept because
#   its refusal on DREAMT is evidence: the BVP copy in the 100 Hz file never matches
#   the 64 Hz BVP at one offset (r <= 0.25), which is how the drift was found.
#   Coarse: cross-correlate heart-period series (ECG on 100 Hz clock, PPG on 64 Hz
#   clock) over +-max_lag. Fine: in n_windows windows, correlate the 100 Hz BVP copy
#   with the 64 Hz BVP over coarse +-fine_range in 10 ms steps.
# EDGE CASES: raises if any window's correlation is below min_r, so a bad offset
#   can never flow downstream.
# NOT HANDLED: clock drift. It assumes one constant offset, which DREAMT violates.
# CHECKED BY: tests/test_beats.py::test_clock_offset_recovered and
#   ::test_clock_offset_refuses_unrelated_signal
def find_clock_offset(ppg_times64, ecg_times100, bvp64, t64, bvp100, t100,
                      max_lag=4 * 3600, n_windows=5, win_s=600, fine_range=5.0, min_r=0.9):
    """Constant offset t64 = t100 + offset (superseded; see comment)."""
    from scipy.signal import correlate, correlation_lags
    fs = 4.0
    gp, hp = _period_series(ppg_times64, ppg_times64[1], ppg_times64[-1], fs)
    ge, he = _period_series(ecg_times100, ecg_times100[1], ecg_times100[-1], fs)
    c = correlate(hp, he, mode="full", method="fft")
    k = correlation_lags(hp.size, he.size, mode="full")
    off = gp[0] - ge[0] + k / fs
    ok = np.abs(off) <= max_lag
    i = np.argmax(np.where(ok, c, -np.inf))
    coarse = float(off[i])
    coarse_r = float(c[i] / he.size)

    t64 = np.asarray(t64, "float64"); t100 = np.asarray(t100, "float64")
    b64 = np.asarray(bvp64, "float64"); b100 = np.asarray(bvp100, "float64")
    centres = np.linspace(t100[0] + 0.1 * (t100[-1] - t100[0]),
                          t100[-1] - 0.1 * (t100[-1] - t100[0]) - win_s, n_windows)
    n_shift = int(round(fine_range * 100))
    offsets, rs = [], []
    for c0 in centres:
        grid = np.arange(c0, c0 + win_s, 0.01)
        a = np.interp(grid, t100, b100)
        a = (a - a.mean()) / a.std()
        grid_wide = np.arange(c0 + coarse - fine_range, c0 + win_s + coarse + fine_range, 0.01)
        b = np.interp(grid_wide, t64, b64)
        best_r, best_s = -np.inf, 0
        for sh in range(0, 2 * n_shift + 1):
            seg = b[sh: sh + a.size]
            if seg.size < a.size:
                break
            r = np.mean(a * (seg - seg.mean()) / seg.std())
            if r > best_r:
                best_r, best_s = r, sh
        offsets.append(coarse - fine_range + best_s * 0.01)
        rs.append(best_r)
    offsets, rs = np.array(offsets), np.array(rs)
    if rs.min() < min_r:
        raise ValueError(f"clock offset not trusted: window correlations {np.round(rs, 3)}")
    return {"coarse_s": coarse, "coarse_r": coarse_r, "offsets_s": offsets, "r": rs,
            "offset_s": float(np.median(offsets)),
            "spread_ms": float((offsets.max() - offsets.min()) * 1000)}


# LOGIC: find the constant offset DREAMT used to place the labels. Take the first
#   stage change in the 100 Hz file; try every 64 Hz change with the same label as
#   its partner; keep the offset under which most of the first n_check 100 Hz
#   changes have a 64 Hz change with the same label at the shifted time (within 1 s).
# EDGE CASES: returns (nan, 0.0) if no candidate matches; the share tells how
#   convincing the match is (1.0 in all 3 ECG people).
# NOT HANDLED: only the first n_check changes are used, so it describes how the
#   labels were placed at the start, not whether that held all night.
# CHECKED BY: notebook 03 (share 1.0 for S042, S022, S035);
#   tests/test_timing.py::test_label_offset_recovered
def label_offset(stage100, t100, stage64, t64, n_check=20):
    """Returns (offset_s, share_of_changes_matched)."""
    def changes(st, t):
        st = np.asarray(st, dtype=object)
        i = np.flatnonzero(st[1:] != st[:-1]) + 1
        return np.asarray(t)[i], st[i]
    c1t, c1s = changes(stage100, t100)
    c2t, c2s = changes(stage64, t64)
    c1t, c1s = c1t[:n_check], c1s[:n_check]
    best = (0.0, float("nan"))
    for t2 in c2t[c2s == c1s[0]]:
        off = t2 - c1t[0]
        j = np.clip(np.searchsorted(c2t, c1t + off), 0, c2t.size - 1)
        jm = np.clip(j - 1, 0, None)
        k = np.where(np.abs(c2t[j] - (c1t + off)) < np.abs(c2t[jm] - (c1t + off)), j, jm)
        ok = (np.abs(c2t[k] - (c1t + off)) < 1.0) & (c2s[k] == c1s)
        if ok.mean() > best[0]:
            best = (float(ok.mean()), float(off))
    return best[1], best[0]


# LOGIC: in one window of reference beats (ECG R-peaks, 100 Hz clock), try each
#   candidate offset; move the reference beats by it and pair each with the next
#   beat on the other clock. Score = share of those delays within +-band of their
#   median. When the clocks agree inside the window the delay (pulse arrival time)
#   is constant and the score is high.
# EDGE CASES: offsets that put no beats in range are skipped; a window with no
#   lock returns its best (low) share, so callers can weigh or drop it.
# NOT HANDLED: drift inside the window smears the lock (about 0.4 s per 5 min at
#   0.13%), so the share stays well below 1 even when the offset is right.
# CHECKED BY: notebook 03 (offsets on straight lines in 3 people);
#   tests/test_timing.py::test_local_lock_offset_recovered
def local_lock_offset(ref_times, beat_times, offsets, t_start, win_s=300, band=0.03):
    """Returns a dict: offset_s, lock_share, delay_s, n_ref."""
    r = np.asarray(ref_times, "float64")
    r = r[(r >= t_start) & (r < t_start + win_s)]
    p = np.asarray(beat_times, "float64")
    best = {"offset_s": float("nan"), "lock_share": 0.0, "delay_s": float("nan"), "n_ref": int(r.size)}
    for o in offsets:
        j = np.searchsorted(p, r + o)
        ok = j < p.size
        d = p[j[ok]] - (r[ok] + o)
        if d.size == 0:
            continue
        med = np.median(d)
        share = float(np.mean(np.abs(d - med) < band))
        if share > best["lock_share"]:
            best.update(offset_s=float(o), lock_share=share, delay_s=float(med))
    return best


# LOGIC: fit offset(t100) = a + s * t100 through the window offsets with Theil-Sen
#   (median of all pairwise slopes; intercept = median of y - s*x, method="joint",
#   because the default intercept uses median(y) and an outlier moves it), so one
#   bad window cannot pull the line. This
#   replaces the earlier lock-share >= 0.5 cut-off, which dropped S022 entirely.
#   Returns a ClockModel plus the largest absolute residual.
# EDGE CASES: windows with NaN offsets are dropped; fewer than 3 usable windows
#   raises, since a line through 2 points cannot be checked.
# NOT HANDLED: does not decide whether a residual is "too big"; it reports
#   max_resid_s and the caller (and the figure) judges.
# CHECKED BY: tests/test_timing.py::test_fit_clock_model_ignores_outlier
def fit_clock_model(t100_windows, offsets):
    """Returns a ClockModel fitted on this person's windows."""
    from scipy.stats import theilslopes
    x = np.asarray(t100_windows, "float64")
    y = np.asarray(offsets, "float64")
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        raise ValueError("need at least 3 windows with an offset")
    s, a, _, _ = theilslopes(y[ok], x[ok], method="joint")
    resid = y[ok] - (a + s * x[ok])
    return ClockModel(a=float(a), s=float(s), estimated=False,
                      max_resid_s=float(np.abs(resid).max()))


# LOGIC: the label misalignment as a straight line on the 64 Hz clock:
#   m(t64) = m0 + m1 * t64, positive = wristband signal LATER than its label.
#   From a person's own fit: DREAMT assigned sleep-lab time t100 = t64 - label_off,
#   and that moment's wrist signal is really at clock.to64(t100), so
#   m(t64) = clock.to64(t64 - label_off) - t64 = (a - label_off * (1 + s)) + s * t64.
# EDGE CASES: none; pure arithmetic.
# NOT HANDLED: which clock is "right"; the sleep-lab clock (labels + ECG) is taken
#   as the reference.
# CHECKED BY: tests/test_timing.py::test_misalignment_line_matches_definition
def misalignment_line(clock, label_off):
    """Returns (m0, m1) for m(t64) = m0 + m1 * t64."""
    return clock.a - label_off * (1.0 + clock.s), clock.s


# LOGIC: for people without ECG, borrow the misalignment line: m0 and m1 are the
#   means over the people with ECG. The result is marked estimated.
# EDGE CASES: raises on an empty list.
# NOT HANDLED: whether the borrowed line is right for that person; it cannot be
#   checked without their ECG (CODE_PLAN decision 3 would add more ECG people).
# CHECKED BY: tests/test_timing.py::test_pool_misalignment_mean
def pool_misalignment(lines):
    """lines: list of (m0, m1). Returns (m0_mean, m1_mean)."""
    if not lines:
        raise ValueError("no misalignment lines to pool")
    arr = np.asarray(lines, "float64")
    return float(arr[:, 0].mean()), float(arr[:, 1].mean())


# LOGIC: move each scored epoch to where its wristband signal really is. The label
#   stays with the scorer's 30 s page; the page's start on the 64 Hz clock moves by
#   m(start) = m0 + m1 * start. Adds t_start_wrist and start_sample_wrist; features
#   read the wrist signal from there. Epochs whose moved window runs past either end
#   of the recording get keep=False.
# EDGE CASES: m changes about 40 ms across one epoch at 0.13% drift, so the start
#   value is used for the whole epoch (within about 3 samples at 64 Hz).
# NOT HANDLED: non-linear drift (the per-person fit reports max_resid_s).
# CHECKED BY: tests/test_timing.py::test_correct_epoch_times_shift
def correct_epoch_times(epochs, t64, m0, m1, fs=64, epoch_s=30):
    """Returns a copy of epochs with t_start_wrist and start_sample_wrist."""
    out = epochs.copy()
    t64 = np.asarray(t64, "float64")
    t_start = t64[out["start_sample"].to_numpy()]
    out["t_start_wrist"] = t_start + m0 + m1 * t_start
    out["start_sample_wrist"] = np.round((out["t_start_wrist"] - t64[0]) * fs).astype(int)
    outside = (out["start_sample_wrist"] < 0) | (out["start_sample_wrist"] + fs * epoch_s > t64.size)
    out.loc[outside, "keep"] = False
    out.loc[outside, "start_sample_wrist"] = out.loc[outside, "start_sample"]
    return out
