"""Signal quality per epoch, and the gate.

Logic approved in CODE_PLAN.md group C.
Thresholds live in Config and are first guesses, to be tuned on S042 only.
"""
import numpy as np
import pandas as pd


# LOGIC: average Pearson correlation between each beat's shape and the epoch's mean
#   beat shape. Cut a fixed segment around each peak (pre_s before, post_s after),
#   average them into a template, correlate each segment with it, return the mean.
#   Clean PPG gives similar beats (near 1); noise or motion gives dissimilar ones.
# EDGE CASES: NaN with fewer than 3 complete segments; segments running off the
#   signal are dropped; a flat segment (zero spread) is dropped.
# NOT HANDLED: genuinely changing beat shape (e.g. ectopic beats) also lowers it.
# CHECKED BY: tests/test_quality.py::test_template_r_clean_vs_noise
def template_r(x, peak_idx, fs=64, pre_s=0.25, post_s=0.40):
    x = np.asarray(x, "float64")
    a, b = int(round(pre_s * fs)), int(round(post_s * fs))
    segs = [x[i - a: i + b] for i in peak_idx if i - a >= 0 and i + b <= x.size]
    segs = [s for s in segs if s.std() > 0]
    if len(segs) < 3:
        return float("nan")
    segs = np.vstack(segs)
    tpl = segs.mean(axis=0)
    r = [np.corrcoef(s, tpl)[0, 1] for s in segs]
    return float(np.mean(r))


# LOGIC: quality measures per epoch, read from where the epoch's wrist signal is
#   (start_sample_wrist): number of beats, share of intervals kept (intervals ending
#   inside the epoch), beat rate, whether the beat count is plausible (rate 30-200
#   bpm and within 2 beats of epoch length / median kept interval), template_r.
# EDGE CASES: epochs with no intervals get pct_kept NaN and plausible False.
# NOT HANDLED: motion is a separate column added by the caller (acc_motion).
# CHECKED BY: tests/test_quality.py::test_epoch_quality_counts
def epoch_quality(bvp_filt, beat_idx, intervals, start_samples, t64_0=0.0, fs=64, epoch_s=30):
    beat_idx = np.asarray(beat_idx, int)
    it = intervals["t"].to_numpy()
    ik = intervals["keep"].to_numpy()
    ii = intervals["ibi"].to_numpy()
    n = fs * epoch_s
    rows = []
    for s in np.asarray(start_samples, int):
        pk = beat_idx[(beat_idx >= s) & (beat_idx < s + n)]
        t0, t1 = t64_0 + s / fs, t64_0 + (s + n) / fs
        m = (it >= t0) & (it < t1)
        n_int = int(m.sum())
        pct_kept = float(ik[m].mean()) if n_int else float("nan")
        rate = 60.0 * pk.size / epoch_s
        med = np.median(ii[m & ik]) if (m & ik).any() else float("nan")
        expect = epoch_s / med if np.isfinite(med) else float("nan")
        plausible = bool(30 <= rate <= 200 and np.isfinite(expect) and abs(pk.size - expect) <= 2)
        rows.append({"n_beats": int(pk.size), "pct_kept": pct_kept, "rate_bpm": rate,
                     "plausible": plausible, "template_r": template_r(bvp_filt, pk, fs)})
    return pd.DataFrame(rows)


# LOGIC: an epoch passes when its beat count is plausible, at least q_min_kept of its
#   intervals are kept, its template_r is at least q_min_template_r, and its motion
#   is at most q_max_motion. All thresholds come from Config.
# EDGE CASES: NaN in any measure fails the gate.
# NOT HANDLED: tuning; thresholds are set on S042 against ECG agreement, then fixed.
# CHECKED BY: tests/test_quality.py::test_quality_gate
def quality_gate(q, motion, config):
    motion = np.asarray(motion, "float64")
    return (q["plausible"].to_numpy()
            & (q["pct_kept"].to_numpy() >= config.q_min_kept)
            & (q["template_r"].to_numpy() >= config.q_min_template_r)
            & (motion <= config.q_max_motion))
