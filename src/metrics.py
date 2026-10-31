"""Metrics. No scikit-learn needed.

Logic approved in CODE_PLAN.md group G and
VALIDATION_PLAN sections 9 and 10.
"""
import numpy as np
import pandas as pd


# LOGIC: confusion matrix counts, rows = true stage, columns = predicted, in the
#   given label order.
# EDGE CASES: labels not in `labels` are ignored.
# NOT HANDLED: normalisation; the caller divides by row sums for the figure.
# CHECKED BY: tests/test_splits_metrics.py::test_kappa_known
def confusion(y, yhat, labels):
    idx = {l: i for i, l in enumerate(labels)}
    m = np.zeros((len(labels), len(labels)), int)
    for a, b in zip(y, yhat):
        if a in idx and b in idx:
            m[idx[a], idx[b]] += 1
    return m


# LOGIC: Cohen's kappa from the confusion matrix: (observed agreement - agreement
#   expected by chance) / (1 - chance agreement).
# EDGE CASES: if chance agreement is 1 (one class only), returns NaN.
# NOT HANDLED: weighted kappa (not in the plan).
# CHECKED BY: tests/test_splits_metrics.py::test_kappa_known
def cohen_kappa(y, yhat, labels):
    m = confusion(y, yhat, labels).astype(float)
    n = m.sum()
    if n == 0:
        return float("nan")
    po = np.trace(m) / n
    pe = (m.sum(0) * m.sum(1)).sum() / n ** 2
    return float("nan") if pe == 1 else float((po - pe) / (1 - pe))


# LOGIC: F1 per class from the confusion matrix: 2 * precision * recall / (p + r).
# EDGE CASES: a class never predicted and never true gives NaN.
# NOT HANDLED: none.
# CHECKED BY: tests/test_splits_metrics.py::test_f1_known
def per_class_f1(y, yhat, labels):
    m = confusion(y, yhat, labels).astype(float)
    tp = np.diag(m); fp = m.sum(0) - tp; fn = m.sum(1) - tp
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = 2 * tp / (2 * tp + fp + fn)
    return pd.Series(f1, index=labels)


# LOGIC: kappa computed separately for each person (the primary endpoint unit).
# EDGE CASES: a person with one class only gets NaN.
# NOT HANDLED: none.
# CHECKED BY: tests/test_splits_metrics.py::test_kappa_per_person
def kappa_per_person(df, labels, person_col="subject", y_col="y", yhat_col="yhat"):
    return df.groupby(person_col)[[y_col, yhat_col]].apply(lambda g: cohen_kappa(g[y_col], g[yhat_col], labels))


# LOGIC: bootstrap over people: resample the per-person values with replacement
#   n_boot times (fixed seed), take the statistic each time, report the 2.5% and
#   97.5% points. People, never epochs, are the independent unit.
# EDGE CASES: NaN values are dropped first.
# NOT HANDLED: none.
# CHECKED BY: tests/test_splits_metrics.py::test_bootstrap_ci_contains_median
def bootstrap_ci(values, stat=np.median, n_boot=10000, seed=20261011):
    v = np.asarray(values, "float64"); v = v[np.isfinite(v)]
    rng = np.random.default_rng(seed)
    boots = np.array([stat(rng.choice(v, v.size, replace=True)) for _ in range(n_boot)])
    return float(stat(v)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


# LOGIC: paired comparison of two models on the same people: per-person difference
#   (a - b), bootstrap CI of its median, and a two-sided bootstrap p-value
#   (2 * the smaller share of bootstrap medians on either side of 0) for Holm.
# EDGE CASES: people with NaN in either are dropped.
# NOT HANDLED: the equivalence margin (VALIDATION_PLAN decision); the caller compares
#   the CI with it.
# CHECKED BY: tests/test_splits_metrics.py::test_paired_difference_sign
def paired_difference(a, b, n_boot=10000, seed=20261011):
    a = pd.Series(a); b = pd.Series(b)
    d = (a - b).dropna().to_numpy()
    rng = np.random.default_rng(seed)
    boots = np.array([np.median(rng.choice(d, d.size, replace=True)) for _ in range(n_boot)])
    p = 2 * min(np.mean(boots <= 0), np.mean(boots >= 0))
    return {"median": float(np.median(d)), "lo": float(np.percentile(boots, 2.5)),
            "hi": float(np.percentile(boots, 97.5)), "p": float(min(p, 1.0)), "n": int(d.size)}


# LOGIC: Holm correction: sort p-values, multiply the i-th smallest by (m - i),
#   enforce that adjusted values never decrease, cap at 1.
# EDGE CASES: none.
# NOT HANDLED: none.
# CHECKED BY: tests/test_splits_metrics.py::test_holm_known
def holm(pvalues):
    p = np.asarray(pvalues, "float64"); m = p.size
    order = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(run, 1.0)
    return adj
