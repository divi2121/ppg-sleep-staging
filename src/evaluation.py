"""Model inputs, the fold loop, scoring, and logistic regression.

Moved here unchanged on 2026-10-23 from notebook 06 (cells make_xy,
macro_f1_per_person, run_folds, LogRegFeatures, tune_settings, rem_rates), so that later
notebooks import them instead of copying. Logic was approved in notebook 06.
"""
import time
import warnings

import numpy as np
import pandas as pd

from metrics import per_class_f1
from splits import assert_no_person_overlap

CLASSES = ("W", "NREM", "REM")
LABELS = [0, 1, 2]
FORBIDDEN = {"stage3", "stage5", "subject", "epoch", "t_start", "pass"}


# LOGIC: check the columns exist and none is a label, ID or the gate; drop rows with no
#   label; a label outside `classes` raises; X is float64 with NaN kept; y holds integer
#   codes in the order of `classes`; subjects and index come from the kept rows.
# EDGE CASES: NaN label dropped; label outside classes raises; leaky column raises
#   (checked before the missing-column check); missing column raises KeyError.
# NOT HANDLED: filtering on the quality gate (decision 3 in CODE_PLAN is still open).
# CHECKED BY: tests/test_evaluation.py::test_make_xy
def make_xy(df, cols, label="stage3", classes=CLASSES):
    """Returns X (n, p) float64 with NaN kept, y (n,) int codes, subjects (n,), index (n,)."""
    cols = list(cols)
    bad = [c for c in cols if c in FORBIDDEN]
    if bad:
        raise ValueError(f"label, ID or gate columns cannot be features: {bad}")
    missing = [c for c in cols + [label, "subject"] if c not in df.columns]
    if missing:
        raise KeyError(f"missing columns: {missing}")
    lab = df[label]
    ok = lab.notna()
    unknown = set(lab[ok].unique()) - set(classes)
    if unknown:
        raise ValueError(f"label values outside classes: {sorted(unknown)}")
    sub = df.loc[ok]
    code = {c: i for i, c in enumerate(classes)}
    X = sub[cols].to_numpy("float64")
    y = sub[label].map(code).to_numpy(int)
    return X, y, sub["subject"].to_numpy(object), sub.index.to_numpy()


# LOGIC: per person, F1 of each stage from the confusion matrix (per_class_f1), then the
#   mean over stages (VALIDATION_PLAN A4, decided 2026-10-17).
# EDGE CASES: stage in neither y nor yhat is left out; predicted but absent scores 0;
#   a person whose stages are all NaN gets NaN.
# NOT HANDLED: weighting stages; every stage counts equally by design.
# CHECKED BY: tests/test_evaluation.py::test_macro_f1_per_person
def macro_f1_per_person(preds, labels=LABELS, person_col="subject", y_col="y", yhat_col="yhat"):
    """Returns a table indexed by person with macro_f1 and n_stages (stages averaged)."""
    rows = []
    for s, g in preds.groupby(person_col, sort=True):
        f1 = per_class_f1(g[y_col], g[yhat_col], list(labels))
        rows.append({"subject": s, "macro_f1": float(f1.mean()), "n_stages": int(f1.notna().sum())})
    return pd.DataFrame(rows).set_index("subject")


# LOGIC: the one loop every model goes through. Check no person sits in two of train,
#   tune and test. Per fold: refuse if a class is missing from the training people, make
#   a fresh model, fit on train (models with uses_tune = True also get the tune rows),
#   predict the test people, keep one row per test epoch.
# EDGE CASES: class missing from train raises; model_factory is called once per fold so
#   no state leaks between folds.
# NOT HANDLED: class probabilities; parallel folds.
# CHECKED BY: tests/test_evaluation.py::test_run_folds
def run_folds(model_factory, X, y, subjects, index, folds, name, classes=CLASSES, verbose=True):
    """Returns a table with columns subject, row, y, yhat, fold, model."""
    assert_no_person_overlap(folds)
    out, t0 = [], time.time()
    for i, f in enumerate(folds):
        tr = np.isin(subjects, f["train"]); tu = np.isin(subjects, f["tune"]); te = np.isin(subjects, f["test"])
        if len(np.unique(y[tr])) < len(classes):
            raise ValueError(f"fold {i}: a class is missing from the training people")
        model = model_factory()
        if getattr(model, "uses_tune", False):
            model.fit(X[tr], y[tr], tune=(X[tu], y[tu]))
        else:
            model.fit(X[tr], y[tr])
        yhat = np.asarray(model.predict(X[te])).astype(int)
        out.append(pd.DataFrame({"subject": subjects[te], "row": index[te], "y": y[te],
                                 "yhat": yhat, "fold": i, "model": name}))
        if verbose:
            print(f"{name} fold {i + 1}/{len(folds)} done, {time.time() - t0:.0f}s elapsed", end="\r")
    if verbose:
        print()
    return pd.concat(out, ignore_index=True)


# LOGIC: logistic regression cannot take NaN. Inside the model only, fill each NaN with
#   the training median of its column and add a 0/1 missing flag per column; standardise
#   with training statistics; L2 logistic regression with balanced class weights.
# EDGE CASES: all-NaN column in training gets median 0; zero-variance column keeps sd 1.
# NOT HANDLED: class weights are "balanced" (decision 4 in CODE_PLAN is still open).
# CHECKED BY: tests/test_evaluation.py::test_logreg_features
class LogRegFeatures:
    def __init__(self, C=1.0):
        self.C = C

    def _prep(self, X):
        X = np.asarray(X, "float64")
        Xf = np.where(np.isnan(X), self.med_, X)
        return np.hstack([(Xf - self.mu_) / self.sd_, np.isnan(X).astype("float64")])

    def fit(self, X, y):
        from sklearn.linear_model import LogisticRegression
        X = np.asarray(X, "float64")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(X, axis=0)
        self.med_ = np.where(np.isnan(med), 0.0, med)
        Xf = np.where(np.isnan(X), self.med_, X)
        self.mu_ = Xf.mean(0); sd = Xf.std(0); self.sd_ = np.where(sd == 0, 1.0, sd)
        self.model_ = LogisticRegression(C=self.C, class_weight="balanced", max_iter=500).fit(self._prep(X), y)
        return self

    def predict(self, X):
        return self.model_.predict(self._prep(X))


# LOGIC: for each candidate setting, fit on the training people of each fold, score on
#   that fold's tuning people (mean of per-person macro F1), average over folds, sort.
#   Test people are never used.
# EDGE CASES: settings are dicts passed to model_factory(**setting); a tuning person with a
#   missing stage is handled by macro_f1_per_person.
# NOT HANDLED: parallel fits.
# CHECKED BY: tests/test_evaluation.py::test_tune_settings
def tune_settings(model_factory, settings, X, y, subjects, folds, labels=LABELS, verbose=True):
    """Returns the settings with tune_macro_f1 and setting (index in the list), best first."""
    rows = []
    for k, s in enumerate(settings):
        scores = []
        for f in folds:
            tr = np.isin(subjects, f["train"]); tu = np.isin(subjects, f["tune"])
            m = model_factory(**s).fit(X[tr], y[tr])
            p = pd.DataFrame({"subject": subjects[tu], "y": y[tu],
                              "yhat": np.asarray(m.predict(X[tu])).astype(int)})
            scores.append(macro_f1_per_person(p, labels)["macro_f1"].mean())
        rows.append({**s, "tune_macro_f1": float(np.nanmean(scores)), "setting": k})
        if verbose:
            print(f"setting {k + 1}/{len(settings)}: tune macro F1 {rows[-1]['tune_macro_f1']:.3f}", end="\r")
    if verbose:
        print()
    return pd.DataFrame(rows).sort_values("tune_macro_f1", ascending=False).reset_index(drop=True)


# LOGIC: REM recall = share of true REM epochs predicted REM; false-REM rate = share of
#   true NREM epochs predicted REM. Both pooled over all test epochs.
# EDGE CASES: no REM (or no NREM) epochs gives NaN.
# CHECKED BY: tests/test_evaluation.py::test_rem_rates
def rem_rates(preds, rem=2, nrem=1):
    y, yh = preds["y"].to_numpy(), preds["yhat"].to_numpy()
    r, n = y == rem, y == nrem
    return (float((yh[r] == rem).mean()) if r.any() else np.nan,
            float((yh[n] == rem).mean()) if n.any() else np.nan)


# LOGIC: the index of the fold whose test list holds this person, so each person is only
#   ever scored or replayed by the model that was not trained or tuned on them (live plan
#   Task 5, 2026-10-24).
# EDGE CASES: a person in no test list raises KeyError rather than falling back to any model.
# NOT HANDLED: a person in two test lists (make_person_folds never does this) returns the first.
# CHECKED BY: tests/test_evaluation.py::test_model_for_returns_the_fold_that_tested_the_person
def model_for(person, folds):
    for i, f in enumerate(folds):
        if person in f["test"]:
            return i
    raise KeyError(f"{person} is not a test person in any fold")


# LOGIC: read the folds shipped in models/live/folds.json and return them in the same shape
#   make_person_folds returns: a list of {"train", "tune", "test"} (lists of person IDs), one
#   per fold, in fold order. The shuffle was done once with the seed stored in the file, so
#   nothing here needs the features table (which is derived from restricted data).
# EDGE CASES: fold numbers in the file must be 0, 1, 2, ... in order, because the position in
#   the list is also the number of the model file (live_fold<i>.pt); otherwise ValueError.
# NOT HANDLED: the file is not checked against the data; a test compares it with
#   make_person_folds when the features table exists.
# CHECKED BY: tests/test_evaluation.py::test_load_folds_roundtrip, test_shipped_folds_file_shape,
#   test_shipped_folds_equal_make_person_folds
def load_folds(path):
    import json
    d = json.load(open(path))
    if [f["fold"] for f in d["folds"]] != list(range(len(d["folds"]))):
        raise ValueError("fold numbers in the file must be 0, 1, 2, ... in order")
    return [{k: list(f[k]) for k in ("train", "tune", "test")} for f in d["folds"]]
