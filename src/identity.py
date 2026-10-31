"""Identity audit: how much do features recognise the person instead of the stage.

Logic approved in CODE_PLAN.md group H and
VALIDATION_PLAN section 11 (adapted from Lin et al., arXiv 2606.06647).
"""
import numpy as np
import pandas as pd


# LOGIC: can one epoch's features tell who the person is? Split each night into
#   10-minute blocks; train a classifier on even blocks and test on odd blocks, so
#   neighbouring epochs never sit on both sides. Report accuracy and chance (1 / number
#   of people). Rows with NaN features are dropped.
# EDGE CASES: model_factory lets the same split run for "predict person" and
#   "predict stage", as the plan compares the two.
# NOT HANDLED: the default model (scikit-learn logistic regression) is not tuned.
# CHECKED BY: tests/test_identity.py::test_identify_person_split (with a stand-in
#   classifier, so the split logic is tested on its own)
def identify_person(X, target, t, block_s=600, model_factory=None):
    X = pd.DataFrame(X).reset_index(drop=True)
    target = pd.Series(target).reset_index(drop=True)
    ok = X.notna().all(axis=1).to_numpy()
    X, target, t = X[ok], target[ok], np.asarray(t, "float64")[ok]
    train = identity_blocks(t, block_s) % 2 == 0
    if model_factory is None:
        def model_factory():
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    m = model_factory().fit(X[train], target[train])
    acc = float(np.mean(m.predict(X[~train]) == target[~train].to_numpy()))
    return {"accuracy": acc, "chance": 1.0 / target.nunique(), "n_test": int((~train).sum())}


# LOGIC: block number of each epoch: floor(time / block_s). Even blocks train, odd test.
# EDGE CASES: none.
# NOT HANDLED: none.
# CHECKED BY: tests/test_identity.py::test_identify_person_split
def identity_blocks(t, block_s=600):
    return np.floor(np.asarray(t, "float64") / block_s).astype(int)


# LOGIC: per feature, the share of variance explained by person and by stage (eta
#   squared: between-group sum of squares / total). Null for person: shuffle person
#   labels n_shuffle times (fixed seed) and take the 95th percentile.
# EDGE CASES: NaN values are dropped per feature.
# NOT HANDLED: person and stage are not separated jointly (two one-way splits).
# CHECKED BY: tests/test_identity.py::test_variance_split_person_signal
def variance_split(X, person, stage, n_shuffle=1000, seed=20261011):
    X = pd.DataFrame(X); person = np.asarray(person); stage = np.asarray(stage)
    rng = np.random.default_rng(seed)
    rows = []
    for c in X.columns:
        v = X[c].to_numpy("float64"); ok = np.isfinite(v)
        v, p, s = v[ok], person[ok], stage[ok]
        null = [_eta2(v, rng.permutation(p)) for _ in range(n_shuffle)]
        rows.append({"feature": c, "eta2_person": _eta2(v, p), "eta2_stage": _eta2(v, s),
                     "eta2_person_null95": float(np.percentile(null, 95))})
    return pd.DataFrame(rows).set_index("feature")


def _eta2(v, groups):
    """Between-group sum of squares over total sum of squares."""
    tot = ((v - v.mean()) ** 2).sum()
    if tot == 0:
        return float("nan")
    df = pd.DataFrame({"v": v, "g": groups})
    gm = df.groupby("g")["v"].agg(["mean", "size"])
    return float((gm["size"] * (gm["mean"] - v.mean()) ** 2).sum() / tot)


# LOGIC: remove each person's own average from their features. Offline: the mean over
#   the whole night (uses no stage labels). Causal: the running mean of the night so
#   far. The honest evaluation is then rerun on the result.
# EDGE CASES: NaN-aware means.
# NOT HANDLED: offline needs a full night first (noted for deployment).
# CHECKED BY: tests/test_identity.py::test_remove_person_mean
def remove_person_mean(X, person, causal=False):
    X = pd.DataFrame(X).copy()
    g = X.groupby(np.asarray(person), sort=False)
    mu = g.transform(lambda s: s.expanding(1).mean()) if causal else g.transform("mean")
    return X - mu
