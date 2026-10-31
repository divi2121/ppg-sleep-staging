"""Tests for evaluation.py. Ported from the notebook 06 test cells, 2026-10-23."""
import numpy as np
import pandas as pd
import pytest

from evaluation import (FORBIDDEN, LABELS, LogRegFeatures, macro_f1_per_person, make_xy,
                        rem_rates, run_folds, tune_settings)
from models import MajorityBaseline


def _toy():
    return pd.DataFrame({
        "subject": ["S1", "S1", "S2", "S2"], "epoch": [0, 1, 0, 1],
        "t_start": [0., 30., 0., 30.], "pass": [True] * 4,
        "stage5": ["W", "N2", "R", "N3"],
        "hr_mean": [60., np.nan, 70., 72.], "motion": [.1, .2, .3, .4],
        "stage3": ["W", "NREM", "REM", np.nan],
    })


def test_make_xy():
    X, y, s, i = make_xy(_toy(), ["hr_mean", "motion"])
    assert X.shape == (3, 2) and np.isnan(X[1, 0]) and list(i) == [0, 1, 2]
    assert list(y) == [0, 1, 2]
    assert list(make_xy(_toy(), ["motion", "hr_mean"])[0][:, 0]) == [.1, .2, .3]
    for bad in FORBIDDEN:
        with pytest.raises(ValueError):
            make_xy(_toy(), ["motion", bad])
    d = _toy(); d.loc[0, "stage3"] = "N9"
    with pytest.raises(ValueError):
        make_xy(d, ["motion"])
    with pytest.raises(KeyError):
        make_xy(_toy(), ["nope"])


def test_macro_f1_per_person():
    p = pd.DataFrame({"subject": "A", "y": [0, 1, 2, 2], "yhat": [0, 1, 2, 2]})
    assert macro_f1_per_person(p, LABELS).loc["A", "macro_f1"] == 1.0
    p = pd.DataFrame({"subject": "A", "y": [0, 0, 1, 1], "yhat": [0, 1, 1, 1]})
    r = macro_f1_per_person(p, LABELS).loc["A"]
    assert r.macro_f1 == pytest.approx((2 / 3 + 4 / 5) / 2) and r.n_stages == 2
    p = pd.DataFrame({"subject": "A", "y": [0, 0], "yhat": [0, 2]})
    r = macro_f1_per_person(p, LABELS).loc["A"]
    assert r.macro_f1 == pytest.approx((2 / 3 + 0) / 2) and r.n_stages == 2


def test_run_folds():
    subj = np.repeat(np.array(["a", "b", "c", "d", "e", "f"], dtype=object), 4)
    y = np.tile([0, 0, 1, 2], 6); X = np.zeros((24, 2)); idx = np.arange(24) + 100
    folds = [{"test": ["a"], "tune": ["b"], "train": ["c", "d", "e", "f"]},
             {"test": ["b"], "tune": ["c"], "train": ["a", "d", "e", "f"]}]
    r = run_folds(lambda: MajorityBaseline(), X, y, subj, idx, folds, "m", verbose=False)
    assert sorted(r.subject.unique()) == ["a", "b"] and len(r) == 8
    assert set(r.row) == set(idx[np.isin(subj, ["a", "b"])])
    assert (r.yhat == 0).all()
    with pytest.raises(AssertionError):
        run_folds(lambda: MajorityBaseline(), X, y, subj, idx,
                  [{"test": ["a"], "tune": ["b"], "train": ["a", "c"]}], "m", verbose=False)
    y2 = y.copy(); y2[np.isin(subj, ["c", "d", "e", "f"])] = 0
    with pytest.raises(ValueError):
        run_folds(lambda: MajorityBaseline(), X, y2, subj, idx, folds[:1], "m", verbose=False)


def test_logreg_features():
    rng = np.random.default_rng(1)
    yy = rng.integers(0, 3, 600); XX = rng.normal(size=(600, 4)) + 2.0 * yy[:, None]
    XX[rng.random(XX.shape) < 0.1] = np.nan
    m = LogRegFeatures(C=1.0).fit(XX, yy)
    assert (m.predict(XX) == yy).mean() > 0.8
    Xn = XX[:5].copy(); Xn[0, 0] = np.nan
    P = m._prep(Xn)
    assert np.isclose(P[0, 0], (m.med_[0] - m.mu_[0]) / m.sd_[0]) and P[0, 4] == 1.0
    LogRegFeatures().fit(np.full((30, 2), np.nan), np.arange(30) % 3)


def test_tune_settings():
    rng = np.random.default_rng(2)
    subj = np.repeat(np.array([f"p{i}" for i in range(8)], dtype=object), 60)
    y = rng.integers(0, 3, subj.size); X = rng.normal(size=(subj.size, 3)) + 2.0 * y[:, None]
    folds = [{"test": ["p0"], "tune": ["p1", "p2"], "train": [f"p{i}" for i in range(3, 8)]},
             {"test": ["p3"], "tune": ["p4", "p5"], "train": ["p0", "p1", "p2", "p6", "p7"]}]
    res = tune_settings(lambda **s: LogRegFeatures(**s), [{"C": 1e-9}, {"C": 1.0}], X, y, subj, folds,
                        verbose=False)
    assert res.iloc[0]["C"] == 1.0 and res.iloc[0]["tune_macro_f1"] > res.iloc[1]["tune_macro_f1"]


def test_rem_rates():
    p = pd.DataFrame({"y": [2, 2, 2, 2, 1, 1, 1, 1, 1, 0], "yhat": [2, 2, 1, 0, 2, 1, 1, 1, 1, 0]})
    assert rem_rates(p) == (0.5, 0.2)
    assert np.isnan(rem_rates(p[p.y != 2])[0])


def test_model_for_returns_the_fold_that_tested_the_person():
    # live plan Task 5 (2026-10-24): a person is only ever decoded by the model that never trained on them
    from evaluation import model_for
    folds = [{"train": ["A", "B"], "tune": ["C"], "test": ["D"]},
             {"train": ["C", "D"], "tune": ["A"], "test": ["B"]}]
    assert model_for("D", folds) == 0 and model_for("B", folds) == 1
    import pytest
    with pytest.raises(KeyError):
        model_for("Z", folds)


# ---- live plan section H item 3 (2026-10-26): folds read from models/live/folds.json ----
import json as _json
from pathlib import Path as _Path

_SHIPPED = _Path(__file__).resolve().parents[1] / "models" / "live" / "folds.json"


def test_load_folds_roundtrip(tmp_path):
    from evaluation import load_folds, model_for
    d = {"seed": 1, "n_folds": 2, "n_tune": 1, "folds": [
        {"fold": 0, "train": ["A", "B"], "tune": ["C"], "test": ["D"]},
        {"fold": 1, "train": ["C", "D"], "tune": ["A"], "test": ["B"]}]}
    (tmp_path / "f.json").write_text(_json.dumps(d))
    folds = load_folds(tmp_path / "f.json")
    assert folds == [{"train": ["A", "B"], "tune": ["C"], "test": ["D"]},
                     {"train": ["C", "D"], "tune": ["A"], "test": ["B"]}]
    assert model_for("B", folds) == 1
    d["folds"][0]["fold"], d["folds"][1]["fold"] = 1, 0                     # numbers out of order
    (tmp_path / "g.json").write_text(_json.dumps(d))
    with pytest.raises(ValueError):
        load_folds(tmp_path / "g.json")


def test_shipped_folds_file_shape():
    from evaluation import load_folds
    folds = load_folds(_SHIPPED)
    assert len(folds) == 5                                                  # 5 folds
    assert [(len(f["train"]), len(f["tune"]), len(f["test"])) for f in folds] == [(24, 4, 7)] * 5   # 24 + 4 + 7 = 35
    tests = [p for f in folds for p in f["test"]]
    assert len(tests) == 35 and len(set(tests)) == 35                        # each person tested exactly once
    for f in folds:
        assert len(set(f["train"]) | set(f["tune"]) | set(f["test"])) == 35  # no person in two parts of a fold


def test_shipped_folds_equal_make_person_folds():
    from evaluation import load_folds
    from splits import make_person_folds
    feats = _SHIPPED.parents[2] / "data" / "derived" / "features_all35.csv"
    if not feats.exists():
        pytest.skip("features_all35.csv is not available (it is derived from restricted data)")
    people = sorted(pd.read_csv(feats, usecols=["subject"]).subject.unique())
    d = _json.loads(_SHIPPED.read_text())
    fresh = make_person_folds(people, n_folds=d["n_folds"], n_tune=d["n_tune"], seed=d["seed"])
    shipped = load_folds(_SHIPPED)
    for a, b in zip(fresh, shipped):
        assert {k: sorted(a[k]) for k in ("train", "tune", "test")} == b
