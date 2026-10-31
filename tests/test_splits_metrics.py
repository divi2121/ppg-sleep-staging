"""Tests for splits, metrics and models."""
import numpy as np
import pandas as pd
import pytest

from metrics import bootstrap_ci, cohen_kappa, confusion, holm, kappa_per_person, paired_difference, per_class_f1
from models import GBMModel, MajorityBaseline
from splits import assert_no_person_overlap, make_loro_folds, make_person_folds, make_random_folds

SUBJ = [f"S{i:03d}" for i in range(35)]


def test_person_folds_cover_once_no_overlap():
    folds = make_person_folds(SUBJ, n_folds=10, n_tune=4, seed=1)
    tested = [s for f in folds for s in f["test"]]
    assert sorted(tested) == SUBJ
    assert all(len(f["tune"]) == 4 for f in folds)


def test_loro_folds_cover_once_tune_has_n3():
    n3 = SUBJ[:16]
    folds = make_loro_folds(SUBJ, n3, n_tune=4, n_tune_n3=2, seed=1)
    assert len(folds) == 35
    assert sorted(s for f in folds for s in f["test"]) == SUBJ
    for f in folds:
        assert len(f["test"]) == 1 and len(f["tune"]) == 4 and len(f["train"]) == 30
        assert sum(s in n3 for s in f["tune"]) == 2
    # tuning duty is spread: no one tunes far more often than the rotation allows
    counts = pd.Series([s for f in folds for s in f["tune"]]).value_counts()
    assert counts.max() - counts.min() <= 2
    assert make_loro_folds(SUBJ, n3, seed=1) == folds


def test_loro_folds_raise_without_enough_n3():
    with pytest.raises(ValueError):
        make_loro_folds(SUBJ, SUBJ[:2], n_tune=4, n_tune_n3=2)


def test_overlap_is_caught():
    with pytest.raises(AssertionError):
        assert_no_person_overlap([{"test": ["A"], "tune": ["B"], "train": ["A", "C"]}])


def test_random_folds_balanced():
    f = make_random_folds(1000, 10, seed=1)
    assert np.bincount(f).tolist() == [100] * 10


def test_kappa_known():
    y = ["W", "W", "N", "N"]
    assert cohen_kappa(y, y, ["W", "N"]) == 1.0
    assert cohen_kappa(y, ["W", "N", "W", "N"], ["W", "N"]) == pytest.approx(0.0)
    assert confusion(y, ["W", "N", "W", "N"], ["W", "N"]).tolist() == [[1, 1], [1, 1]]


def test_f1_known():
    f = per_class_f1(["A", "A", "B"], ["A", "B", "B"], ["A", "B"])
    assert f["A"] == pytest.approx(2 / 3) and f["B"] == pytest.approx(2 / 3)


def test_kappa_per_person():
    df = pd.DataFrame({"subject": ["A"] * 4 + ["B"] * 4, "y": ["W", "N"] * 4,
                       "yhat": ["W", "N"] * 2 + ["N", "W"] * 2})
    k = kappa_per_person(df, ["W", "N"])
    assert k["A"] == 1.0 and k["B"] == -1.0


def test_bootstrap_ci_contains_median():
    v = np.random.default_rng(0).normal(0.5, 0.1, 40)
    med, lo, hi = bootstrap_ci(v, n_boot=2000)
    assert lo < med < hi


def test_paired_difference_sign():
    a = np.linspace(0.6, 0.8, 30); b = a - 0.1
    r = paired_difference(a, b, n_boot=2000)
    assert r["median"] == pytest.approx(0.1) and r["lo"] > 0 and r["p"] < 0.01


def test_holm_known():
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_majority_baseline():
    m = MajorityBaseline().fit(None, ["N2", "N2", "W"])
    assert list(m.predict([1, 2])) == ["N2", "N2"]


def test_gbm_learns_separable():
    pytest.importorskip("sklearn")
    X = np.r_[np.zeros((50, 2)), np.ones((50, 2))]; y = np.r_[["A"] * 50, ["B"] * 50]
    assert (GBMModel(max_iter=20).fit(X, y).predict(X) == y).all()
