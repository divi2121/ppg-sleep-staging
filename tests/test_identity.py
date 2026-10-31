"""Tests for identity.py."""
import numpy as np
import pandas as pd
import pytest

from identity import identify_person, identity_blocks, remove_person_mean, variance_split


class _Nearest:
    """Tiny stand-in classifier so the split logic is testable without scikit-learn."""
    def fit(self, X, y):
        self.c = pd.DataFrame(X).groupby(np.asarray(y)).mean(); return self
    def predict(self, X):
        d = ((pd.DataFrame(X).values[:, None, :] - self.c.values[None]) ** 2).sum(-1)
        return self.c.index.values[d.argmin(1)]


def test_identify_person_split():
    assert list(identity_blocks([0, 599, 600, 1300])) == [0, 0, 1, 2]
    rng = np.random.default_rng(0)
    people = np.repeat(["A", "B", "C"], 200)
    X = pd.DataFrame({"f": np.repeat([0.0, 5.0, 10.0], 200) + rng.normal(0, 0.1, 600)})
    t = np.tile(np.arange(200) * 30.0, 3)
    r = identify_person(X, people, t, model_factory=_Nearest)
    assert r["accuracy"] == 1.0 and r["chance"] == pytest.approx(1 / 3)


def test_variance_split_person_signal():
    rng = np.random.default_rng(0)
    person = np.repeat(["A", "B"], 100); stage = np.tile(["W", "N"], 100)
    X = pd.DataFrame({"f": np.where(person == "A", 0.0, 3.0) + rng.normal(0, 0.1, 200)})
    v = variance_split(X, person, stage, n_shuffle=200)
    assert v.loc["f", "eta2_person"] > 0.9 > v.loc["f", "eta2_person_null95"]
    assert v.loc["f", "eta2_stage"] < 0.05


def test_remove_person_mean():
    X = pd.DataFrame({"f": [1.0, 3.0, 10.0, 14.0]}); p = ["A", "A", "B", "B"]
    assert remove_person_mean(X, p).f.tolist() == [-1.0, 1.0, -2.0, 2.0]
    assert remove_person_mean(X, p, causal=True).f.tolist() == [0.0, 1.0, 0.0, 2.0]
