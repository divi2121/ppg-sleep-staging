"""Tests for groups.py. Ported from the notebook 06 test cell, 2026-10-24."""
import numpy as np
import pandas as pd
import pytest

from groups import build_group


class _FakeRec:
    def __init__(self, n, drop):
        keep = np.ones(n, bool); keep[drop] = False
        self.epochs = pd.DataFrame({"keep": keep, "start_sample_wrist": np.arange(n) * 1920})
        self.t64 = np.arange(n * 1920 + 1) / 64.0


def test_build_group_fake_recordings(tmp_path):
    base = pd.DataFrame({"subject": ["A"] * 3 + ["B"] * 4, "epoch": [0, 2, 3, 0, 1, 2, 3],
                         "pass": [True, False, True, True, True, True, False]})
    fake = {"A": _FakeRec(4, [1]), "B": _FakeRec(4, [])}
    comp = lambda rec, ss, ts: pd.DataFrame({"feat": ts.astype(float)})
    G = build_group("t", comp, True, ["B", "A"], fake.get, base, tmp_path)   # build order differs
    assert list(G.index) == list(base.index)
    assert np.allclose(G["feat"], [0, np.nan, 90, 0, 30, 60, np.nan], equal_nan=True)
    assert np.isnan(G.loc[[1, 6], "feat_z"]).all()                          # gated epochs are NaN
    assert G.loc[0, "feat_z"] == pytest.approx((0 - 45) / np.std([0, 90], ddof=1))
    G2 = build_group("t", comp, True, ["B", "A"], fake.get, base, tmp_path)  # loads the file
    assert np.allclose(G.to_numpy(float), G2.to_numpy(float), equal_nan=True)
    with pytest.raises(ValueError):
        build_group("u", comp, True, ["A"], fake.get, base, tmp_path)        # a person missing
    with pytest.raises(ValueError):
        build_group("v", lambda r, s, t: pd.DataFrame({"pass": t}), False, ["B", "A"], fake.get, base, tmp_path)
