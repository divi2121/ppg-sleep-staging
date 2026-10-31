"""Tests for sequence.py."""
import numpy as np
import pytest

from sequence import BiGRUSequence, balanced_weights, effective_number_weights, make_sequences


def test_balanced_weights():
    w = balanced_weights([10, 30, 60])                    # 100 / (3 * n)
    assert w == pytest.approx([100 / 30, 100 / 90, 100 / 180])
    assert balanced_weights([0, 5])[0] == 0.0
    with pytest.raises(ValueError):
        BiGRUSequence(weighting="nope")


def test_make_sequences_order_and_fill():
    X = np.array([[1.0], [np.nan], [3.0], [4.0]])
    y = np.array([0, 1, 2, 0]); subj = np.array(["A", "A", "B", "A"], dtype=object)
    idx = np.array([10, 11, 12, 13]); ep = np.array([5, 1, 0, 3])
    seqs = make_sequences(X, y, subj, idx, ep, medians=np.array([9.0]))
    a = seqs[0]
    assert a["subject"] == "A" and list(a["row"]) == [11, 13, 10]          # sorted by epoch 1, 3, 5
    assert list(a["y"]) == [1, 0, 0]
    assert a["x"][0, 0] == 9.0 and a["x"][0, 1] == 1.0                     # NaN -> median, flag 1
    assert a["x"][1, 0] == 4.0 and a["x"][1, 1] == 0.0
    assert seqs[1]["subject"] == "B" and list(seqs[1]["row"]) == [12]


def test_effective_number_weights():
    w = effective_number_weights([1, 1], beta=0.5)
    assert w == pytest.approx([1.0, 1.0])                                  # equal counts, equal weights
    w = effective_number_weights([1, 2], beta=0.5)                         # (0.5/0.5, 0.5/0.75) = (1, 2/3)
    assert w == pytest.approx(np.array([1.0, 2 / 3]) * 2 / (5 / 3))
    assert effective_number_weights([0, 5])[0] == 0.0


def _context_task(n_nights, T, rng):
    # The label of each epoch is the sign of the feature 3 epochs EARLIER, so a model that
    # sees one epoch at a time cannot do better than chance; one with memory can.
    seqs = []
    for i in range(n_nights):
        x = rng.normal(size=T)
        y = np.r_[np.zeros(3, int), (x[:-3] > 0).astype(int)]
        seqs.append({"subject": f"s{i}", "x": x[:, None].astype("float32"), "y": y, "row": np.arange(T)})
    return seqs


def test_bigru_uses_context():
    rng = np.random.default_rng(0)
    train, val, test = _context_task(16, 200, rng), _context_task(4, 200, rng), _context_task(4, 200, rng)
    m = BiGRUSequence(n_classes=2, max_epochs=40, batch_nights=4, lr=1e-2, conv_ch=8, kernel=1, hidden=16)
    m.fit(train, val)
    acc = np.mean(np.concatenate(m.predict(test)) == np.concatenate([s["y"] for s in test]))
    assert acc > 0.9                                                       # memory solves it
    # the same-epoch feature alone carries no information about the label:
    same = np.concatenate([(s["x"][:, 0] > 0).astype(int) for s in test])
    assert abs(np.mean(same == np.concatenate([s["y"] for s in test])) - 0.5) < 0.08


def test_spec_encoder_learns():
    # The label is which rhythm row is bright in channel 0 of the picture (row 10 or row 40);
    # the feature column is pure noise, so only the encoder can solve it.
    rng = np.random.default_rng(1)

    def nights(n, T):
        out = []
        for i in range(n):
            y = rng.integers(0, 2, T)
            z = rng.normal(size=(T, 2, 30, 64)).astype("float32")
            z[np.arange(T), 0, :, np.where(y == 1, 40, 10)] += 3.0
            out.append({"subject": f"s{i}", "x": rng.normal(size=(T, 1)).astype("float32"), "y": y,
                        "row": np.arange(T), "spec": z.astype("float16")})
        return out
    train, val, test = nights(8, 40), nights(2, 40), nights(2, 40)
    m = BiGRUSequence(n_classes=2, max_epochs=20, batch_nights=4, lr=3e-3, weighting="balanced").fit(train, val)
    acc = np.mean(np.concatenate(m.predict(test)) == np.concatenate([s["y"] for s in test]))
    assert acc > 0.95
    assert m.model_.enc is not None and m.model_.conv.in_channels == 1 + 64


# ---- live plan Task 4 (2026-10-24): forward-only model ----

def _small_causal():
    nights = _context_task(8, 60, np.random.default_rng(0))
    m = BiGRUSequence(n_classes=2, max_epochs=3, causal=True, conv_ch=8, hidden=8).fit(nights[:6], nights[6:])
    return m, nights


def test_causal_model_ignores_future():
    m, nights = _small_causal()
    x = nights[0]["x"].copy()
    a = m.predict([{**nights[0], "x": x}])[0]
    x[30:] = 99.0                                                 # change everything after epoch 29
    b = m.predict([{**nights[0], "x": x}])[0]
    assert np.array_equal(a[:30], b[:30])


def test_stepper_matches_whole_night():
    m, nights = _small_causal()
    st = m.stepper()
    assert np.array_equal([st.step(v) for v in nights[0]["x"]], m.predict([nights[0]])[0])


def test_save_load_roundtrip(tmp_path):
    m, nights = _small_causal()
    m.save(tmp_path / "m.pt", extra={"cols": ["a"], "medians": [1.5]})
    m2 = BiGRUSequence.load(tmp_path / "m.pt")
    assert np.array_equal(m2.predict([nights[0]])[0], m.predict([nights[0]])[0])
    assert m2.extra == {"cols": ["a"], "medians": [1.5]} and m2.causal


class _NotPlainData:
    """A custom object: pickle can save it, weights_only loading must refuse it."""


def test_load_refuses_a_file_with_code(tmp_path):
    import pickle
    import pytest
    import torch
    m, _ = _small_causal()
    m.save(tmp_path / "ok.pt")
    BiGRUSequence.load(tmp_path / "ok.pt")                                   # a normal file still loads
    torch.save({"state": {}, "n_in": 1, "spec": False, "n_classes": 2, "causal": True, "model_kw": {},
                "extra": _NotPlainData()}, tmp_path / "bad.pt")
    with pytest.raises(pickle.UnpicklingError):
        BiGRUSequence.load(tmp_path / "bad.pt")
