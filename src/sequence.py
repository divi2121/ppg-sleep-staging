"""Sequence models over the night: one feature vector per 30 s epoch, read in order.

Architecture and training settings follow Tallal et al. 2026
(Sensors 26:5746), section 3.6.2 and Table 5: temporal CNN (64 channels), BiGRU (64 units per
direction), dropout 0.25, linear head; AdamW, lr 1e-3, weight decay 1e-4, up to 30 epochs,
effective-number weighted cross-entropy with beta 0.999, best epoch chosen on validation
macro F1. Not stated in the paper and chosen here: CNN kernel size 5, whole nights as
sequences, batch of 4 nights.
"""
import numpy as np
import pandas as pd


# LOGIC: split the epoch rows into one sequence per person, in epoch order. Missing values
#   become the training median of the column, plus one 0/1 missing flag per column (as in
#   LogRegFeatures), so the model can tell a filled value from a real one.
# EDGE CASES: an all-NaN training column gets median 0; rows are sorted by epoch inside
#   each person whatever their input order; row indices are kept so predictions map back.
# NOT HANDLED: gaps in the epoch numbers (dropped epochs) are not marked; the sequence
#   simply continues across them.
# CHECKED BY: tests/test_sequence.py::test_make_sequences_order_and_fill
def make_sequences(X, y, subjects, index, epochs, medians):
    """Returns a list of dicts {subject, x (T, 2p) float32, y (T,) int64, row (T,)}."""
    X = np.asarray(X, "float64")
    filled = np.where(np.isnan(X), medians, X)
    feats = np.hstack([filled, np.isnan(X).astype("float64")]).astype("float32")
    out = []
    for s in pd.unique(subjects):
        m = np.flatnonzero(subjects == s)
        m = m[np.argsort(np.asarray(epochs)[m], kind="stable")]
        out.append({"subject": s, "x": feats[m], "y": np.asarray(y)[m].astype("int64"),
                    "row": np.asarray(index)[m]})
    return out


# LOGIC: class weights by the effective number of samples (Cui et al. 2019), the weighting
#   Tallal used: w_c = (1 - beta) / (1 - beta ** n_c), scaled so the weights sum to the
#   number of classes.
# EDGE CASES: a class with 0 samples gets weight 0.
# NOT HANDLED: none.
# CHECKED BY: tests/test_sequence.py::test_effective_number_weights
def effective_number_weights(counts, beta=0.999):
    n = np.asarray(counts, "float64")
    w = np.where(n > 0, (1 - beta) / (1 - beta ** n), 0.0)
    return w * len(n) / w.sum()


# LOGIC: balanced class weights, as scikit-learn's class_weight="balanced" (used by our
#   logistic regression): w_c = n_total / (n_classes * n_c). Added 2026-10-23 because the
#   effective-number weights gave REM only about 1.2 x the weight of NREM and the BiGRU
#   stopped predicting REM.
# EDGE CASES: a class with 0 samples gets weight 0.
# NOT HANDLED: none.
# CHECKED BY: tests/test_sequence.py::test_balanced_weights
def balanced_weights(counts):
    n = np.asarray(counts, "float64")
    return np.where(n > 0, n.sum() / (len(n) * np.where(n > 0, n, 1)), 0.0)


def _torch_model(n_in, n_classes, conv_ch=64, kernel=5, hidden=64, dropout=0.25, spec=False, enc_dim=64,
                 causal=False):
    import torch
    import torch.nn as nn

    class SpecEncoder(nn.Module):
        # LOGIC: turns one epoch's spectrogram (2 channels x 30 frames x 64 rhythms) into
        #   enc_dim numbers. Three blocks of 3x3 convolution, batch norm and ReLU (16, 32,
        #   enc_dim channels); the first two halve time and rhythm (max pool); then the mean
        #   over what is left. Added 2026-10-24 (notebook 09); about 23,000 weights.
        # EDGE CASES: only real epochs are passed in (padding is removed by the caller), so
        #   padded epochs never enter the batch-norm statistics.
        # NOT HANDLED: the encoder sees one epoch at a time; context comes from the BiGRU.
        # CHECKED BY: tests/test_sequence.py::test_spec_encoder_learns
        def __init__(self):
            super().__init__()
            def block(a, b, pool):
                layers = [nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU()]
                return layers + ([nn.MaxPool2d(2)] if pool else [])
            self.net = nn.Sequential(*block(2, 16, True), *block(16, 32, True), *block(32, enc_dim, False),
                                     nn.AdaptiveAvgPool2d(1), nn.Flatten())

        def forward(self, z):                         # z: (n, 2, 30, 64)
            return self.net(z)

    class CNNBiGRU(nn.Module):
        # LOGIC: a 1D convolution over neighbouring epochs picks out local patterns; a
        #   bidirectional GRU then reads the whole night forwards and backwards; dropout and
        #   a linear layer give one score per class for every epoch.
        #   causal=True (live, 2026-10-24): the convolution is padded on the left only, so
        #   output t sees epochs t-kernel+1 .. t, and the GRU reads forwards only. Then
        #   output t depends on epochs 0 .. t and nothing later.
        # EDGE CASES: padding keeps the sequence length; padded epochs are masked in the loss.
        # NOT HANDLED: offline (causal=False) the backward GRU reads the future.
        # CHECKED BY: tests/test_sequence.py::test_bigru_uses_context, test_causal_model_ignores_future
        def __init__(self):
            super().__init__()
            self.enc = SpecEncoder() if spec else None
            self.causal, self.kernel = causal, kernel
            self.conv = nn.Conv1d(n_in + (enc_dim if spec else 0), conv_ch, kernel,
                                  padding=0 if causal else kernel // 2)
            self.act = nn.ReLU()
            self.gru = nn.GRU(conv_ch, hidden, batch_first=True, bidirectional=not causal)
            self.drop = nn.Dropout(dropout)
            self.head = nn.Linear((1 if causal else 2) * hidden, n_classes)

        def forward(self, x, z=None, mask=None):      # x: (batch, T, n_in); z: (batch, T, 2, 30, 64)
            if self.enc is not None:                  # encode real epochs only, zeros for padding
                e = torch.zeros(x.shape[0], x.shape[1], enc_dim)
                e[mask] = self.enc(z[mask])
                x = torch.cat([x, e], dim=2)
            xc = x.transpose(1, 2)
            if self.causal:                           # kernel - 1 zeros before the first epoch
                xc = torch.nn.functional.pad(xc, (self.kernel - 1, 0))
            h = self.act(self.conv(xc)).transpose(1, 2)
            h, _ = self.gru(h)
            return self.head(self.drop(h))            # (batch, T, n_classes)

    return CNNBiGRU()


def _pad(seqs):
    """Returns x, y, and (spec, mask) or (None, None) when the nights carry no "spec"."""
    import torch
    T = max(len(s["y"]) for s in seqs)
    x = np.zeros((len(seqs), T, seqs[0]["x"].shape[1]), "float32")
    y = np.full((len(seqs), T), -1, "int64")
    has = "spec" in seqs[0]
    z = np.zeros((len(seqs), T) + seqs[0]["spec"].shape[1:], "float32") if has else None
    for i, s in enumerate(seqs):
        x[i, :len(s["y"])] = s["x"]; y[i, :len(s["y"])] = s["y"]
        if has:
            z[i, :len(s["y"])] = s["spec"]
    if not has:
        return torch.from_numpy(x), torch.from_numpy(y), None, None
    return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(z), torch.from_numpy(y >= 0)


# LOGIC: train on the training nights, check macro F1 on the validation nights after every
#   pass, keep the weights of the best pass (Tallal's selection rule), predict test nights.
#   Batches are whole nights, padded; padded epochs carry label -1 and are ignored by the
#   loss. Fixed seed.
# EDGE CASES: a class absent from training gets weight 0 (run_folds-style callers check
#   classes first); validation macro F1 is computed per epoch pooled over validation nights.
#   When the nights carry a "spec" array (T, 2, 30, 64), the spectrogram encoder is added and
#   its 64 numbers per epoch are joined to the features (2026-10-24).
# NOT HANDLED: learning-rate schedules; multiple seeds (one seed per call).
# CHECKED BY: tests/test_sequence.py::test_bigru_uses_context, test_spec_encoder_learns
class BiGRUSequence:
    def __init__(self, n_classes=3, max_epochs=30, batch_nights=4, lr=1e-3, weight_decay=1e-4,
                 beta=0.999, weighting="effective", seed=20261011, causal=False, **model_kw):
        self.causal = causal
        self.n_classes, self.max_epochs, self.batch_nights = n_classes, max_epochs, batch_nights
        self.lr, self.weight_decay, self.beta, self.seed, self.model_kw = lr, weight_decay, beta, seed, model_kw
        if weighting not in ("effective", "balanced"):
            raise ValueError("weighting must be 'effective' or 'balanced'")
        self.weighting = weighting

    def fit(self, train, val):
        import torch
        from sklearn.metrics import f1_score
        torch.manual_seed(self.seed); rng = np.random.default_rng(self.seed)
        counts = np.bincount(np.concatenate([s["y"] for s in train]), minlength=self.n_classes)
        w = (effective_number_weights(counts, self.beta) if self.weighting == "effective"
             else balanced_weights(counts))
        w = torch.tensor(w, dtype=torch.float32)
        loss_fn = torch.nn.CrossEntropyLoss(weight=w, ignore_index=-1)
        self.n_in_, self.spec_ = train[0]["x"].shape[1], "spec" in train[0]
        self.model_ = _torch_model(self.n_in_, self.n_classes, spec=self.spec_, causal=self.causal, **self.model_kw)
        opt = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        best, state, self.history_ = -1.0, None, []
        yv = np.concatenate([s["y"] for s in val])
        for ep in range(self.max_epochs):
            self.model_.train()
            order = rng.permutation(len(train))
            for b in range(0, len(order), self.batch_nights):
                x, y, z, mask = _pad([train[i] for i in order[b:b + self.batch_nights]])
                opt.zero_grad()
                out = self.model_(x, z, mask)
                loss_fn(out.reshape(-1, self.n_classes), y.reshape(-1)).backward()
                opt.step()
            f1 = f1_score(yv, np.concatenate(self.predict(val)), average="macro",
                          labels=list(range(self.n_classes)), zero_division=0)
            self.history_.append(f1)
            if f1 > best:
                best, state = f1, {k: v.detach().clone() for k, v in self.model_.state_dict().items()}
        self.model_.load_state_dict(state)
        self.best_val_f1_, self.best_epoch_ = best, int(np.argmax(self.history_)) + 1
        return self

    def predict(self, seqs):
        import torch
        self.model_.eval(); out = []
        with torch.no_grad():
            for s in seqs:
                x, _, z, mask = _pad([s])
                out.append(self.model_(x, z, mask).argmax(-1)[0].numpy())
        return out

    # LOGIC: decode one epoch at a time (live, 2026-10-24). Keeps the last kernel - 1 input
    #   vectors (zeros at the start, like the left padding) and the GRU's hidden state; each
    #   step runs the convolution on the last kernel inputs, one GRU step and the head.
    # EDGE CASES: only for causal=True models without a spectrogram; others raise.
    # NOT HANDLED: none.
    # CHECKED BY: tests/test_sequence.py::test_stepper_matches_whole_night
    def stepper(self):
        if not self.causal or self.spec_:
            raise ValueError("stepper needs a causal model without a spectrogram")
        return _Stepper(self.model_, self.n_in_)

    # LOGIC: save the weights and every setting needed to rebuild the model, plus `extra`
    #   (the caller's feature names and training medians), in one torch file.
    # EDGE CASES: load refuses a file that holds anything but tensors, lists, dicts, numbers and
    #   strings (weights_only=True), so a tampered model file cannot run code when loaded.
    # NOT HANDLED: files from other code versions are not checked.
    # CHECKED BY: tests/test_sequence.py::test_save_load_roundtrip, test_load_refuses_a_file_with_code
    def save(self, path, extra=None):
        import torch
        torch.save({"state": self.model_.state_dict(), "n_in": self.n_in_, "spec": self.spec_,
                    "n_classes": self.n_classes, "causal": self.causal, "model_kw": self.model_kw,
                    "extra": extra}, path)

    @classmethod
    def load(cls, path):
        import torch
        d = torch.load(path, weights_only=True)    # refuse anything but tensors and plain values (2026-10-26)
        m = cls(n_classes=d["n_classes"], causal=d["causal"], **d["model_kw"])
        m.n_in_, m.spec_, m.extra = d["n_in"], d["spec"], d["extra"]
        m.model_ = _torch_model(m.n_in_, m.n_classes, spec=m.spec_, causal=m.causal, **m.model_kw)
        m.model_.load_state_dict(d["state"]); m.model_.eval()
        return m


class _Stepper:
    def __init__(self, net, n_in):
        import torch
        net.eval()
        self.net, self.h = net, None
        self.buf = torch.zeros(1, n_in, net.kernel - 1)

    def step(self, x_t):
        import torch
        with torch.no_grad():
            x = torch.as_tensor(np.asarray(x_t, "float32")).reshape(1, -1, 1)
            win = torch.cat([self.buf, x], dim=2)
            self.buf = win[:, :, 1:]
            h = self.net.act(self.net.conv(win)).transpose(1, 2)     # (1, 1, conv_ch)
            out, self.h = self.net.gru(h, self.h)
            return int(self.net.head(out)[0, 0].argmax())
