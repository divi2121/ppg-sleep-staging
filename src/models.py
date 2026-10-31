"""Models, all with fit(X, y) / predict(X).

Logic approved in CODE_PLAN.md group F.
Not yet written: the PyTorch CNN and the PaPaGei / Pulse-PPG wrappers. They wait for
the installs (CODE_PLAN decision 4).
"""
import numpy as np


# LOGIC: the floor every model must clear: always predict the most common stage in
#   the training labels.
# EDGE CASES: ties go to the first label in sorted order.
# NOT HANDLED: none.
# CHECKED BY: tests/test_splits_metrics.py::test_majority_baseline
class MajorityBaseline:
    def fit(self, X, y):
        vals, counts = np.unique(np.asarray(y), return_counts=True)
        self.label_ = vals[np.argmax(counts)]
        return self

    def predict(self, X):
        return np.full(len(X), self.label_, dtype=object)


# LOGIC: gradient boosting on the epoch features (scikit-learn
#   HistGradientBoostingClassifier), which accepts NaN, so missing features need no
#   filling. Classes are weighted by inverse frequency so rare stages count.
# EDGE CASES: settings come in as keyword arguments, to be tuned on tune people only.
# NOT HANDLED: tuning; settings are chosen on tune people only. scikit-learn 1.9.1
#   is in the venv as a dependency of neurokit2 (no separate install was made).
# CHECKED BY: tests/test_splits_metrics.py::test_gbm_learns_separable
class GBMModel:
    def __init__(self, **params):
        self.params = {"class_weight": "balanced", "random_state": 20261011, **params}

    def fit(self, X, y):
        from sklearn.ensemble import HistGradientBoostingClassifier
        self.model_ = HistGradientBoostingClassifier(**self.params).fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)
