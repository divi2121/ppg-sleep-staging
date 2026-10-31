"""Train/test splits.

Logic approved in CODE_PLAN.md group E.
"""
import numpy as np


# LOGIC: honest folds by person. Shuffle the people with a fixed seed and deal them
#   into n_folds test groups, so each person is tested exactly once. For each fold,
#   n_tune people are drawn (same seed) from the remaining people for tuning and
#   early stopping; the rest train.
# EDGE CASES: raises if a fold would leave fewer than 1 training person.
# NOT HANDLED: stratifying folds by apnea band or sex (not in the plan).
# CHECKED BY: tests/test_splits_metrics.py::test_person_folds_cover_once_no_overlap
def make_person_folds(subjects, n_folds=10, n_tune=4, seed=20261011):
    """Returns a list of dicts: {"test": [...], "tune": [...], "train": [...]}."""
    subjects = sorted(subjects)
    rng = np.random.default_rng(seed)
    order = list(rng.permutation(subjects))
    tests = [order[i::n_folds] for i in range(n_folds)]
    folds = []
    for test in tests:
        rest = [s for s in subjects if s not in test]
        tune = list(rng.choice(rest, size=n_tune, replace=False))
        train = [s for s in rest if s not in tune]
        if not train:
            raise ValueError("no training people left")
        folds.append({"test": sorted(test), "tune": sorted(tune), "train": train})
    assert_no_person_overlap(folds)
    return folds


# LOGIC: leave-one-recording-out (one night per person in DREAMT, so also one person
#   out), decided 2026-10-11 (amendment A3). Each person is the test set once. For each
#   fold, n_tune people are drawn from the other people for tuning and early stopping:
#   n_tune_n3 from the people who have any N3, the rest from those who have none. The
#   draw rotates through a seeded shuffle of each group, so tuning duty is spread
#   evenly instead of the same people being picked again; the test person is skipped.
#   Everyone else trains.
# EDGE CASES: raises if there are fewer than n_tune_n3 + 1 N3 people or fewer than
#   n_tune - n_tune_n3 + 1 non-N3 people (the test person may be in either group).
# NOT HANDLED: REM balance in the tuning set (6 people have no REM); not asked for.
# CHECKED BY: tests/test_splits_metrics.py::test_loro_folds_cover_once_tune_has_n3
def make_loro_folds(subjects, n3_subjects, n_tune=4, n_tune_n3=2, seed=20261011):
    """Returns a list of dicts: {"test": [s], "tune": [...], "train": [...]}."""
    subjects = sorted(subjects)
    n3 = sorted(set(n3_subjects) & set(subjects))
    other = [s for s in subjects if s not in n3]
    if len(n3) < n_tune_n3 + 1 or len(other) < n_tune - n_tune_n3 + 1:
        raise ValueError("not enough people with and without N3 for the tuning sets")
    rng = np.random.default_rng(seed)
    pools = [(list(rng.permutation(n3)), n_tune_n3), (list(rng.permutation(other)), n_tune - n_tune_n3)]
    pos = [0, 0]
    folds = []
    for test in subjects:
        tune = []
        for k, (pool, need) in enumerate(pools):
            got = 0
            while got < need:
                s = pool[pos[k] % len(pool)]
                pos[k] += 1
                if s != test and s not in tune:
                    tune.append(s); got += 1
        train = [s for s in subjects if s != test and s not in tune]
        folds.append({"test": [test], "tune": sorted(tune), "train": train})
    assert_no_person_overlap(folds)
    return folds


# LOGIC: naive folds for the inflated comparison: each epoch gets a fold number at
#   random (fixed seed), ignoring who it came from.
# EDGE CASES: none.
# NOT HANDLED: by design, people appear in train and test; it is reported as inflated.
# CHECKED BY: tests/test_splits_metrics.py::test_random_folds_balanced
def make_random_folds(n_epochs, n_folds=10, seed=20261011):
    rng = np.random.default_rng(seed)
    return rng.permutation(np.arange(n_epochs) % n_folds)


# LOGIC: raise if any person appears in more than one of train, tune and test within
#   a fold. This protects the headline result.
# EDGE CASES: none.
# NOT HANDLED: none.
# CHECKED BY: tests/test_splits_metrics.py::test_overlap_is_caught
def assert_no_person_overlap(folds):
    for i, f in enumerate(folds):
        a, b, c = set(f["test"]), set(f["tune"]), set(f["train"])
        if (a & b) or (a & c) or (b & c):
            raise AssertionError(f"fold {i}: a person appears in two of train/tune/test")
