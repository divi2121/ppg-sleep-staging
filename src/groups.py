"""Building a feature group for all people, and loading people without rerunning the detector.

Moved here on 2026-10-24 from notebook 06 section 12 (build_group) and the
beats-cache cells (load_beats, load_rec_cached), unchanged except that paths are arguments.
"""
import gc
from pathlib import Path

import numpy as np
import pandas as pd

from features import context_features, normalise_per_person


# LOGIC: run the beat detector once for everyone and save each beat's sample index and
#   refined time; later calls read the file.
# EDGE CASES: read with float_precision="round_trip": pandas' default parser changed 4,883
#   of 32,137 S042 beat times by up to 3.6e-12 s (found 2026-10-21).
# NOT HANDLED: a change to Config (filter, detector) does not invalidate the file.
# CHECKED BY: notebook 06 S042 check (cached and fresh beats, intervals, amplitude identical)
def load_beats(people, cache, data_root):
    """Returns {person: (beat sample indices, beat times)}; detects and saves on first call."""
    from recording import Recording
    cache = Path(cache)
    if cache.exists():
        B = pd.read_csv(cache, float_precision="round_trip")
    else:
        parts = []
        for s in people:
            rec = Recording.load(s, data_root)
            parts.append(pd.DataFrame({"subject": s, "idx": rec.beat_idx, "t": rec.beat_times}))
            del rec; gc.collect()
        B = pd.concat(parts, ignore_index=True)
        B.to_csv(cache, index=False)
    return {s: (g["idx"].to_numpy(int), g["t"].to_numpy("float64")) for s, g in B.groupby("subject")}


def load_rec_cached(s, beats, data_root):
    """Recording for person s with its beats taken from the cache, not detected again."""
    from recording import Recording
    if s not in beats:
        raise KeyError(f"{s} is not in the beats cache")
    rec = Recording.load(s, data_root)
    rec.__dict__["_beats"] = beats[s]
    rec._failed = []
    return rec


# LOGIC: build one feature group for all people. For each person, load the Recording, take
#   the same kept epochs as the epoch table, call compute(rec, ss, ts), which returns one row
#   of raw columns per epoch. Rows are lined up with the base table by (subject, epoch); any
#   mismatch raises. For a heart group (gated) the columns are NaN where base "pass" is
#   False. Then per-person z score over the night, and _past4 / _next4 of each z column.
#   Cached as features_plus_<name>.csv; an existing file is loaded, not rebuilt.
# EDGE CASES: compute returning the wrong number of rows raises; keys not matching base
#   raise; a new column name already in base raises.
# NOT HANDLED: causal scaling and causal context (offline only).
# CHECKED BY: tests/test_groups.py::test_build_group_fake_recordings
def build_group(name, compute, gated, people, load, base, out_dir):
    """Returns a DataFrame on base's index: raw, _z, _z_past4, _z_next4 columns."""
    path = Path(out_dir) / f"features_plus_{name}.csv"
    if path.exists():
        G = pd.read_csv(path)
    else:
        parts = []
        for s in people:
            rec = load(s)
            ep = rec.epochs; k = ep.keep.to_numpy(bool)
            ss = ep.start_sample_wrist.to_numpy()[k]
            f = compute(rec, ss, rec.t64[ss]).reset_index(drop=True)
            if len(f) != k.sum():
                raise ValueError(f"{s}: compute gave {len(f)} rows for {k.sum()} epochs")
            f.insert(0, "epoch", np.flatnonzero(k)); f.insert(0, "subject", s)
            parts.append(f)
            del rec; gc.collect()
        G = pd.concat(parts, ignore_index=True)
    raw = [c for c in G.columns if c not in ("subject", "epoch") and not c.endswith(("_z", "_past4", "_next4"))]
    clash = [c for c in raw if c in base.columns]
    if clash:
        raise ValueError(f"columns already in the epoch table: {clash}")
    key = base[["subject", "epoch", "pass"]]
    M = key.reset_index().merge(G, on=["subject", "epoch"], how="left", validate="one_to_one", indicator=True)
    if len(G) != len(key) or (M["_merge"] != "both").any():
        raise ValueError(f"{name}: rows do not match the epoch table one to one")
    M = M.set_index("index").drop(columns="_merge"); M.index.name = None
    if not path.exists():
        if gated:
            M.loc[~M["pass"].to_numpy(bool), raw] = np.nan
        Zg = normalise_per_person(M, raw); Zg.columns = [c + "_z" for c in raw]
        M = pd.concat([M, Zg], axis=1)
        M = pd.concat([M, context_features(M, list(Zg.columns), k=4)], axis=1)
        M.drop(columns="pass").to_csv(path, index=False)
    return M.drop(columns=["subject", "epoch", "pass"])
