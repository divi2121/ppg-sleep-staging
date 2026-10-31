"""One person's night: loads the files once and caches everything derived from them.

Logic approved in CODE_PLAN.md section 1.
Calls the plain functions in preprocess, beats and timing; adds no maths of its own.
"""
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

from config import Config
from preprocess import (acc_motion, filter_bvp, find_scoring_offset,
                        find_wristband_files, make_epochs)
from beats import clean_intervals, detect_beats_chunked, device_beats_from_ibi, ecg_rpeaks
from timing import (correct_epoch_times, fit_clock_model, label_offset,
                    local_lock_offset, misalignment_line)

D100_REL = "physionet.org/files/dreamt/2.2.0/data_100Hz"


# LOGIC: one person's night. load() finds the 64 Hz file (and the 100 Hz file if it
#   exists). Every derived quantity is a cached property: computed on first use from
#   the plain functions, with the settings in self.config, then kept. The subject ID
#   travels with every output so results cannot be mixed between people.
#   Label timing follows config.label_timing: "none" uses the scorer's epochs as
#   DREAMT placed them; "corrected" moves each epoch by the misalignment line (own
#   fit if ECG exists, else the pooled line passed in as pooled_line).
# EDGE CASES: has_ecg is False when no 100 Hz file exists; ECG properties then raise.
#   "corrected" without ECG and without pooled_line raises rather than guessing.
# NOT HANDLED: memory: a 100 Hz file is about 1 GB on disk; only TIMESTAMP, ECG and
#   Sleep_Stage are read.
# CHECKED BY: tests/test_structure.py::test_recording_on_synthetic_file
class Recording:
    def __init__(self, subject, path64, path100=None, config=Config(), pooled_line=None):
        self.subject = subject
        self.path64 = Path(path64)
        self.path100 = Path(path100) if path100 else None
        self.config = config
        self.pooled_line = pooled_line

    @classmethod
    def load(cls, subject, data_root, config=Config(), pooled_line=None):
        files = find_wristband_files(data_root)
        p100 = Path(data_root) / D100_REL / f"{subject}_PSG_df.csv"
        return cls(subject, files[subject], p100 if p100.exists() else None, config, pooled_line)

    def __repr__(self):
        return f"Recording({self.subject}, ecg={self.has_ecg}, label_timing={self.config.label_timing})"

    # ---- wristband (64 Hz) ----
    @cached_property
    def wrist(self):
        df = pd.read_csv(self.path64)
        df["Sleep_Stage"] = df["Sleep_Stage"].astype(str)
        return df

    @cached_property
    def t64(self):
        return self.wrist["TIMESTAMP"].to_numpy("float64")

    @cached_property
    def stage(self):
        return self.wrist["Sleep_Stage"].to_numpy(object)

    @cached_property
    def scoring(self):
        return find_scoring_offset(self.stage, self.config.fs, self.config.epoch_s)

    @cached_property
    def epochs_as_labelled(self):
        return make_epochs(self.stage, self.config.fs, self.config.epoch_s, self.scoring["offset"])

    @cached_property
    def misalignment(self):
        """(m0, m1) and whether it was estimated, for label_timing="corrected"."""
        if self.has_ecg:
            return (*misalignment_line(self.clock, self.label_off), False)
        if self.pooled_line is None:
            raise ValueError(f"{self.subject}: no ECG and no pooled_line for correction")
        return (*self.pooled_line, True)

    @cached_property
    def epochs(self):
        """Epochs used downstream. Always has start_sample_wrist (where to read the signal)."""
        ep = self.epochs_as_labelled
        if self.config.label_timing == "none":
            ep = ep.copy()
            ep["start_sample_wrist"] = ep["start_sample"]
            return ep
        if self.config.label_timing == "corrected":
            m0, m1, _ = self.misalignment
            return correct_epoch_times(ep, self.t64, m0, m1, self.config.fs, self.config.epoch_s)
        raise ValueError(f"unknown label_timing {self.config.label_timing!r}")

    @cached_property
    def bvp_filt(self):
        c = self.config
        return filter_bvp(self.wrist["BVP"].to_numpy(), c.fs, c.filt_low, c.filt_high, c.filt_order, c.causal)

    # LOGIC: beats are detected only from the scorer's anchor minus detect_pre_anchor_s
    #   (filter and context settling), in chunks (detect_beats_chunked). Beats in P are
    #   never used, and P is where the detector crashed (S080, S102).
    # EDGE CASES: failed chunk spans are kept in detector_failed and reported.
    # NOT HANDLED: nothing is filled in for failed spans.
    # CHECKED BY: tests/test_structure.py::test_recording_on_synthetic_file
    @cached_property
    def _beats(self):
        c = self.config
        start = max(0, self.scoring["anchor"] - c.detect_pre_anchor_s * c.fs)
        idx, t, failed = detect_beats_chunked(self.bvp_filt, c.fs, c.detector, start,
                                              c.detect_chunk_s, c.detect_overlap_s)
        self._failed = [(a + self.t64[0], b + self.t64[0]) for a, b in failed]
        return idx, t + self.t64[0]

    @property
    def detector_failed(self):
        """Spans (s, 64 Hz clock) where the detector failed; empty if none."""
        self._beats
        return self._failed

    @property
    def beat_idx(self):
        return self._beats[0]

    @property
    def beat_times(self):
        return self._beats[1]

    @cached_property
    def intervals(self):
        c = self.config
        return clean_intervals(self.beat_times, c.ibi_lo, c.ibi_hi, c.ibi_rel, c.ibi_win)

    @cached_property
    def device_beats(self):
        return device_beats_from_ibi(self.wrist["IBI"].to_numpy(), self.t64)

    @cached_property
    def motion(self):
        w = self.wrist
        return acc_motion(w.ACC_X, w.ACC_Y, w.ACC_Z, self.epochs["start_sample_wrist"].to_numpy(),
                          self.config.fs, self.config.epoch_s)

    # LOGIC: True for times (64 Hz clock) that fall inside a kept epoch's wrist window
    #   [t at start_sample_wrist, + epoch_s). Used to restrict beats to scored sleep.
    # EDGE CASES: times before the first epoch or after the last are False.
    # NOT HANDLED: none.
    # CHECKED BY: tests/test_structure.py::test_recording_on_synthetic_file
    def in_scored(self, times):
        ep = self.epochs[self.epochs.keep]
        starts = self.t64[ep["start_sample_wrist"].to_numpy()]
        times = np.asarray(times, "float64")
        i = np.searchsorted(starts, times, side="right") - 1
        ok = i >= 0
        out = np.zeros(times.size, bool)
        out[ok] = times[ok] < starts[i[ok]] + self.config.epoch_s
        return out

    # ---- sleep lab (100 Hz), only for people with ECG ----
    @property
    def has_ecg(self):
        return self.path100 is not None

    @cached_property
    def psg(self):
        if not self.has_ecg:
            raise ValueError(f"{self.subject} has no 100 Hz file")
        df = pd.read_csv(self.path100, usecols=["TIMESTAMP", "ECG", "Sleep_Stage"])
        df["Sleep_Stage"] = df["Sleep_Stage"].astype(str)
        return df

    @cached_property
    def t100(self):
        return self.psg["TIMESTAMP"].to_numpy("float64")

    @cached_property
    def ecg_beats100(self):
        _, t = ecg_rpeaks(self.psg["ECG"].to_numpy(), fs=100)
        return t + self.t100[0]

    @cached_property
    def label_off(self):
        off, share = label_offset(self.psg["Sleep_Stage"].to_numpy(object), self.t100, self.stage, self.t64)
        if share < 0.9:
            raise ValueError(f"{self.subject}: label offset not convincing (share {share:.2f})")
        return off

    @cached_property
    def drift_windows(self):
        """Lock offset in 12 five-minute windows across the ECG recording."""
        offs = np.arange(self.label_off - 30, self.label_off + 90, 0.01)
        rows = []
        for c in np.linspace(self.t100[0] + 600, self.t100[-1] - 900, 12):
            rows.append({"t100": c, **local_lock_offset(self.ecg_beats100, self.beat_times, offs, c)})
        return pd.DataFrame(rows)

    @cached_property
    def clock(self):
        w = self.drift_windows
        return fit_clock_model(w["t100"], w["offset_s"])

    @cached_property
    def ecg_beats(self):
        """R-peaks on the 64 Hz clock."""
        return self.clock.to64(self.ecg_beats100)
