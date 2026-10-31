"""The streaming sleep-stage decoder (live plan Task 6).

Design: docs/live_decoding_design.md.

ASSUMPTIONS, all stated here once:
- Input arrives as chunks of any length, columns STREAM_COLS, at Config.fs (64 Hz), from
  the first sample of the recording (so the forward filter settles exactly as in the batch).
- The decoder is told where the epochs start (epoch_starts, sample indices). In a replay
  these are the scorer's epochs. With epoch_starts=None they are a fixed 30 s clock from the
  first sample, which is what a real device would use.
- Beats are looked for from beat_start onwards (in a replay: 300 s before the first scored
  epoch, as offline), fed to StreamingBeats in 1 s blocks counted from beat_start.
- The stage for an epoch is decided once the beat detector has been fed live_lag_s + 1 s
  past the epoch's end, so every beat inside the epoch is confirmed. Features use beats up
  to the epoch end only. This is the decision delay you accepted ("ok for now").
- Each feature is z-scored with that night's running mean and spread so far, including
  the current epoch, starting after 10 values (RunningNormaliser(min_n=10)); this equals
  normalise_per_person(causal=True, min_periods=10) used to train the model.
- Missing z values are filled with the model's training medians, plus a missing flag,
  exactly as in training (make_sequences).
- "Abstain" means the epoch failed the quality gate. The model still gives its best
  guess, and its memory is still updated, so the night goes on.
"""
import time
from dataclasses import dataclass, field

import numpy as np

from beats import clean_intervals
from live_features import RAW_COLS, features_for_epochs
from sequence import BiGRUSequence
from stream import CausalFilter, RunningNormaliser, StreamingBeats

STREAM_COLS = ["BVP", "ACC_X", "ACC_Y", "ACC_Z", "TEMP", "EDA"]
CLASSES = ("W", "NREM", "REM")


@dataclass
class EpochResult:
    epoch: int                 # position in epoch_starts
    stage: str                 # "W", "NREM" or "REM": the model's best guess
    abstain: bool              # True when the quality gate failed: shown as "no guess"
    ms: float                  # time to compute this epoch's features and stage
    features: dict = field(default_factory=dict)   # raw RAW_COLS values and n_beats, for checks


# LOGIC: holds the forward filter, the live beat detector, a recent slice of all signals,
#   every confirmed beat, one running normaliser per feature and the model's memory. Each
#   push appends samples, feeds the beat detector in 1 s blocks, then decides every epoch
#   that is now ready: features from the recent slice (features_for_epochs, shared with
#   the batch builder), scaling, filling, one model step.
#   With epoch_starts=None (2026-10-26) epochs follow a fixed 30 s clock from the first
#   sample, and beat_start defaults to 0: for a stream with no sleep-lab scorer.
# EDGE CASES: chunk length does not matter; an epoch whose end has not arrived when the
#   stream finishes gives no output; a detector failure gives no beats for that second.
# NOT HANDLED: a gap or a change of sampling rate in the stream; several nights in one
#   decoder (make a new one per night).
# CHECKED BY: tests/test_live.py (incomplete epoch, beat at the epoch edge, equal to batch)
class LiveDecoder:
    def __init__(self, model_path, cfg, epoch_starts=None, beat_start=0, detector=None, ring_s=120):
        if not cfg.causal:
            raise ValueError("LiveDecoder needs Config(causal=True)")
        self.cfg, self.fs = cfg, cfg.fs
        self.model = BiGRUSequence.load(model_path)
        self.cols = list(self.model.extra["cols"])
        if self.cols != [c + "_z" for c in RAW_COLS]:
            raise ValueError("the model was not trained on the live feature columns")
        self.medians = np.asarray(self.model.extra["medians"], "float64")
        self.stepper = self.model.stepper()
        self.filt = CausalFilter(cfg.fs, cfg.filt_low, cfg.filt_high, cfg.filt_order)
        self.beats = StreamingBeats(cfg.fs, cfg.live_buffer_s, cfg.live_lag_s, detector)
        self.norm = {c: RunningNormaliser(min_n=10) for c in RAW_COLS}
        self.epoch_starts = None if epoch_starts is None else np.asarray(epoch_starts, int)
        self.beat_start = int(beat_start)
        self.n_ring = int(ring_s * cfg.fs)
        self.ring = np.zeros((0, 6)); self.ring_start = 0        # recent samples, BVP already filtered
        self.n = 0                                                # samples received so far
        self.fed = self.beat_start                                # next sample to give the beat detector
        self.beat_idx, self.beat_t = [], []
        self.next_epoch = 0
        self.wait = int((cfg.live_lag_s + 1) * cfg.fs)

    def push(self, chunk):
        chunk = np.array(chunk, dtype="float64", ndmin=2)
        chunk[:, 0] = self.filt.process(chunk[:, 0])
        self.ring = np.vstack([self.ring, chunk])
        self.n += len(chunk)
        self._feed_beats(final=False)
        out = self._decide(final=False)
        # trim by whole seconds only: hr_minutes builds a 1 s grid in the slice's frame, and
        # it must sit on the same seconds as in the batch (bug found by test_live_matches_batch_causal)
        drop = (len(self.ring) - self.n_ring) // self.fs * self.fs
        if drop > 0:
            self.ring = self.ring[drop:]; self.ring_start += drop
        return out

    def finish(self):
        """The stream has ended: feed the last part-second and decide complete epochs."""
        self._feed_beats(final=True)
        return self._decide(final=True)

    def _feed_beats(self, final):
        while self.fed < self.n and (self.fed + self.fs <= self.n or final):
            b = min(self.fed + self.fs, self.n)
            blk = self.ring[self.fed - self.ring_start: b - self.ring_start, 0]
            idx, t = self.beats.push(blk)
            self.beat_idx.append(idx + self.beat_start); self.beat_t.append(t + self.beat_start / self.fs)
            self.fed = b

    def _next_start(self):
        """Start sample of the next epoch to decide, or None when the given list is used up."""
        if self.epoch_starts is None:                 # fixed clock: epoch k covers [30k, 30k + 30) s from sample 0
            return self.next_epoch * self.fs * self.cfg.epoch_s
        if self.next_epoch < len(self.epoch_starts):
            return int(self.epoch_starts[self.next_epoch])
        return None

    def _decide(self, final):
        out = []
        n_ep = self.fs * self.cfg.epoch_s
        while (s := self._next_start()) is not None:
            ready = self.fed >= s + n_ep + self.wait or (final and self.n >= s + n_ep)
            if not ready:
                break
            out.append(self._one_epoch(s))
            self.next_epoch += 1
        return out

    def _one_epoch(self, s):
        t0 = time.perf_counter()
        fs, off = self.fs, self.ring_start
        bi = np.concatenate(self.beat_idx) if self.beat_idx else np.array([], int)
        bt = np.concatenate(self.beat_t) if self.beat_t else np.array([])
        iv = clean_intervals(bt, self.cfg.ibi_lo, self.cfg.ibi_hi, self.cfg.ibi_rel, self.cfg.ibi_win, causal=True)
        iv = iv.assign(t=iv["t"] - off / fs)                      # into the frame of the recent slice
        sig = {"bvp_filt": self.ring[:, 0], "acc_x": self.ring[:, 1], "acc_y": self.ring[:, 2],
               "acc_z": self.ring[:, 3], "temp": self.ring[:, 4], "eda": self.ring[:, 5]}
        f = features_for_epochs(sig, bi - off, iv, [s - off], self.cfg).iloc[0]
        z = []
        for c in RAW_COLS:
            v = float(f[c])
            self.norm[c].update(v)
            z.append(self.norm[c].transform(v) if np.isfinite(v) else np.nan)
        z = np.asarray(z)
        x = np.r_[np.where(np.isnan(z), self.medians, z), np.isnan(z).astype("float64")]
        k = self.stepper.step(x)
        feats = {c: float(f[c]) for c in RAW_COLS}; feats["n_beats"] = int(f["n_beats"])
        return EpochResult(epoch=self.next_epoch, stage=CLASSES[k], abstain=not bool(f["pass"]),
                           ms=1000 * (time.perf_counter() - t0), features=feats)


# LOGIC: decode one whole night in-process, as a live stream would arrive (64-sample chunks
#   from the first sample of the file), with the fold model that never saw this person.
#   Worker for a process pool (notebook 10 sections 4 and 5).
# EDGE CASES: epochs that end within live_lag_s + 1 s of the end of the file are decided
#   by finish(), with whatever beats exist.
# NOT HANDLED: real time; this runs as fast as the computer allows.
# CHECKED BY: notebook 10 section 4 (equal to the batch-causal predictions on a real night)
def run_night(person, data_root, cfg, model_path, chunk=64):
    import resource
    from recording import Recording
    rec = Recording.load(person, data_root, cfg)
    ep = rec.epochs; k = ep.keep.to_numpy(bool); ss = ep.start_sample_wrist.to_numpy()[k]
    beat_start = max(0, rec.scoring["anchor"] - cfg.detect_pre_anchor_s * cfg.fs)
    raw = rec.wrist[STREAM_COLS].to_numpy("float64")
    del rec
    dec = LiveDecoder(model_path, cfg, epoch_starts=ss, beat_start=beat_start)
    t0 = time.perf_counter(); out = []
    for i in range(0, len(raw), chunk):
        out += dec.push(raw[i:i + chunk])
    out += dec.finish()
    wall = time.perf_counter() - t0
    rows = [{"subject": person, "epoch_pos": r.epoch, "stage": r.stage, "abstain": r.abstain, "ms": r.ms,
             **{f"raw_{c}": v for c, v in r.features.items()}} for r in out]
    import pandas as pd
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6   # bytes on macOS
    return pd.DataFrame(rows), {"subject": person, "wall_s": wall, "seconds_of_signal": len(raw) / cfg.fs,
                                "peak_mb_worker": peak_mb}


# LOGIC: the batch-causal stages saved by notebook 10 section 3 for one person, in epoch
#   order: the rows of features_all35.csv that belong to the person are looked up in
#   preds_live_5fold.csv (column "row" is the row number of features_all35.csv).
# EDGE CASES: either file missing, the person absent, or any epoch without a saved stage
#   gives None, so a caller can skip the comparison. Added 2026-10-26 for the public demo,
#   where these derived files do not exist.
# NOT HANDLED: a features file in a different order from the one the predictions were made on.
# CHECKED BY: tests/test_live.py::test_saved_batch_stages
def saved_batch_stages(person, derived_dir):
    from pathlib import Path
    import pandas as pd
    d = Path(derived_dir)
    f, p = d / "features_all35.csv", d / "preds_live_5fold.csv"
    if not (f.exists() and p.exists()):
        return None
    subj = pd.read_csv(f, usecols=["subject"]).subject
    rows = subj.index[subj == person]
    preds = pd.read_csv(p)
    saved = preds[preds.subject == person].set_index("row").yhat
    if len(rows) == 0 or saved.empty:
        return None
    out = saved.reindex(rows).to_numpy()
    return None if np.isnan(out).any() else out.astype(int)


# LOGIC: share of epochs on which two stage sequences agree, over the shorter length (the
#   live run stops a few epochs early when the replay ends), and how many epochs that was.
# EDGE CASES: no epochs to compare gives (nan, 0).
# NOT HANDLED: a shift between the sequences; the caller aligns them from epoch 0.
# CHECKED BY: tests/test_live.py::test_agreement_known
def agreement(live, batch):
    n = min(len(live), len(batch))
    if n == 0:
        return float("nan"), 0
    return float(np.mean(np.asarray(live[:n]) == np.asarray(batch[:n]))), n
