"""Causal and streaming parts (VALIDATION_PLAN section 8b).

Logic approved in CODE_PLAN.md group I.
These hold state between chunks, which is why they are classes.
"""
import time

import numpy as np


# LOGIC: the forward-only band-pass filter, fed in chunks. Keeps the filter's
#   internal state (zi) between calls, so filtering a night chunk by chunk gives the
#   same output as filter_bvp(..., causal=True) on the whole night.
# EDGE CASES: starts from zero state, exactly like scipy's sosfilt on the whole signal.
# NOT HANDLED: NaN in a chunk propagates; the 64 Hz BVP has none.
# CHECKED BY: tests/test_stream.py::test_causal_filter_chunks_equal_whole
class CausalFilter:
    def __init__(self, fs=64, low=0.5, high=8.0, order=2):
        from scipy.signal import butter
        self.sos = butter(order, [low, high], btype="band", fs=fs, output="sos")
        self.zi = np.zeros((self.sos.shape[0], 2))

    def process(self, chunk):
        from scipy.signal import sosfilt
        y, self.zi = sosfilt(self.sos, np.asarray(chunk, "float64"), zi=self.zi)
        return y


# LOGIC: running mean and spread of everything seen so far (Welford's method). update()
#   adds a value; transform() z-scores a value with the current statistics. Used for
#   causal per-person scaling.
# EDGE CASES: transform returns NaN until min_n values have been seen, or if the
#   spread is 0.
# NOT HANDLED: one scalar stream per instance; use one per feature.
# CHECKED BY: tests/test_stream.py::test_running_normaliser_matches_expanding
class RunningNormaliser:
    def __init__(self, min_n=10):
        self.n, self.mean, self.m2, self.min_n = 0, 0.0, 0.0, min_n

    def update(self, x):
        if not np.isfinite(x):
            return
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self.m2 += d * (x - self.mean)

    def transform(self, x):
        if self.n < self.min_n:
            return float("nan")
        sd = np.sqrt(self.m2 / (self.n - 1))
        return float("nan") if sd == 0 else float((x - self.mean) / sd)


# LOGIC: collects incoming samples and releases a full epoch (fs * epoch_s samples)
#   as soon as one is complete, keeping the remainder for the next epoch.
# EDGE CASES: a chunk can complete several epochs at once; all are returned in order.
# NOT HANDLED: alignment to the scorer's grid; the caller starts the stream at the
#   scorer offset.
# CHECKED BY: tests/test_stream.py::test_stream_buffer_releases_epochs
class StreamBuffer:
    def __init__(self, fs=64, epoch_s=30):
        self.n = fs * epoch_s
        self.buf = np.empty(0)

    def push(self, chunk):
        self.buf = np.concatenate([self.buf, np.asarray(chunk, "float64")])
        out = []
        while self.buf.size >= self.n:
            out.append(self.buf[:self.n]); self.buf = self.buf[self.n:]
        return out


# LOGIC: beat detection on a live stream (live plan Task 2, 2026-10-24). Keeps the last
#   buffer_s of filtered signal. After every push it runs the detector on that buffer
#   and returns the peaks that are now at least lag_s old and later than the last one
#   returned (by more than min_gap_s), each refined to sub-sample time (refine_peaks).
#   A peak is only returned once it is old enough that later samples can no longer move
#   it, so each beat is returned once, in order.
# EDGE CASES: a detector that raises on a buffer gives no beats for that push; the
#   same beats are found again on the next push, while they are still in the buffer.
#   min_gap_s (0.25 s, below the 0.3 s shortest interval kept) stops a peak found one
#   sample apart on two pushes from being returned twice.
# NOT HANDLED: a beat that stays undetected until it has left the buffer is lost;
#   buffer_s must exceed lag_s plus the chunk length.
# CHECKED BY: tests/test_stream.py::test_streaming_beats_match_whole_signal,
#   test_streaming_beats_survive_detector_crash
class StreamingBeats:
    def __init__(self, fs=64, buffer_s=30, lag_s=5.0, detector=None, min_gap_s=0.25):
        if buffer_s <= lag_s + 1:
            raise ValueError("buffer_s must exceed lag_s plus a 1 s chunk")
        if detector is None:
            def detector(seg):
                import neurokit2 as nk
                return np.asarray(nk.ppg_findpeaks(seg, sampling_rate=fs, method="msptdfast")["PPG_Peaks"], int)
        self.fs, self.detector = fs, detector
        self.n_buf, self.lag, self.gap = int(buffer_s * fs), int(lag_s * fs), int(min_gap_s * fs)
        self.buf = np.zeros(0)
        self.n_seen, self.last = 0, -np.inf

    def push(self, chunk):
        """Returns (new beat indices since the stream started, their times in s)."""
        from beats import refine_peaks
        self.buf = np.concatenate([self.buf, np.asarray(chunk, "float64")])[-self.n_buf:]
        self.n_seen += len(chunk)
        offset = self.n_seen - self.buf.size
        try:
            rel = np.asarray(self.detector(self.buf), int)
        except Exception:
            return np.array([], int), np.array([])
        new = []
        for r in np.sort(rel):
            i = r + offset
            if i < self.n_seen - self.lag and i > self.last + self.gap:
                new.append(r); self.last = i
        new = np.asarray(new, int)
        return new + offset, refine_peaks(self.buf, new, self.fs) + offset / self.fs


# LOGIC: in-process stand-in for a live stream: yields the signal in chunks of
#   chunk_size samples. With realtime=True it sleeps so chunks arrive at the true
#   sampling rate (times speed).
# EDGE CASES: the last chunk may be shorter.
# NOT HANDLED: network transport; lsl_replay does that once pylsl is installed.
# CHECKED BY: tests/test_stream.py::test_replay_chunks_cover_signal
def replay(signal, chunk_size=64, fs=64, realtime=False, speed=1.0):
    x = np.asarray(signal)
    for i in range(0, x.size, chunk_size):
        if realtime:
            time.sleep(chunk_size / fs / speed)
        yield x[i:i + chunk_size]


# LOGIC: replay one night over LSL as if from a live wristband: open an LSL outlet named
#   `name` (type "Wristband", one channel per column of `signal`, fs Hz) and push chunks at
#   the true rate times speed. Updated 2026-10-24 (live plan Task 8): pylsl is installed,
#   and the live decoder needs all six wrist columns (BVP, ACC_X/Y/Z, TEMP, EDA), not only BVP.
# EDGE CASES: a 1-D signal is sent as one channel; sending starts only once a receiver has
#   connected, or after wait_s seconds if none does (then it sends anyway). A fixed wait lost the
#   first part of the signal at high replay speeds (found 2026-10-26 by test_demo_script.py).
# NOT HANDLED: LSL timestamps are the sender's clock; the decoder ignores them and counts
#   samples, so a dropped chunk would shift time (not seen on one machine).
# CHECKED BY: tests/test_stream.py::test_lsl_replay_roundtrip (skipped when LSL cannot resolve)
def lsl_replay(signal, fs=64, chunk_size=64, name="DREAMT_replay", speed=1.0, wait_s=30.0):
    from pylsl import StreamInfo, StreamOutlet
    x = np.asarray(signal, "float64")
    x = x[:, None] if x.ndim == 1 else x
    outlet = StreamOutlet(StreamInfo(name, "Wristband", x.shape[1], fs, "float32", name), chunk_size=chunk_size, max_buffered=1800)
    outlet.wait_for_consumers(wait_s)       # nothing is sent until a receiver has connected (or wait_s passes)
    for i in range(0, len(x), chunk_size):
        time.sleep(chunk_size / fs / speed)
        outlet.push_chunk(x[i:i + chunk_size].tolist())
    time.sleep(1.0)
