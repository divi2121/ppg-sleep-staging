"""Tests for stream.py."""
import numpy as np
import pandas as pd

from preprocess import filter_bvp
from stream import CausalFilter, RunningNormaliser, StreamBuffer, StreamingBeats, replay


def test_causal_filter_chunks_equal_whole():
    x = np.random.default_rng(0).normal(size=64 * 120)
    f = CausalFilter()
    y = np.concatenate([f.process(c) for c in replay(x, chunk_size=97)])
    assert np.allclose(y, filter_bvp(x, causal=True))


def test_running_normaliser_matches_expanding():
    v = np.random.default_rng(1).normal(size=50)
    rn = RunningNormaliser(min_n=5)
    out = []
    for x in v:
        rn.update(x); out.append(rn.transform(x))
    s = pd.Series(v)
    exp = ((s - s.expanding(5).mean()) / s.expanding(5).std()).to_numpy()
    assert np.allclose(np.array(out)[4:], exp[4:]) and np.isnan(out[0])


def test_stream_buffer_releases_epochs():
    b = StreamBuffer(fs=64, epoch_s=30)
    assert b.push(np.zeros(1000)) == []
    ep = b.push(np.arange(4000.0))
    assert len(ep) == 2 and ep[0].size == 1920 and b.buf.size == 5000 - 3840


def test_replay_chunks_cover_signal():
    x = np.arange(1000)
    assert np.array_equal(np.concatenate(list(replay(x, chunk_size=64))), x)


def _local_max(seg):
    return np.flatnonzero((seg[1:-1] > seg[:-2]) & (seg[1:-1] >= seg[2:]) & (seg[1:-1] > 0.5)) + 1


def test_streaming_beats_match_whole_signal():
    # live plan Task 2 (2026-10-24): fed 1 s at a time, every beat older than lag_s is
    # returned exactly once, at the same sample as detecting on the whole signal
    fs = 64; t = np.arange(fs * 120) / fs
    x = np.sin(2 * np.pi * 1.2 * t) ** 21                        # sharp peaks at 1.2 Hz
    whole = _local_max(x)
    sb = StreamingBeats(fs=fs, buffer_s=10, lag_s=2.0, detector=_local_max)
    out = [sb.push(x[i:i + fs]) for i in range(0, x.size, fs)]
    got = np.concatenate([o[0] for o in out])
    assert np.array_equal(got, whole[whole < x.size - 2 * fs])  # all but the last lag_s, no duplicates
    times = np.concatenate([o[1] for o in out])
    assert np.allclose(times, got / fs, atol=0.5 / fs)          # refined times stay within half a sample


def test_streaming_beats_survive_detector_crash():
    calls = {"n": 0}
    def det(seg):
        calls["n"] += 1
        if calls["n"] == 3:
            raise ValueError("bad buffer")
        return np.flatnonzero(seg > 0.99)
    sb = StreamingBeats(fs=64, buffer_s=10, lag_s=2.0, detector=det)
    x = np.zeros(64 * 30); x[::64] = 1.0
    got = np.concatenate([sb.push(x[i:i + 64])[0] for i in range(0, x.size, 64)])
    # the crashed push loses nothing: later buffers still hold those spikes
    assert np.array_equal(got, np.arange(0, x.size - 2 * 64, 64))


def test_lsl_replay_roundtrip():
    # live plan Task 8 (2026-10-24): six channels go out over LSL and come back unchanged
    import threading
    import pytest
    pylsl = pytest.importorskip("pylsl")
    from stream import lsl_replay
    x = np.arange(64 * 3 * 6, dtype=float).reshape(-1, 6)       # 3 s, 6 channels, every value different
    th = threading.Thread(target=lsl_replay, args=(x,), kwargs={"name": "test_rt", "speed": 4.0}, daemon=True)
    th.start()
    found = pylsl.resolve_byprop("name", "test_rt", timeout=5)
    if not found:
        pytest.skip("LSL could not resolve a local stream on this machine")
    inlet = pylsl.StreamInlet(found[0]); got = []
    t_end = __import__("time").time() + 10
    while len(got) < len(x) and __import__("time").time() < t_end:
        chunk, _ = inlet.pull_chunk(timeout=1.0)
        got += chunk
    th.join(timeout=5)
    got = np.asarray(got)
    assert got.shape[1] == 6 and len(got) >= len(x) - 64          # the first chunk may be sent before the inlet opens
    assert np.allclose(got, x[len(x) - len(got):])
