"""Live demo: decode a wrist signal as it arrives over LSL.

Three modes added 2026-10-26.

    uv run python scripts/live_demo.py --person S014 --minutes 20 --speed 120   # needs DREAMT
    uv run python scripts/live_demo.py --synthetic --minutes 20 --speed 120     # needs no data
    uv run python scripts/live_demo.py --lsl MY_STREAM                          # your own stream

ASSUMPTIONS:
- --person: a DREAMT person (data/raw or DREAMT_ROOT) is decoded only by the fold model that was
  not trained or tuned on them (evaluation.model_for). The models and fold lists are in models/live/.
  The replay starts 60 s before the beat detector's start (5 min before the first scored epoch),
  not at the start of the file, to save time. The forward filter settles within seconds, so this
  should change almost nothing; the printed agreement with the saved batch run measures it.
  The comparison runs only if features_all35.csv and preds_live_5fold.csv exist in data/derived.
  It checks the streaming code, not sleep accuracy.
- --synthetic: a made-up pulse (src/synthetic.py), replayed over LSL. The stages it gives mean
  nothing, because the model never saw a made-up pulse. The demo shows streaming, timing and
  "no guess" only. It needs at least 15 minutes: each feature is scaled with the night so far
  and needs 10 epochs (5 minutes) first.
- --lsl NAME: any LSL stream called NAME with six channels (BVP, ACC_X, ACC_Y, ACC_Z, TEMP, EDA)
  at 64 Hz. Epochs follow a fixed 30 s clock from the first sample received. Decoded by fold 0's
  model. It stops after 15 s without data.
- LSL carries the six wrist columns; the decoder counts samples and ignores LSL time stamps.
- The models were trained on DREAMT (restricted data); the data itself is not in the repo.
"""
import argparse
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from config import Config                      # noqa: E402
from evaluation import load_folds, model_for   # noqa: E402
from live import STREAM_COLS, LiveDecoder, agreement, saved_batch_stages      # noqa: E402
from recording import Recording                # noqa: E402
from stream import lsl_replay                  # noqa: E402
from synthetic import synthetic_stream         # noqa: E402

DATA_ROOT = Path(os.environ.get("DREAMT_ROOT", ROOT / "data" / "raw"))
DERIVED = ROOT / "data" / "derived"
MODELS = ROOT / "models" / "live"          # shipped with the repo: five fold models and folds.json
FS = 64


# LOGIC: find an LSL stream by name and open an inlet on it.
# EDGE CASES: no stream within timeout seconds, or one that is not 6 channels at 64 Hz, ends the
#   program with a clear message (exit code 3 and 2).
# NOT HANDLED: several streams with the same name (the first is used).
# CHECKED BY: tests/test_demo_script.py (synthetic mode); by hand for --lsl
def open_inlet(name, timeout=10):
    import pylsl
    found = pylsl.resolve_byprop("name", name, timeout=timeout)
    if not found:
        print(f"LSL stream '{name}' not found")
        sys.exit(3)
    info = found[0]
    if info.channel_count() != 6 or abs(info.nominal_srate() - FS) > 1e-6:
        print(f"stream '{name}' has {info.channel_count()} channels at {info.nominal_srate():g} Hz; "
              f"the decoder needs {STREAM_COLS} at {FS} Hz")
        sys.exit(2)
    inlet = pylsl.StreamInlet(info, max_buflen=1800)
    inlet.open_stream(timeout=timeout)       # connects now, so a replay sender can start sending
    return inlet


# LOGIC: pull chunks from the inlet into the decoder and print one line per epoch decided.
#   Stops when `expected` samples have arrived (a replay) or after idle_s seconds without data.
# EDGE CASES: epochs still undecided at the end are decided by dec.finish().
# NOT HANDLED: a gap in the stream (samples are counted, not time stamps).
# CHECKED BY: tests/test_demo_script.py (synthetic mode)
def decode_inlet(inlet, dec, expected=None, idle_s=15):
    got, results, last = 0, [], time.time()

    def show(r):
        results.append(r)
        label = f"no guess ({r.stage})" if r.abstain else r.stage
        print(f"  epoch {r.epoch:4d}  {label:18s} {r.ms:6.1f} ms", flush=True)

    while expected is None or got < expected:
        chunk, _ = inlet.pull_chunk(timeout=1.0)
        if chunk:
            last = time.time(); got += len(chunk)
            for r in dec.push(chunk):
                show(r)
        elif time.time() - last > idle_s:
            break
    for r in dec.finish():
        show(r)
    return got, results


def summary(got, expected, results):
    of = f" of {expected}" if expected else ""
    if not results:
        print(f"received {got}{of} samples | decoded no epochs")
        return
    print(f"received {got}{of} samples | decoded {len(results)} epochs | "
          f"median {np.median([r.ms for r in results]):.1f} ms per epoch | "
          f"no guess on {np.mean([r.abstain for r in results]):.0%} of epochs")


def replay_thread(raw, name, speed):
    threading.Thread(target=lsl_replay, args=(raw,), kwargs={"name": name, "speed": speed}, daemon=True).start()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--person", help="a DREAMT person, for example S014 (needs the DREAMT files)")
    mode.add_argument("--synthetic", action="store_true", help="a made-up signal; needs no data")
    mode.add_argument("--lsl", metavar="NAME", help="decode an existing LSL stream called NAME")
    ap.add_argument("--minutes", type=float, default=20, help="minutes to replay (--person: of scored night)")
    ap.add_argument("--speed", type=float, default=120, help="replay speed, times real time")
    a = ap.parse_args()
    cfg = Config(causal=True)

    if a.lsl:
        dec = LiveDecoder(MODELS / "live_fold0.pt", cfg)
        print(f"decoding LSL stream '{a.lsl}' with fold 0's model, fixed 30 s clock from the first sample")
        got, results = decode_inlet(open_inlet(a.lsl), dec)
        summary(got, None, results)
        return

    if a.synthetic:
        if a.minutes < 15:
            sys.exit("--synthetic needs --minutes 15 or more: each feature is scaled with the night so far "
                     "and needs 10 epochs (5 minutes) first")
        raw = synthetic_stream(a.minutes)
        print("SYNTHETIC SIGNAL: the stages below mean nothing, the model never saw a made-up pulse. "
              "This shows streaming, timing and 'no guess' only.")
        name = "wrist_synthetic"
        dec = LiveDecoder(MODELS / "live_fold0.pt", cfg)
        replay_thread(raw, name, a.speed)
        got, results = decode_inlet(open_inlet(name), dec, expected=len(raw))
        summary(got, len(raw), results)
        return

    folds = load_folds(MODELS / "folds.json")
    fold = model_for(a.person, folds)
    rec = Recording.load(a.person, DATA_ROOT, cfg)
    ep = rec.epochs; k = ep.keep.to_numpy(bool); ss = ep.start_sample_wrist.to_numpy()[k]
    beat_start = max(0, rec.scoring["anchor"] - cfg.detect_pre_anchor_s * cfg.fs)
    r0 = max(0, beat_start - 60 * cfg.fs)
    r1 = min(len(rec.wrist), int(ss[0] + a.minutes * 60 * cfg.fs + (cfg.live_lag_s + 2) * cfg.fs))
    raw = rec.wrist[STREAM_COLS].to_numpy("float64")[r0:r1]
    dec = LiveDecoder(MODELS / f"live_fold{fold}.pt", cfg, epoch_starts=ss - r0, beat_start=beat_start - r0)
    print(f"{a.person}: decoded by fold {fold}'s model, which never saw this person. "
          f"Replaying {len(raw) / cfg.fs / 60:.1f} min at {a.speed:g}x over LSL.")
    name = f"wrist_{a.person}"
    replay_thread(raw, name, a.speed)
    got, results = decode_inlet(open_inlet(name), dec, expected=len(raw))
    summary(got, len(raw), results)
    batch = saved_batch_stages(a.person, DERIVED)
    if batch is None:
        print("no saved batch predictions found, so no comparison (timing only)")
    else:
        live = np.array([{"W": 0, "NREM": 1, "REM": 2}[r.stage] for r in results])
        share, n = agreement(live, batch)
        print(f"same stage as the saved batch-causal run: {share:.1%} of {n} epochs "
              "(this checks the streaming code, it is not sleep accuracy)")


if __name__ == "__main__":
    main()
