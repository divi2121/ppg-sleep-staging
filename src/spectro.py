"""Spectrograms of the wrist pulse and movement, following Tallal 2026 (DREAMT, STFT branch).

Raw signals come from the 64 Hz wrist file (BVP, ACC_X/Y/Z).
Both are brought to 32 Hz, turned into one spectrum per second (10 s window), and cut into
30 frames per epoch. Settings not stated in the paper are marked "our choice".
"""
import numpy as np
import pandas as pd
from scipy.signal import butter, cheby1, get_window, sosfiltfilt


# LOGIC: Tallal's pulse preparation. Low-pass at 12 Hz (4th-order Chebyshev type I,
#   zero-phase; ripple 0.5 dB is our choice, the paper does not give it), keep every
#   2nd sample (64 -> 32 Hz), band-pass 0.1 to 8 Hz (5th-order Butterworth, zero-phase).
#   Then scale with the centred 300 s window: (x - median) / IQR, and clip to +-20.
# EDGE CASES: an IQR below min_iqr (a flat stretch) gives 0 there, not a blow-up of
#   rounding noise; under 60 s of samples in the window uses what exists.
# NOT HANDLED: NaN in the raw BVP (not checked yet on DREAMT); causal use (centred window).
# CHECKED BY: tests/test_spectro.py::test_prepare_pulse_scale_and_clip,
#   test_flat_signal_is_floor
def prepare_pulse(bvp, fs=64, norm_s=300, clip=20.0, min_iqr=1e-6):
    x = np.asarray(bvp, "float64")
    x = sosfiltfilt(cheby1(4, 0.5, 12, btype="low", fs=fs, output="sos"), x)[::2]
    f2 = fs / 2
    x = sosfiltfilt(butter(5, [0.1, 8], btype="band", fs=f2, output="sos"), x)
    r = pd.Series(x).rolling(int(norm_s * f2), center=True, min_periods=int(60 * f2))
    med = r.median().to_numpy(); iqr = (r.quantile(0.75) - r.quantile(0.25)).to_numpy()
    out = np.where(iqr > min_iqr, (x - med) / np.where(iqr > min_iqr, iqr, 1.0), 0.0)
    return np.clip(out, -clip, clip).astype("float32")


# LOGIC: Tallal's movement preparation. Each axis scaled with the person's whole-night
#   median and IQR, then the magnitude sqrt(x^2 + y^2 + z^2), then every 2nd sample
#   (64 -> 32 Hz; the paper says ACC was "restored to its native 32 Hz").
# EDGE CASES: an axis with IQR below min_iqr is centred but not divided.
# NOT HANDLED: our notes say ACC really updates about 8 times a second and is held, so
#   rhythms above about 4 Hz carry no movement; we keep only rhythms below 4 Hz anyway.
# CHECKED BY: tests/test_spectro.py::test_prepare_movement_scale
def prepare_movement(acc_x, acc_y, acc_z, min_iqr=1e-6):
    axes = []
    for a in (acc_x, acc_y, acc_z):
        a = np.asarray(a, "float64")
        q1, med, q3 = np.percentile(a, [25, 50, 75])
        axes.append((a - med) / (q3 - q1 if q3 - q1 > min_iqr else 1.0))
    return np.sqrt(sum(a ** 2 for a in axes))[::2].astype("float32")


# LOGIC: one spectrum per second. Frame i uses the win_s window centred on the middle of
#   second i (samples from (i + 0.5 - win_s/2) s to (i + 0.5 + win_s/2) s), with a Blackman
#   taper, a zero-padded nfft-point transform, power |X|^2, the first n_bins rhythms
#   (0 to n_bins x fs/nfft Hz; 64 x 32/512 = 0 to 4 Hz), and the natural log of
#   power + floor. Before the first and after the last sample the signal counts as 0.
# EDGE CASES: a signal of all zeros gives exactly log(floor) everywhere.
# NOT HANDLED: centring the window is our choice; the paper says only "10 s, 9 s overlap".
# CHECKED BY: tests/test_spectro.py::test_sine_peaks, test_epoch_timing
def spectrogram(x, fs=32, win_s=10, nfft=512, n_bins=64, floor=1e-8):
    x = np.asarray(x, "float64")
    n_sec = x.size // fs
    w = int(win_s * fs); lead = w // 2 - fs // 2           # samples before the second's start
    xp = np.concatenate([np.zeros(lead), x, np.zeros(w)])
    frames = np.lib.stride_tricks.sliding_window_view(xp, w)[::fs][:n_sec]
    X = np.fft.rfft(frames * get_window("blackman", w), n=nfft, axis=1)[:, :n_bins]
    return np.log(np.abs(X) ** 2 + floor).astype("float32")


# LOGIC: cut per-second spectrograms into epochs. An epoch starting at 64 Hz sample s
#   starts at second s // fs; it takes that frame and the next epoch_s - 1. Pulse and
#   movement are stacked as 2 channels: shape (n_epochs, 2, epoch_s, n_bins).
# EDGE CASES: frames past the end of the file are filled with log(floor), like silence.
# NOT HANDLED: an epoch start between whole seconds is rounded down (at most 1 s early).
# CHECKED BY: tests/test_spectro.py::test_epoch_timing
def epoch_spectrograms(pulse_spec, move_spec, start_samples, fs=64, epoch_s=30, floor=1e-8):
    S = np.stack([pulse_spec, move_spec])                    # (2, n_sec, n_bins)
    n_sec = S.shape[1]
    out = np.full((len(start_samples), 2, epoch_s, S.shape[2]), np.log(floor), "float32")
    for i, s in enumerate(np.asarray(start_samples, int)):
        a = s // fs; b = min(a + epoch_s, n_sec)
        if b > a:
            out[i, :, : b - a] = S[:, a:b]
    return out
