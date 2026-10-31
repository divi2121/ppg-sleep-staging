"""Tests for spectro.py. Every expected value is worked out by hand."""
import numpy as np
import pytest

from spectro import epoch_spectrograms, prepare_movement, prepare_pulse, spectrogram

FLOOR = np.log(1e-8)


def test_sine_peaks():
    # rhythm k = k x 32 / 512 = k x 0.0625 Hz
    t = np.arange(32 * 120) / 32                                 # 2 min at 32 Hz
    for f, k in [(1.0, 16), (1.2, 19), (3.5, 56)]:               # on a rhythm, between 19 and 20, near the top
        S = spectrogram(np.sin(2 * np.pi * f * t))
        assert S.shape == (120, 64)                              # one frame per second, 64 rhythms
        assert (S[10:110].argmax(axis=1) == k).all()             # frames away from the file edges


def test_flat_signal_is_floor():
    # a constant has only a 0 Hz rhythm; the 0.1 Hz high-pass removes it
    x = prepare_pulse(np.full(64 * 600, 100.0))
    assert np.abs(x).max() == 0.0                                # flat IQR gives 0, not amplified noise
    assert (spectrogram(x) == np.float32(FLOOR)).all()


def test_prepare_pulse_scale_and_clip():
    t = np.arange(64 * 900) / 64                                 # 15 min at 64 Hz
    x = prepare_pulse(1000.0 + 50.0 * np.sin(2 * np.pi * 1.2 * t))
    mid = x[32 * 300: 32 * 600]                                  # away from filter and window edges
    # the IQR of a sine with amplitude A is A x sqrt(2), so the scaled amplitude is 1/sqrt(2)
    assert np.abs(mid).max() == pytest.approx(1 / np.sqrt(2), rel=0.02)
    assert abs(mid.mean()) < 0.01                                # offset of 1000 removed
    spike = 1000.0 + 50.0 * np.sin(2 * np.pi * 1.2 * t); spike[64 * 450] += 1e6
    assert prepare_pulse(spike).max() == pytest.approx(20.0)     # clipped


def test_prepare_movement_scale():
    n = 64 * 10
    x = np.tile([0.0, 2.0], n // 2)                              # median 1, IQR 2 -> values -0.5, +0.5
    m = prepare_movement(x, np.zeros(n), np.zeros(n))            # y and z: IQR 0, centred only -> 0
    assert m.size == n // 2                                      # 64 -> 32 Hz
    assert np.allclose(m, 0.5)                                   # magnitude of -0.5 or +0.5; every 2nd sample


def test_epoch_timing():
    # a 1.2 Hz sine only in seconds 90 to 119 (epoch 3 of epochs starting at 0, 30, 60, ...)
    t = np.arange(32 * 180) / 32
    x = np.where((t >= 90) & (t < 120), np.sin(2 * np.pi * 1.2 * t), 0.0)
    S = spectrogram(x)
    E = epoch_spectrograms(S, S, np.arange(6) * 30 * 64)         # epoch starts in 64 Hz samples
    assert E.shape == (6, 2, 30, 64)
    assert (E[3, 0, 5:25].argmax(axis=1) == 19).all()            # the sine is in epoch 3
    # frame i covers seconds i - 4.5 to i + 5.5, so second 84 is the last frame before the
    # sine and second 125 the first after it; those frames see only zeros
    assert (E[2, 0, :25] == np.float32(FLOOR)).all()             # seconds 60 to 84
    assert (E[2, 0, 25:, 19] > FLOOR).all()                      # seconds 85 to 89 see its start
    assert (E[4, 0, 5:] == np.float32(FLOOR)).all()              # seconds 125 to 149
    assert (E[1] == np.float32(FLOOR)).all()
    E = epoch_spectrograms(S, S, [170 * 64])                     # runs 20 s past the end
    assert (E[0, :, 10:] == np.float32(FLOOR)).all()
