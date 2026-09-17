"""
Tests for CineFlow NLE Studio: Fairlight-Style Audio DAW Suite.
Verifies parametric EQ curve calculation, compressor/limiter dynamics, dialogue auto-ducking, and VU metering.
"""

import numpy as np
import pytest
from modules.audio_engine import (
    FairlightAudioEngine,
    EQBand,
    CompressorSettings,
    LimiterSettings,
    AutoDuckingSettings,
)


def test_fairlight_initialization():
    engine = FairlightAudioEngine()
    assert "A1" in engine.channels
    assert "A2" in engine.channels
    assert "A3" in engine.channels
    assert "A4" in engine.channels


def test_parametric_eq_curve():
    engine = FairlightAudioEngine()
    bands = [
        EQBand(frequency_hz=80.0, gain_db=0.0, q_factor=0.707, filter_type="hpf"),
        EQBand(frequency_hz=1000.0, gain_db=6.0, q_factor=1.5, filter_type="bell"),
        EQBand(frequency_hz=12000.0, gain_db=3.0, q_factor=0.707, filter_type="high_shelf"),
    ]
    curve = engine.calculate_eq_curve(bands, num_points=64)
    assert len(curve) == 64
    assert curve[0]["freq"] < 50.0
    assert curve[-1]["freq"] > 15000.0

    # Near 1000Hz, curve should show a positive boost
    boost_point = min(curve, key=lambda pt: abs(pt["freq"] - 1000.0))
    assert boost_point["db"] > 3.0


def test_compressor_and_limiter():
    engine = FairlightAudioEngine()
    # High amplitude signal (exceeding threshold)
    t = np.linspace(0, 1.0, 4800, endpoint=False)
    loud_signal = np.sin(2 * np.pi * 440 * t) * 0.9

    compressor = CompressorSettings(threshold_db=-12.0, ratio=4.0, makeup_gain_db=0.0)
    limiter = LimiterSettings(ceiling_db=-1.0)

    processed = engine.apply_dynamics(loud_signal, compressor=compressor, limiter=limiter)
    assert processed.shape == loud_signal.shape
    # Maximum amplitude should not exceed limiter ceiling (-1.0dB ~ 0.891)
    assert np.max(np.abs(processed)) <= 0.90


def test_dialogue_auto_ducking():
    engine = FairlightAudioEngine(sample_rate=48000)
    n_samples = 48000  # 1 second
    # Dialogue active in first half second, silent in second half
    dialogue = np.zeros(n_samples, dtype=np.float32)
    dialogue[:24000] = 0.5  # loud speech

    settings = AutoDuckingSettings(
        duck_amount_db=-14.0,
        threshold_db=-30.0,
        attack_ms=20.0,
        release_ms=100.0,
    )
    envelope = engine.compute_ducking_envelope(dialogue, n_samples, settings)

    # During dialogue, envelope should attenuate (< 0.50)
    assert envelope[10000] < 0.40
    # After speech ends, envelope should ramp back up towards 1.0
    assert envelope[-1] > envelope[10000]


def test_vu_metering():
    engine = FairlightAudioEngine()
    silent = np.zeros(1000, dtype=np.float32)
    stats_silent = engine.calculate_peak_rms(silent)
    assert stats_silent["peakDb"] <= -50.0
    assert stats_silent["clipped"] is False

    loud = np.full(1000, 1.05, dtype=np.float32)
    stats_loud = engine.calculate_peak_rms(loud)
    assert stats_loud["clipped"] is True
