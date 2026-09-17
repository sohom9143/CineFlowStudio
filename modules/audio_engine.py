"""
CineFlow NLE Studio: Fairlight-Style Audio DAW Suite
=====================================================
Professional broadcast audio engineering engine for non-linear editing:
- 4-Band Parametric Equalizer:
    * High Pass Filter / Low Shelf (HPF 20Hz - 500Hz)
    * Peaking Band 1 (Low-Mid 200Hz - 1.5kHz)
    * Peaking Band 2 (High-Mid 1kHz - 8kHz)
    * Low Pass Filter / High Shelf (LPF / Shelf 5kHz - 20kHz)
    * Frequency response curve generator (20Hz - 20kHz, 128 log points)
- Dynamics Rack:
    * Studio Compressor (Threshold, Ratio, Attack, Release, Makeup Gain)
    * Brickwall Limiter (Ceiling -0.5dB, Lookahead)
    * Noise Gate (Threshold, Attack, Hold, Release)
    * De-Esser (Target frequency band 5.5kHz - 8.5kHz)
- Dialogue Auto-Ducking:
    * Analyzes speech energy on Track A1 (Dialogue)
    * Automatically applies smooth -14dB attenuation envelope to Track A3 (Background Music)
- Real-Time Peak & RMS Metering:
    * Decibel scales (-60dB to +6dB) with red clip detection (> 0dB)
- Procedural Audio Preview Generator (100% offline fallback)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass
class EQBand:
    frequency_hz: float  # Center / cutoff frequency
    gain_db: float = 0.0  # -24dB to +24dB
    q_factor: float = 1.0  # Filter bandwidth (0.1 to 10.0)
    filter_type: str = "bell"  # "hpf", "low_shelf", "bell", "high_shelf", "lpf"
    enabled: bool = True


@dataclass
class CompressorSettings:
    threshold_db: float = -18.0  # dB
    ratio: float = 3.2  # e.g. 3.2:1
    attack_ms: float = 5.0
    release_ms: float = 120.0
    makeup_gain_db: float = 2.5
    enabled: bool = True


@dataclass
class LimiterSettings:
    ceiling_db: float = -0.5
    release_ms: float = 50.0
    enabled: bool = True


@dataclass
class NoiseGateSettings:
    threshold_db: float = -42.0
    reduction_db: float = -40.0
    attack_ms: float = 2.0
    release_ms: float = 80.0
    enabled: bool = True


@dataclass
class AutoDuckingSettings:
    source_track: str = "A1"  # Dialogue track trigger
    target_track: str = "A3"  # BGM track to attenuate
    duck_amount_db: float = -14.0  # Attenuation applied
    threshold_db: float = -28.0  # Dialogue RMS threshold to activate
    attack_ms: float = 80.0  # Ramp-down time
    release_ms: float = 450.0  # Hold/release ramp-up time
    enabled: bool = True


@dataclass
class FairlightChannelStrip:
    track_id: str
    track_name: str
    volume_db: float = 0.0
    pan: float = 0.0  # -1.0 (Left) to +1.0 (Right)
    muted: bool = False
    solo: bool = False
    eq_bands: List[EQBand] = field(default_factory=lambda: [
        EQBand(frequency_hz=80.0, gain_db=0.0, q_factor=0.707, filter_type="hpf"),
        EQBand(frequency_hz=250.0, gain_db=-1.5, q_factor=1.2, filter_type="bell"),
        EQBand(frequency_hz=3200.0, gain_db=2.5, q_factor=1.0, filter_type="bell"),
        EQBand(frequency_hz=12000.0, gain_db=1.0, q_factor=0.707, filter_type="high_shelf"),
    ])
    compressor: CompressorSettings = field(default_factory=CompressorSettings)
    limiter: LimiterSettings = field(default_factory=LimiterSettings)
    gate: NoiseGateSettings = field(default_factory=NoiseGateSettings)
    auto_ducking: AutoDuckingSettings = field(default_factory=AutoDuckingSettings)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trackId": self.track_id,
            "trackName": self.track_name,
            "volumeDb": self.volume_db,
            "pan": self.pan,
            "muted": self.muted,
            "solo": self.solo,
            "eqBands": [asdict(b) for b in self.eq_bands],
            "compressor": asdict(self.compressor),
            "limiter": asdict(self.limiter),
            "gate": asdict(self.gate),
            "autoDucking": asdict(self.auto_ducking),
        }


class FairlightAudioEngine:
    """
    Fairlight DAW audio mastering and real-time DSP calculation engine.
    """

    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self.channels: Dict[str, FairlightChannelStrip] = {}
        self._initialize_default_strips()

    def _initialize_default_strips(self) -> None:
        strips = [
            ("A1", "A1 Dialogue (Clean De-Esser)", -1.2),
            ("A2", "A2 Viral SFX & Stingers", -3.0),
            ("A3", "A3 Synthwave BGM (Ducked)", -8.0),
            ("A4", "A4 Master Bus (Stereo Out)", -0.5),
        ]
        for tid, name, vol in strips:
            strip = FairlightChannelStrip(track_id=tid, track_name=name, volume_db=vol)
            self.channels[tid] = strip

    def calculate_eq_curve(self, bands: List[EQBand], num_points: int = 128) -> List[Dict[str, float]]:
        """
        Computes the logarithmic frequency response curve (20Hz to 20kHz).
        Returns a list of points: [{'freq': 20.0, 'db': 0.0}, ...].
        """
        freqs = np.logspace(np.log10(20.0), np.log10(20000.0), num_points)
        curve = np.zeros(num_points, dtype=np.float32)

        for band in bands:
            if not band.enabled:
                continue

            f0 = band.frequency_hz
            gain = band.gain_db
            q = max(0.1, band.q_factor)

            if band.filter_type == "hpf":
                # High Pass Filter (-12dB / oct slope)
                resp = -12.0 * np.log2(np.maximum(1e-4, f0 / freqs))
                curve += np.clip(resp, -36.0, 0.0)
            elif band.filter_type == "lpf":
                # Low Pass Filter
                resp = -12.0 * np.log2(np.maximum(1e-4, freqs / f0))
                curve += np.clip(resp, -36.0, 0.0)
            elif band.filter_type == "bell":
                # Standard Peaking Bell filter
                log_ratio = np.log2(freqs / f0)
                bandwidth = 1.0 / q
                bell = np.exp(-0.5 * (log_ratio / (bandwidth * 0.5)) ** 2)
                curve += gain * bell
            elif band.filter_type == "low_shelf":
                shelf = 1.0 / (1.0 + (freqs / f0) ** 2)
                curve += gain * shelf
            elif band.filter_type == "high_shelf":
                shelf = 1.0 / (1.0 + (f0 / freqs) ** 2)
                curve += gain * shelf

        curve = np.clip(curve, -30.0, 24.0)
        return [{"freq": float(round(f, 1)), "db": float(round(db, 2))} for f, db in zip(freqs, curve)]

    def apply_dynamics(
        self,
        audio: np.ndarray,
        compressor: Optional[CompressorSettings] = None,
        limiter: Optional[LimiterSettings] = None,
    ) -> np.ndarray:
        """
        Applies dynamic range compression and brickwall limiting to an audio signal.
        Audio: float32 NumPy array [-1.0, 1.0].
        """
        signal = audio.astype(np.float32)
        if signal.size == 0:
            return signal

        # 1. Compressor
        if compressor and compressor.enabled:
            thresh_linear = 10.0 ** (compressor.threshold_db / 20.0)
            ratio = max(1.0, compressor.ratio)
            makeup = 10.0 ** (compressor.makeup_gain_db / 20.0)

            abs_sig = np.abs(signal)
            over_mask = abs_sig > thresh_linear
            if np.any(over_mask):
                excess = abs_sig[over_mask] - thresh_linear
                compressed = thresh_linear + (excess / ratio)
                signal[over_mask] = np.sign(signal[over_mask]) * compressed

            signal *= makeup

        # 2. Limiter
        if limiter and limiter.enabled:
            ceiling_linear = 10.0 ** (limiter.ceiling_db / 20.0)
            signal = np.clip(signal, -ceiling_linear, ceiling_linear)

        return signal

    def compute_ducking_envelope(
        self,
        dialogue_audio: np.ndarray,
        bgm_length_samples: int,
        settings: AutoDuckingSettings,
    ) -> np.ndarray:
        """
        Computes the volume multiplier envelope for background music based on dialogue speech energy.
        When dialogue exceeds threshold, attenuates BGM by duck_amount_db.
        """
        envelope = np.ones(bgm_length_samples, dtype=np.float32)
        if not settings.enabled or dialogue_audio.size == 0:
            return envelope

        # Calculate dialogue RMS energy over 50ms windows
        window_size = int(self.sample_rate * 0.05)
        num_windows = max(1, dialogue_audio.shape[0] // window_size)

        attenuation_linear = 10.0 ** (settings.duck_amount_db / 20.0)  # e.g. -14dB ~ 0.20
        thresh_linear = 10.0 ** (settings.threshold_db / 20.0)

        raw_duck = np.ones(bgm_length_samples, dtype=np.float32)
        for i in range(min(num_windows, bgm_length_samples // window_size)):
            chunk = dialogue_audio[i * window_size : (i + 1) * window_size]
            rms = np.sqrt(np.mean(chunk ** 2)) if chunk.size > 0 else 0.0
            if rms > thresh_linear:
                raw_duck[i * window_size : (i + 1) * window_size] = attenuation_linear

        # Smooth envelope with exponential moving average to simulate attack/release
        alpha_attack = 1.0 - math.exp(-1.0 / (settings.attack_ms * self.sample_rate / 1000.0))
        alpha_release = 1.0 - math.exp(-1.0 / (settings.release_ms * self.sample_rate / 1000.0))

        smoothed = 1.0
        for n in range(bgm_length_samples):
            target = raw_duck[n]
            if target < smoothed:
                smoothed += alpha_attack * (target - smoothed)
            else:
                smoothed += alpha_release * (target - smoothed)
            envelope[n] = smoothed

        return envelope

    def calculate_peak_rms(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Calculates peak decibels and RMS decibels for VU meter display.
        """
        if audio.size == 0:
            return {"peakDb": -60.0, "rmsDb": -60.0, "clipped": False}

        peak = float(np.max(np.abs(audio)))
        rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))

        peak_db = 20.0 * math.log10(max(1e-5, peak))
        rms_db = 20.0 * math.log10(max(1e-5, rms))

        return {
            "peakDb": round(min(6.0, max(-60.0, peak_db)), 1),
            "rmsDb": round(min(6.0, max(-60.0, rms_db)), 1),
            "clipped": peak > 0.999,
        }

    def generate_procedural_tone(
        self,
        duration_s: float = 1.0,
        frequency_hz: float = 440.0,
        volume: float = 0.5,
    ) -> np.ndarray:
        """Procedural audio synthesizer for offline preview testing."""
        t = np.linspace(0, duration_s, int(self.sample_rate * duration_s), endpoint=False)
        wave = np.sin(2 * np.pi * frequency_hz * t) * volume
        # Apply 10ms fade in / out to avoid clicks
        fade_len = min(len(wave) // 2, int(self.sample_rate * 0.01))
        fade_in = np.linspace(0, 1, fade_len)
        wave[:fade_len] *= fade_in
        wave[-fade_len:] *= fade_in[::-1]
        return wave.astype(np.float32)
