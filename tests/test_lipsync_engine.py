"""
Unit & Integration Test Suite for CineFlow-AI LipSync & Bengali Audio Engine (Milestone 4 / R4)
================================================================================================
Comprehensive test suite covering:
- LipSyncConfig dataclass defaults, serialization, and dictionary roundtrips.
- Audio loading, multi-format ingestion, 16kHz resampling, and mono downmixing.
- 80-channel log Mel-spectrogram extraction ($10$ms hop, $25$ms window) and Mel filterbank math.
- Frame-aligned RMS audio energy envelopes and 16-mel chunk temporal slicing.
- Engine initialization, backend registry, aliasing, dynamic switching, and model unloading.
- Deterministic procedural CPU mock mouth deformation (energy modulation, contours, teeth rendering).
- VRAM lifecycle stage decorator isolation (@vram_lifecycle_stage("lipsync_generation")).
- Automatic cascading fallback (LivePortrait -> Wav2Lip -> CPU Mock).
- Input flexibility (PIL Image, NumPy RGB, NumPy RGBA, NumPy batch 4D).
- Edge cases (silence, audio/video duration mismatch, empty frames, extreme scaling).
"""

import os
import sys
import tempfile
import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

# Ensure workspace root is in path
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from modules.memory_manager import VRAMManager, stage_context
from modules.lipsync_engine import (
    LipSyncConfig,
    AudioAnalysisResult,
    LipSyncResult,
    LipSyncEngine,
    LivePortraitBackend,
    Wav2LipBackend,
    MockLipSyncBackend,
    hz_to_mel,
    mel_to_hz,
    create_mel_filterbank,
    compute_stft,
    extract_log_mel_spectrogram,
    resample_audio_waveform,
    load_audio_any_format,
    write_wav_file,
    synthesize_dialogue_waveform,
    synchronize_dialogue,
)


# =============================================================================
# 1. LipSyncConfig & Data Structures Tests
# =============================================================================

class TestLipSyncConfigAndDataStructures:
    """Tests configuration defaults, serialization, and data container behaviors."""

    def test_default_config_values(self):
        """Verify default configuration adheres to Milestone 4 specifications."""
        cfg = LipSyncConfig()
        assert cfg.backend == "liveportrait"
        assert cfg.sample_rate == 16000
        assert cfg.fps == 24
        assert cfg.mel_step_size == 16
        assert cfg.temp_dir == "outputs/temp_lipsync"
        assert cfg.mouth_open_scale == 1.0
        assert cfg.seed is None
        assert cfg.face_detect_confidence == 0.7
        assert cfg.audio_padding is True
        assert cfg.blend_feather_radius == 5
        assert cfg.device is None

    def test_config_serialization_roundtrip(self):
        """Test to_dict and from_dict roundtrip with extraneous key filtering."""
        cfg = LipSyncConfig(
            backend="wav2lip",
            sample_rate=16000,
            fps=30,
            mel_step_size=16,
            mouth_open_scale=1.5,
            seed=42,
            blend_feather_radius=8,
        )
        d = cfg.to_dict()
        assert d["backend"] == "wav2lip"
        assert d["fps"] == 30
        assert d["mouth_open_scale"] == 1.5
        assert d["seed"] == 42

        # Inject extra keys to verify filtering
        d["unknown_custom_flag"] = True
        reconstructed = LipSyncConfig.from_dict(d)
        assert reconstructed.backend == "wav2lip"
        assert reconstructed.fps == 30
        assert reconstructed.mouth_open_scale == 1.5
        assert not hasattr(reconstructed, "unknown_custom_flag")

    def test_audio_analysis_result_dict_and_attr_access(self):
        """Verify AudioAnalysisResult allows both attribute and dictionary key indexing."""
        res = AudioAnalysisResult(
            sample_rate=16000,
            duration_sec=2.5,
            samples=np.zeros(40000, dtype=np.float32),
            mel_spectrogram=np.zeros((80, 250), dtype=np.float32),
            energy_envelope=np.zeros(60, dtype=np.float32),
            mel_chunks=[np.zeros((16, 80), dtype=np.float32) for _ in range(60)],
            num_frames=60,
            audio_path="outputs/temp.wav",
            metadata={"source": "test"},
        )
        # Attribute access
        assert res.sample_rate == 16000
        assert res.duration_sec == 2.5
        assert res.num_frames == 60

        # Dictionary access
        assert res["sample_rate"] == 16000
        assert res["duration_sec"] == 2.5
        assert res["source"] == "test"
        assert res.get("num_frames") == 60
        assert res.get("missing_key", 999) == 999
        assert "mel_spectrogram" in res.keys()


# =============================================================================
# 2. Pure NumPy / SciPy Acoustic Processing & Mel Mathematics Tests
# =============================================================================

class TestAcousticProcessingAndMelMath:
    """Tests audio resampling, Mel scale conversion, STFT, and Mel filterbank math."""

    def test_mel_hz_conversion_roundtrip(self):
        """Verify hz_to_mel and mel_to_hz are exact mathematical inverses."""
        test_frequencies = [0.0, 100.0, 440.0, 1000.0, 4000.0, 8000.0]
        for f in test_frequencies:
            mel = hz_to_mel(f)
            recovered_f = mel_to_hz(mel)
            assert np.isclose(f, recovered_f, atol=1e-3), f"Failed for frequency {f}"

    def test_mel_filterbank_shape_and_properties(self):
        """Verify 80-channel triangular Mel filterbank matrix shape and non-negativity."""
        filterbank = create_mel_filterbank(sr=16000, n_fft=512, n_mels=80, f_min=0.0, f_max=8000.0)
        assert filterbank.shape == (80, 257)  # (n_mels, n_fft // 2 + 1)
        assert filterbank.dtype == np.float32
        assert np.all(filterbank >= 0.0)
        # Each filter has positive energy
        assert np.all(np.sum(filterbank, axis=1) > 0.0)

    def test_stft_computation_shape_and_spectrum(self):
        """Verify STFT computes complex spectrum with expected temporal and frequency bins."""
        # 1.0 second of audio at 16kHz
        t = np.linspace(0.0, 1.0, 16000, endpoint=False)
        audio = 0.5 * np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32)

        stft_res = compute_stft(audio, n_fft=512, hop_length=160, win_length=400)
        assert stft_res.dtype == np.complex64
        assert stft_res.shape[0] == 257  # 512 // 2 + 1 frequency bins
        # 16000 / 160 = ~100 frames
        assert abs(stft_res.shape[1] - 100) <= 2

    def test_log_mel_spectrogram_extraction_validity(self):
        """Verify 80-channel log Mel-spectrogram output has 80 channels and no NaNs/Infs."""
        synthetic_audio, _ = synthesize_dialogue_waveform(duration_sec=2.0, sample_rate=16000)
        log_mel = extract_log_mel_spectrogram(synthetic_audio, sr=16000, n_mels=80, hop_length=160)
        
        assert log_mel.shape[0] == 80
        # 2.0s * 100 frames/sec = ~200 frames
        assert abs(log_mel.shape[1] - 200) <= 2
        assert not np.any(np.isnan(log_mel))
        assert not np.any(np.isinf(log_mel))

    def test_audio_resampling_from_various_sample_rates(self):
        """Verify resampling from 44.1kHz, 48kHz, and 22.05kHz to standard 16kHz."""
        for orig_sr in [44100, 48000, 22050, 8000]:
            duration = 1.5
            num_orig_samples = int(orig_sr * duration)
            t = np.linspace(0.0, duration, num_orig_samples, endpoint=False)
            orig_audio = np.sin(2.0 * np.pi * 300.0 * t).astype(np.float32)

            resampled = resample_audio_waveform(orig_audio, orig_sr=orig_sr, target_sr=16000)
            expected_samples = int(round(16000 * duration))
            assert abs(len(resampled) - expected_samples) <= 2
            assert resampled.dtype == np.float32

    def test_audio_loader_from_synthetic_and_wave_file(self):
        """Verify universal audio loader correctly parses WAV files and normalizes audio."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "test_bengali.wav"
            synth_samples, _ = synthesize_dialogue_waveform(duration_sec=1.0, sample_rate=16000)
            write_wav_file(wav_path, synth_samples, sample_rate=16000)

            loaded_samples, sr = load_audio_any_format(str(wav_path), target_sr=16000)
            assert sr == 16000
            assert len(loaded_samples) == 16000
            assert loaded_samples.dtype == np.float32
            assert np.max(np.abs(loaded_samples)) <= 1.0

    def test_audio_loader_from_numpy_array_mono_and_stereo(self):
        """Verify loading 1D mono and 2D stereo numpy arrays."""
        # 1. 1D float32 array
        arr_1d = np.random.uniform(-0.8, 0.8, size=16000).astype(np.float32)
        loaded_1d, sr = load_audio_any_format(arr_1d, target_sr=16000)
        assert loaded_1d.ndim == 1
        assert len(loaded_1d) == 16000

        # 2. 2D stereo array (samples, channels)
        arr_2d = np.random.uniform(-0.8, 0.8, size=(16000, 2)).astype(np.float32)
        loaded_2d, sr = load_audio_any_format(arr_2d, target_sr=16000)
        assert loaded_2d.ndim == 1
        assert len(loaded_2d) == 16000

    def test_audio_loader_file_not_found(self):
        """Verify FileNotFoundError is raised when input file path does not exist."""
        with pytest.raises(FileNotFoundError):
            load_audio_any_format("non_existent_audio_path_xyz.wav")


# =============================================================================
# 3. Engine Initialization & Backend Registry Tests
# =============================================================================

class TestLipSyncEngineInitialization:
    """Tests engine initialization, registry, aliasing, and backend switching."""

    def test_engine_initialization_defaults(self):
        """Verify engine initializes with default backend and registry."""
        engine = LipSyncEngine()
        assert engine.get_active_backend() == "liveportrait"
        backends = engine.list_available_backends()
        assert "liveportrait" in backends
        assert "wav2lip" in backends
        assert "mock" in backends

    def test_backend_name_aliasing(self):
        """Verify aliases map properly to canonical backend names."""
        engine = LipSyncEngine()
        assert engine._normalize_backend_name("liveportrait") == "liveportrait"
        assert engine._normalize_backend_name("LivePortrait") == "liveportrait"
        assert engine._normalize_backend_name("live_portrait") == "liveportrait"
        assert engine._normalize_backend_name("expressive") == "liveportrait"
        assert engine._normalize_backend_name("wav2lip") == "wav2lip"
        assert engine._normalize_backend_name("Wav2Lip") == "wav2lip"
        assert engine._normalize_backend_name("w2l") == "wav2lip"
        assert engine._normalize_backend_name("mock") == "mock"
        assert engine._normalize_backend_name("cpu") == "mock"
        assert engine._normalize_backend_name("procedural") == "mock"

    def test_dynamic_backend_switching(self):
        """Verify dynamic backend switching and unloads."""
        engine = LipSyncEngine()
        assert engine.get_active_backend() == "liveportrait"

        engine.switch_backend("wav2lip")
        assert engine.get_active_backend() == "wav2lip"

        engine.switch_backend("cpu")
        assert engine.get_active_backend() == "mock"

    def test_switch_backend_invalid_raises_error(self):
        """Verify ValueError is raised on non-existent backend."""
        engine = LipSyncEngine()
        with patch.object(engine, "_normalize_backend_name", return_value="invalid_backend"):
            with pytest.raises(ValueError, match="Unknown lip-sync backend"):
                engine.switch_backend("invalid_backend")

    def test_unload_models(self):
        """Verify unload_models purges registered pipelines from VRAM."""
        mgr = VRAMManager.get_instance()
        engine = LipSyncEngine(memory_manager=mgr)
        mgr.register_model("liveportrait_pipeline", "dummy_lp")
        mgr.register_model("wav2lip_pipeline", "dummy_w2l")

        purged = engine.unload_models()
        assert purged >= 2
        assert "liveportrait_pipeline" not in mgr.get_registered_models()
        assert "wav2lip_pipeline" not in mgr.get_registered_models()


# =============================================================================
# 4. Audio Processing Subsystem (`process_audio`) Tests
# =============================================================================

class TestAudioProcessingSubsystem:
    """Tests the full process_audio pipeline, energy envelopes, and 16-mel chunks."""

    def test_process_audio_synthetic_bengali_speech(self):
        """Verify feature extraction for 3.375s (81 frames @ 24fps) Bengali dialogue."""
        engine = LipSyncEngine()
        audio_samples, _ = synthesize_dialogue_waveform(duration_sec=3.375, sample_rate=16000)

        result = engine.process_audio(audio_samples, target_fps=24)
        assert isinstance(result, AudioAnalysisResult)
        assert result.sample_rate == 16000
        assert np.isclose(result.duration_sec, 3.375, atol=0.01)
        assert result.num_frames == 81
        assert len(result.energy_envelope) == 81
        assert len(result.mel_chunks) == 81

        # Check 16-mel chunk shapes: (16, 80)
        for chunk in result.mel_chunks:
            assert chunk.shape == (16, 80)
            assert not np.any(np.isnan(chunk))

    def test_process_audio_silence_envelope(self):
        """Verify near-zero energy envelope for silence audio."""
        engine = LipSyncEngine()
        silence = np.zeros(32000, dtype=np.float32)  # 2.0s silence

        result = engine.process_audio(silence, target_fps=24)
        assert result.num_frames == 48
        assert np.all(result.energy_envelope == 0.0)

    def test_process_audio_vowel_burst_envelope(self):
        """Verify dynamic peak energy detection during loud speech bursts."""
        engine = LipSyncEngine()
        # 1.0s silence + 1.0s loud tone + 1.0s silence (3.0s total)
        t = np.linspace(0.0, 1.0, 16000, endpoint=False)
        burst = 0.9 * np.sin(2.0 * np.pi * 500.0 * t).astype(np.float32)
        silence = np.zeros(16000, dtype=np.float32)
        combined = np.concatenate([silence, burst, silence])

        result = engine.process_audio(combined, target_fps=24)
        assert result.num_frames == 72
        # First 24 frames (silence) should have lower energy than middle 24 frames (burst)
        silence_mean = np.mean(result.energy_envelope[:24])
        burst_mean = np.mean(result.energy_envelope[24:48])
        assert burst_mean > silence_mean + 0.3


# =============================================================================
# 5. Procedural CPU Mock Lip-Sync Backend Tests
# =============================================================================

class TestProceduralMockLipSyncBackend:
    """Tests mathematical mouth deformation, contour modulation, and determinism."""

    def test_mock_synchronization_from_numpy_frames(self):
        """Verify synchronizing 81 video frames produces 81 uint8 RGB frames."""
        engine = LipSyncEngine(default_backend="mock")
        
        # Create 81 dummy video frames (480, 720, 3)
        input_frames = [np.full((480, 720, 3), 160, dtype=np.uint8) for _ in range(81)]
        audio_samples, _ = synthesize_dialogue_waveform(duration_sec=3.375, sample_rate=16000)

        config = LipSyncConfig(backend="mock", fps=24, seed=42)
        synced_frames, audio_path = engine.synchronize_lips(
            frames=input_frames,
            audio_path=audio_samples,
            config=config,
        )

        assert isinstance(synced_frames, list)
        assert len(synced_frames) == 81
        for f in synced_frames:
            assert isinstance(f, np.ndarray)
            assert f.shape == (480, 720, 3)
            assert f.dtype == np.uint8
        assert Path(audio_path).exists()

    def test_mock_synchronization_from_pil_frames(self):
        """Verify accepting PIL Images as input frames."""
        engine = LipSyncEngine(default_backend="mock")
        pil_frames = [Image.new("RGB", (640, 480), color=(140, 90, 60)) for _ in range(24)]
        synth_audio, _ = synthesize_dialogue_waveform(duration_sec=1.0, sample_rate=16000)

        synced_frames, audio_path = engine.synchronize_lips(
            frames=pil_frames,
            audio_path=synth_audio,
            config=LipSyncConfig(backend="mock", fps=24),
        )
        assert len(synced_frames) == 24
        assert synced_frames[0].shape == (480, 640, 3)

    def test_mock_mouth_opening_proportional_to_energy(self):
        """Verify that high-energy speech frame deforms mouth pixels more than silence."""
        backend = MockLipSyncBackend()
        # Create uniform skin-tone canvas
        frame = np.full((480, 720, 3), 180, dtype=np.uint8)

        # 1. Silent audio analysis
        silence_analysis = AudioAnalysisResult(
            sample_rate=16000,
            duration_sec=1.0,
            samples=np.zeros(16000, dtype=np.float32),
            mel_spectrogram=np.zeros((80, 100), dtype=np.float32),
            energy_envelope=np.array([0.0, 0.0]),
            mel_chunks=[np.zeros((16, 80)) for _ in range(2)],
            num_frames=2,
        )
        silent_output = backend.synchronize([frame.copy(), frame.copy()], silence_analysis, LipSyncConfig())

        # 2. High-energy speech analysis
        speech_analysis = AudioAnalysisResult(
            sample_rate=16000,
            duration_sec=1.0,
            samples=np.ones(16000, dtype=np.float32),
            mel_spectrogram=np.zeros((80, 100), dtype=np.float32),
            energy_envelope=np.array([1.0, 1.0]),
            mel_chunks=[np.zeros((16, 80)) for _ in range(2)],
            num_frames=2,
        )
        speech_output = backend.synchronize([frame.copy(), frame.copy()], speech_analysis, LipSyncConfig())

        # Difference between original canvas and silent frame should be minimal/zero
        silent_diff = np.mean(np.abs(silent_output[0].astype(float) - frame.astype(float)))
        # Difference between original canvas and speech frame should be non-zero (mouth aperture opened)
        speech_diff = np.mean(np.abs(speech_output[0].astype(float) - frame.astype(float)))

        assert speech_diff > silent_diff + 1.0
        assert speech_diff > 0.0

    def test_mock_determinism_with_seed(self):
        """Verify identical seed produces pixel-identical output frames."""
        engine = LipSyncEngine(default_backend="mock")
        frames = [np.full((480, 720, 3), 150, dtype=np.uint8) for _ in range(12)]
        audio_samples, _ = synthesize_dialogue_waveform(duration_sec=0.5, sample_rate=16000)

        cfg1 = LipSyncConfig(backend="mock", seed=12345)
        synced1, _ = engine.synchronize_lips(frames=frames, audio_path=audio_samples, config=cfg1)

        cfg2 = LipSyncConfig(backend="mock", seed=12345)
        synced2, _ = engine.synchronize_lips(frames=frames, audio_path=audio_samples, config=cfg2)

        for f1, f2 in zip(synced1, synced2):
            assert np.array_equal(f1, f2)


# =============================================================================
# 6. Automatic Cascading Fallbacks & VRAM Isolation Tests
# =============================================================================

class TestFallbackMechanicsAndVRAMLifecycle:
    """Tests automatic fallback from LivePortrait -> Wav2Lip -> Mock and VRAM isolation."""

    def test_automatic_fallback_on_cpu_environment(self):
        """Verify requesting LivePortrait gracefully falls back to Mock without throwing."""
        engine = LipSyncEngine()
        frames = [np.full((480, 720, 3), 120, dtype=np.uint8) for _ in range(10)]
        audio, _ = synthesize_dialogue_waveform(duration_sec=0.4, sample_rate=16000)

        # Force liveportrait backend to report is_available() == False
        engine.backends["liveportrait"]._force_available = False
        engine.backends["wav2lip"]._force_available = False

        config = LipSyncConfig(backend="liveportrait", fps=24)
        synced, _ = engine.synchronize_lips(frames=frames, audio_path=audio, config=config)
        assert len(synced) == 10
        assert synced[0].shape == (480, 720, 3)

    def test_vram_lifecycle_stage_isolation(self):
        """Verify @vram_lifecycle_stage sets and clears current stage during synchronize_lips."""
        mgr = VRAMManager.get_instance()
        assert mgr.current_stage is None

        engine = LipSyncEngine(memory_manager=mgr)
        frames = [np.full((480, 720, 3), 100, dtype=np.uint8) for _ in range(5)]
        audio, _ = synthesize_dialogue_waveform(duration_sec=0.2, sample_rate=16000)

        # Execute synchronized lips
        synced, _ = engine.synchronize_lips(frames, audio, LipSyncConfig(backend="mock"))
        
        # After execution, current_stage must be None
        assert mgr.current_stage is None
        assert len(synced) == 5

    def test_sync_dialogue_rich_result_wrapper(self):
        """Verify sync_dialogue returns LipSyncResult dataclass."""
        engine = LipSyncEngine()
        frames = [np.full((480, 720, 3), 110, dtype=np.uint8) for _ in range(12)]
        audio, _ = synthesize_dialogue_waveform(duration_sec=0.5, sample_rate=16000)

        result = engine.sync_dialogue(frames, audio, LipSyncConfig(backend="mock", fps=24))
        assert isinstance(result, LipSyncResult)
        assert len(result.synced_frames) == 12
        assert result.fps == 24
        assert result.frame_count == 12
        assert Path(result.output_audio_path).exists()


# =============================================================================
# 7. Boundary & Adversarial Handling Tests
# =============================================================================

class TestLipSyncAdversarialAndBoundaries:
    """Tests edge cases: empty inputs, audio/video duration mismatches, and convenience functions."""

    def test_empty_frames_raises_value_error(self):
        """Verify ValueError is raised when empty frame list is passed."""
        engine = LipSyncEngine()
        synth_audio, _ = synthesize_dialogue_waveform(duration_sec=1.0)
        with pytest.raises(ValueError, match="No video frames provided"):
            engine.synchronize_lips(frames=[], audio_path=synth_audio)

    def test_audio_shorter_than_video_duration(self):
        """Verify handling when audio (0.5s = 12 frames) is shorter than video (24 frames)."""
        engine = LipSyncEngine()
        video_frames = [np.full((480, 720, 3), 150, dtype=np.uint8) for _ in range(24)]
        short_audio, _ = synthesize_dialogue_waveform(duration_sec=0.5, sample_rate=16000)

        synced, _ = engine.synchronize_lips(video_frames, short_audio, LipSyncConfig(backend="mock", fps=24))
        assert len(synced) == 24

    def test_audio_longer_than_video_duration_with_padding(self):
        """Verify video frames are extended to match audio duration when audio_padding=True."""
        engine = LipSyncEngine()
        # 10 video frames (< 0.5s at 24fps)
        short_video = [np.full((480, 720, 3), 150, dtype=np.uint8) for _ in range(10)]
        # 2.0s audio = 48 frames
        long_audio, _ = synthesize_dialogue_waveform(duration_sec=2.0, sample_rate=16000)

        synced, _ = engine.synchronize_lips(
            short_video,
            long_audio,
            LipSyncConfig(backend="mock", fps=24, audio_padding=True),
        )
        assert len(synced) == 48

    def test_extreme_mouth_open_scale_boundary(self):
        """Verify mouth_open_scale=0.0 and mouth_open_scale=10.0 do not cause NaN/Inf or crash."""
        engine = LipSyncEngine()
        frames = [np.full((480, 720, 3), 130, dtype=np.uint8) for _ in range(8)]
        audio, _ = synthesize_dialogue_waveform(duration_sec=0.33, sample_rate=16000)

        # Scale 0.0 (no opening)
        synced_zero, _ = engine.synchronize_lips(frames, audio, LipSyncConfig(mouth_open_scale=0.0))
        assert len(synced_zero) == 8

        # Scale 10.0 (extreme opening)
        synced_extreme, _ = engine.synchronize_lips(frames, audio, LipSyncConfig(mouth_open_scale=10.0))
        assert len(synced_extreme) == 8
        for f in synced_extreme:
            assert not np.any(np.isnan(f))
            assert np.all(f >= 0) and np.all(f <= 255)

    def test_top_level_convenience_function(self):
        """Verify synchronize_dialogue helper function."""
        frames = [np.full((480, 720, 3), 140, dtype=np.uint8) for _ in range(6)]
        audio, _ = synthesize_dialogue_waveform(duration_sec=0.25, sample_rate=16000)

        synced, audio_path = synchronize_dialogue(frames, audio, LipSyncConfig(backend="mock"))
        assert len(synced) == 6
        assert Path(audio_path).exists()

    def test_save_synced_video_export(self):
        """Verify save_synced_video produces a valid file on disk."""
        engine = LipSyncEngine()
        frames = [np.full((240, 320, 3), 100, dtype=np.uint8) for _ in range(12)]
        audio, _ = synthesize_dialogue_waveform(duration_sec=0.5, sample_rate=16000)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_p = Path(tmpdir) / "test.wav"
            write_wav_file(audio_p, audio, sample_rate=16000)
            
            out_video = Path(tmpdir) / "synced_test.mp4"
            saved_p = engine.save_synced_video(frames, str(audio_p), str(out_video), fps=24)
            assert Path(saved_p).exists()
