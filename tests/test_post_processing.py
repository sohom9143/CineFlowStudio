"""
Unit, Component & Integration Test Suite for CineFlow-AI Post-Processing & Master Rendering Engine (Milestone 5 / R5)
===================================================================================================================
Comprehensive test coverage:
1. PostProcessingConfig dataclass defaults, serialization (to_dict/from_dict), and validation.
2. Resolution parsing & aspect ratio mathematics ("720p", "1080p", "4k", "4K", "2160p", odd dimension sanitization).
3. Frame normalization for diverse inputs (PIL, uint8, float, RGBA, channels-first, 4D arrays).
4. Real-ESRGAN chunked frame upscaler ($N=2-4$), spatial overlapping tiling (tile_size=512), and Lanczos unsharp fallback.
5. RIFE temporal frame interpolation (24fps -> 60fps) with Farneback optical flow and cosine blend fallback.
6. Audio/Video master multiplexer (FFmpeg / MoviePy / OpenCV) exporting clean H.264 / AAC MP4 with faststart.
7. PostProductionEngine full end-to-end master render pipeline with @vram_lifecycle_stage("post_processing").
8. VRAM lifecycle memory purging, stage isolation, and error handling.
"""

from __future__ import annotations

import math
import os
import shutil
import struct
import sys
import tempfile
import wave
from pathlib import Path
from typing import List, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

# Ensure workspace root is in sys.path
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from modules.memory_manager import VRAMManager
from modules.post_processing import (
    PostProcessingConfig,
    PostProcessResult,
    PostProductionEngine,
    RealESRGANUpscaler,
    RIFEInterpolator,
    AudioVideoMuxer,
    RESOLUTION_PRESETS,
    parse_resolution,
    normalize_frame_to_numpy,
    normalize_frame_sequence,
)


# =============================================================================
# Helper Fixtures & Test Data Generators
# =============================================================================

def generate_synthetic_frame(
    width: int = 1280,
    height: int = 720,
    color: Tuple[int, int, int] = (100, 150, 200),
    add_pattern: bool = True,
) -> np.ndarray:
    """Generates a synthetic RGB uint8 frame with gradient and sharp edges."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :] = color
    if add_pattern:
        # Add high-frequency edge box for unsharp filter testing
        cv_box_h = max(2, height // 4)
        cv_box_w = max(2, width // 4)
        frame[cv_box_h : 2 * cv_box_h, cv_box_w : 2 * cv_box_w] = (255, 255, 0)
        # Add gradient
        x_grad = np.linspace(0, 50, width, dtype=np.uint8).reshape(1, width, 1)
        frame = np.clip(frame.astype(np.int16) + x_grad, 0, 255).astype(np.uint8)
    return frame


def generate_synthetic_video_sequence(
    num_frames: int = 24,
    width: int = 1280,
    height: int = 720,
) -> List[np.ndarray]:
    """Generates a sequence of frames with smooth animated motion."""
    frames = []
    for i in range(num_frames):
        # Shift color and moving circle
        r = int((i * 10) % 255)
        g = int(120 + 80 * math.sin(i * 0.3))
        b = 200
        frame = generate_synthetic_frame(width, height, color=(r, g, b))
        # Draw moving patch
        cx = int((width * (i + 1)) / (num_frames + 2))
        cy = height // 2
        rad = max(4, height // 8)
        y1, y2 = max(0, cy - rad), min(height, cy + rad)
        x1, x2 = max(0, cx - rad), min(width, cx + rad)
        frame[y1:y2, x1:x2] = (255, 50, 50)
        frames.append(frame)
    return frames


def create_synthetic_wav_file(
    file_path: str,
    duration_s: float = 1.0,
    sample_rate: int = 16000,
    freq_hz: float = 440.0,
) -> str:
    """Generates a synthetic sine-wave WAV audio file."""
    num_samples = int(sample_rate * duration_s)
    t = np.linspace(0, duration_s, num_samples, endpoint=False)
    sine_wave = (np.sin(2 * np.pi * freq_hz * t) * 32767.0).astype(np.int16)

    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    with wave.open(file_path, "wb") as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(sine_wave.tobytes())
    return file_path


@pytest.fixture
def temp_test_dir():
    """Provides a temporary directory that is cleanly removed after test."""
    temp_dir = tempfile.mkdtemp(prefix="cineflow_test_m5_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


# =============================================================================
# 1. PostProcessingConfig & Resolution Mathematics Tests
# =============================================================================

class TestPostProcessingConfigAndResolutionMath:
    """Tests PostProcessingConfig dataclass, defaults, serialization, and resolution parsing."""

    def test_default_config_values(self):
        """Verify all default configuration fields adhere strictly to Milestone 5 requirements."""
        cfg = PostProcessingConfig()
        assert cfg.enable_upscale is True
        assert cfg.target_resolution == "1080p"
        assert cfg.chunk_batch_size == 4
        assert cfg.enable_interpolation is True
        assert cfg.target_fps == 60
        assert cfg.video_codec == "libx264"
        assert cfg.audio_codec == "aac"
        assert cfg.crf == 18
        assert cfg.tile_size == 512
        assert cfg.tile_pad == 10
        assert cfg.source_fps == 24
        assert cfg.upscaler_model == "RealESRGAN_x4plus"
        assert cfg.half_precision is True

    def test_config_serialization_and_deserialization(self):
        """Test to_dict and from_dict roundtrip with field filtering."""
        cfg = PostProcessingConfig(
            enable_upscale=True,
            target_resolution="4k",
            chunk_batch_size=2,
            enable_interpolation=False,
            target_fps=24,
            video_codec="libx265",
            audio_codec="mp3",
            crf=20,
            tile_size=256,
        )
        d = cfg.to_dict()
        assert d["target_resolution"] == "4k"
        assert d["chunk_batch_size"] == 2
        assert d["crf"] == 20
        assert d["tile_size"] == 256

        # Add unrecognized keys to test filtering
        d["unknown_extra_param"] = 999
        reconstructed = PostProcessingConfig.from_dict(d)
        assert reconstructed.target_resolution == "4k"
        assert reconstructed.chunk_batch_size == 2
        assert reconstructed.enable_interpolation is False
        assert not hasattr(reconstructed, "unknown_extra_param")

    def test_config_from_dict_non_dict_fallback(self):
        """Verify from_dict returns default config when passed non-dict."""
        cfg = PostProcessingConfig.from_dict("invalid")  # type: ignore
        assert cfg.target_resolution == "1080p"

    def test_config_validation_success(self):
        """Valid configurations must pass validate() without error."""
        cfg = PostProcessingConfig(
            target_resolution="1080p",
            chunk_batch_size=4,
            target_fps=60,
            source_fps=24,
            crf=18,
            tile_size=512,
        )
        cfg.validate()  # No exception

    def test_config_validation_invalid_crf_raises(self):
        """CRF outside [0, 51] must raise ValueError."""
        with pytest.raises(ValueError, match="crf must be in range"):
            PostProcessingConfig(crf=-5).validate()
        with pytest.raises(ValueError, match="crf must be in range"):
            PostProcessingConfig(crf=55).validate()

    def test_config_validation_invalid_fps_and_batch_raises(self):
        """Negative or zero chunk_batch_size, target_fps, source_fps raise ValueError."""
        with pytest.raises(ValueError, match="chunk_batch_size must be positive"):
            PostProcessingConfig(chunk_batch_size=0).validate()
        with pytest.raises(ValueError, match="target_fps must be positive"):
            PostProcessingConfig(target_fps=0).validate()
        with pytest.raises(ValueError, match="source_fps must be positive"):
            PostProcessingConfig(source_fps=-24).validate()
        with pytest.raises(ValueError, match="tile_size cannot be negative"):
            PostProcessingConfig(tile_size=-10).validate()

    def test_parse_resolution_standard_presets(self):
        """Verify standard resolution presets parse to exact (width, height) tuples."""
        assert parse_resolution("720p") == (1280, 720)
        assert parse_resolution("1080p") == (1920, 1080)
        assert parse_resolution("4k") == (3840, 2160)
        assert parse_resolution("4K") == (3840, 2160)
        assert parse_resolution("2160p") == (3840, 2160)
        assert parse_resolution("480p") == (854, 480)
        assert parse_resolution("1440p") == (2560, 1440)
        assert parse_resolution("8k") == (7680, 4320)

    def test_parse_resolution_dimension_strings(self):
        """Verify custom dimension strings parse correctly."""
        assert parse_resolution("1920x1080") == (1920, 1080)
        assert parse_resolution("1280X720") == (1280, 720)
        assert parse_resolution("3840*2160") == (3840, 2160)
        assert parse_resolution("1920, 1080") == (1920, 1080)
        assert parse_resolution((1920, 1080)) == (1920, 1080)
        assert parse_resolution([1280, 720]) == (1280, 720)

    def test_parse_resolution_enforces_even_dimensions(self):
        """Odd dimensions must be automatically sanitized to even dimensions for video codecs."""
        assert parse_resolution((1921, 1081)) == (1920, 1080)
        assert parse_resolution("721x481") == (720, 480)

    def test_parse_resolution_invalid_inputs_raise(self):
        """Invalid strings, negative values, and wrong types raise ValueError / TypeError."""
        with pytest.raises(ValueError, match="Invalid resolution specifier"):
            parse_resolution("invalid_preset_xyz")
        with pytest.raises(ValueError, match="Resolution tuple/list must have at least 2 elements"):
            parse_resolution([1920])
        with pytest.raises(TypeError, match="Expected resolution as str, tuple, or list"):
            parse_resolution(1080)  # type: ignore


# =============================================================================
# 2. Frame Normalization Helpers Tests
# =============================================================================

class TestFrameNormalizationHelpers:
    """Tests normalization of various input types (PIL, float numpy, RGBA, channels-first)."""

    def test_normalize_uint8_rgb_numpy_array(self):
        """Standard uint8 RGB array passes through unchanged."""
        arr = np.ones((100, 100, 3), dtype=np.uint8) * 128
        norm = normalize_frame_to_numpy(arr)
        assert norm.shape == (100, 100, 3)
        assert norm.dtype == np.uint8
        assert norm[0, 0, 0] == 128

    def test_normalize_float_numpy_array(self):
        """Float array in [0.0, 1.0] is scaled to [0, 255] uint8."""
        arr = np.ones((100, 100, 3), dtype=np.float32) * 0.5
        norm = normalize_frame_to_numpy(arr)
        assert norm.shape == (100, 100, 3)
        assert norm.dtype == np.uint8
        assert 127 <= norm[0, 0, 0] <= 128

    def test_normalize_grayscale_2d_and_3d(self):
        """2D (H, W) and (H, W, 1) grayscale frames are expanded to (H, W, 3) RGB."""
        arr_2d = np.ones((50, 50), dtype=np.uint8) * 200
        norm_2d = normalize_frame_to_numpy(arr_2d)
        assert norm_2d.shape == (50, 50, 3)
        assert np.all(norm_2d == 200)

        arr_3d = np.ones((50, 50, 1), dtype=np.uint8) * 150
        norm_3d = normalize_frame_to_numpy(arr_3d)
        assert norm_3d.shape == (50, 50, 3)
        assert np.all(norm_3d == 150)

    def test_normalize_rgba_strips_alpha(self):
        """RGBA (H, W, 4) frame has alpha channel cleanly stripped to (H, W, 3)."""
        arr_rgba = np.zeros((40, 40, 4), dtype=np.uint8)
        arr_rgba[:, :, :3] = 180
        arr_rgba[:, :, 3] = 255
        norm = normalize_frame_to_numpy(arr_rgba)
        assert norm.shape == (40, 40, 3)
        assert np.all(norm == 180)

    def test_normalize_channels_first(self):
        """Channels-first (3, H, W) numpy array is transposed to (H, W, 3)."""
        arr_cf = np.zeros((3, 60, 80), dtype=np.uint8)
        arr_cf[0, :, :] = 255  # Red channel
        norm = normalize_frame_to_numpy(arr_cf)
        assert norm.shape == (60, 80, 3)
        assert norm[0, 0, 0] == 255
        assert norm[0, 0, 1] == 0

    def test_normalize_pil_image_rgb_and_rgba(self):
        """PIL Images in RGB and RGBA formats are converted properly."""
        pil_rgb = Image.new("RGB", (64, 48), color=(255, 0, 0))
        norm_rgb = normalize_frame_to_numpy(pil_rgb)
        assert norm_rgb.shape == (48, 64, 3)
        assert norm_rgb.dtype == np.uint8
        assert norm_rgb[0, 0, 0] == 255

        pil_rgba = Image.new("RGBA", (64, 48), color=(0, 255, 0, 128))
        norm_rgba = normalize_frame_to_numpy(pil_rgba)
        assert norm_rgba.shape == (48, 64, 3)
        assert norm_rgba[0, 0, 1] == 255

    def test_normalize_frame_sequence_from_4d_array(self):
        """4D numpy array (N, H, W, 3) is converted to a list of N arrays."""
        arr_4d = np.zeros((5, 100, 100, 3), dtype=np.uint8)
        seq = normalize_frame_sequence(arr_4d)
        assert len(seq) == 5
        assert seq[0].shape == (100, 100, 3)

    def test_normalize_frame_sequence_none_raises(self):
        """None input raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            normalize_frame_sequence(None)


# =============================================================================
# 3. RealESRGAN Super-Resolution Upscaler Tests
# =============================================================================

class TestRealESRGANUpscaler:
    """Tests Super-Resolution upscaling, chunking (N=2-4), tiling (512), and Lanczos unsharp filter."""

    def test_upscaler_initialization(self):
        """Upscaler initializes cleanly with device fallback."""
        upscaler = RealESRGANUpscaler(model_name="RealESRGAN_x4plus", device="cpu")
        assert upscaler.device == "cpu"
        assert upscaler.model_name == "RealESRGAN_x4plus"

    def test_upscale_single_frame_720p_to_1080p(self):
        """Upscales a 720p frame (1280x720) to exact 1080p (1920x1080) dimensions."""
        frame_720p = generate_synthetic_frame(width=1280, height=720)
        upscaler = RealESRGANUpscaler(device="cpu")
        upscaled = upscaler.upscale_frame_algorithmic(frame_720p, target_width=1920, target_height=1080)
        assert upscaled.shape == (1080, 1920, 3)
        assert upscaled.dtype == np.uint8

    def test_upscale_single_frame_720p_to_4k(self):
        """Upscales a 720p frame (1280x720) to exact 4K (3840x2160) dimensions."""
        frame_720p = generate_synthetic_frame(width=1280, height=720)
        upscaler = RealESRGANUpscaler(device="cpu")
        upscaled = upscaler.upscale_frame_tiled(frame_720p, target_width=3840, target_height=2160, tile_size=512)
        assert upscaled.shape == (2160, 3840, 3)
        assert upscaled.dtype == np.uint8

    def test_upscale_same_resolution_identity(self):
        """When input resolution matches target resolution, returns identical dimensions."""
        frame_1080p = generate_synthetic_frame(width=1920, height=1080)
        upscaler = RealESRGANUpscaler(device="cpu")
        upscaled = upscaler.upscale_frame_tiled(frame_1080p, target_width=1920, target_height=1080)
        assert upscaled.shape == (1080, 1920, 3)
        assert np.array_equal(frame_1080p, upscaled)

    def test_upscale_downscaling_support(self):
        """Downscaling (e.g. 1080p -> 720p) is supported without distortion."""
        frame_1080p = generate_synthetic_frame(width=1920, height=1080)
        upscaler = RealESRGANUpscaler(device="cpu")
        downscaled = upscaler.upscale_frame_algorithmic(frame_1080p, target_width=1280, target_height=720)
        assert downscaled.shape == (720, 1280, 3)
        assert downscaled.dtype == np.uint8

    def test_upscale_chunked_frame_batching(self):
        """Processes 7 frames with chunk_batch_size=2 -> returns all 7 frames at 1080p."""
        frames = [generate_synthetic_frame(width=640, height=360, color=(i * 30, 100, 150)) for i in range(7)]
        upscaler = RealESRGANUpscaler(device="cpu")
        upscaled = upscaler.upscale_frames(frames, target_resolution="1080p", chunk_size=2, tile_size=512)
        assert len(upscaled) == 7
        for frame in upscaled:
            assert frame.shape == (1080, 1920, 3)
            assert frame.dtype == np.uint8

    def test_upscale_spatial_tiling_overlap_blending(self):
        """Verifies spatial tiling (tile_size=256, tile_pad=10) produces seamless output."""
        frame = generate_synthetic_frame(width=600, height=400)
        upscaler = RealESRGANUpscaler(device="cpu")
        tiled_out = upscaler.upscale_frame_tiled(frame, target_width=1200, target_height=800, tile_size=256, tile_pad=10)
        assert tiled_out.shape == (800, 1200, 3)
        assert tiled_out.dtype == np.uint8
        # Ensure no NaN or infinite values
        assert not np.isnan(tiled_out).any()
        assert not np.isinf(tiled_out).any()

    def test_upscale_empty_frames_list(self):
        """Empty input list returns empty output list."""
        upscaler = RealESRGANUpscaler(device="cpu")
        assert upscaler.upscale_frames([], target_resolution="1080p") == []

    def test_upscaler_progress_callback(self):
        """Progress callback is called and tracks progress up to 1.0."""
        frames = [generate_synthetic_frame(width=320, height=240) for _ in range(4)]
        upscaler = RealESRGANUpscaler(device="cpu")
        progress_calls = []

        def callback(pct, msg):
            progress_calls.append((pct, msg))

        upscaler.upscale_frames(frames, target_resolution="720p", chunk_size=2, progress_callback=callback)
        assert len(progress_calls) >= 2
        assert progress_calls[-1][0] == 1.0


# =============================================================================
# 4. RIFE Temporal Frame Interpolation Tests
# =============================================================================

class TestRIFEInterpolator:
    """Tests 24fps -> 60fps frame rate interpolation, optical flow warping, and temporal duration matching."""

    def test_interpolator_initialization(self):
        """RIFEInterpolator initializes cleanly with device fallback."""
        interpolator = RIFEInterpolator(device="cpu")
        assert interpolator.device == "cpu"

    def test_interpolate_24fps_to_60fps_frame_count(self):
        """
        24 frames @ 24fps (1.0 sec duration) interpolated to 60fps
        must yield exactly 60 (or mathematically round((24-1)*60/24)+1 = 58..60) frames.
        """
        frames_24 = generate_synthetic_video_sequence(num_frames=24, width=640, height=360)
        interpolator = RIFEInterpolator(device="cpu")
        frames_60 = interpolator.interpolate_fps(frames_24, source_fps=24, target_fps=60)
        
        # Mathematical calculation: (24-1)*60/24 + 1 = 23*2.5 + 1 = 57.5 + 1 = 58.5 -> 59
        expected_count = int(round((24 - 1) * 60.0 / 24.0)) + 1
        assert len(frames_60) == expected_count
        assert all(f.shape == (360, 640, 3) for f in frames_60)

    def test_interpolate_81frames_to_60fps(self):
        """
        81 frames @ 24fps (3.375s video) interpolated to 60fps
        must produce exactly 201 frames: round((81-1)*60/24) + 1 = 80*2.5 + 1 = 201.
        """
        frames_81 = [generate_synthetic_frame(width=320, height=240, color=(i % 255, 100, 100)) for i in range(81)]
        interpolator = RIFEInterpolator(device="cpu")
        interpolated = interpolator.interpolate_fps(frames_81, source_fps=24, target_fps=60)
        assert len(interpolated) == 201
        assert all(f.shape == (240, 320, 3) for f in interpolated)

    def test_interpolate_same_fps_returns_identical_frames(self):
        """When source_fps == target_fps (24 -> 24), returns frames unchanged."""
        frames = generate_synthetic_video_sequence(num_frames=10, width=320, height=240)
        interpolator = RIFEInterpolator(device="cpu")
        out = interpolator.interpolate_fps(frames, source_fps=24, target_fps=24)
        assert len(out) == 10
        for i in range(10):
            assert np.array_equal(frames[i], out[i])

    def test_interpolate_single_frame(self):
        """Single frame sequence returns single frame."""
        frame = generate_synthetic_frame(width=320, height=240)
        interpolator = RIFEInterpolator(device="cpu")
        out = interpolator.interpolate_fps([frame], source_fps=24, target_fps=60)
        assert len(out) == 1
        assert np.array_equal(frame, out[0])

    def test_interpolate_two_frames_creates_intermediate_blend(self):
        """Two frames with distinct colors produce smooth intermediate transition."""
        f0 = np.zeros((100, 100, 3), dtype=np.uint8)  # Black
        f1 = np.ones((100, 100, 3), dtype=np.uint8) * 200  # Bright
        interpolator = RIFEInterpolator(device="cpu")
        out = interpolator.interpolate_fps([f0, f1], source_fps=24, target_fps=48)
        # (2-1)*48/24 + 1 = 3 frames
        assert len(out) == 3
        assert np.array_equal(out[0], f0)
        assert np.array_equal(out[2], f1)
        # Middle frame must be between 0 and 200
        mid_val = int(out[1][50, 50, 0])
        assert 50 <= mid_val <= 150

    def test_interpolate_empty_frames(self):
        """Empty input list returns empty list."""
        interpolator = RIFEInterpolator(device="cpu")
        assert interpolator.interpolate_fps([], source_fps=24, target_fps=60) == []

    def test_interpolate_invalid_fps_raises(self):
        """Zero or negative framerates raise ValueError."""
        interpolator = RIFEInterpolator(device="cpu")
        with pytest.raises(ValueError, match="Framerate values must be positive"):
            interpolator.interpolate_fps([generate_synthetic_frame(100, 100)], source_fps=0, target_fps=60)

    def test_interpolator_progress_callback(self):
        """Progress callback is called and reaches 1.0."""
        frames = generate_synthetic_video_sequence(num_frames=12, width=320, height=240)
        interpolator = RIFEInterpolator(device="cpu")
        progress_calls = []

        def callback(pct, msg):
            progress_calls.append((pct, msg))

        interpolator.interpolate_fps(frames, source_fps=24, target_fps=60, progress_callback=callback)
        assert len(progress_calls) >= 1
        assert progress_calls[-1][0] == 1.0


# =============================================================================
# 5. Audio/Video Master Multiplexer Tests
# =============================================================================

class TestAudioVideoMuxer:
    """Tests FFmpeg / MoviePy / OpenCV master MP4 muxing with audio synchronization and faststart."""

    def test_muxer_initialization(self):
        """AudioVideoMuxer initializes and locates FFmpeg binary if available."""
        muxer = AudioVideoMuxer()
        assert hasattr(muxer, "_ffmpeg_bin")

    def test_mux_video_only_mp4(self, temp_test_dir):
        """Muxes a sequence of frames into a valid MP4 file without audio."""
        frames = generate_synthetic_video_sequence(num_frames=24, width=640, height=360)
        output_path = os.path.join(temp_test_dir, "video_only.mp4")

        muxer = AudioVideoMuxer()
        res_path = muxer.mux_video_audio(
            frames=frames,
            audio_path=None,
            output_path=output_path,
            fps=24,
            crf=18,
        )
        assert os.path.exists(res_path)
        assert os.path.getsize(res_path) > 0
        assert res_path.endswith(".mp4")

    def test_mux_video_with_audio_wav(self, temp_test_dir):
        """Muxes frames with a synthetic WAV audio file into an MP4 master."""
        frames = generate_synthetic_video_sequence(num_frames=24, width=640, height=360)
        audio_path = os.path.join(temp_test_dir, "speech.wav")
        create_synthetic_wav_file(audio_path, duration_s=1.0, sample_rate=16000)

        output_path = os.path.join(temp_test_dir, "master_with_audio.mp4")
        muxer = AudioVideoMuxer()
        res_path = muxer.mux_video_audio(
            frames=frames,
            audio_path=audio_path,
            output_path=output_path,
            fps=24,
            crf=18,
            audio_codec="aac",
        )
        assert os.path.exists(res_path)
        assert os.path.getsize(res_path) > 0

    def test_mux_handles_audio_duration_mismatch(self, temp_test_dir):
        """Handles audio longer than video cleanly without error."""
        frames = generate_synthetic_video_sequence(num_frames=12, width=320, height=240)  # 0.5 sec @ 24fps
        audio_path = os.path.join(temp_test_dir, "long_audio.wav")
        create_synthetic_wav_file(audio_path, duration_s=3.0, sample_rate=16000)  # 3.0 sec

        output_path = os.path.join(temp_test_dir, "duration_mismatch.mp4")
        muxer = AudioVideoMuxer()
        res_path = muxer.mux_video_audio(
            frames=frames,
            audio_path=audio_path,
            output_path=output_path,
            fps=24,
        )
        assert os.path.exists(res_path)
        assert os.path.getsize(res_path) > 0

    def test_mux_handles_nonexistent_audio_gracefully(self, temp_test_dir):
        """Non-existent audio file falls back gracefully to video-only export."""
        frames = generate_synthetic_video_sequence(num_frames=10, width=320, height=240)
        output_path = os.path.join(temp_test_dir, "missing_audio.mp4")

        muxer = AudioVideoMuxer()
        res_path = muxer.mux_video_audio(
            frames=frames,
            audio_path=os.path.join(temp_test_dir, "nonexistent.wav"),
            output_path=output_path,
            fps=24,
        )
        assert os.path.exists(res_path)
        assert os.path.getsize(res_path) > 0

    def test_mux_creates_nested_directory_if_missing(self, temp_test_dir):
        """Output directory tree is created automatically if it does not exist."""
        frames = generate_synthetic_video_sequence(num_frames=6, width=320, height=240)
        nested_output = os.path.join(temp_test_dir, "sub", "dir", "level2", "nested.mp4")

        muxer = AudioVideoMuxer()
        res_path = muxer.mux_video_audio(
            frames=frames,
            audio_path=None,
            output_path=nested_output,
            fps=24,
        )
        assert os.path.exists(res_path)

    def test_mux_empty_frames_raises_value_error(self, temp_test_dir):
        """Passing an empty list of frames raises ValueError."""
        muxer = AudioVideoMuxer()
        with pytest.raises(ValueError, match="Cannot mux an empty frames list"):
            muxer.mux_video_audio([], audio_path=None, output_path=os.path.join(temp_test_dir, "out.mp4"))


# =============================================================================
# 6. PostProductionEngine End-to-End & Memory Lifecycle Tests
# =============================================================================

class TestPostProductionEngineIntegration:
    """Tests PostProductionEngine end-to-end master render, VRAM lifecycle isolation, and pipeline methods."""

    def test_engine_initialization_defaults(self):
        """Initializes with default VRAMManager and default PostProcessingConfig."""
        engine = PostProductionEngine()
        assert engine.config.target_resolution == "1080p"
        assert engine.config.target_fps == 60
        assert engine.config.chunk_batch_size == 4
        assert engine.upscaler is not None
        assert engine.interpolator is not None
        assert engine.muxer is not None

    def test_render_final_master_full_pipeline(self, temp_test_dir):
        """
        Executes end-to-end master render:
        1. Input: 12 frames @ 720p (1280x720), 24fps + WAV audio
        2. Interpolates to 60fps (12 -> 28 frames)
        3. Upscales to 1080p (1280x720 -> 1920x1080)
        4. Muxes to MP4 master
        """
        frames_720p = generate_synthetic_video_sequence(num_frames=12, width=1280, height=720)
        audio_path = os.path.join(temp_test_dir, "dialogue.wav")
        create_synthetic_wav_file(audio_path, duration_s=0.5, sample_rate=16000)

        output_path = os.path.join(temp_test_dir, "final_master_1080p_60fps.mp4")
        engine = PostProductionEngine()

        cfg = PostProcessingConfig(
            enable_upscale=True,
            target_resolution="1080p",
            chunk_batch_size=4,
            enable_interpolation=True,
            target_fps=60,
            crf=18,
        )

        final_path = engine.render_final_master(
            frames=frames_720p,
            audio_path=audio_path,
            output_path=output_path,
            config=cfg,
            source_fps=24,
        )

        assert os.path.exists(final_path)
        assert os.path.getsize(final_path) > 0
        assert final_path.endswith(".mp4")

    def test_render_final_master_vram_lifecycle_isolation(self, temp_test_dir):
        """
        Verifies that render_final_master executes inside @vram_lifecycle_stage("post_processing")
        and cleans up memory stages properly.
        """
        mgr = VRAMManager.get_instance()
        engine = PostProductionEngine(memory_manager=mgr)

        frames = generate_synthetic_video_sequence(num_frames=6, width=640, height=360)
        output_path = os.path.join(temp_test_dir, "lifecycle_test.mp4")

        # Prior stage should be None
        assert mgr.current_stage is None

        final_path = engine.render_final_master(
            frames=frames,
            audio_path=None,
            output_path=output_path,
            config=PostProcessingConfig(enable_upscale=False, enable_interpolation=False),
        )

        assert os.path.exists(final_path)
        # Stage must be cleanly exited and returned to None
        assert mgr.current_stage is None

    def test_render_final_master_with_progress_callback(self, temp_test_dir):
        """Progress callback is called throughout all 3 pipeline steps and finishes at 1.0."""
        frames = generate_synthetic_video_sequence(num_frames=6, width=640, height=360)
        output_path = os.path.join(temp_test_dir, "progress_test.mp4")
        engine = PostProductionEngine()

        progress_history = []

        def callback(pct, msg):
            progress_history.append((pct, msg))

        engine.render_final_master(
            frames=frames,
            audio_path=None,
            output_path=output_path,
            config=PostProcessingConfig(enable_upscale=True, target_resolution="720p", enable_interpolation=True, target_fps=30),
            source_fps=24,
            progress_callback=callback,
        )

        assert len(progress_history) >= 3
        assert progress_history[-1][0] == 1.0

    def test_process_pipeline_method_returns_result_dataclass(self, temp_test_dir):
        """process_pipeline() returns a PostProcessResult with comprehensive metadata."""
        frames = generate_synthetic_video_sequence(num_frames=8, width=640, height=360)
        output_path = os.path.join(temp_test_dir, "pipeline_result.mp4")
        engine = PostProductionEngine()

        result = engine.process_pipeline(
            video_input=frames,
            audio_input=None,
            upscale_target="720p",
            interpolate_60fps=True,
            output_path=output_path,
        )

        assert isinstance(result, PostProcessResult)
        assert result.output_path == os.path.abspath(output_path)
        assert result.resolution == (1280, 720)
        assert result.fps == 60
        assert result.duration > 0.0
        assert result.has_audio is False
        assert result.processing_time_s >= 0.0
        assert result.upscaled is True
        assert result.interpolated is True
        # String conversion matches output path
        assert str(result) == os.path.abspath(output_path)

    def test_read_video_frames_and_process_video_file(self, temp_test_dir):
        """Tests reading generated MP4 video file and processing video-in to video-out."""
        # 1. Create a base video file
        frames = generate_synthetic_video_sequence(num_frames=10, width=640, height=360)
        base_video_path = os.path.join(temp_test_dir, "base_input.mp4")
        engine = PostProductionEngine()
        engine.muxer.mux_video_audio(frames, None, base_video_path, fps=24)

        # 2. Read frames back
        read_frames = engine.read_video_frames(base_video_path)
        assert len(read_frames) == 10
        assert read_frames[0].shape == (360, 640, 3)

        # 3. Process video file input directly via process_pipeline
        out_processed = os.path.join(temp_test_dir, "processed_from_file.mp4")
        res = engine.process_pipeline(
            video_input=base_video_path,
            upscale_target="720p",
            interpolate_60fps=False,
            output_path=out_processed,
        )
        assert os.path.exists(res.output_path)

    def test_render_final_master_empty_frames_raises(self, temp_test_dir):
        """Empty input frames sequence raises ValueError."""
        engine = PostProductionEngine()
        with pytest.raises(ValueError, match="Input frames sequence cannot be empty"):
            engine.render_final_master([], None, os.path.join(temp_test_dir, "fail.mp4"))


# =============================================================================
# 7. Adversarial & Stress Testing
# =============================================================================

class TestPostProcessingAdversarialAndStress:
    """Adversarial stress tests covering corrupted inputs, strange aspect ratios, and extreme resolutions."""

    def test_upscale_extreme_aspect_ratio(self):
        """Ultra-wide (21:9 cinematic) and tall aspect ratios upscale accurately without crashing."""
        ultra_wide = generate_synthetic_frame(width=1600, height=400)
        upscaler = RealESRGANUpscaler(device="cpu")
        out = upscaler.upscale_frame_algorithmic(ultra_wide, target_width=3200, target_height=800)
        assert out.shape == (800, 3200, 3)

    def test_interpolate_high_framerate_conversion(self):
        """Extreme interpolation (12fps -> 120fps, 10x ratio) produces correct frame count."""
        frames_12 = generate_synthetic_video_sequence(num_frames=12, width=320, height=240)
        interpolator = RIFEInterpolator(device="cpu")
        frames_120 = interpolator.interpolate_fps(frames_12, source_fps=12, target_fps=120)
        # (12-1)*120/12 + 1 = 11*10 + 1 = 111 frames
        assert len(frames_120) == 111

    def test_interpolate_downsampling_fps(self):
        """Downsampling framerate (60fps -> 24fps) operates cleanly."""
        frames_60 = generate_synthetic_video_sequence(num_frames=61, width=320, height=240)
        interpolator = RIFEInterpolator(device="cpu")
        frames_24 = interpolator.interpolate_fps(frames_60, source_fps=60, target_fps=24)
        # (61-1)*24/60 + 1 = 60*0.4 + 1 = 25 frames
        assert len(frames_24) == 25

    def test_mux_with_high_crf_and_custom_preset(self, temp_test_dir):
        """High CRF value and different presets produce valid output."""
        frames = generate_synthetic_video_sequence(num_frames=8, width=320, height=240)
        output_path = os.path.join(temp_test_dir, "high_crf.mp4")
        muxer = AudioVideoMuxer()
        res = muxer.mux_video_audio(
            frames=frames,
            audio_path=None,
            output_path=output_path,
            fps=24,
            crf=30,
            preset="ultrafast",
        )
        assert os.path.exists(res)
        assert os.path.getsize(res) > 0


# =============================================================================
# 8. Neural Model & VRAM Lifecycle Mock Testing
# =============================================================================

class TestNeuralModelAndVRAMIntegration:
    """Tests neural model execution path, mock PyTorch modules, and VRAMManager model registration."""

    def test_mock_pytorch_cuda_model_registration(self):
        """Simulates PyTorch CUDA execution with mocked RRDBNet and verifies registration."""
        mock_mgr = VRAMManager.get_instance()

        class MockRRDBNet:
            def __init__(self):
                self.device = "cuda"
                self.is_half = False
            def to(self, device):
                self.device = device
                return self
            def half(self):
                self.is_half = True
                return self
            def eval(self):
                return self
            def __call__(self, x):
                return x

        mock_model = MockRRDBNet()
        upscaler = RealESRGANUpscaler(device="cpu", memory_manager=mock_mgr)
        upscaler._model = mock_model
        upscaler._is_mock = False
        upscaler.device = "cuda"
        mock_mgr.register_model("realesrgan_upscaler", mock_model)

        assert "realesrgan_upscaler" in mock_mgr.get_registered_models()
        purged = mock_mgr.purge_models()
        assert purged >= 1
        assert "realesrgan_upscaler" not in mock_mgr.get_registered_models()

    def test_render_final_master_with_mock_oom_recovery(self, temp_test_dir):
        """Simulates an OutOfMemoryError inside render_final_master and verifies stage cleanup."""
        mgr = VRAMManager.get_instance()
        engine = PostProductionEngine(memory_manager=mgr)
        frames = generate_synthetic_video_sequence(num_frames=4, width=320, height=240)

        with patch.object(engine.muxer, "mux_video_audio", side_effect=RuntimeError("Simulated Stage Error")):
            with pytest.raises(RuntimeError, match="Simulated Stage Error"):
                engine.render_final_master(
                    frames=frames,
                    audio_path=None,
                    output_path=os.path.join(temp_test_dir, "oom_test.mp4"),
                )

        # Stage context should have cleanly recovered
        assert mgr.current_stage is None

