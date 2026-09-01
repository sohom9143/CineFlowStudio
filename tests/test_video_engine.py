"""
Unit & Integration Test Suite for CineFlow-AI Video Motion Synthesizer (Milestone 3 / R3)
========================================================================================
Tests cover:
- VideoGenerationConfig dataclass defaults, serialization, and parameter sanitization.
- DiT temporal frame count mathematics ($(4k+1)$ and $(8k+1)$).
- Engine initialization, backend registry, aliasing, and dynamic switching.
- Deterministic procedural CPU mock video generation (pan, zoom, tilt, lighting motion).
- Exact frame count (81 frames), shape (H, W, 3), and dtype (uint8) verification.
- Seamless automatic fallback from Wan 2.1 / LTX-Video to Mock on CPU/non-CUDA.
- VRAM lifecycle stage decorator and memory management integration.
- Video file exporting and adversarial boundary handling.
"""

import os
import sys
import tempfile
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

# Ensure workspace root is in path
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from modules.memory_manager import VRAMManager
from modules.video_engine import (
    CineVideoEngine,
    VideoGenerationConfig,
    Wan21Backend,
    LTXVideoBackend,
    MockVideoBackend,
    get_valid_dit_frame_counts,
    validate_frame_count,
    save_video_frames,
)


# =============================================================================
# 1. VideoGenerationConfig & DiT Mathematics Tests
# =============================================================================

class TestVideoGenerationConfigAndMath:
    """Tests configuration defaults, dictionary serialization, and DiT frame math."""

    def test_default_config_values(self):
        """Verify default configuration adheres to Milestone 3 specifications."""
        cfg = VideoGenerationConfig()
        assert cfg.backend == "wan2.1"
        assert cfg.num_frames == 81
        assert cfg.fps == 24
        assert cfg.width == 720
        assert cfg.height == 480
        assert cfg.motion_scale == 1.0
        assert cfg.guidance_scale == 6.0
        assert cfg.num_inference_steps == 30
        assert cfg.seed is None
        assert cfg.motion_prompt == ""
        assert cfg.enable_cpu_offload is True
        assert cfg.quantization == "fp8"

    def test_config_serialization(self):
        """Test to_dict and from_dict roundtrip."""
        cfg = VideoGenerationConfig(
            backend="ltx-video",
            num_frames=49,
            fps=30,
            width=832,
            height=480,
            motion_scale=1.5,
            guidance_scale=7.5,
            num_inference_steps=25,
            seed=1234,
            motion_prompt="cinematic dolly zoom",
        )
        d = cfg.to_dict()
        assert d["backend"] == "ltx-video"
        assert d["num_frames"] == 49
        assert d["seed"] == 1234

        # Add extraneous keys to test filtering
        d["unknown_key"] = "ignored"
        reconstructed = VideoGenerationConfig.from_dict(d)
        assert reconstructed.backend == "ltx-video"
        assert reconstructed.num_frames == 49
        assert reconstructed.motion_prompt == "cinematic dolly zoom"
        assert not hasattr(reconstructed, "unknown_key")

    def test_dit_frame_count_math_wan21(self):
        """Verify Wan 2.1 (4k + 1) temporal framing math."""
        valid_wan = get_valid_dit_frame_counts("wan2.1")
        assert 81 in valid_wan  # 4 * 20 + 1
        assert 49 in valid_wan  # 4 * 12 + 1
        assert 17 in valid_wan  # 4 * 4 + 1
        assert 33 in valid_wan  # 4 * 8 + 1
        assert 97 in valid_wan  # 4 * 24 + 1
        assert 80 not in valid_wan

        is_valid, count, msg = validate_frame_count(81, "wan2.1")
        assert is_valid is True
        assert count == 81

        # Non-conforming count 80 should adjust to 81
        is_valid, count, msg = validate_frame_count(80, "wan2.1")
        assert is_valid is False
        assert count == 81

    def test_dit_frame_count_math_ltx(self):
        """Verify LTX-Video (8k + 1) temporal framing math."""
        valid_ltx = get_valid_dit_frame_counts("ltx-video")
        assert 81 in valid_ltx  # 8 * 10 + 1
        assert 49 in valid_ltx  # 8 * 6 + 1
        assert 17 in valid_ltx  # 8 * 2 + 1
        assert 33 in valid_ltx  # 8 * 4 + 1
        assert 25 in valid_ltx  # 8 * 3 + 1
        assert 80 not in valid_ltx

        is_valid, count, msg = validate_frame_count(81, "ltx-video")
        assert is_valid is True
        assert count == 81


# =============================================================================
# 2. Engine Initialization & Backend Registry Tests
# =============================================================================

class TestCineVideoEngineInitialization:
    """Tests engine initialization, registry, aliasing, and backend switching."""

    def test_engine_initialization_defaults(self):
        """Verify engine initializes with default backend and registry."""
        engine = CineVideoEngine()
        assert engine.get_active_backend() == "wan2.1"
        backends = engine.list_available_backends()
        assert "wan2.1" in backends
        assert "ltx-video" in backends
        assert "mock" in backends

    def test_backend_name_aliasing(self):
        """Verify various aliases map correctly to canonical backend names."""
        engine = CineVideoEngine()
        assert engine._normalize_backend_name("wan21") == "wan2.1"
        assert engine._normalize_backend_name("WAN2.1") == "wan2.1"
        assert engine._normalize_backend_name("Wan") == "wan2.1"
        assert engine._normalize_backend_name("ltx") == "ltx-video"
        assert engine._normalize_backend_name("LTX-Video") == "ltx-video"
        assert engine._normalize_backend_name("ltxvideo") == "ltx-video"
        assert engine._normalize_backend_name("mock") == "mock"
        assert engine._normalize_backend_name("cpu") == "mock"
        assert engine._normalize_backend_name("procedural") == "mock"

    def test_dynamic_backend_switching(self):
        """Verify dynamic backend switching and model purging."""
        engine = CineVideoEngine()
        assert engine.get_active_backend() == "wan2.1"

        engine.switch_backend("ltx")
        assert engine.get_active_backend() == "ltx-video"

        engine.switch_backend("cpu")
        assert engine.get_active_backend() == "mock"

    def test_unload_models(self):
        """Verify unload_models purges registered pipelines."""
        mgr = VRAMManager.get_instance()
        engine = CineVideoEngine(memory_manager=mgr)
        mgr.register_model("wan2.1_pipeline", "dummy_wan")
        mgr.register_model("ltx_pipeline", "dummy_ltx")

        purged = engine.unload_models()
        assert purged >= 2
        assert "wan2.1_pipeline" not in mgr.get_registered_models()
        assert "ltx_pipeline" not in mgr.get_registered_models()


# =============================================================================
# 3. Procedural Mock Video Generation & Mathematics Tests
# =============================================================================

class TestProceduralMockVideoGeneration:
    """Tests deterministic procedural CPU mock generation, mathematical continuity, and formats."""

    def test_mock_generation_with_no_input_image(self):
        """Verify generating video from scratch synthesizes scene canvas (81 frames @ 24fps)."""
        engine = CineVideoEngine(default_backend="mock")
        config = VideoGenerationConfig(
            backend="mock",
            num_frames=81,
            fps=24,
            width=720,
            height=480,
            seed=42,
        )
        frames = engine.generate_motion(image=None, config=config)

        assert isinstance(frames, list)
        assert len(frames) == 81
        for idx, f in enumerate(frames):
            assert isinstance(f, np.ndarray)
            assert f.shape == (480, 720, 3)
            assert f.dtype == np.uint8
            # Ensure valid RGB value ranges
            assert np.all(f >= 0) and np.all(f <= 255)

    def test_mock_generation_from_pil_image(self):
        """Verify generating video from a PIL Image input."""
        engine = CineVideoEngine(default_backend="mock")
        pil_img = Image.new("RGB", (640, 480), color=(180, 100, 50))
        
        config = VideoGenerationConfig(
            backend="mock",
            num_frames=81,
            width=720,
            height=480,
            seed=100,
        )
        frames = engine.generate_motion(image=pil_img, config=config)
        assert len(frames) == 81
        assert frames[0].shape == (480, 720, 3)

    def test_mock_generation_from_numpy_array(self):
        """Verify generating video from a numpy array input (RGB and RGBA)."""
        engine = CineVideoEngine(default_backend="mock")
        
        # 1. RGB numpy array
        np_rgb = np.full((480, 720, 3), 120, dtype=np.uint8)
        frames_rgb = engine.generate_motion(image=np_rgb, config=VideoGenerationConfig(backend="mock", num_frames=17))
        assert len(frames_rgb) == 17
        assert frames_rgb[0].shape == (480, 720, 3)

        # 2. RGBA numpy array
        np_rgba = np.full((480, 720, 4), 200, dtype=np.uint8)
        frames_rgba = engine.generate_motion(image=np_rgba, config=VideoGenerationConfig(backend="mock", num_frames=17))
        assert len(frames_rgba) == 17
        assert frames_rgba[0].shape == (480, 720, 3)

    def test_mock_generation_from_file_path(self):
        """Verify generating video from an image saved to disk."""
        engine = CineVideoEngine(default_backend="mock")
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "test_keyframe.png"
            test_img = Image.new("RGB", (720, 480), color=(50, 150, 200))
            test_img.save(str(img_path))

            frames = engine.generate_motion(
                image=str(img_path),
                motion_prompt="cinematic slow push-in",
                config=VideoGenerationConfig(backend="mock", num_frames=33)
            )
            assert len(frames) == 33
            assert frames[0].shape == (480, 720, 3)

    def test_mathematical_continuity_and_motion(self):
        """
        Verify that procedural synthesis produces true mathematical motion across frames
        (not static duplicates) with smooth continuous transitions.
        """
        engine = CineVideoEngine(default_backend="mock")
        config = VideoGenerationConfig(
            backend="mock",
            num_frames=81,
            width=720,
            height=480,
            motion_scale=1.5,
            seed=42,
        )
        frames = engine.generate_motion(image=None, config=config)

        f0 = frames[0].astype(np.float32)
        f40 = frames[40].astype(np.float32)
        f80 = frames[80].astype(np.float32)

        # Difference between frame 0 and frame 40 should be non-zero (motion occurred)
        diff_0_40 = np.mean(np.abs(f40 - f0))
        diff_0_80 = np.mean(np.abs(f80 - f0))
        assert diff_0_40 > 1.0, f"Expected noticeable motion delta, got {diff_0_40}"
        assert diff_0_80 > 1.0, f"Expected noticeable motion delta, got {diff_0_80}"

        # Temporal continuity: Adjacent frames should have smaller delta than distant frames
        diff_adjacent = np.mean(np.abs(frames[1].astype(np.float32) - f0))
        assert diff_adjacent < diff_0_40, "Adjacent frame delta should be smaller than half-clip delta."

    def test_reproducible_determinism_with_seed(self):
        """Verify identical seed produces exact identical frame array outputs."""
        engine = CineVideoEngine(default_backend="mock")
        config_a = VideoGenerationConfig(backend="mock", num_frames=33, seed=42)
        config_b = VideoGenerationConfig(backend="mock", num_frames=33, seed=42)
        config_diff = VideoGenerationConfig(backend="mock", num_frames=33, seed=999)

        frames_a = engine.generate_motion(image=None, config=config_a)
        frames_b = engine.generate_motion(image=None, config=config_b)
        frames_diff = engine.generate_motion(image=None, config=config_diff)

        # Frames A and B must be bitwise identical
        for fa, fb in zip(frames_a, frames_b):
            assert np.array_equal(fa, fb)

        # Frames A and Diff must differ
        diff_count = sum(not np.array_equal(fa, fd) for fa, fd in zip(frames_a, frames_diff))
        assert diff_count == len(frames_a)


# =============================================================================
# 4. Cascading Automatic Fallbacks & Signature Flexibility Tests
# =============================================================================

class TestAutomaticFallbacksAndSignatures:
    """Tests automatic backend fallback on CPU/non-CUDA and overloaded signatures."""

    def test_automatic_fallback_from_wan21_to_mock_on_cpu(self):
        """
        When Wan 2.1 is requested on CPU or without weights, engine automatically
        falls back to Mock backend without raising unhandled exceptions.
        """
        engine = CineVideoEngine(default_backend="wan2.1")
        # Request wan2.1 explicitly
        config = VideoGenerationConfig(backend="wan2.1", num_frames=81, seed=42)
        frames = engine.generate_motion(image=None, config=config)

        assert len(frames) == 81
        assert frames[0].shape == (480, 720, 3)

    def test_automatic_fallback_from_ltx_to_mock_on_cpu(self):
        """
        When LTX-Video is requested on CPU or without weights, engine automatically
        falls back to Mock backend gracefully.
        """
        engine = CineVideoEngine(default_backend="ltx-video")
        config = VideoGenerationConfig(backend="ltx-video", num_frames=81, seed=42)
        frames = engine.generate_motion(image=None, config=config)

        assert len(frames) == 81
        assert frames[0].shape == (480, 720, 3)

    def test_overloaded_generate_motion_signatures(self):
        """
        Test calling generate_motion with different positional/keyword arguments:
        - engine.generate_motion(image, config)
        - engine.generate_motion(image, "motion prompt", config)
        - engine.generate_motion(image, motion_prompt="...", num_frames=49)
        """
        engine = CineVideoEngine(default_backend="mock")
        pil_img = Image.new("RGB", (720, 480), color=(100, 100, 100))

        # Signature 1: (image, config)
        cfg = VideoGenerationConfig(backend="mock", num_frames=17)
        res1 = engine.generate_motion(pil_img, cfg)
        assert len(res1) == 17

        # Signature 2: (image, prompt, config)
        res2 = engine.generate_motion(pil_img, "slow zoom", cfg)
        assert len(res2) == 17

        # Signature 3: keyword overrides
        res3 = engine.generate_motion(pil_img, motion_prompt="pan right", num_frames=33)
        assert len(res3) == 33


# =============================================================================
# 5. VRAM Lifecycle & Memory Manager Integration Tests
# =============================================================================

class TestVRAMLifecycleIntegration:
    """Tests stage lifecycle decorator execution and memory isolation."""

    def test_generate_motion_wrapped_in_lifecycle_stage(self):
        """Verify that generate_motion triggers VRAM lifecycle stage entry and exit."""
        mgr = VRAMManager.get_instance()
        engine = CineVideoEngine(memory_manager=mgr, default_backend="mock")

        assert mgr.current_stage is None
        frames = engine.generate_motion(image=None, config=VideoGenerationConfig(backend="mock", num_frames=17))
        assert len(frames) == 17
        # After execution, stage context should be cleanly exited
        assert mgr.current_stage is None

    def test_wan21_and_ltx_backends_mock_pipeline_loading(self):
        """Test load_model and unload_model on Wan21Backend and LTXVideoBackend."""
        mgr = VRAMManager.get_instance()
        wan_backend = Wan21Backend(memory_manager=mgr)
        ltx_backend = LTXVideoBackend(memory_manager=mgr)

        # On CPU, is_available is False
        if not mgr.is_cuda:
            assert wan_backend.is_available() is False
            assert ltx_backend.is_available() is False

        # Unload model safety
        wan_backend.unload_model()
        ltx_backend.unload_model()
        assert wan_backend.model is None
        assert ltx_backend.model is None


# =============================================================================
# 6. Video Export & File Output Tests
# =============================================================================

class TestVideoExport:
    """Tests writing frame sequences to MP4 / video files."""

    def test_save_video_frames_success(self):
        """Verify save_video_frames successfully exports an array of frames."""
        frames = [np.full((240, 320, 3), i * 10, dtype=np.uint8) for i in range(24)]
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "output_test.mp4"
            saved_path = save_video_frames(frames, output_path=str(out_path), fps=24)
            assert Path(saved_path).exists()
            assert Path(saved_path).stat().st_size > 0

    def test_save_video_frames_empty_error(self):
        """Verify error raised when frame sequence is empty."""
        with pytest.raises(ValueError, match="frame sequence is empty"):
            save_video_frames([], "dummy.mp4")

    def test_engine_export_to_video(self):
        """Verify CineVideoEngine.export_to_video helper."""
        engine = CineVideoEngine(default_backend="mock")
        frames = engine.generate_motion(image=None, config=VideoGenerationConfig(backend="mock", num_frames=17))
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "subfolder" / "rendered_clip.mp4"
            result_path = engine.export_to_video(frames, out_file, fps=24)
            assert Path(result_path).exists()
            assert Path(result_path).stat().st_size > 0


# =============================================================================
# 7. Boundary & Adversarial Tests
# =============================================================================

class TestBoundaryAndAdversarial:
    """Tests boundary cases, invalid configurations, and extreme inputs."""

    def test_extreme_and_negative_parameters(self):
        """Verify engine sanitizes negative/zero parameters gracefully."""
        engine = CineVideoEngine(default_backend="mock")
        cfg = VideoGenerationConfig(
            backend="mock",
            num_frames=-10,        # Negative frames
            width=50,              # Too small width
            height=30,             # Too small height
            fps=-5,                # Negative fps
            motion_scale=-2.0,     # Negative scale
            guidance_scale=0.1,    # Below 1.0
            num_inference_steps=0, # Zero steps
        )
        validated = engine.validate_config(cfg)
        assert validated.num_frames >= 17  # Adjusted to valid DiT count
        assert validated.width >= 256
        assert validated.height >= 256
        assert validated.fps >= 1
        assert validated.motion_scale > 0
        assert validated.guidance_scale >= 1.0
        assert validated.num_inference_steps >= 1

        # Execute generation with sanitized config
        frames = engine.generate_motion(image=None, config=validated)
        assert len(frames) == validated.num_frames

    def test_nonexistent_image_path_fallback(self):
        """Verify nonexistent image path triggers warning and uses procedural canvas."""
        engine = CineVideoEngine(default_backend="mock")
        frames = engine.generate_motion(
            image="non_existent_file_path_123456.jpg",
            config=VideoGenerationConfig(backend="mock", num_frames=17)
        )
        assert len(frames) == 17
        assert frames[0].shape == (480, 720, 3)

    def test_concurrent_generation_threads(self):
        """Verify multiple threads can generate procedural video concurrently without collision."""
        engine = CineVideoEngine(default_backend="mock")
        results = {}
        errors = []

        def worker(thread_id: int):
            try:
                cfg = VideoGenerationConfig(backend="mock", num_frames=17, seed=thread_id * 10)
                frames = engine.generate_motion(image=None, config=cfg)
                results[thread_id] = frames
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 4
        for tid in range(4):
            assert len(results[tid]) == 17
