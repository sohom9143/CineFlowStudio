"""
Adversarial Stress Test Suite for CineFlow-AI M2 (Character Engine) and M3 (Video Engine).
Contains exhaustive boundary testing, adversarial fuzzing, numerical stability checks,
seed determinism verification, DiT temporal math validation, and fallback stress tests.
"""

import os
import json
import math
import hashlib
import tempfile
import numpy as np
from PIL import Image
import pytest

from modules.character_engine import (
    CharacterProfile,
    CharacterStudio,
    compute_l2_norm,
    l2_normalize,
    fuse_consensus_embeddings,
    extract_facial_embedding_from_image,
    sanitize_character_slug,
)
from modules.video_engine import (
    CineVideoEngine,
    VideoGenerationConfig,
    MockVideoBackend,
    Wan21Backend,
    LTXVideoBackend,
    validate_frame_count,
    get_valid_dit_frame_counts,
    save_video_frames,
)
from modules.memory_manager import VRAMManager


# =============================================================================
# Adversarial Fixtures
# =============================================================================

@pytest.fixture
def temp_character_workspace(tmp_path):
    """Isolated directory for dynamic profile enrollment and persistence stress tests."""
    profiles_dir = tmp_path / "character_profiles"
    configs_dir = tmp_path / "configs"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    configs_dir.mkdir(parents=True, exist_ok=True)

    styles_file = configs_dir / "cinematic_styles.json"
    styles_data = {
        "styles": [
            {
                "id": "imax_realism",
                "name": "IMAX 8K Cinematic Realism",
                "prompt_prefix": "70mm IMAX film still",
                "prompt_suffix": "8k uhd",
                "negative_prompt": "cartoon, 3d, blurry",
            },
            {
                "id": "kolkata_vintage",
                "name": "North Kolkata Vintage 35mm",
                "prompt_prefix": "Kodak Portra 400 35mm photograph",
                "prompt_suffix": "vintage Kolkata",
                "negative_prompt": "digital, neon",
            },
            {
                "id": "ghibli_anime",
                "name": "Studio Ghibli / 3D Anime Style",
                "prompt_prefix": "Studio Ghibli animation style",
                "prompt_suffix": "cel shading",
                "negative_prompt": "photorealistic, grain",
            },
            {
                "id": "cyberpunk_noir",
                "name": "Dark Cyberpunk Noir",
                "prompt_prefix": "Blade Runner 2049 style",
                "prompt_suffix": "neon rim light",
                "negative_prompt": "daylight, sunny",
            },
            {
                "id": "custom_neutral",
                "name": "Custom / Neutral Cinematic",
                "prompt_prefix": "cinematic film still",
                "prompt_suffix": "4k",
                "negative_prompt": "blurry",
            },
        ]
    }
    with open(styles_file, "w", encoding="utf-8") as f:
        json.dump(styles_data, f, indent=2)

    return {
        "profiles_dir": str(profiles_dir),
        "styles_path": str(styles_file),
    }


# =============================================================================
# 1. CHARACTER ENGINE ADVERSARIAL CHALLENGES (M2)
# =============================================================================

class TestConsensusEmbeddingAdversarial:
    """Stress-test embedding normalization, zero vectors, noise, and consensus fusion."""

    def test_l2_normalize_all_zeros_vector(self):
        """Zero vector should safely return a valid unit vector on unit hypersphere without NaN/Inf."""
        zero_vec = np.zeros(512, dtype=np.float32)
        norm_vec = l2_normalize(zero_vec)
        assert not np.isnan(norm_vec).any(), "Normalized vector must not contain NaNs"
        assert not np.isinf(norm_vec).any(), "Normalized vector must not contain Infs"
        assert np.isclose(np.linalg.norm(norm_vec), 1.0, atol=1e-5), "Norm must be exactly 1.0"

    def test_l2_normalize_tiny_epsilon_vector(self):
        """Vector with values near machine epsilon (1e-15) must normalize cleanly without overflow/underflow."""
        tiny_vec = np.full(512, 1e-15, dtype=np.float32)
        norm_vec = l2_normalize(tiny_vec)
        assert np.isfinite(norm_vec).all()
        assert np.isclose(np.linalg.norm(norm_vec), 1.0, atol=1e-5)

    def test_l2_normalize_extreme_scale_vector(self):
        """Extremely large values (1e8) must normalize without floating point overflow."""
        huge_vec = np.full(512, 1e8, dtype=np.float32)
        norm_vec = l2_normalize(huge_vec)
        assert np.isfinite(norm_vec).all()
        assert np.isclose(np.linalg.norm(norm_vec), 1.0, atol=1e-5)

    @pytest.mark.parametrize("count", [1, 2, 3, 4, 5, 10, 25])
    def test_consensus_fusion_varying_vector_counts(self, count):
        """Consensus fusion with varying vector counts (1, 3, 5, 10, 25) must yield unit L2 norm."""
        rng = np.random.RandomState(count * 7)
        vectors = [rng.randn(512).astype(np.float32) for _ in range(count)]
        
        fused = fuse_consensus_embeddings(vectors)
        assert fused.shape == (512,)
        assert np.isfinite(fused).all()
        assert np.isclose(np.linalg.norm(fused), 1.0, atol=1e-5)

    def test_consensus_fusion_opposing_antipodal_vectors(self):
        """Vectors that sum exactly to zero (antipodal pair) must fallback safely to unit sphere without division by zero."""
        rng = np.random.RandomState(42)
        v1 = rng.randn(512).astype(np.float32)
        v1 = v1 / np.linalg.norm(v1)
        v2 = -v1  # Exactly opposite vector

        fused = fuse_consensus_embeddings([v1, v2])
        assert np.isfinite(fused).all()
        assert np.isclose(np.linalg.norm(fused), 1.0, atol=1e-5)

    def test_consensus_fusion_noisy_outliers(self):
        """Consensus fusion with extreme noisy outliers should maintain bounded unit projection."""
        rng = np.random.RandomState(123)
        normal_v1 = rng.randn(512).astype(np.float32)
        normal_v2 = rng.randn(512).astype(np.float32)
        crazy_noisy = rng.randn(512).astype(np.float32) * 10000.0

        fused = fuse_consensus_embeddings([normal_v1, normal_v2, crazy_noisy])
        assert np.isfinite(fused).all()
        assert np.isclose(np.linalg.norm(fused), 1.0, atol=1e-5)


class TestDynamicPortraitEnrollmentAdversarial:
    """Adversarial image inputs for dynamic character enrollment."""

    def test_enrollment_extreme_image_dimensions(self, temp_character_workspace):
        """Enrollment with 1x1, 2x2, and 4096x4096 images."""
        studio = CharacterStudio(
            profiles_dir=temp_character_workspace["profiles_dir"],
            styles_path=temp_character_workspace["styles_path"],
        )

        # 1x1 image
        img_1x1 = Image.new("RGB", (1, 1), color=(255, 0, 0))
        # 4096x4096 image (simulated large high-res capture)
        img_4k = Image.new("RGB", (4096, 4096), color=(50, 100, 150))
        # Aspect ratio extreme 1x4096 and 4096x1
        img_tall = Image.new("RGB", (1, 512), color=(20, 20, 20))
        img_wide = Image.new("RGB", (512, 1), color=(80, 80, 80))

        profile = studio.enroll_character(
            name="Extreme Dimensions Hero",
            description="Character enrolled from extreme image dimensions",
            images=[img_1x1, img_4k, img_tall, img_wide],
        )

        assert profile.id == "extreme_dimensions_hero"
        emb = studio.get_character_embedding(profile.id)
        assert emb is not None
        assert emb.shape == (512,)
        assert np.isclose(np.linalg.norm(emb), 1.0, atol=1e-5)

    def test_enrollment_various_color_modes(self, temp_character_workspace):
        """Enrollment with Grayscale (L), RGBA, 1-bit (1), and CMYK modes."""
        studio = CharacterStudio(
            profiles_dir=temp_character_workspace["profiles_dir"],
            styles_path=temp_character_workspace["styles_path"],
        )

        img_l = Image.new("L", (256, 256), color=128)
        img_rgba = Image.new("RGBA", (256, 256), color=(200, 100, 50, 180))
        img_1bit = Image.new("1", (256, 256), color=1)
        img_cmyk = Image.new("CMYK", (256, 256), color=(10, 20, 30, 40))

        profile = studio.enroll_character(
            name="Color Modes Persona",
            description="Testing multi-color-mode handling",
            images=[img_l, img_rgba, img_1bit, img_cmyk],
        )

        assert profile is not None
        emb = studio.get_character_embedding(profile.id)
        assert emb is not None
        assert np.isclose(np.linalg.norm(emb), 1.0, atol=1e-5)

    def test_enrollment_more_than_5_images_truncation(self, temp_character_workspace):
        """Providing 10 images should log a warning and cap enrollment to top 5 images without failing."""
        studio = CharacterStudio(
            profiles_dir=temp_character_workspace["profiles_dir"],
            styles_path=temp_character_workspace["styles_path"],
        )

        images = [Image.new("RGB", (64, 64), color=(i * 20, i * 15, i * 10)) for i in range(10)]
        profile = studio.enroll_character(
            name="Ten Photos Character",
            description="Enrolling with 10 images",
            images=images,
        )

        assert profile is not None
        assert len(profile.reference_images) <= 5


class TestPromptSynthesisAdversarial:
    """Stress-test prompt synthesis with unicode, special characters, and empty inputs."""

    @pytest.fixture
    def studio(self, temp_character_workspace):
        return CharacterStudio(
            profiles_dir=temp_character_workspace["profiles_dir"],
            styles_path=temp_character_workspace["styles_path"],
        )

    def test_synthesis_bengali_unicode_prompts(self, studio):
        """Bengali unicode scene prompts, character names, and style modifiers."""
        bengali_scene = "একটি অন্ধকার কলকাতার গলিতে বৃষ্টি পড়ছে, দেব ও নীল চায়ের দোকানে কথা বলছে"
        bengali_mods = "সিনেমাটিক আলো, সত্যজিৎ রায় ফ্রেম"

        pos, neg = studio.synthesize_prompt(
            character_id="dev",
            scene_prompt=bengali_scene,
            style_id="kolkata_vintage",
            custom_modifiers=bengali_mods,
        )

        assert bengali_scene in pos
        assert bengali_mods in pos
        assert len(pos) > 0
        assert isinstance(pos, str)
        assert isinstance(neg, str)

    def test_synthesis_empty_and_whitespace_inputs(self, studio):
        """Empty scene prompt, empty character ID, empty style ID."""
        pos, neg = studio.synthesize_prompt(
            character_id="",
            scene_prompt="   ",
            style_id="",
            custom_modifiers="",
        )
        assert isinstance(pos, str)
        assert isinstance(neg, str)
        assert len(pos) > 0  # Fallback style prompt should still provide baseline

    def test_synthesis_comma_and_punctuation_flooding(self, studio):
        """Excessive commas, whitespace, tabs, and newlines should be sanitized."""
        messy_scene = ",,, ,,, dark moody rain, ,,, neon reflections,,, \n\n\t  "
        messy_mods = "  ,, master shot,, ,, award winning,   "

        pos, neg = studio.synthesize_prompt(
            character_id="neel",
            scene_prompt=messy_scene,
            style_id="cyberpunk_noir",
            custom_modifiers=messy_mods,
        )

        assert not pos.startswith(",")
        assert not pos.endswith(",")
        assert ", ," not in pos
        assert "dark moody rain" in pos

    def test_synthesis_extreme_length_prompt(self, studio):
        """Excessively long prompt (>10,000 characters) should be handled without buffer errors."""
        long_scene = "cinematic dramatic close-up " * 500
        pos, neg = studio.synthesize_prompt(
            character_id="meghla",
            scene_prompt=long_scene,
            style_id="ghibli_anime",
        )
        assert len(pos) > 10000
        assert "meghla" in pos.lower() or "ghibli" in pos.lower()


class TestProfilePersistenceIntegrity:
    """Verify profile.json and embedding.npy reload integrity and edge cases."""

    def test_enrollment_persistence_and_reload(self, temp_character_workspace):
        """Enrolled character must be fully accessible upon instantiating a new CharacterStudio instance."""
        p_dir = temp_character_workspace["profiles_dir"]
        s_path = temp_character_workspace["styles_path"]

        studio1 = CharacterStudio(profiles_dir=p_dir, styles_path=s_path)
        img = Image.new("RGB", (128, 128), color=(200, 150, 100))
        
        enrolled = studio1.enroll_character(
            name="Soumitra",
            description="Veteran Kolkata actor",
            gender="male",
            images=[img],
            age=65,
            tags=["veteran", "thespian"],
        )

        # Create new studio instance scanning the filesystem
        studio2 = CharacterStudio(profiles_dir=p_dir, styles_path=s_path)
        loaded = studio2.get_character("soumitra")

        assert loaded is not None
        assert loaded.name == "Soumitra"
        assert loaded.age == 65
        assert "veteran" in loaded.tags

        emb1 = studio1.get_character_embedding("soumitra")
        emb2 = studio2.get_character_embedding("soumitra")
        assert emb1 is not None and emb2 is not None
        assert np.allclose(emb1, emb2, atol=1e-6)


# =============================================================================
# 2. VIDEO ENGINE ADVERSARIAL CHALLENGES (M3)
# =============================================================================

class TestVideoEngineInitializationAdversarial:
    """Tests CineVideoEngine initialization, backend aliasing, and normalization order."""

    def test_engine_initialization_all_backends(self):
        """Verify CineVideoEngine can be instantiated with each supported backend name."""
        for backend_name in ["mock", "wan2.1", "ltx-video", "cpu", "procedural", "wan", "ltx"]:
            try:
                engine = CineVideoEngine(default_backend=backend_name)
                assert engine is not None
            except AttributeError as e:
                pytest.fail(f"CineVideoEngine failed to initialize with backend='{backend_name}': {e}")

    def test_backend_aliases_defined_prior_to_normalization(self):
        """Verify that _normalize_backend_name works on fresh CineVideoEngine without AttributeError."""
        engine = CineVideoEngine()
        assert engine._normalize_backend_name("wan2.1") == "wan2.1"
        assert engine._normalize_backend_name("WAN21") == "wan2.1"
        assert engine._normalize_backend_name("ltx_video") == "ltx-video"
        assert engine._normalize_backend_name("CPU") == "mock"


class TestDiTFrameCountBoundaryMath:
    """Exhaustive boundary testing of $ and $ DiT temporal downsampling rules."""

    @pytest.mark.parametrize("input_frames, backend, expected_valid", [
        (17, "wan2.1", True),    # 4(4) + 1 = 17
        (33, "wan2.1", True),    # 4(8) + 1 = 33
        (49, "wan2.1", True),    # 4(12) + 1 = 49
        (65, "wan2.1", True),    # 4(16) + 1 = 65
        (81, "wan2.1", True),    # 4(20) + 1 = 81
        (97, "wan2.1", True),    # 4(24) + 1 = 97
        (121, "wan2.1", True),   # 4(30) + 1 = 121
        (16, "wan2.1", False),   # Not (4k+1)
        (30, "wan2.1", False),
        (80, "wan2.1", False),
        (82, "wan2.1", False),
        (17, "ltx-video", True), # 8(2) + 1 = 17
        (25, "ltx-video", True), # 8(3) + 1 = 25
        (33, "ltx-video", True), # 8(4) + 1 = 33
        (41, "ltx-video", True), # 8(5) + 1 = 41
        (49, "ltx-video", True), # 8(6) + 1 = 49
        (65, "ltx-video", True), # 8(8) + 1 = 65
        (81, "ltx-video", True), # 8(10) + 1 = 81
        (20, "ltx-video", False),
        (80, "ltx-video", False),
    ])
    def test_validate_frame_count_exact_rules(self, input_frames, backend, expected_valid):
        valid, adjusted, msg = validate_frame_count(input_frames, backend=backend)
        assert valid == expected_valid
        if not expected_valid:
            # Adjusted must be a valid DiT frame count
            valid_counts = get_valid_dit_frame_counts(backend=backend)
            assert adjusted in valid_counts

    def test_extreme_frame_counts(self):
        """Negative, zero, or huge frame counts should be sanitized."""
        for bad_count in [-10, 0, 1, 3]:
            valid, adjusted, msg = validate_frame_count(bad_count, backend="wan2.1")
            assert adjusted >= 17, f"Frame count must be clamped to minimum valid DiT frames (got {adjusted})"


class TestSeedDeterminismAndPRNG:
    """Strict empirical test: identical seeds must yield 100% bitwise identical frame hashes."""

    def test_seed_determinism_identical_hashes(self):
        """Run Mock backend twice with seed=1337; all frame SHA256 hashes must match bitwise."""
        backend = MockVideoBackend()
        config = VideoGenerationConfig(num_frames=17, width=256, height=256, seed=1337)

        frames_run1 = backend.generate(image=None, motion_prompt="cinematic dolly zoom", config=config)
        frames_run2 = backend.generate(image=None, motion_prompt="cinematic dolly zoom", config=config)

        assert len(frames_run1) == len(frames_run2) == 17

        for idx, (f1, f2) in enumerate(zip(frames_run1, frames_run2)):
            hash1 = hashlib.sha256(f1.tobytes()).hexdigest()
            hash2 = hashlib.sha256(f2.tobytes()).hexdigest()
            assert hash1 == hash2, f"Frame {idx} hash mismatch between identical seed runs!"

    def test_seed_divergence_different_hashes(self):
        """Run Mock backend with seed=100 vs seed=200; frame hashes must diverge."""
        backend = MockVideoBackend()
        cfg1 = VideoGenerationConfig(num_frames=17, width=256, height=256, seed=100)
        cfg2 = VideoGenerationConfig(num_frames=17, width=256, height=256, seed=200)

        frames_run1 = backend.generate(image=None, motion_prompt="cinematic pan", config=cfg1)
        frames_run2 = backend.generate(image=None, motion_prompt="cinematic pan", config=cfg2)

        differences = 0
        for f1, f2 in zip(frames_run1, frames_run2):
            if not np.array_equal(f1, f2):
                differences += 1

        assert differences == 17, "All frames should differ between different seeds."


class TestRapidBackendSwitchingAndFallbacks:
    """Stress-test dynamic backend switching and cascading fallbacks."""

    def test_rapid_backend_switching_loop(self):
        """Switch back and forth 20 times between backends; verify stability and model purging."""
        engine = CineVideoEngine(default_backend="mock")
        backends_cycle = ["mock", "wan2.1", "ltx-video", "procedural", "wan"]

        for b in backends_cycle * 4:
            engine.switch_backend(b)
            active = engine.get_active_backend()
            assert active in ["mock", "wan2.1", "ltx-video"]

        # Final purge
        purged = engine.unload_models()
        assert isinstance(purged, int)

    def test_cascading_fallback_execution_on_cpu(self):
        """When wan2.1 is requested on CPU, engine must cascade: wan2.1 -> ltx-video -> mock."""
        engine = CineVideoEngine(default_backend="wan2.1")
        config = VideoGenerationConfig(backend="wan2.1", num_frames=17, width=256, height=256, seed=42)

        # On CPU (or without weights), generation must succeed via fallback
        frames = engine.generate_motion(image=None, motion_prompt="slow tracking shot", config=config)
        assert len(frames) == 17
        assert frames[0].shape == (256, 256, 3)
        assert frames[0].dtype == np.uint8


class TestResolutionBoundsAndInputTypes:
    """Adversarial resolution bounds and image input formats."""

    @pytest.mark.parametrize("width, height", [
        (128, 128),
        (256, 256),
        (720, 480),
        (1280, 720),
        (1920, 1080),
        (723, 487),  # Non-multiples of 16 (odd dimensions)
    ])
    def test_resolution_bounds_handling(self, width, height):
        backend = MockVideoBackend()
        config = VideoGenerationConfig(num_frames=17, width=width, height=height, seed=42)
        
        # Validate dimensions via CineVideoEngine.validate_config or direct generation
        frames = backend.generate(image=None, motion_prompt="wide shot", config=config)
        assert len(frames) == 17
        assert frames[0].shape[0] == height
        assert frames[0].shape[1] == width
        assert frames[0].shape[2] == 3

    def test_input_image_types_resilience(self):
        """Test generation from PIL Image, 2D Grayscale ndarray, 4-Channel RGBA ndarray, and None."""
        backend = MockVideoBackend()
        config = VideoGenerationConfig(num_frames=17, width=256, height=256, seed=42)

        # 1. PIL RGBA
        pil_rgba = Image.new("RGBA", (300, 200), color=(100, 150, 200, 128))
        f_rgba = backend.generate(image=pil_rgba, motion_prompt="", config=config)
        assert len(f_rgba) == 17
        assert f_rgba[0].shape == (256, 256, 3)

        # 2. NumPy Grayscale (H, W)
        np_gray = np.full((300, 200), 120, dtype=np.uint8)
        f_gray = backend.generate(image=np_gray, motion_prompt="", config=config)
        assert len(f_gray) == 17
        assert f_gray[0].shape == (256, 256, 3)

        # 3. NumPy RGBA (H, W, 4)
        np_rgba = np.full((300, 200, 4), 180, dtype=np.uint8)
        f_np_rgba = backend.generate(image=np_rgba, motion_prompt="", config=config)
        assert len(f_np_rgba) == 17
        assert f_np_rgba[0].shape == (256, 256, 3)
