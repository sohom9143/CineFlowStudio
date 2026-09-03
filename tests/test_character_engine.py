"""
Unit & Integration Test Suite for CineFlow-AI Character Engine & Face Bank (Milestone 2 / R2)
=============================================================================================
Tests:
- CharacterProfile dataclass serialization, deserialization, and alias handling
- Facial embedding mathematics: L2-normalization, consensus fusion, 512-D vectors
- Feature extraction from PIL Images, NumPy arrays, and file paths
- CharacterStudio lifecycle, styles listing, and preset validation (5 cinematic styles)
- Pre-configured Face Bank profiles (Dev, Neel, Meghla, Cha Kaku)
- Hierarchical prompt synthesis and negative prompt deduplication
- Dynamic portrait enrollment with consensus ArcFace vectors and filesystem persistence
- Procedural character keyframe generation with deterministic seeds and style rendering
- VRAM lifecycle stage decorator integration and error handling
"""

import os
import json
import shutil
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
    render_procedural_character_view,
    image_to_base64_data_uri,
)
from modules.memory_manager import VRAMManager


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_workspace(tmp_path):
    """Creates a temporary workspace directory for isolated filesystem testing."""
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
                "description": "70mm IMAX photorealism",
                "prompt_prefix": "70mm IMAX film still, cinematic hyper-realism",
                "prompt_suffix": "8k uhd, dslr, subtle film grain",
                "negative_prompt": "cartoon, 3d, render, illustration, blurry",
                "guidance_scale": 7.5,
                "num_inference_steps": 30,
            },
            {
                "id": "kolkata_vintage",
                "name": "North Kolkata Vintage 35mm",
                "description": "Vintage 35mm Kolkata aesthetic",
                "prompt_prefix": "Kodak Portra 400 35mm film photograph, vintage Kolkata",
                "prompt_suffix": "organic 35mm film grain, muted warm palette",
                "negative_prompt": "digital, ultra-clean, neon, plastic",
                "guidance_scale": 7.0,
                "num_inference_steps": 28,
            },
            {
                "id": "ghibli_anime",
                "name": "Studio Ghibli / 3D Anime Style",
                "description": "Hand-painted anime aesthetic",
                "prompt_prefix": "Studio Ghibli cinematic animation style",
                "prompt_suffix": "cel shading, masterpiece anime art",
                "negative_prompt": "photorealistic, live action photo, noisy film grain",
                "guidance_scale": 8.0,
                "num_inference_steps": 32,
            },
            {
                "id": "cyberpunk_noir",
                "name": "Dark Cyberpunk Noir",
                "description": "Moody neo-noir dystopian aesthetic",
                "prompt_prefix": "Blade Runner 2049 cyberpunk noir style",
                "prompt_suffix": "anamorphic lens streak, atmospheric rain mist",
                "negative_prompt": "daylight, cheerful, pastel colors",
                "guidance_scale": 8.0,
                "num_inference_steps": 30,
            },
            {
                "id": "custom_neutral",
                "name": "Custom / Neutral Cinematic",
                "description": "Neutral baseline",
                "prompt_prefix": "cinematic film still",
                "prompt_suffix": "cinematic lighting, 4k",
                "negative_prompt": "blurry, low quality",
                "guidance_scale": 7.5,
                "num_inference_steps": 25,
            },
        ]
    }
    with open(styles_file, "w", encoding="utf-8") as f:
        json.dump(styles_data, f, indent=2)

    return {
        "root": tmp_path,
        "profiles_dir": str(profiles_dir),
        "styles_path": str(styles_file),
    }


@pytest.fixture
def default_studio():
    """Returns a CharacterStudio instance referencing the default repository files."""
    return CharacterStudio(
        profiles_dir="character_profiles",
        styles_path="configs/cinematic_styles.json",
    )


# =============================================================================
# Test CharacterProfile Dataclass
# =============================================================================

class TestCharacterProfileDataclass:
    def test_character_profile_creation_and_defaults(self):
        profile = CharacterProfile(
            id="test_hero",
            name="Test Hero",
            description="A test character",
        )
        assert profile.id == "test_hero"
        assert profile.name == "Test Hero"
        assert profile.gender == "neutral"
        assert profile.prompt_prefix == ""
        assert profile.negative_prompt == ""
        assert profile.reference_images == []
        assert profile.tags == []

    def test_character_profile_serialization_roundtrip(self):
        profile = CharacterProfile(
            id="dev",
            name="Dev",
            description="Protagonist indie filmmaker",
            gender="male",
            prompt_prefix="cinematic portrait of Dev",
            negative_prompt="blurry, distorted",
            embedding_path="embedding.npy",
            reference_images=["ref_primary.png", "ref_1.png"],
            created_at="2026-08-25T00:00:00Z",
            age=28,
            tags=["protagonist", "filmmaker"],
        )
        d = profile.to_dict()
        assert d["id"] == "dev"
        assert d["default_prompt_prefix"] == "cinematic portrait of Dev"
        assert d["embedding_file"] == "embedding.npy"

        restored = CharacterProfile.from_dict(d)
        assert restored.id == profile.id
        assert restored.name == profile.name
        assert restored.age == 28
        assert restored.gender == "male"
        assert restored.prompt_prefix == "cinematic portrait of Dev"
        assert restored.embedding_path == "embedding.npy"
        assert restored.reference_images == ["ref_primary.png", "ref_1.png"]

    def test_character_profile_alias_handling(self):
        legacy_data = {
            "id": "legacy_char",
            "name": "Legacy Character",
            "description": "Legacy format description",
            "default_prompt_prefix": "legacy prompt prefix",
            "embedding_file": "custom_embedding.npy",
            "reference_images": "single_image.png",
            "age": "35",
        }
        p = CharacterProfile.from_dict(legacy_data)
        assert p.id == "legacy_char"
        assert p.prompt_prefix == "legacy prompt prefix"
        assert p.embedding_path == "custom_embedding.npy"
        assert p.reference_images == ["single_image.png"]
        assert p.age == 35


# =============================================================================
# Test Embedding Mathematics & Normalization
# =============================================================================

class TestEmbeddingMathematics:
    def test_l2_normalize_single_vector(self):
        rng = np.random.RandomState(42)
        raw_vec = rng.randn(512).astype(np.float32)
        norm_before = compute_l2_norm(raw_vec)
        assert norm_before > 0.0

        normalized = l2_normalize(raw_vec)
        assert normalized.shape == (512,)
        assert normalized.dtype == np.float32
        assert np.isclose(compute_l2_norm(normalized), 1.0, atol=1e-6)

    def test_l2_normalize_zero_vector_fallback(self):
        zero_vec = np.zeros(512, dtype=np.float32)
        normalized = l2_normalize(zero_vec)
        assert normalized.shape == (512,)
        assert np.isclose(compute_l2_norm(normalized), 1.0, atol=1e-6)

    def test_fuse_consensus_embeddings_single(self):
        rng = np.random.RandomState(101)
        v1 = rng.randn(512).astype(np.float32) * 5.0
        consensus = fuse_consensus_embeddings([v1])

        assert consensus.shape == (512,)
        assert np.isclose(compute_l2_norm(consensus), 1.0, atol=1e-6)
        expected = l2_normalize(v1)
        np.testing.assert_allclose(consensus, expected, atol=1e-5)

    def test_fuse_consensus_embeddings_multiple(self):
        rng = np.random.RandomState(202)
        vectors = [rng.randn(512).astype(np.float32) for _ in range(4)]
        consensus = fuse_consensus_embeddings(vectors)

        assert consensus.shape == (512,)
        assert np.isclose(compute_l2_norm(consensus), 1.0, atol=1e-6)

        # Verify math manually: mean of normalized vectors, then re-normalized
        normalized_inputs = [l2_normalize(v) for v in vectors]
        mean_vec = np.mean(normalized_inputs, axis=0)
        expected = l2_normalize(mean_vec)
        np.testing.assert_allclose(consensus, expected, atol=1e-5)

    def test_fuse_consensus_embeddings_empty_raises(self):
        with pytest.raises(ValueError, match="Cannot fuse consensus embeddings from an empty list"):
            fuse_consensus_embeddings([])

    def test_fuse_consensus_embeddings_invalid_dim_raises(self):
        invalid_vec = np.ones(128, dtype=np.float32)
        with pytest.raises(ValueError, match="Expected 512-dimensional embedding vector"):
            fuse_consensus_embeddings([invalid_vec])


# =============================================================================
# Test Feature Extraction From Images
# =============================================================================

class TestFeatureExtraction:
    def test_extract_from_pil_image(self):
        img = Image.new("RGB", (256, 256), color=(120, 150, 180))
        emb = extract_facial_embedding_from_image(img)
        assert emb.shape == (512,)
        assert emb.dtype == np.float32
        assert np.isclose(compute_l2_norm(emb), 1.0, atol=1e-6)

    def test_extract_from_numpy_array(self):
        arr = np.full((128, 128, 3), 100, dtype=np.uint8)
        emb = extract_facial_embedding_from_image(arr)
        assert emb.shape == (512,)
        assert np.isclose(compute_l2_norm(emb), 1.0, atol=1e-6)

    def test_extract_from_rgba_numpy_array(self):
        arr = np.full((100, 100, 4), 200, dtype=np.uint8)
        emb = extract_facial_embedding_from_image(arr)
        assert emb.shape == (512,)
        assert np.isclose(compute_l2_norm(emb), 1.0, atol=1e-6)

    def test_extract_from_image_file_path(self, tmp_path):
        img_path = str(tmp_path / "test_face.png")
        img = Image.new("RGB", (200, 200), color=(80, 100, 120))
        img.save(img_path)

        emb = extract_facial_embedding_from_image(img_path)
        assert emb.shape == (512,)
        assert np.isclose(compute_l2_norm(emb), 1.0, atol=1e-6)

    def test_extract_from_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            extract_facial_embedding_from_image("non_existent_file_123.jpg")

    def test_extract_from_invalid_type_raises(self):
        with pytest.raises(TypeError):
            extract_facial_embedding_from_image(12345)  # type: ignore


# =============================================================================
# Test CharacterStudio Styles & Pre-Configured Face Bank
# =============================================================================

class TestCharacterStudioPreconfigured:
    def test_styles_presets_loaded(self, default_studio):
        styles = default_studio.list_styles()
        assert len(styles) >= 5

        style_ids = [s["id"] for s in styles]
        expected_ids = ["imax_realism", "kolkata_vintage", "ghibli_anime", "custom_neutral"]
        for expected in expected_ids:
            assert expected in style_ids
        assert ("scifi_cyberpunk" in style_ids or "cyberpunk_noir" in style_ids)

    def test_get_style_resolution(self, default_studio):
        style = default_studio.get_style("kolkata_vintage")
        assert style is not None
        assert "North Kolkata Vintage 35mm" in style["name"]
        assert "Portra" in style["prompt_prefix"]
        assert style["guidance_scale"] > 0

        # Case-insensitive lookup
        style_upper = default_studio.get_style("IMAX_REALISM")
        assert style_upper is not None
        assert style_upper["id"] == "imax_realism"

        # Lookup by display name
        style_by_name = default_studio.get_style("dark cyberpunk noir")
        assert style_by_name is not None
        assert style_by_name["id"] in ("cyberpunk_noir", "scifi_cyberpunk")

    def test_user_based_clean_state(self, default_studio):
        """Verifies CineFlow has purged legacy prebuilt characters."""
        char_ids = [c.id for c in default_studio.list_characters()]
        assert "dev" not in char_ids
        assert "neel" not in char_ids
        assert "meghla" not in char_ids
        assert "cha_kaku" not in char_ids

    def test_user_character_enrollment_and_tree(self, default_studio):
        """Tests dynamic user-based enrollment and automatic facial consistency tree generation."""
        kira = default_studio.enroll_character(
            name="Kira Thorne",
            description="Cyberpunk courier with neon cyan hair and amber eyes",
            gender="female",
            tags=["cyberpunk", "protagonist"],
            default_prompt_prefix="cinematic portrait of Kira Thorne",
        )
        try:
            assert kira is not None
            assert kira.id == "kira_thorne"
            assert kira.name == "Kira Thorne"
            assert kira.gender == "female"

            # Check 512-D normalized consensus embedding
            emb = default_studio.get_character_embedding("kira_thorne")
            assert emb is not None
            assert emb.shape == (512,)
            assert np.isclose(compute_l2_norm(emb), 1.0, atol=1e-5)

            # Check facial consistency JSON tree
            tree = kira.facial_consistency_tree
            assert tree is not None
            assert "consensus_embedding" in tree
            assert len(tree["consensus_embedding"]) == 512
            assert "keyframe_anchors" in tree
            assert "grit" in tree["keyframe_anchors"]
            assert "action" in tree["keyframe_anchors"]
            assert "dialogue" in tree["keyframe_anchors"]
            assert "noir" in tree["keyframe_anchors"]
            assert tree["identity_confidence"] >= 0.85
            assert "wardrobe_lock" in tree
            assert "lighting_blend" in tree
            assert "voice_profile" in tree
        finally:
            default_studio.db.delete_character("kira_thorne")
            default_studio.reload_profiles()


# =============================================================================
# Test Hierarchical Prompt Synthesis
# =============================================================================

class TestPromptSynthesis:
    def test_prompt_synthesis_full_hierarchy(self, default_studio):
        test_char = default_studio.enroll_character(
            name="Alex",
            description="Cinematic test protagonist",
            gender="neutral",
            default_prompt_prefix="cinematic portrait of Alex",
        )
        try:
            pos_prompt, neg_prompt = default_studio.synthesize_prompt(
                character_id=test_char.id,
                scene_prompt="standing on Howrah bridge under rain",
                style_id="kolkata_vintage",
                custom_modifiers="dramatic lighting, cinematic composition",
            )

            # Style prefix should appear first
            assert pos_prompt.startswith("Kodak Portra 400 35mm")
            # Character prefix included
            assert "cinematic portrait of Alex" in pos_prompt
            # Scene prompt included
            assert "standing on Howrah bridge under rain" in pos_prompt
            assert "dramatic lighting, cinematic composition" in pos_prompt
            # Style suffix included at end
            assert pos_prompt.endswith("Satyajit Ray cinematic framing")

            # Negative prompt should merge character and style negatives
            assert "blurry" in neg_prompt
            assert "digital" in neg_prompt
            # Check no consecutive double commas
            assert ", ," not in pos_prompt
            assert ", ," not in neg_prompt
        finally:
            default_studio.db.delete_character(test_char.id)
            default_studio.reload_profiles()

    def test_prompt_synthesis_negative_prompt_deduplication(self, default_studio):
        pos, neg = default_studio.synthesize_prompt(
            character_id=None,
            scene_prompt="walking in alley",
            style_id="imax_realism",
        )
        tokens = [t.strip().lower() for t in neg.split(",") if t.strip()]
        # Verify deduplication
        assert len(tokens) == len(set(tokens))

    def test_prompt_synthesis_unknown_character_or_style(self, default_studio):
        pos, neg = default_studio.synthesize_prompt(
            character_id="unknown_char",
            scene_prompt="a simple scene",
            style_id="unknown_style",
        )
        # Should gracefully fall back
        assert "a simple scene" in pos
        assert isinstance(neg, str)


# =============================================================================
# Test Dynamic Face Enrollment
# =============================================================================

class TestDynamicFaceEnrollment:
    def test_enroll_character_single_image(self, temp_workspace):
        studio = CharacterStudio(
            profiles_dir=temp_workspace["profiles_dir"],
            styles_path=temp_workspace["styles_path"],
        )

        test_img = Image.new("RGB", (256, 256), color=(140, 90, 60))
        profile = studio.enroll_character(
            name="Ayan Banerjee",
            description="Lead actor in thriller series",
            images=[test_img],
            gender="male",
            age=30,
            tags=["actor", "thriller"],
        )

        assert profile.id == "ayan_banerjee"
        assert profile.name == "Ayan Banerjee"
        assert profile.age == 30
        assert profile.gender == "male"
        assert len(profile.reference_images) == 1

        # Check filesystem persistence
        profile_dir = os.path.join(temp_workspace["profiles_dir"], "ayan_banerjee")
        assert os.path.exists(profile_dir)
        assert os.path.exists(os.path.join(profile_dir, "profile.json"))
        assert os.path.exists(os.path.join(profile_dir, "embedding.npy"))
        assert os.path.exists(os.path.join(profile_dir, "ref_primary.png"))

        # Check embedding properties
        saved_emb = np.load(os.path.join(profile_dir, "embedding.npy"))
        assert saved_emb.shape == (512,)
        assert np.isclose(compute_l2_norm(saved_emb), 1.0, atol=1e-6)

        # Check cached retrieval
        cached = studio.get_character("ayan_banerjee")
        assert cached is not None
        assert cached.name == "Ayan Banerjee"
        cached_emb = studio.get_character_embedding("ayan_banerjee")
        assert cached_emb is not None
        np.testing.assert_allclose(saved_emb, cached_emb)

    def test_enroll_character_multi_image_consensus(self, temp_workspace):
        studio = CharacterStudio(
            profiles_dir=temp_workspace["profiles_dir"],
            styles_path=temp_workspace["styles_path"],
        )

        img1 = Image.new("RGB", (300, 300), color=(100, 120, 150))
        img2 = np.full((250, 250, 3), 130, dtype=np.uint8)
        img3 = Image.new("RGB", (200, 200), color=(90, 110, 140))

        profile = studio.enroll_character(
            name="Rupa Sen",
            description="Documentary producer",
            images=[img1, img2, img3],
            gender="female",
            age=29,
        )

        assert profile.id == "rupa_sen"
        assert len(profile.reference_images) == 3

        emb = studio.get_character_embedding("rupa_sen")
        assert emb is not None
        assert emb.shape == (512,)
        assert np.isclose(compute_l2_norm(emb), 1.0, atol=1e-5)

    def test_enroll_character_validation_errors(self, temp_workspace):
        studio = CharacterStudio(
            profiles_dir=temp_workspace["profiles_dir"],
            styles_path=temp_workspace["styles_path"],
        )

        # Empty name
        with pytest.raises(ValueError, match="Character name cannot be empty"):
            studio.enroll_character(name="", description="test", images=[Image.new("RGB", (50, 50))])

        # Empty images list
        with pytest.raises(ValueError, match="At least 1 portrait image must be provided"):
            studio.enroll_character(name="Test Char", description="test", images=[])


# =============================================================================
# Test Character Frame Generation & VRAM Lifecycle
# =============================================================================

class TestCharacterFrameGeneration:
    def test_generate_procedural_frame_dimensions_and_mode(self, default_studio):
        frame = default_studio.generate_character_frame(
            character_id="dev",
            scene_prompt="sitting at vintage tea stall",
            style_id="kolkata_vintage",
            width=720,
            height=480,
            seed=42,
        )

        assert isinstance(frame, Image.Image)
        assert frame.mode == "RGB"
        assert frame.size == (720, 480)

    def test_generate_frame_deterministic_seed(self, default_studio):
        frame1 = default_studio.generate_character_frame(
            character_id="meghla",
            scene_prompt="investigating old mansion",
            style_id="imax_realism",
            width=512,
            height=512,
            seed=12345,
        )
        frame2 = default_studio.generate_character_frame(
            character_id="meghla",
            scene_prompt="investigating old mansion",
            style_id="imax_realism",
            width=512,
            height=512,
            seed=12345,
        )
        frame3 = default_studio.generate_character_frame(
            character_id="meghla",
            scene_prompt="investigating old mansion",
            style_id="imax_realism",
            width=512,
            height=512,
            seed=99999,
        )

        arr1 = np.asarray(frame1)
        arr2 = np.asarray(frame2)
        arr3 = np.asarray(frame3)

        # Same seed must be 100% byte identical
        np.testing.assert_array_equal(arr1, arr2)
        # Different seeds must produce different pixels
        assert not np.array_equal(arr1, arr3)

    def test_generate_frame_across_all_cinematic_styles(self, default_studio):
        styles = ["imax_realism", "kolkata_vintage", "ghibli_anime", "cyberpunk_noir", "custom_neutral"]
        for style_id in styles:
            frame = default_studio.generate_character_frame(
                character_id="dev",
                scene_prompt="cinematic portrait shot",
                style_id=style_id,
                width=640,
                height=360,
                seed=777,
            )
            assert isinstance(frame, Image.Image)
            assert frame.size == (640, 360)

    def test_vram_lifecycle_isolation_during_generation(self, default_studio):
        mgr = VRAMManager.get_instance()
        assert mgr.current_stage is None

        # Execute frame generation wrapped in @vram_lifecycle_stage("character_generation")
        frame = default_studio.generate_character_frame(
            character_id="cha_kaku",
            scene_prompt="pouring hot tea from kettle",
            style_id="kolkata_vintage",
            seed=888,
        )
        assert isinstance(frame, Image.Image)
        # Stage must be cleanly exited and reset to None
        assert mgr.current_stage is None


# =============================================================================
# Test Slug Sanitization & Edge Cases
# =============================================================================

class TestCharacterEngineEdgeCases:
    def test_sanitize_character_slug(self):
        assert sanitize_character_slug("Dev Banerjee") == "dev_banerjee"
        assert sanitize_character_slug("Cha Kaku (North Kolkata)") == "cha_kaku_north_kolkata"
        assert sanitize_character_slug("  $$Special#Character!!  ") == "special_character"
        assert sanitize_character_slug("") == "character"

    def test_corrupted_profile_json_gracefully_handled(self, temp_workspace):
        studio = CharacterStudio(
            profiles_dir=temp_workspace["profiles_dir"],
            styles_path=temp_workspace["styles_path"],
        )
        # Create corrupted profile directory
        corrupt_dir = os.path.join(temp_workspace["profiles_dir"], "corrupt_char")
        os.makedirs(corrupt_dir, exist_ok=True)
        with open(os.path.join(corrupt_dir, "profile.json"), "w") as f:
            f.write("{invalid json syntax,,")

        # Reload should not crash
        studio.reload_profiles()
        assert "corrupt_char" not in [c.id for c in studio.list_characters()]

    def test_enrollment_with_5_images_and_grayscale_mode(self, temp_workspace):
        studio = CharacterStudio(
            profiles_dir=temp_workspace["profiles_dir"],
            styles_path=temp_workspace["styles_path"],
        )
        # Create 5 images of different modes
        imgs = [
            Image.new("RGB", (100, 100), (10, 20, 30)),
            Image.new("L", (120, 120), 128),
            Image.new("RGBA", (140, 140), (40, 50, 60, 200)),
            np.zeros((150, 150, 3), dtype=np.uint8),
            Image.new("RGB", (160, 160), (70, 80, 90)),
        ]
        profile = studio.enroll_character(
            name="Max Images Char",
            description="Character with 5 distinct reference images",
            images=imgs,
        )
        assert len(profile.reference_images) == 5
        emb = studio.get_character_embedding("max_images_char")
        assert emb is not None
        assert np.isclose(compute_l2_norm(emb), 1.0, atol=1e-5)

    def test_missing_styles_config_fallback(self, tmp_path):
        # Point to non-existent file
        non_existent = str(tmp_path / "does_not_exist_styles.json")
        studio = CharacterStudio(
            profiles_dir=str(tmp_path / "profiles"),
            styles_path=non_existent,
        )
        styles = studio.list_styles()
        assert len(styles) >= 5
        assert studio.get_style("imax_realism") is not None

    def test_frame_generation_various_resolutions(self, default_studio):
        resolutions = [(480, 320), (720, 480), (1080, 720), (256, 256)]
        for w, h in resolutions:
            frame = default_studio.generate_character_frame(
                character_id="dev",
                scene_prompt="test prompt",
                width=w,
                height=h,
                seed=42,
            )
            assert frame.size == (w, h)


# =============================================================================
# Test Multi-Angle 3/4-Side Views & 360° Character Turntable Viewer
# =============================================================================

class TestMultiAngle360Viewer:
    def test_enroll_character_with_3_sides_dict(self, temp_workspace):
        studio = CharacterStudio(
            profiles_dir=temp_workspace["profiles_dir"],
            styles_path=temp_workspace["styles_path"],
        )

        front_img = Image.new("RGB", (200, 200), (220, 180, 150))
        left_img = Image.new("RGB", (200, 200), (200, 160, 140))
        right_img = Image.new("RGB", (200, 200), (210, 170, 145))

        profile = studio.enroll_character(
            name="Satyajit Hero",
            description="3-angle character for 360 inspection",
            views={
                "front": front_img,
                "left": left_img,
                "right": right_img,
            },
            gender="male",
        )

        assert profile.id == "satyajit_hero"
        assert "front" in profile.views
        assert "left" in profile.views
        assert "right" in profile.views
        assert profile.views["front"] == "ref_front.png"
        assert profile.views["left"] == "ref_left.png"
        assert profile.views["right"] == "ref_right.png"

        # Check embedding normalization
        emb = studio.get_character_embedding("satyajit_hero")
        assert emb is not None
        assert np.isclose(compute_l2_norm(emb), 1.0, atol=1e-5)

        # Check get_character_views
        views = studio.get_character_views("satyajit_hero")
        assert "front" in views
        assert "left" in views
        assert "right" in views
        assert "back" in views  # Synthesized back

        # Check 360 turntable frames
        frames = studio.generate_360_turntable_frames("satyajit_hero", num_frames=16, target_size=(256, 256))
        assert len(frames) == 16
        for f in frames:
            assert isinstance(f, Image.Image)
            assert f.size == (256, 256)

        # Check 360 HTML widget generation
        html = studio.generate_360_viewer_html("satyajit_hero", width=300, height=300, num_frames=16)
        assert "Satyajit Hero" in html
        assert "360° Turntable" in html
        assert "data:image/jpeg;base64," in html
        assert "slider_" in html
        assert "snapAngle_" in html
        assert "togglePlay_" in html

    def test_enroll_character_with_4_sides_keywords(self, temp_workspace):
        studio = CharacterStudio(
            profiles_dir=temp_workspace["profiles_dir"],
            styles_path=temp_workspace["styles_path"],
        )

        f_img = Image.new("RGB", (150, 150), (240, 190, 160))
        l_img = Image.new("RGB", (150, 150), (220, 180, 150))
        r_img = Image.new("RGB", (150, 150), (230, 185, 155))
        b_img = Image.new("RGB", (150, 150), (40, 30, 30))

        profile = studio.enroll_character(
            name="Ananya Roy",
            description="4-angle character with full 360 coverage",
            image_front=f_img,
            image_left=l_img,
            image_right=r_img,
            image_back=b_img,
            gender="female",
        )

        assert profile.id == "ananya_roy"
        assert len(profile.views) == 4
        assert "front" in profile.views
        assert "left" in profile.views
        assert "right" in profile.views
        assert "back" in profile.views

        # Check files exist on disk
        char_dir = os.path.join(temp_workspace["profiles_dir"], "ananya_roy")
        assert os.path.exists(os.path.join(char_dir, "ref_front.png"))
        assert os.path.exists(os.path.join(char_dir, "ref_left.png"))
        assert os.path.exists(os.path.join(char_dir, "ref_right.png"))
        assert os.path.exists(os.path.join(char_dir, "ref_back.png"))
        assert os.path.exists(os.path.join(char_dir, "ref_primary.png"))
        assert os.path.exists(os.path.join(char_dir, "embedding.npy"))
        assert os.path.exists(os.path.join(char_dir, "profile.json"))

        # Verify JSON serialization includes views
        with open(os.path.join(char_dir, "profile.json"), "r") as pf:
            p_data = json.load(pf)
        assert "views" in p_data
        assert p_data["views"]["front"] == "ref_front.png"

    def test_default_face_bank_360_turntable(self, default_studio):
        characters = default_studio.list_characters()
        assert len(characters) >= 4

        for char in characters:
            views = default_studio.get_character_views(char.id)
            assert "front" in views
            assert "left" in views
            assert "right" in views
            assert "back" in views

            frames = default_studio.generate_360_turntable_frames(char.id, num_frames=8, target_size=(200, 200))
            assert len(frames) == 8

            html = default_studio.generate_360_viewer_html(char.id, width=300, height=300, num_frames=8)
            assert char.name in html
            assert "360° Turntable" in html

    def test_procedural_multi_angle_rendering(self):
        for angle in ["front", "left", "right", "back"]:
            view_img = render_procedural_character_view(
                character_id="test_char",
                view_angle=angle,
                width=240,
                height=240,
            )
            assert isinstance(view_img, Image.Image)
            assert view_img.size == (240, 240)
            assert view_img.mode == "RGB"

    def test_enroll_character_validation_errors(self, temp_workspace):
        studio = CharacterStudio(
            profiles_dir=temp_workspace["profiles_dir"],
            styles_path=temp_workspace["styles_path"],
        )

        with pytest.raises(ValueError, match="Character name cannot be empty"):
            studio.enroll_character(name="", description="Empty", images=[Image.new("RGB", (50, 50))])

        with pytest.raises(ValueError, match="At least 1 portrait image"):
            studio.enroll_character(name="Valid Name", description="No images", images=[])


