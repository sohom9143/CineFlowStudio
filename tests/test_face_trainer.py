"""
Unit and Integration Tests for On-Demand Character Face Adapter Training Engine
==============================================================================
Tests:
1. CharacterFaceAdapter dataclass serialization & deserialization.
2. Synthetic multi-angle and photometric augmentation generation.
3. Unit L2 norm validation (||e||_2 = 1.0) and identity dispersion loss computation.
4. adapter_weights.npz file persistence and reload verification.
5. Status reporting through get_face_adapter_status.
"""

import json
import os
import tempfile
import numpy as np
import pytest
from PIL import Image

from modules.character_engine import (
    CharacterStudio,
    CharacterProfile,
    CharacterFaceAdapter,
    compute_l2_norm,
)


class TestFaceAdapterDataclass:
    def test_face_adapter_to_dict_and_from_dict(self):
        adapter = CharacterFaceAdapter(
            character_id="dev",
            is_trained=True,
            training_loss=0.0125,
            augmentation_count=16,
            trained_at="2026-08-30T12:00:00Z",
            identity_confidence=0.98,
        )
        d = adapter.to_dict()
        assert d["character_id"] == "dev"
        assert d["is_trained"] is True
        assert d["training_loss"] == 0.0125
        assert d["augmentation_count"] == 16

        reconstructed = CharacterFaceAdapter.from_dict(d)
        assert reconstructed.character_id == "dev"
        assert reconstructed.is_trained is True
        assert reconstructed.training_loss == 0.0125


class TestCharacterStudioFaceTrainer:
    @pytest.fixture
    def studio_workspace(self, tmp_path):
        profiles_dir = str(tmp_path / "profiles")
        styles_path = str(tmp_path / "styles.json")
        styles_data = [
            {"id": "imax_realism", "name": "IMAX Realism", "prompt_prefix": "70mm IMAX", "negative_prompt": "blurry"}
        ]
        with open(styles_path, "w", encoding="utf-8") as f:
            json.dump(styles_data, f)
        
        studio = CharacterStudio(profiles_dir=profiles_dir, styles_path=styles_path)
        return studio, profiles_dir

    def test_train_face_adapter_newly_enrolled_character(self, studio_workspace):
        studio, profiles_dir = studio_workspace
        
        # 1. Enroll character with 3 views
        f = Image.new("RGB", (128, 128), (210, 160, 120))
        l = Image.new("RGB", (128, 128), (200, 150, 110))
        r = Image.new("RGB", (128, 128), (215, 165, 125))

        profile = studio.enroll_character(
            name="Vikram",
            description="Intense spaceship captain",
            views={"front": f, "left": l, "right": r},
            gender="male",
        )
        assert profile.id == "vikram"

        # 2. Check initial status
        init_status = studio.get_face_adapter_status("vikram")
        assert init_status["is_trained"] is False

        # 3. Train face adapter with 8x augmentations
        train_result = studio.train_character_face_adapter("vikram", augmentation_factor=8)
        assert train_result["is_trained"] is True
        assert train_result["augmentation_count"] >= 24  # 3 views * (1 + 8) = 27
        assert "training_loss" in train_result
        assert train_result["training_loss"] >= 0.0

        # 4. Verify adapter_weights.npz on disk
        adapter_npz = os.path.join(profiles_dir, "vikram", "adapter_weights.npz")
        assert os.path.exists(adapter_npz)

        with np.load(adapter_npz) as data:
            assert "consensus_embedding" in data
            consensus = data["consensus_embedding"]
            assert np.isclose(np.linalg.norm(consensus), 1.0, atol=1e-5)
            assert "augmented_features" in data
            assert len(data["augmented_features"]) == train_result["augmentation_count"]

        # 5. Verify updated in-memory profile and status
        updated_status = studio.get_face_adapter_status("vikram")
        assert updated_status["is_trained"] is True
        assert "🟢" in updated_status["status_badge"]

    def test_train_face_adapter_nonexistent_character_raises(self, studio_workspace):
        studio, _ = studio_workspace
        with pytest.raises(ValueError, match="not found"):
            studio.train_character_face_adapter("nonexistent_char")
