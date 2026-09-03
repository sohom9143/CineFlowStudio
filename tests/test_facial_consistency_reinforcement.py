"""
tests/test_facial_consistency_reinforcement.py

Comprehensive test suite verifying:
1. User-based character architecture (zero legacy prebuilt characters).
2. MongoDB / Local JSON Document Store consistency.
3. 512-D ArcFace Consensus Vector and Biometric Consistency Tree schema.
4. Adaptive facial reinforcement on every generated video ("stronger with every video").
5. Senior Designer Web Studio REST APIs and WebSocket/HTTP contracts.
"""

import os
import shutil
import tempfile
import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from modules.character_database import CharacterDatabase, get_character_database
from modules.character_engine import CharacterStudio, CharacterProfile, compute_l2_norm
from app import CineFlowApp, build_fastapi_app


@pytest.fixture
def temp_db_dir(tmp_path):
    """Provides an isolated directory and clean local database file."""
    storage_dir = tmp_path / "database"
    profiles_dir = tmp_path / "character_profiles"
    storage_dir.mkdir(parents=True, exist_ok=True)
    profiles_dir.mkdir(parents=True, exist_ok=True)
    return {
        "storage_dir": str(storage_dir),
        "profiles_dir": str(profiles_dir),
    }


@pytest.fixture
def clean_db(temp_db_dir):
    """Creates a clean CharacterDatabase instance."""
    return CharacterDatabase(
        storage_dir=temp_db_dir["storage_dir"],
        profiles_dir=temp_db_dir["profiles_dir"],
        mongo_uri=None,
    )


class TestFacialConsistencyTreeAndDatabase:
    def test_database_crud_operations(self, clean_db):
        # Clean initial state
        assert len(clean_db.list_characters()) == 0

        # Save character
        char_data = {
            "id": "kira_thorne",
            "name": "Kira Thorne",
            "gender": "female",
            "description": "Cyberpunk courier",
            "tags": ["cyberpunk", "protagonist"],
        }
        clean_db.save_character(char_data)
        assert clean_db.count() == 1

        # Retrieve character
        retrieved = clean_db.get_character("kira_thorne")
        assert retrieved is not None
        assert retrieved["name"] == "Kira Thorne"
        assert retrieved["gender"] == "female"

        # Delete character
        deleted = clean_db.delete_character("kira_thorne")
        assert deleted is True
        assert clean_db.count() == 0

    def test_build_facial_consistency_tree_schema(self, clean_db):
        fake_emb = np.random.randn(512).astype(np.float32)
        fake_emb = fake_emb / np.linalg.norm(fake_emb)

        tree = clean_db.build_facial_consistency_tree(
            character_id="kira_thorne",
            name="Kira Thorne",
            embedding=fake_emb,
            views={"front": "ref_front.png", "left": "ref_left.png"},
            traits={"facial_structure": "sharp jawline", "complexion_and_skin": "neon rim"},
        )

        assert tree["version"] == "2.0"
        assert tree["character_id"] == "kira_thorne"
        assert tree["identity_confidence"] == 0.88
        assert tree["generation_reinforcement_count"] == 0
        assert len(tree["consensus_embedding"]) == 512
        assert np.isclose(np.linalg.norm(np.array(tree["consensus_embedding"])), 1.0, atol=1e-5)

        # 4 Keyframe Anchor Poses
        anchors = tree["keyframe_anchors"]
        assert "grit" in anchors
        assert "action" in anchors
        assert "dialogue" in anchors
        assert "noir" in anchors

        # Wardrobe & Lighting Blends
        assert "wardrobe_lock" in tree
        assert tree["wardrobe_lock"]["lock_fidelity"] == 0.94
        assert "lighting_blend" in tree
        assert tree["lighting_blend"]["blend_range"] == 0.72

        # Voice Profile
        assert "voice_profile" in tree

    def test_adaptive_facial_reinforcement_strengthening(self, clean_db):
        """Tests that every generated video deepens character identity fidelity."""
        fake_emb = np.ones(512, dtype=np.float32)
        fake_emb = fake_emb / np.linalg.norm(fake_emb)

        tree = clean_db.build_facial_consistency_tree(
            character_id="kira_thorne",
            name="Kira Thorne",
            embedding=fake_emb,
        )
        clean_db.save_character({
            "id": "kira_thorne",
            "name": "Kira Thorne",
            "facial_consistency_tree": tree,
        })

        initial_conf = tree["identity_confidence"]
        assert initial_conf == 0.88
        assert tree["generation_reinforcement_count"] == 0

        # Shot 1 generated
        gen_emb_1 = np.ones(512, dtype=np.float32)
        gen_emb_1 = gen_emb_1 / np.linalg.norm(gen_emb_1)
        tree_1 = clean_db.reinforce_character_facial_consistency(
            character_id="kira_thorne",
            generated_frame_embedding=gen_emb_1,
            prompt="Kira walking in neon rain",
            shot_metadata={"shot_id": "shot_01", "engine": "Wan 2.1 DiT"},
        )

        assert tree_1["generation_reinforcement_count"] == 1
        assert tree_1["identity_confidence"] > initial_conf
        assert len(tree_1["reinforcement_history"]) == 2  # 1 initial enrollment + 1 shot
        assert tree_1["reinforcement_history"][-1]["prompt"] == "Kira walking in neon rain"

        # Shot 2 generated
        tree_2 = clean_db.reinforce_character_facial_consistency(
            character_id="kira_thorne",
            prompt="Kira speaking in noir alleyway",
            shot_metadata={"shot_id": "shot_02", "engine": "Wan 2.1 DiT"},
        )

        assert tree_2["generation_reinforcement_count"] == 2
        assert tree_2["identity_confidence"] > tree_1["identity_confidence"]
        assert len(tree_2["reinforcement_history"]) == 3  # 1 enrollment + 2 shots
        assert tree_2["reinforcement_history"][-1]["prompt"] == "Kira speaking in noir alleyway"


class TestStudioFastAPIRoutes:
    @pytest.fixture
    def test_app(self, tmp_path):
        app_inst = CineFlowApp()
        fastapi_app = build_fastapi_app(app_inst)
        return TestClient(fastapi_app)

    def test_root_serves_studio_html(self, test_app):
        res = test_app.get("/")
        assert res.status_code == 200
        assert "Synthai AI" in res.text or "CineFlow" in res.text or "Digital Actors" in res.text

    def test_telemetry_endpoint(self, test_app):
        res = test_app.get("/api/telemetry")
        assert res.status_code == 200
        data = res.json()
        assert "gpu" in data
        assert "credits" in data
        assert data["credits"] == 4850

    def test_characters_rest_crud(self, test_app):
        # 1. Enroll new digital actor via REST API
        enroll_res = test_app.post(
            "/api/characters",
            data={
                "name": "Marcus Vance",
                "tag": "@vance-synth",
                "gender": "Male",
                "description": "Hard-boiled cyber detective",
                "voice_tone": "ElevenLabs: Low Gritty",
            },
        )
        assert enroll_res.status_code == 200
        actor = enroll_res.json()
        assert actor["id"] == "marcus_vance"
        assert actor["name"] == "Marcus Vance"
        assert "facial_consistency_tree" in actor
        assert actor["facial_consistency_tree"]["identity_confidence"] >= 0.85

        # 2. Get character by ID
        get_res = test_app.get(f"/api/characters/{actor['id']}")
        assert get_res.status_code == 200
        assert get_res.json()["name"] == "Marcus Vance"

        # 3. Clean up
        del_res = test_app.delete(f"/api/characters/{actor['id']}")
        assert del_res.status_code == 200
