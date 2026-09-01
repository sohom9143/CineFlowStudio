"""
Unit and Integration Tests for Gemini Multimodal AI Agent & Character Intelligence
=================================================================================
Tests:
1. Model alias resolution & API key status reporting.
2. Multimodal character feature extraction across single & 360 multi-angle photos.
3. Offline heuristic fallback consistency and tag string generation.
4. Mocked online Gemini multimodal API execution and JSON response schema validation.
5. Director-grade scene prompt refinement with character trait conditioning.
6. End-to-end CharacterStudio and CharacterProfile integration.
"""

import json
import os
import tempfile
import unittest.mock as mock
from typing import Dict, Any

import numpy as np
import pytest
from PIL import Image

from modules.agent_gemini import (
    CharacterGeminiAgent,
    resolve_gemini_model_name,
    pil_image_to_jpeg_bytes,
)
from modules.character_engine import CharacterProfile, CharacterStudio, compute_l2_norm


# =============================================================================
# Test Model Resolution & Status
# =============================================================================

class TestGeminiAgentInitializationAndModelResolution:
    def test_model_alias_resolution(self):
        assert resolve_gemini_model_name("gemini-3.5-flash") == "gemini-2.5-flash"
        assert resolve_gemini_model_name("3.5-flash") == "gemini-2.5-flash"
        assert resolve_gemini_model_name("gemini-2.5-flash") == "gemini-2.5-flash"
        assert resolve_gemini_model_name("gemini-2.0-flash") == "gemini-2.0-flash"
        assert resolve_gemini_model_name("gemini-1.5-flash") == "gemini-1.5-flash"
        assert resolve_gemini_model_name("gemini-1.5-pro") == "gemini-1.5-pro"
        assert resolve_gemini_model_name("") == "gemini-2.5-flash"

    def test_agent_status_without_api_key(self):
        agent = CharacterGeminiAgent(api_key=None, model_name="gemini-2.5-flash")
        agent.api_key = None
        status = agent.get_status()
        assert status["enabled"] is True
        assert status["has_api_key"] is False
        assert status["is_available"] is False
        assert "Heuristic" in status["mode"]

    def test_agent_set_api_key(self):
        agent = CharacterGeminiAgent(api_key=None)
        agent.set_api_key("AIzaSyTestMockKey1234567890")
        assert agent.api_key == "AIzaSyTestMockKey1234567890"
        status = agent.get_status()
        assert status["has_api_key"] is True
        assert "AIza" in status["masked_key"]


# =============================================================================
# Test Character Multimodal Trait Extraction (Heuristic & Mocked Online)
# =============================================================================

class TestGeminiAgentMultimodalCharacterAnalysis:
    def test_heuristic_single_photo_analysis(self):
        agent = CharacterGeminiAgent(api_key=None)
        # Create test portrait
        img = Image.new("RGB", (300, 300), color=(210, 160, 120))
        
        traits = agent.analyze_character_multimodal(
            images_or_views=img,
            character_name="Anirban",
            context_hints="Charismatic Kolkata detective in tailored trenchcoat",
            gender_hint="male",
        )

        assert "prompt_prefix" in traits
        assert "Anirban" in traits["prompt_prefix"]
        assert "facial_structure" in traits
        assert "eyes_and_gaze" in traits
        assert "hair_and_grooming" in traits
        assert "complexion_and_skin" in traits
        assert "wardrobe_and_culture" in traits
        assert "character_tag_string" in traits
        assert traits["agent_mode"] == "heuristic_fallback"

    def test_heuristic_360_multi_angle_analysis(self):
        agent = CharacterGeminiAgent(api_key=None)
        views = {
            "front": Image.new("RGB", (200, 200), (220, 170, 130)),
            "left": Image.new("RGB", (200, 200), (210, 165, 125)),
            "right": Image.new("RGB", (200, 200), (215, 168, 128)),
            "back": Image.new("RGB", (200, 200), (30, 20, 20)),
        }

        traits = agent.analyze_character_multimodal(
            images_or_views=views,
            character_name="Debolina",
            context_hints="Classical vocalist in traditional handloom saree",
            gender_hint="female",
        )

        assert "Debolina" in traits["prompt_prefix"]
        assert "woman" in traits["prompt_prefix"] or "Debolina" in traits["character_synopsis"]
        assert len(traits["character_tag_string"]) > 5

    def test_mocked_online_gemini_vision_call(self):
        agent = CharacterGeminiAgent(api_key="AIzaSyMockKeyForTesting123456789")
        
        mock_response_json = {
            "facial_structure": "Sharp chiseled jawline with prominent cheekbones and straight nose bridge",
            "eyes_and_gaze": "Intense dark almond eyes with piercing focused cinematic gaze",
            "hair_and_grooming": "Wavy raven-black hair parted neatly with clean sideburns",
            "complexion_and_skin": "Warm olive-golden complexion with subtle filmic micro-texture",
            "wardrobe_and_culture": "Dark tailored indigo blazer with crisp white open collar shirt",
            "cinematic_presence": "Magnetic, enigmatic aura of an intellectual film noir protagonist",
            "prompt_prefix": "cinematic portrait of Dev, sharp chiseled jawline, intense dark almond eyes, wavy raven-black hair, warm olive-golden skin, 8k resolution, IMAX 70mm",
            "negative_prompt_additions": "plastic skin, asymmetrical eyes, cartoonish lighting",
            "character_synopsis": "Dev, a charismatic 30s Bengali protagonist with sharp features.",
            "character_tag_string": "Dev, sharp jawline, dark almond eyes, raven hair, indigo blazer",
        }

        mock_client = mock.MagicMock()
        mock_response = mock.MagicMock()
        mock_response.text = json.dumps(mock_response_json)
        mock_client.generate_content.return_value = mock_response
        agent._client = mock_client
        agent.enabled = True

        with mock.patch.object(agent, "is_available", return_value=True):
            img = Image.new("RGB", (100, 100), (150, 120, 90))
            result = agent.analyze_character_multimodal(
                images_or_views=img,
                character_name="Dev",
                gender_hint="male",
            )

        assert result["facial_structure"] == mock_response_json["facial_structure"]
        assert result["prompt_prefix"] == mock_response_json["prompt_prefix"]
        assert result["agent_mode"] == "gemini_multimodal"


# =============================================================================
# Test Scene Prompt Refinement
# =============================================================================

class TestGeminiAgentScenePromptRefinement:
    def test_heuristic_scene_prompt_enhancement(self):
        agent = CharacterGeminiAgent(api_key=None)
        
        enhanced = agent.refine_scene_prompt(
            scene_prompt="Dev walks down a foggy alley",
            character_name="Dev",
            style_preset="kolkata_vintage",
        )

        assert "Dev" in enhanced
        assert "foggy alley" in enhanced or "Dev walks down a foggy alley" in enhanced
        assert "1970s" in enhanced or "35mm" in enhanced or "celluloid" in enhanced

    def test_mocked_gemini_scene_refinement(self):
        agent = CharacterGeminiAgent(api_key="AIzaSyMockKeyForTesting123456789")
        
        mock_client = mock.MagicMock()
        mock_response = mock.MagicMock()
        mock_response.text = "A breathtaking cinematic master shot of Dev walking through the misty, lantern-lit North Kolkata tram tracks at dusk, captured on 35mm anamorphic lens with rich golden hour chiaroscuro."
        mock_client.generate_content.return_value = mock_response
        agent._client = mock_client

        with mock.patch.object(agent, "is_available", return_value=True):
            result = agent.refine_scene_prompt(
                scene_prompt="Dev walking in tram depot",
                character_name="Dev",
                style_preset="kolkata_vintage",
            )

        assert "breathtaking cinematic master shot" in result
        assert "Dev" in result


# =============================================================================
# Test CharacterStudio Integration with Gemini Agent
# =============================================================================

class TestCharacterStudioGeminiIntegration:
    @pytest.fixture
    def studio_with_temp_workspace(self, tmp_path):
        profiles_dir = str(tmp_path / "profiles")
        styles_path = str(tmp_path / "styles.json")
        styles_data = [
            {
                "id": "imax_realism",
                "name": "IMAX 8K Cinematic Realism",
                "prompt_prefix": "masterpiece shot on IMAX 70mm, 8k resolution",
                "prompt_suffix": "photorealistic, hyperdetailed",
                "negative_prompt": "blurry, low quality",
            }
        ]
        with open(styles_path, "w", encoding="utf-8") as f:
            json.dump(styles_data, f)
        
        studio = CharacterStudio(profiles_dir=profiles_dir, styles_path=styles_path)
        return studio, profiles_dir

    def test_enroll_character_extracts_gemini_traits(self, studio_with_temp_workspace):
        studio, profiles_dir = studio_with_temp_workspace
        
        front = Image.new("RGB", (150, 150), (220, 170, 130))
        left = Image.new("RGB", (150, 150), (210, 160, 120))
        right = Image.new("RGB", (150, 150), (215, 165, 125))

        profile = studio.enroll_character(
            name="Siddhartha",
            description="Young intellectual philosopher in 1970s College Street",
            views={"front": front, "left": left, "right": right},
            gender="male",
        )

        assert profile.id == "siddhartha"
        assert profile.gemini_traits is not None
        assert "facial_structure" in profile.gemini_traits
        assert "prompt_prefix" in profile.gemini_traits
        assert np.isclose(compute_l2_norm(studio.get_character_embedding("siddhartha")), 1.0, atol=1e-5)

        # Check saved profile.json contains gemini_traits
        profile_json = os.path.join(profiles_dir, "siddhartha", "profile.json")
        with open(profile_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "gemini_traits" in data
        assert data["gemini_traits"]["facial_structure"] is not None

    def test_refine_scene_prompt_with_gemini_method(self, studio_with_temp_workspace):
        studio, _ = studio_with_temp_workspace
        
        front = Image.new("RGB", (100, 100), (200, 150, 110))
        studio.enroll_character(name="Rajat", description="Coffee house debater", images=[front])

        enhanced = studio.refine_scene_prompt_with_gemini(
            scene_prompt="Rajat argues passionately at the wooden table",
            character_id="rajat",
            style_id="imax_realism",
        )
        assert "Rajat" in enhanced
        assert len(enhanced) > 20
