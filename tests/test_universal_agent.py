"""
Unit and Integration Tests for Universal AI Director Agent & Ingredient Engine
=============================================================================
Tests:
1. Ingredient decomposition across universal genres (Cyberpunk, Fantasy, Anime, Horror, Commercial, Vintage, IMAX).
2. Intent classification and orchestration plan synthesis (Video vs Image vs LipSync vs Train).
3. Bengali / Banglish conversational prompt understanding.
4. Universal style preset resolution and model parameters.
"""

import pytest
from modules.agent_gemini import CharacterGeminiAgent, resolve_gemini_model_name


class TestUniversalIngredientDecomposition:
    @pytest.fixture
    def agent(self):
        return CharacterGeminiAgent(api_key=None)

    def test_cyberpunk_ingredient_decomposition(self, agent):
        res = agent.decompose_prompt_ingredients(
            user_prompt="Dev walks through rainy cyberpunk alleyway with neon signs and flying cars",
            character_name="Dev",
            style_preset="scifi_cyberpunk",
        )
        assert res["detected_genre"] == "Sci-Fi Cyberpunk"
        assert "Dev" in res["master_positive_prompt"]
        assert "neon" in res["lighting_setup"].lower() or "cyberpunk" in res["environment_setting"].lower()
        assert "camera_lens" in res
        assert "atmosphere_weather" in res

    def test_high_fantasy_ingredient_decomposition(self, agent):
        res = agent.decompose_prompt_ingredients(
            user_prompt="Arya in regal cloak standing on top of royal castle overlooking magical valley",
            character_name="Arya",
            style_preset="high_fantasy",
        )
        assert res["detected_genre"] == "High Fantasy & Mythological Epic"
        assert "Arya" in res["master_positive_prompt"]
        assert "god rays" in res["lighting_setup"].lower() or "fantasy" in res["environment_setting"].lower()

    def test_anime_ghibli_ingredient_decomposition(self, agent):
        res = agent.decompose_prompt_ingredients(
            user_prompt="Meghla watching vibrant summer clouds in animated field",
            character_name="Meghla",
            style_preset="ghibli_anime",
        )
        assert res["detected_genre"] == "Studio Ghibli & Modern Anime"
        assert "Meghla" in res["master_positive_prompt"]

    def test_gothic_horror_ingredient_decomposition(self, agent):
        res = agent.decompose_prompt_ingredients(
            user_prompt="Neel walking through eerie dark gothic catacombs with candle",
            character_name="Neel",
            style_preset="gothic_horror",
        )
        assert res["detected_genre"] == "Dark Gothic Horror"
        assert "fog" in res["atmosphere_weather"].lower() or "candle" in res["lighting_setup"].lower()

    def test_commercial_luxury_ingredient_decomposition(self, agent):
        res = agent.decompose_prompt_ingredients(
            user_prompt="Close up portrait for luxury perfume commercial in studio",
            character_name="Zara",
            style_preset="commercial_studio",
        )
        assert res["detected_genre"] == "Commercial Studio & Luxury Advertising"
        assert "softbox" in res["lighting_setup"].lower() or "studio" in res["environment_setting"].lower()


class TestAutonomousIntentOrchestration:
    @pytest.fixture
    def agent(self):
        return CharacterGeminiAgent(api_key=None)

    def test_video_intent_orchestration(self, agent):
        plan = agent.orchestrate_generation_plan(
            natural_prompt="Create an action video of Dev in cyberpunk city with neon lighting",
            available_characters=[{"id": "dev", "name": "Dev"}, {"id": "neel", "name": "Neel"}],
        )
        assert plan["target_action"] == "video"
        assert plan["character_id"] == "dev"
        assert plan["style_id"] == "scifi_cyberpunk"
        assert "master_positive_prompt" in plan
        assert "recommended_params" in plan
        assert plan["recommended_params"]["frames"] == 49

    def test_image_keyframe_intent_orchestration(self, agent):
        plan = agent.orchestrate_generation_plan(
            natural_prompt="Generate a still portrait photo of Neel in a vintage library",
            available_characters=[{"id": "dev", "name": "Dev"}, {"id": "neel", "name": "Neel"}],
        )
        assert plan["target_action"] == "image_keyframe"
        assert plan["character_id"] == "neel"
        assert plan["recommended_params"]["frames"] == 1

    def test_bengali_prompt_intent_orchestration(self, agent):
        plan = agent.orchestrate_generation_plan(
            natural_prompt="Dev er ekta cyberpunk sci-fi video banao jekhane brishti porche",
            available_characters=[{"id": "dev", "name": "Dev"}],
        )
        assert plan["target_action"] == "video"
        assert plan["character_id"] == "dev"
        assert plan["style_id"] == "scifi_cyberpunk"
        assert "Dev" in plan["master_positive_prompt"]
