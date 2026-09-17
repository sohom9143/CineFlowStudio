"""
Tests for CineFlow NLE Studio: Next-Gen Agentic Scripting & AI Protocol Assistant.
Verifies multimodal perception (silence/scene/beat detection), interactive plan proposal ("AI asks before editing"),
and autonomous step-by-step agentic execution loop.
"""

import numpy as np
import pytest
from modules.nle_timeline import NLETimeline, Clip
from modules.color_engine import CineColorEngine
from modules.audio_engine import FairlightAudioEngine
from modules.meme_vault import MemeVault
from modules.agentic_scripting_engine import (
    AgenticScriptingEngine,
    EditingPlan,
    PlanStepStatus,
)


@pytest.fixture
def setup_agentic_engine():
    timeline = NLETimeline(fps=60.0)
    # Add initial dialogue and b-roll
    timeline.add_clip(Clip(
        id="hero_clip",
        name="Take01.mov",
        track_id="V1",
        source_path="take01.mov",
        start_frame=0,
        duration_frames=1800,  # 30 seconds
    ))
    timeline.add_clip(Clip(
        id="dialogue_clip",
        name="Speech.wav",
        track_id="A1",
        source_path="speech.wav",
        start_frame=0,
        duration_frames=1800,
    ))

    color_engine = CineColorEngine()
    audio_engine = FairlightAudioEngine()
    meme_vault = MemeVault()

    engine = AgenticScriptingEngine(
        timeline=timeline,
        color_engine=color_engine,
        audio_engine=audio_engine,
        meme_vault=meme_vault,
    )
    return engine


def test_multimodal_perceptions(setup_agentic_engine):
    engine = setup_agentic_engine

    # 1. Silence VAD detection
    silences = engine.detect_silence_intervals(None)
    assert len(silences) >= 2
    assert silences[0][0] < silences[0][1]

    # 2. Scene cut detection
    cuts = engine.detect_scene_cuts(None)
    assert len(cuts) >= 2

    # 3. Beat detection
    beats = engine.detect_rhythm_beats(None)
    assert len(beats) >= 5


def test_ai_asks_before_editing_plan_proposal(setup_agentic_engine):
    engine = setup_agentic_engine
    user_prompt = "Auto-cut silences, grade with Kodak 2383, insert Hera Pheri meme on joke, add Bengali subtitles, and duck BGM."

    # Propose plan
    plan = engine.propose_editing_plan(user_prompt)
    assert isinstance(plan, EditingPlan)
    assert plan.approved_by_user is False  # AI has NOT executed yet; waiting for approval!
    assert len(plan.steps) >= 5

    # Check that key requested actions are in the plan
    actions = [s.action_type for s in plan.steps]
    assert "cut_silence" in actions
    assert "insert_meme" in actions
    assert "color_grade" in actions
    assert "generate_captions" in actions
    assert "audio_ducking" in actions


def test_autonomous_agentic_execution_loop(setup_agentic_engine):
    engine = setup_agentic_engine
    user_prompt = "Cut silences, insert meme, and color grade."

    plan = engine.propose_editing_plan(user_prompt)
    assert plan.is_completed is False

    # Execute step 1 individually
    step1_res = engine.execute_plan_step(plan.plan_id, 0)
    assert step1_res["step"]["status"] == PlanStepStatus.COMPLETED.value

    # Execute remaining full plan in agentic loop
    full_result = engine.execute_full_plan_loop(plan.plan_id)
    assert full_result["isCompleted"] is True

    # Verify timeline was actually modified:
    # 1. Track V3 should now have the inserted meme clip
    assert len(engine.timeline.tracks["V3"].clips) >= 1
    # 2. Track V1 clips should have the LUT applied
    v1_clips = engine.timeline.tracks["V1"].clips
    assert any(c.lut_name is not None for c in v1_clips)
