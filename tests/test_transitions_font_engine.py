"""
Tests for CineFlow NLE Studio: Transitions & Multilingual Typography / Font Engine.
Verifies transition easing, frame blending (whip pan, glitch, dissolve), Bengali fonts, and subtitle presets.
"""

import numpy as np
import pytest
from modules.transitions_engine import TransitionsAndFontEngine


def test_transitions_list():
    engine = TransitionsAndFontEngine()
    transitions = engine.list_transitions()
    assert len(transitions) >= 8
    trans_ids = [t["id"] for t in transitions]
    assert "cross_dissolve" in trans_ids
    assert "whip_pan_left" in trans_ids
    assert "glitch_rgb" in trans_ids
    assert "dip_to_black" in trans_ids


def test_fonts_list():
    engine = TransitionsAndFontEngine()
    fonts = engine.list_fonts()
    font_families = [f["family"] for f in fonts]
    assert "Noto Sans Bengali" in font_families
    assert "Hind Siliguri" in font_families
    assert "Plus Jakarta Sans" in font_families
    assert "JetBrains Mono" in font_families


def test_subtitle_presets():
    engine = TransitionsAndFontEngine()
    presets = engine.list_subtitle_presets()
    preset_ids = [p["id"] for p in presets]
    assert "hormozi_pop" in preset_ids
    assert "mrbeast_bold" in preset_ids
    assert "bengali_cinematic" in preset_ids


def test_transition_frame_blending():
    engine = TransitionsAndFontEngine()
    frame_a = np.zeros((32, 32, 3), dtype=np.uint8)  # Black frame
    frame_b = np.full((32, 32, 3), 200, dtype=np.uint8)  # Bright frame

    # Progress 0.0 -> matches A
    blended_0 = engine.blend_frames(frame_a, frame_b, 0.0, "cross_dissolve")
    assert blended_0.mean() < 5

    # Progress 1.0 -> matches B
    blended_1 = engine.blend_frames(frame_a, frame_b, 1.0, "cross_dissolve")
    assert blended_1.mean() > 195

    # Progress 0.5 -> intermediate value
    blended_mid = engine.blend_frames(frame_a, frame_b, 0.5, "cross_dissolve")
    assert 80 < blended_mid.mean() < 120


def test_whip_pan_and_glitch_blending():
    engine = TransitionsAndFontEngine()
    frame_a = np.full((16, 32, 3), 50, dtype=np.uint8)
    frame_b = np.full((16, 32, 3), 220, dtype=np.uint8)

    whip = engine.blend_frames(frame_a, frame_b, 0.5, "whip_pan_left")
    assert whip.shape == frame_a.shape

    glitch = engine.blend_frames(frame_a, frame_b, 0.5, "glitch_rgb")
    assert glitch.shape == frame_a.shape
