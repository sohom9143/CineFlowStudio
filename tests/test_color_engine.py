"""
Tests for CineFlow NLE Studio: DaVinci Resolve-Inspired Color Engine.
Verifies 3-way color wheels, temperature/tint, contrast/pivot, 3D LUTs, and RGB parade scopes.
"""

import numpy as np
import pytest
from PIL import Image
from modules.color_engine import (
    CineColorEngine,
    ColorGradeSettings,
    ColorWheelVector,
)


def test_color_engine_presets_list():
    engine = CineColorEngine()
    presets = engine.list_lut_presets()
    assert len(presets) >= 7
    preset_names = [p["name"] for p in presets]
    assert "Kodak 2383 Film Print" in preset_names
    assert "Teal & Orange Neo" in preset_names
    assert "Matrix Green" in preset_names


def test_color_engine_neutral_grade():
    engine = CineColorEngine()
    settings = ColorGradeSettings()  # All neutral defaults
    frame = np.full((64, 64, 3), 128, dtype=np.uint8)

    graded = engine.apply_color_grade(frame, settings)
    assert graded.shape == (64, 64, 3)
    assert graded.dtype == np.uint8
    # Neutral grade should preserve approximate values
    assert np.abs(graded.astype(int) - 128).mean() < 5


def test_color_engine_temperature_shift():
    engine = CineColorEngine()
    # Warm temperature (+50)
    settings_warm = ColorGradeSettings(temperature=50.0)
    frame = np.full((32, 32, 3), 128, dtype=np.uint8)
    graded_warm = engine.apply_color_grade(frame, settings_warm)

    # Red should increase, Blue should decrease
    assert graded_warm[..., 0].mean() > 128
    assert graded_warm[..., 2].mean() < 128

    # Cool temperature (-50)
    settings_cool = ColorGradeSettings(temperature=-50.0)
    graded_cool = engine.apply_color_grade(frame, settings_cool)
    assert graded_cool[..., 2].mean() > 128
    assert graded_cool[..., 0].mean() < 128


def test_color_engine_color_wheels():
    engine = CineColorEngine()
    # Boost Lift (shadows) Red
    settings = ColorGradeSettings(
        lift=ColorWheelVector(x=0.5, y=0.0, master=0.2),
    )
    dark_frame = np.full((32, 32, 3), 30, dtype=np.uint8)
    graded = engine.apply_color_grade(dark_frame, settings)

    # Shadow should be lifted and red channel boosted
    assert graded[..., 0].mean() > 30


def test_color_engine_lut_application():
    engine = CineColorEngine()
    frame = np.full((32, 32, 3), 100, dtype=np.uint8)

    # Apply Kodak 2383
    settings = ColorGradeSettings(lut_preset="Kodak 2383 Film Print")
    graded = engine.apply_color_grade(frame, settings)
    assert graded.shape == (32, 32, 3)
    assert graded.dtype == np.uint8

    # Apply Matrix Green
    settings_matrix = ColorGradeSettings(lut_preset="Matrix Green")
    graded_matrix = engine.apply_color_grade(frame, settings_matrix)
    # Green should dominate
    assert graded_matrix[..., 1].mean() > graded_matrix[..., 0].mean()


def test_rgb_parade_calculation():
    engine = CineColorEngine()
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    frame[..., 0] = 200  # Strong Red
    frame[..., 1] = 100  # Medium Green
    frame[..., 2] = 50   # Low Blue

    scopes = engine.calculate_rgb_parade(frame, num_bins=64)
    assert "r" in scopes and "g" in scopes and "b" in scopes
    assert "ire_peaks" in scopes
    assert scopes["ire_peaks"]["r"] > scopes["ire_peaks"]["g"]
    assert scopes["ire_peaks"]["g"] > scopes["ire_peaks"]["b"]
