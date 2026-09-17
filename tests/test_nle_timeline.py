"""
Tests for CineFlow NLE Studio: Timeline & Non-Linear Editing Engine.
Verifies tracks, clips, timecode math, razor cutting, ripple trimming, ripple deleting, and snapping.
"""

import pytest
from modules.nle_timeline import (
    NLETimeline,
    Clip,
    Track,
    TrackType,
    TimecodeConverter,
)


def test_timecode_converter():
    # 60 fps tests
    assert TimecodeConverter.frames_to_timecode(0, fps=60.0) == "00:00:00:00"
    assert TimecodeConverter.frames_to_timecode(60, fps=60.0) == "00:00:01:00"
    assert TimecodeConverter.frames_to_timecode(3660, fps=60.0) == "00:01:01:00"
    assert TimecodeConverter.frames_to_timecode(3600 * 60, fps=60.0) == "01:00:00:00"

    # Reverse
    assert TimecodeConverter.timecode_to_frames("00:00:01:00", fps=60.0) == 60
    assert TimecodeConverter.timecode_to_frames("01:04:22:15", fps=60.0) == (1*3600 + 4*60 + 22) * 60 + 15

    # Invalid timecode raises ValueError
    with pytest.raises(ValueError):
        TimecodeConverter.timecode_to_frames("invalid-timecode")


def test_nle_timeline_initialization():
    timeline = NLETimeline(project_name="Test Project", fps=60.0)
    assert timeline.project_name == "Test Project"
    assert timeline.fps == 60.0
    assert "V1" in timeline.tracks
    assert "V2" in timeline.tracks
    assert "V3" in timeline.tracks
    assert "A1" in timeline.tracks
    assert "C1" in timeline.tracks
    assert timeline.total_duration_frames == 0


def test_nle_add_clip_and_duration():
    timeline = NLETimeline(fps=60.0)
    clip1 = Clip(
        id="c1",
        name="Take 1",
        track_id="V1",
        source_path="take1.mov",
        start_frame=0,
        duration_frames=120,
    )
    timeline.add_clip(clip1)
    assert timeline.total_duration_frames == 120

    clip2 = Clip(
        id="c2",
        name="Take 2",
        track_id="V1",
        source_path="take2.mov",
        start_frame=120,
        duration_frames=180,
    )
    timeline.add_clip(clip2)
    assert timeline.total_duration_frames == 300
    assert len(timeline.tracks["V1"].clips) == 2


def test_nle_razor_cut():
    timeline = NLETimeline(fps=60.0)
    clip = Clip(
        id="hero_take",
        name="Hero Take",
        track_id="V1",
        source_path="hero.mov",
        start_frame=0,
        duration_frames=200,
    )
    timeline.add_clip(clip)

    # Perform razor cut at frame 80
    new_clips = timeline.razor_cut(track_id="V1", split_frame=80)
    assert len(new_clips) == 1
    assert len(timeline.tracks["V1"].clips) == 2

    c1 = timeline.tracks["V1"].clips[0]
    c2 = timeline.tracks["V1"].clips[1]

    assert c1.start_frame == 0
    assert c1.duration_frames == 80
    assert c1.end_frame == 80

    assert c2.start_frame == 80
    assert c2.duration_frames == 120
    assert c2.end_frame == 200


def test_nle_ripple_trim_head():
    timeline = NLETimeline(fps=60.0)
    c1 = Clip(id="c1", name="Clip 1", track_id="V1", source_path="c1.mov", start_frame=0, duration_frames=100)
    c2 = Clip(id="c2", name="Clip 2", track_id="V1", source_path="c2.mov", start_frame=100, duration_frames=100)
    timeline.add_clip(c1)
    timeline.add_clip(c2)

    # Trim head of c1 by 20 frames (new start: 20)
    ok = timeline.ripple_trim_head("V1", "c1", 20)
    assert ok is True
    assert c1.start_frame == 0  # In ripple trim, head trimmed shifts start back or duration
    # Subsequent clips shift by delta
    assert timeline.total_duration_frames == 180


def test_nle_ripple_delete():
    timeline = NLETimeline(fps=60.0)
    c1 = Clip(id="c1", name="Clip 1", track_id="V1", source_path="c1.mov", start_frame=0, duration_frames=100)
    c2 = Clip(id="c2", name="Clip 2", track_id="V1", source_path="c2.mov", start_frame=100, duration_frames=150)
    c3 = Clip(id="c3", name="Clip 3", track_id="V1", source_path="c3.mov", start_frame=250, duration_frames=80)
    timeline.add_clip(c1)
    timeline.add_clip(c2)
    timeline.add_clip(c3)

    # Ripple delete c2 (collapsing 150 frames)
    ok = timeline.ripple_delete("V1", "c2")
    assert ok is True
    assert len(timeline.tracks["V1"].clips) == 2
    # c3 should now start at frame 100 instead of 250
    assert timeline.tracks["V1"].clips[1].start_frame == 100


def test_nle_magnetic_snapping():
    timeline = NLETimeline(fps=60.0)
    c1 = Clip(id="c1", name="Clip 1", track_id="V1", source_path="c1.mov", start_frame=0, duration_frames=120)
    timeline.add_clip(c1)

    timeline.snapping_enabled = True
    timeline.snap_tolerance_frames = 5

    # Near frame 120 (e.g. 122) should snap to 120
    assert timeline.snap_frame(122) == 120
    assert timeline.snap_frame(118) == 120
    # Outside tolerance (e.g. 130) should not snap
    assert timeline.snap_frame(130) == 130


def test_nle_compact_ast():
    timeline = NLETimeline(fps=60.0)
    c1 = Clip(id="c1", name="Hero Take", track_id="V1", source_path="c1.mov", start_frame=0, duration_frames=120, tags=["hero"])
    timeline.add_clip(c1)

    ast = timeline.to_compact_ast()
    assert "tracks" in ast
    assert "V1" in ast["tracks"]
    assert len(ast["tracks"]["V1"]["clips"]) == 1
    assert ast["tracks"]["V1"]["clips"][0]["src"] == "Hero Take"
    assert ast["dur_s"] == 2.0
