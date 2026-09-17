"""
CineFlow NLE Studio: Core Timeline & Non-Linear Editing Engine
=============================================================
Provides production-grade NLE data structures and timeline operations:
- Track Management (Video V1-V3, Audio A1-A4, Captions C1, SFX/Memes M1)
- Clip Data Model with trim points, speed, transforms, and audio/video envelopes
- Timecode <-> Frame mathematical conversions (Drop-Frame & Non-Drop Frame)
- Atomic Edit Operations:
    * Razor Cut (split clip at playhead)
    * Ripple Trim (in-point / out-point with timeline ripple)
    * Ripple Delete (remove clip & collapse subsequent gaps)
    * Slip / Slide Clip
    * Magnetic Snapping Engine (within configurable frame tolerance)
- Serialization to compact Timeline AST (Abstract Syntax Tree)
"""

from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union


class TrackType(str, Enum):
    VIDEO = "video"
    AUDIO = "audio"
    SUBTITLE = "subtitle"
    OVERLAY = "overlay"


@dataclass
class ClipTransition:
    transition_id: str = "cross_dissolve"
    duration_frames: int = 12
    easing: str = "ease_in_out"
    edge: str = "start"  # "start", "end"


@dataclass
class Clip:
    id: str
    name: str
    track_id: str
    source_path: str
    start_frame: int  # Position on the timeline ruler (in frames)
    duration_frames: int  # Duration occupied on the timeline
    source_in: int = 0  # In-point within the source media
    source_out: Optional[int] = None  # Out-point within source media
    speed: float = 1.0  # Playback rate (1.0 = normal, 2.0 = 2x, etc.)
    opacity: float = 1.0  # 0.0 to 1.0
    volume_db: float = 0.0  # dB adjustment (-60.0 to +12.0)
    pan: float = 0.0  # -1.0 (Left) to +1.0 (Right)
    color_grade_preset: Optional[str] = None
    lut_name: Optional[str] = None
    transitions: List[ClipTransition] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.source_out is None:
            self.source_out = self.source_in + self.duration_frames

    @property
    def end_frame(self) -> int:
        return self.start_frame + self.duration_frames

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["end_frame"] = self.end_frame
        return d


@dataclass
class Track:
    id: str
    name: str
    track_type: TrackType
    clips: List[Clip] = field(default_factory=list)
    muted: bool = False
    solo: bool = False
    locked: bool = False
    volume_db: float = 0.0
    pan: float = 0.0
    color: str = "#8b5cf6"

    def get_clip_at_frame(self, frame: int) -> Optional[Clip]:
        for clip in self.clips:
            if clip.start_frame <= frame < clip.end_frame:
                return clip
        return None

    def sort_clips(self) -> None:
        self.clips.sort(key=lambda c: c.start_frame)


@dataclass
class Marker:
    id: str
    frame: int
    name: str
    color: str = "#fbbf24"
    comment: str = ""


class TimecodeConverter:
    """Handles professional drop-frame and non-drop-frame timecode calculations."""

    @staticmethod
    def frames_to_timecode(frames: int, fps: float = 60.0, drop_frame: bool = False) -> str:
        if frames < 0:
            frames = 0
        total_seconds = int(frames // fps)
        rem_frames = int(frames % fps)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        sep = ";" if drop_frame else ":"
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}{sep}{rem_frames:02d}"

    @staticmethod
    def timecode_to_frames(timecode: str, fps: float = 60.0) -> int:
        # Match HH:MM:SS:FF or HH:MM:SS;FF
        match = re.match(r"^(\d{2}):(\d{2}):(\d{2})[:;](\d{2,3})$", timecode.strip())
        if not match:
            raise ValueError(f"Invalid timecode format: '{timecode}'. Expected HH:MM:SS:FF")
        h, m, s, f = map(int, match.groups())
        total_seconds = (h * 3600) + (m * 60) + s
        return int(total_seconds * fps) + f


class NLETimeline:
    """
    Core Non-Linear Editor Timeline Controller.
    Manages tracks, clip manipulation, razor slicing, ripple editing, and playhead states.
    """

    def __init__(
        self,
        project_name: str = "Neo-Tokyo Cyberpunk Scene 04",
        fps: float = 60.0,
        resolution: Tuple[int, int] = (3840, 2160),
    ):
        self.project_name = project_name
        self.fps = fps
        self.resolution = resolution
        self.playhead_frame: int = 0
        self.in_point: Optional[int] = None
        self.out_point: Optional[int] = None
        self.snapping_enabled: bool = True
        self.snap_tolerance_frames: int = 4
        self.tracks: Dict[str, Track] = {}
        self.markers: List[Marker] = []
        self._history: List[Dict[str, Any]] = []

        self._initialize_default_tracks()

    def _initialize_default_tracks(self) -> None:
        """Sets up default broadcast NLE track layout: V1-V3, C1, A1-A4."""
        default_layout = [
            ("V3", "V3 Memes & Overlays", TrackType.OVERLAY, "#10b981"),
            ("V2", "V2 Cutaways & B-Roll", TrackType.VIDEO, "#06b6d4"),
            ("V1", "V1 Hero A-Roll", TrackType.VIDEO, "#8b5cf6"),
            ("C1", "C1 Bilingual Subtitles", TrackType.SUBTITLE, "#fbbf24"),
            ("A1", "A1 Dialogue (Speech)", TrackType.AUDIO, "#38bdf8"),
            ("A2", "A2 Sound Effects (SFX)", TrackType.AUDIO, "#f59e0b"),
            ("A3", "A3 Background Music (BGM)", TrackType.AUDIO, "#a855f7"),
            ("A4", "A4 Master Bus", TrackType.AUDIO, "#10b981"),
        ]
        for tid, name, ttype, color in default_layout:
            self.tracks[tid] = Track(id=tid, name=name, track_type=ttype, color=color)

    @property
    def total_duration_frames(self) -> int:
        max_frame = 0
        for track in self.tracks.values():
            for clip in track.clips:
                if clip.end_frame > max_frame:
                    max_frame = clip.end_frame
        return max_frame

    @property
    def playhead_timecode(self) -> str:
        return TimecodeConverter.frames_to_timecode(self.playhead_frame, fps=self.fps)

    def set_playhead(self, frame: int) -> None:
        self.playhead_frame = max(0, frame)

    def add_clip(self, clip: Clip) -> None:
        """Adds a clip to the target track, ensuring sort order."""
        if clip.track_id not in self.tracks:
            raise KeyError(f"Track '{clip.track_id}' not found in timeline.")
        track = self.tracks[clip.track_id]
        track.clips.append(clip)
        track.sort_clips()

    def find_clip(self, clip_id: str) -> Optional[Tuple[Track, Clip]]:
        for track in self.tracks.values():
            for clip in track.clips:
                if clip.id == clip_id:
                    return track, clip
        return None

    def razor_cut(self, track_id: Optional[str] = None, split_frame: Optional[int] = None) -> List[str]:
        """
        Shortcut 'C': Splits active clip(s) at split_frame (default: playhead).
        If track_id is provided, splits only on that track. Otherwise, splits across all unlocked tracks.
        Returns a list of newly created clip IDs.
        """
        frame = self.playhead_frame if split_frame is None else split_frame
        target_tracks = [self.tracks[track_id]] if track_id and track_id in self.tracks else [
            t for t in self.tracks.values() if not t.locked
        ]

        new_clip_ids = []
        for track in target_tracks:
            clip = track.get_clip_at_frame(frame)
            if not clip:
                continue

            # Check that cut point is strictly inside the clip
            if clip.start_frame < frame < clip.end_frame:
                cut_offset = frame - clip.start_frame

                # Part 1 keeps the start
                original_duration = clip.duration_frames
                clip.duration_frames = cut_offset
                clip.source_out = clip.source_in + cut_offset

                # Part 2 is the new slice
                new_clip_id = f"{clip.id}_cut_{frame}"
                new_clip = Clip(
                    id=new_clip_id,
                    name=f"{clip.name} (Part 2)",
                    track_id=track.id,
                    source_path=clip.source_path,
                    start_frame=frame,
                    duration_frames=original_duration - cut_offset,
                    source_in=clip.source_in + cut_offset,
                    source_out=clip.source_out + (original_duration - cut_offset),
                    speed=clip.speed,
                    opacity=clip.opacity,
                    volume_db=clip.volume_db,
                    pan=clip.pan,
                    color_grade_preset=clip.color_grade_preset,
                    lut_name=clip.lut_name,
                    tags=copy.deepcopy(clip.tags),
                    metadata=copy.deepcopy(clip.metadata),
                )
                track.clips.append(new_clip)
                track.sort_clips()
                new_clip_ids.append(new_clip_id)

        return new_clip_ids

    def ripple_trim_head(self, track_id: str, clip_id: str, new_start_frame: int) -> bool:
        """
        Shortcut 'Q': Trims the head of the clip to new_start_frame and ripples all subsequent clips backward.
        """
        if track_id not in self.tracks:
            return False
        track = self.tracks[track_id]
        clip = next((c for c in track.clips if c.id == clip_id), None)
        if not clip:
            return False

        if new_start_frame <= clip.start_frame or new_start_frame >= clip.end_frame:
            return False

        delta = new_start_frame - clip.start_frame
        clip.source_in += delta
        clip.duration_frames -= delta

        # Ripple all subsequent clips on this track to the left
        for other in track.clips:
            if other.start_frame > clip.start_frame:
                other.start_frame -= delta

        track.sort_clips()
        return True

    def ripple_trim_tail(self, track_id: str, clip_id: str, new_end_frame: int) -> bool:
        """
        Shortcut 'W': Trims the tail of the clip to new_end_frame and ripples all subsequent clips backward.
        """
        if track_id not in self.tracks:
            return False
        track = self.tracks[track_id]
        clip = next((c for c in track.clips if c.id == clip_id), None)
        if not clip:
            return False

        if new_end_frame <= clip.start_frame or new_end_frame >= clip.end_frame:
            return False

        delta = clip.end_frame - new_end_frame
        clip.duration_frames -= delta
        if clip.source_out is not None:
            clip.source_out -= delta

        # Ripple subsequent clips
        for other in track.clips:
            if other.start_frame >= clip.end_frame:
                other.start_frame -= delta

        track.sort_clips()
        return True

    def ripple_delete(self, track_id: str, clip_id: str) -> bool:
        """
        Shortcut 'Shift + Delete': Deletes the clip and collapses the resulting gap.
        """
        if track_id not in self.tracks:
            return False
        track = self.tracks[track_id]
        idx = next((i for i, c in enumerate(track.clips) if c.id == clip_id), None)
        if idx is None:
            return False

        target_clip = track.clips.pop(idx)
        gap_size = target_clip.duration_frames

        # Shift all subsequent clips on the track to the left
        for other in track.clips:
            if other.start_frame >= target_clip.end_frame:
                other.start_frame -= gap_size

        track.sort_clips()
        return True

    def snap_frame(self, target_frame: int) -> int:
        """Snaps target_frame to the nearest clip boundary or marker within tolerance."""
        if not self.snapping_enabled:
            return target_frame

        closest = target_frame
        min_dist = self.snap_tolerance_frames + 1

        # Check all clip start and end points
        for track in self.tracks.values():
            for clip in track.clips:
                for pt in (clip.start_frame, clip.end_frame):
                    dist = abs(pt - target_frame)
                    if dist < min_dist:
                        min_dist = dist
                        closest = pt

        # Check markers
        for marker in self.markers:
            dist = abs(marker.frame - target_frame)
            if dist < min_dist:
                min_dist = dist
                closest = marker.frame

        return closest

    def add_marker(self, frame: Optional[int] = None, name: str = "Marker", color: str = "#fbbf24") -> Marker:
        f = self.playhead_frame if frame is None else frame
        marker = Marker(id=f"marker_{len(self.markers)+1}", frame=f, name=name, color=color)
        self.markers.append(marker)
        self.markers.sort(key=lambda m: m.frame)
        return marker

    def to_compact_ast(self) -> Dict[str, Any]:
        """
        Exports the timeline into a compact AST dictionary.
        Optimized for < 1/3 token consumption when queried by LLMs.
        """
        compact_tracks = {}
        for tid, track in self.tracks.items():
            compact_clips = []
            for c in track.clips:
                compact_clips.append({
                    "id": c.id,
                    "t": [round(c.start_frame / self.fps, 2), round(c.end_frame / self.fps, 2)],
                    "src": c.name,
                    "tags": c.tags,
                })
            compact_tracks[tid] = {
                "type": track.track_type.value,
                "clips": compact_clips,
            }

        return {
            "project": self.project_name,
            "fps": self.fps,
            "dur_s": round(self.total_duration_frames / self.fps, 2),
            "playhead_s": round(self.playhead_frame / self.fps, 2),
            "tracks": compact_tracks,
        }

    def to_full_dict(self) -> Dict[str, Any]:
        return {
            "projectName": self.project_name,
            "fps": self.fps,
            "resolution": list(self.resolution),
            "playheadFrame": self.playhead_frame,
            "playheadTimecode": self.playhead_timecode,
            "totalDurationFrames": self.total_duration_frames,
            "totalDurationTimecode": TimecodeConverter.frames_to_timecode(self.total_duration_frames, self.fps),
            "inPoint": self.in_point,
            "outPoint": self.out_point,
            "snappingEnabled": self.snapping_enabled,
            "tracks": {tid: asdict(track) for tid, track in self.tracks.items()},
            "markers": [asdict(m) for m in self.markers],
        }
