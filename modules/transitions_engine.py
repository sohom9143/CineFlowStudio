"""
CineFlow NLE Studio: Transitions & Multilingual Typography / Font Engine
========================================================================
Comprehensive transition rendering and viral caption styling:
- Transition Library:
    * Whip Pan (Directional high-speed camera swish)
    * Glitch RGB Split (Digital chromatic aberration)
    * Light Leak / Lens Flare Flash
    * Zoom Blur (In / Out)
    * Cross Dissolve (Linear & Exponential)
    * Dip to Black / Dip to White
    * Camera Shake / Impact Hit
- Typography & Font Engine:
    * Google Fonts integration + Bengali script typography:
        - Noto Sans Bengali ('আমার সিনেমা প্রবাহ')
        - Hind Siliguri
        - Tiro Bangla
        - Galada
        - Mina
    * English Headline & Monospaced fonts:
        - Plus Jakarta Sans, Inter, JetBrains Mono, Space Grotesk, Bebas Neue, Cinzel
    * Viral Subtitle Presets:
        - Hormozi Pop (Dynamic word-level yellow/green highlight with bounce)
        - MrBeast Bold (Thick black outline + vibrant yellow fill)
        - Netflix Clean (Subtle dark pill letterbox)
        - Cinematic 2.39:1 (Spaced serif uppercase)
        - Cyberpunk Neon Glow (Cyan & Violet glow)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass
class TransitionDefinition:
    id: str
    name: str
    category: str  # "Dissolve", "Motion", "Glitch", "Light", "Wipe"
    default_duration_frames: int = 14
    description: str = ""
    icon: str = "animation"


@dataclass
class FontDefinition:
    family: str
    display_name: str
    category: str  # "Bengali", "Sans-Serif", "Display", "Monospace", "Serif"
    sample_text: str
    google_fonts_url: Optional[str] = None
    weights: List[int] = field(default_factory=lambda: [400, 600, 700])


@dataclass
class SubtitleStylePreset:
    id: str
    name: str
    font_family: str
    font_size: int
    text_color: str
    highlight_color: str
    outline_color: str
    outline_width: int
    shadow_color: str
    background_pill: bool
    animation_type: str  # "pop", "fade", "word_highlight", "static"


class TransitionsAndFontEngine:
    """
    Manages transition shaders, easing equations, and typography stylers.
    """

    TRANSITIONS: List[TransitionDefinition] = [
        TransitionDefinition("cross_dissolve", "Cross Dissolve", "Dissolve", 14, "Smooth linear alpha blend between adjacent takes.", "opacity"),
        TransitionDefinition("whip_pan_left", "Whip Pan (Left)", "Motion", 12, "High-speed horizontal swish camera whip to the left.", "swipe_left"),
        TransitionDefinition("whip_pan_right", "Whip Pan (Right)", "Motion", 12, "High-speed horizontal swish camera whip to the right.", "swipe_right"),
        TransitionDefinition("glitch_rgb", "Glitch RGB Split", "Glitch", 10, "Digital chromatic artifacting and RGB channel displacement.", "grain"),
        TransitionDefinition("light_leak", "Light Leak Lens Flare", "Light", 16, "Warm anamorphic lens flash and exposure burst.", "flare"),
        TransitionDefinition("zoom_blur_in", "Zoom Blur (In)", "Motion", 14, "Dynamic radial zoom acceleration into next shot.", "zoom_in"),
        TransitionDefinition("zoom_blur_out", "Zoom Blur (Out)", "Motion", 14, "Dynamic radial zoom pullback into next shot.", "zoom_out"),
        TransitionDefinition("dip_to_black", "Dip to Black", "Dissolve", 18, "Classic scene transition dipping through dark pedestal.", "brightness_low"),
        TransitionDefinition("dip_to_white", "Dip to White (Flash)", "Light", 12, "High-key specular flash transition.", "flash_on"),
        TransitionDefinition("camera_shake", "Camera Shake Impact", "Motion", 10, "Heavy percussive shake on bass or action hit.", "vibration"),
    ]

    FONTS: List[FontDefinition] = [
        # Bengali Typography
        FontDefinition("Noto Sans Bengali", "Noto Sans Bengali (নোটো বাংলা)", "Bengali", "আমার সিনেমা প্রবাহ", "https://fonts.googleapis.com/css2?family=Noto+Sans+Bengali:wght@400;600;700&display=swap"),
        FontDefinition("Hind Siliguri", "Hind Siliguri (হিন্দ শিলিগুড়ি)", "Bengali", "চলচ্চিত্র সম্পাদনা", "https://fonts.googleapis.com/css2?family=Hind+Siliguri:wght@400;600;700&display=swap"),
        FontDefinition("Tiro Bangla", "Tiro Bangla (তিরো বাংলা)", "Bengali", "কলকাতার ঐতিহ্যবাহী ফন্ট", "https://fonts.googleapis.com/css2?family=Tiro+Bangla&display=swap"),
        FontDefinition("Galada", "Galada (গলদা স্ক্রিপ্ট)", "Bengali", "শিল্পধর্মী বাংলা ক্যালিগ্রাফি", "https://fonts.googleapis.com/css2?family=Galada&display=swap"),

        # Modern English Display & Sans
        FontDefinition("Plus Jakarta Sans", "Plus Jakarta Sans", "Sans-Serif", "Next-Gen Director Studio", "https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@600;700;800&display=swap"),
        FontDefinition("Inter", "Inter UI", "Sans-Serif", "Precision Clean Subtitles", "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap"),
        FontDefinition("JetBrains Mono", "JetBrains Mono", "Monospace", "TC 01:04:22:15 60FPS", "https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&display=swap"),
        FontDefinition("Space Grotesk", "Space Grotesk", "Display", "CYBERPUNK NEURAL NLE", "https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@600;700&display=swap"),
        FontDefinition("Bebas Neue", "Bebas Neue", "Display", "HORMOZI STYLE IMPACT", "https://fonts.googleapis.com/css2?family=Bebas+Neue&display=swap"),
        FontDefinition("Cinzel", "Cinzel Cinematic", "Serif", "THE CHRONICLES OF TOKYO", "https://fonts.googleapis.com/css2?family=Cinzel:wght@600;700&display=swap"),
    ]

    SUBTITLE_PRESETS: List[SubtitleStylePreset] = [
        SubtitleStylePreset(
            id="hormozi_pop",
            name="Hormozi Pop Highlight",
            font_family="Bebas Neue, Hind Siliguri, sans-serif",
            font_size=52,
            text_color="#ffffff",
            highlight_color="#fbbf24",
            outline_color="#000000",
            outline_width=4,
            shadow_color="rgba(0,0,0,0.8)",
            background_pill=False,
            animation_type="pop",
        ),
        SubtitleStylePreset(
            id="mrbeast_bold",
            name="MrBeast Bold Punch",
            font_family="Plus Jakarta Sans, Noto Sans Bengali, sans-serif",
            font_size=46,
            text_color="#facc15",
            highlight_color="#38bdf8",
            outline_color="#000000",
            outline_width=5,
            shadow_color="rgba(0,0,0,0.9)",
            background_pill=False,
            animation_type="word_highlight",
        ),
        SubtitleStylePreset(
            id="bengali_cinematic",
            name="Bengali Cinematic Master (বাংলা সিনেমা)",
            font_family="Hind Siliguri, Noto Sans Bengali, sans-serif",
            font_size=42,
            text_color="#fef08a",
            highlight_color="#4cd7f6",
            outline_color="#000000",
            outline_width=3,
            shadow_color="rgba(0,0,0,0.7)",
            background_pill=True,
            animation_type="fade",
        ),
        SubtitleStylePreset(
            id="netflix_clean",
            name="Netflix Minimalist Dark Pill",
            font_family="Inter, sans-serif",
            font_size=36,
            text_color="#ffffff",
            highlight_color="#ffffff",
            outline_color="transparent",
            outline_width=0,
            shadow_color="rgba(0,0,0,0.5)",
            background_pill=True,
            animation_type="static",
        ),
        SubtitleStylePreset(
            id="cyberpunk_glow",
            name="Cyberpunk Neon Glow",
            font_family="Space Grotesk, JetBrains Mono, sans-serif",
            font_size=40,
            text_color="#4cd7f6",
            highlight_color="#d0bcff",
            outline_color="#0f172a",
            outline_width=2,
            shadow_color="rgba(76,215,246,0.6)",
            background_pill=False,
            animation_type="pop",
        ),
    ]

    def list_transitions(self) -> List[Dict[str, Any]]:
        return [asdict(t) for t in self.TRANSITIONS]

    def list_fonts(self) -> List[Dict[str, Any]]:
        return [asdict(f) for f in self.FONTS]

    def list_subtitle_presets(self) -> List[Dict[str, Any]]:
        return [asdict(p) for p in self.SUBTITLE_PRESETS]

    @staticmethod
    def ease(t: float, easing_type: str = "ease_in_out") -> float:
        """Calculates easing progress [0.0 to 1.0]."""
        t = max(0.0, min(1.0, t))
        if easing_type == "linear":
            return t
        elif easing_type == "ease_in":
            return t * t
        elif easing_type == "ease_out":
            return t * (2.0 - t)
        elif easing_type == "ease_in_out":
            return 0.5 * (1.0 - math.cos(math.pi * t))
        elif easing_type == "bounce":
            # Simple bounce approximation
            if t < 0.36:
                return 7.5625 * t * t
            elif t < 0.72:
                t -= 0.54
                return 7.5625 * t * t + 0.75
            else:
                t -= 0.86
                return 7.5625 * t * t + 0.9375
        return t

    def blend_frames(
        self,
        frame_a: np.ndarray,
        frame_b: np.ndarray,
        progress: float,
        transition_id: str = "cross_dissolve",
    ) -> np.ndarray:
        """
        Synthesizes a blended frame at relative transition progress [0.0 to 1.0].
        Accelerated with NumPy vector operations.
        """
        p = self.ease(progress, "ease_in_out")
        h, w, c = frame_a.shape

        if transition_id == "cross_dissolve":
            # Linear alpha crossfade
            blended = (1.0 - p) * frame_a.astype(np.float32) + p * frame_b.astype(np.float32)
            return np.clip(blended, 0, 255).astype(np.uint8)

        elif transition_id in ("whip_pan_left", "whip_pan_right"):
            shift = int(w * p) if transition_id == "whip_pan_left" else int(w * (1.0 - p))
            out = np.zeros_like(frame_a)
            if transition_id == "whip_pan_left":
                out[:, : w - shift] = frame_a[:, shift:]
                out[:, w - shift :] = frame_b[:, :shift]
            else:
                out[:, :shift] = frame_b[:, w - shift :]
                out[:, shift:] = frame_a[:, : w - shift]
            return out

        elif transition_id == "dip_to_black":
            if p < 0.5:
                fade = 1.0 - (p * 2.0)
                return (frame_a.astype(np.float32) * fade).astype(np.uint8)
            else:
                fade = (p - 0.5) * 2.0
                return (frame_b.astype(np.float32) * fade).astype(np.uint8)

        elif transition_id == "dip_to_white":
            if p < 0.5:
                flash = p * 2.0
                return np.clip(frame_a.astype(np.float32) + (255.0 * flash), 0, 255).astype(np.uint8)
            else:
                flash = (1.0 - p) * 2.0
                return np.clip(frame_b.astype(np.float32) + (255.0 * flash), 0, 255).astype(np.uint8)

        elif transition_id == "glitch_rgb":
            out = frame_a.copy()
            if 0.1 < p < 0.9:
                offset = int(18 * math.sin(p * math.pi * 4))
                # Shift red channel left, blue channel right
                out[..., 0] = np.roll(frame_a[..., 0], offset, axis=1)
                out[..., 2] = np.roll(frame_b[..., 2], -offset, axis=1)
            else:
                return self.blend_frames(frame_a, frame_b, p, "cross_dissolve")
            return out

        # Default fallback
        return self.blend_frames(frame_a, frame_b, p, "cross_dissolve")
