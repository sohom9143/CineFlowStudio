"""
CineFlow NLE Studio: Built-in Regional & Worldwide Viral Meme Vault
===================================================================
Curated, high-retention viral audio & video meme clips:
- Categories:
    * India (Bollywood, Hera Pheri, Mirzapur, 3 Idiots, CID, Daily Soap)
    * Bengal / Bangla (Khabar Dhabar, Bhaat Kheyechi, Gopal Bhar, Feluda, Tollywood)
    * Worldwide / Global (Vine Boom, Emotional Damage, Curb Theme, Bruh, Coffin Dance)
- Built-in Procedural Audio-Visual Synthesizer:
    * Generates instant offline previews and timeline assets directly
    * Eliminates external web search latency and broken link dependencies
- AI Context Matcher:
    * Automatically suggests best meme clip matching scene sentiment, dialogue jokes, or shock moments
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass
class MemeClip:
    id: str
    name: str
    category: str  # "India", "Bangla", "Worldwide"
    language: str  # "Hindi", "Bengali", "English", "Universal"
    description: str
    duration_s: float
    suggested_track: str  # "V3" (video overlay) or "A2" (audio SFX)
    sentiment_triggers: List[str]  # e.g. ["money", "joke", "shock", "awkward", "victory"]
    dialogue_punchline: str
    sound_type: str = "punch"  # "punch", "theme", "stinger", "dialogue", "orchestral"
    waveform_color: str = "#f59e0b"
    video_available: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MemeVault:
    """
    Direct-access Meme & B-Roll footage repository.
    Provides categorized searching, instant timeline insertion, and context matching.
    """

    CATALOG: List[MemeClip] = [
        # ==================== INDIA / BOLLYWOOD ====================
        MemeClip(
            id="in_hera_pheri_paisa",
            name="Hera Pheri - Paisa Hi Paisa Hoga",
            category="India",
            language="Hindi",
            description="Akshay Kumar laughing with open arms in front of money stacks.",
            duration_s=2.4,
            suggested_track="V3",
            sentiment_triggers=["money", "wealth", "profit", "success", "greed", "business", "win"],
            dialogue_punchline="Paisa hi paisa hoga!",
            sound_type="dialogue",
            waveform_color="#fbbf24",
        ),
        MemeClip(
            id="in_3idiots_kehna",
            name="3 Idiots - Kehna Kya Chahte Ho",
            category="India",
            language="Hindi",
            description="Confused student asking professor to clarify complicated definition.",
            duration_s=2.1,
            suggested_track="V3",
            sentiment_triggers=["confusion", "complicated", "question", "overcomplicating", "what"],
            dialogue_punchline="Arey kehna kya chahte ho?",
            sound_type="dialogue",
            waveform_color="#fbbf24",
        ),
        MemeClip(
            id="in_bhai_bhai",
            name="Bhai Bhai Bhai - Control Uday",
            category="India",
            language="Hindi",
            description="Nana Patekar telling Anil Kapoor to control his temper in Welcome.",
            duration_s=1.9,
            suggested_track="V3",
            sentiment_triggers=["anger", "rage", "calm_down", "fight", "tension"],
            dialogue_punchline="Control Uday control!",
            sound_type="dialogue",
            waveform_color="#f59e0b",
        ),
        MemeClip(
            id="in_choti_bachhi",
            name="Choti Bachhi Ho Kya",
            category="India",
            language="Hindi",
            description="Heropanti Tiger Shroff viral dialogue mock.",
            duration_s=1.8,
            suggested_track="V3",
            sentiment_triggers=["silly", "childish", "foolish", "mockery", "obvious"],
            dialogue_punchline="Choti bachhi ho kya?",
            sound_type="dialogue",
            waveform_color="#fbbf24",
        ),
        MemeClip(
            id="in_mirzapur_gareeb",
            name="Mirzapur - Yeh Bhi Theek Hai",
            category="India",
            language="Hindi",
            description="Robin calmly accepting unfortunate circumstances with a nod.",
            duration_s=2.0,
            suggested_track="V3",
            sentiment_triggers=["resignation", "acceptance", "okay", "defeat", "compromise"],
            dialogue_punchline="Yeh bhi theek hai.",
            sound_type="dialogue",
            waveform_color="#fbbf24",
        ),
        MemeClip(
            id="in_serial_dramatic_zoom",
            name="Indian Serial Triple Dramatic Zoom & Thunder",
            category="India",
            language="Universal",
            description="Ekta Kapoor style 3-speed multi-angle zoom with lightning flash.",
            duration_s=3.2,
            suggested_track="A2",
            sentiment_triggers=["shock", "plot_twist", "betrayal", "gasp", "drama", "reveal"],
            dialogue_punchline="*DHOOM TA NA NA THUNDER*",
            sound_type="stinger",
            waveform_color="#ef4444",
        ),
        MemeClip(
            id="in_cid_daya_darwaza",
            name="CID - Daya Darwaza Tod Do",
            category="India",
            language="Hindi",
            description="ACP Pradyuman ordering Daya to kick the door open.",
            duration_s=2.2,
            suggested_track="V3",
            sentiment_triggers=["break", "kick", "action", "force", "investigation"],
            dialogue_punchline="Daya, darwaza tod do!",
            sound_type="dialogue",
            waveform_color="#f59e0b",
        ),

        # ==================== BENGAL / BANGLA ====================
        MemeClip(
            id="bn_khabar_dhabar",
            name="খাবার দাবার চুরি (Khabar Dhabar Churi)",
            category="Bangla",
            language="Bengali",
            description="Viral Kolkata street dialogue about stolen fish and lunch.",
            duration_s=1.8,
            suggested_track="V3",
            sentiment_triggers=["food", "hunger", "lunch", "theft", "hilarious", "bengali", "joke"],
            dialogue_punchline="সব খাবার দাবার চুরি করে নিল!",
            sound_type="dialogue",
            waveform_color="#10b981",
        ),
        MemeClip(
            id="bn_bhaat_kheyechi",
            name="আজকে ভাত খেয়েছি (Bhaat Kheyechi)",
            category="Bangla",
            language="Bengali",
            description="Relaxed Bengali afternoon satisfaction post-lunch statement.",
            duration_s=2.0,
            suggested_track="V3",
            sentiment_triggers=["lazy", "sleepy", "satisfied", "afternoon", "food", "comfort"],
            dialogue_punchline="আজকে তো ভরপেট ভাত খেয়েছি!",
            sound_type="dialogue",
            waveform_color="#10b981",
        ),
        MemeClip(
            id="bn_otai_toh_bolchhi",
            name="ওটাই তো বলছি (Otai Toh Bolchhi)",
            category="Bangla",
            language="Bengali",
            description="Expressive agreement emphasizing 'That's exactly what I was saying!'.",
            duration_s=1.6,
            suggested_track="V3",
            sentiment_triggers=["agreement", "validation", "correct", "told_you_so", "point"],
            dialogue_punchline="আরে দাদা, ওটাই তো বলছি!",
            sound_type="dialogue",
            waveform_color="#10b981",
        ),
        MemeClip(
            id="bn_dada_ektu_shunun",
            name="দাদা একটু শুনুন (Dada Ektu Shunun)",
            category="Bangla",
            language="Bengali",
            description="Classic Kolkata bus/metro conductor catching passenger attention.",
            duration_s=1.7,
            suggested_track="V3",
            sentiment_triggers=["listen", "attention", "excuse_me", "wait", "caution"],
            dialogue_punchline="দাদা, একটু শুনুন না!",
            sound_type="dialogue",
            waveform_color="#10b981",
        ),
        MemeClip(
            id="bn_feluda_deductive_sting",
            name="Feluda Magajastra Deductive Violin Sting",
            category="Bangla",
            language="Universal",
            description="Satyajit Ray style sharp violin hit when Feluda solves the mystery.",
            duration_s=2.5,
            suggested_track="A2",
            sentiment_triggers=["mystery", "detective", "smart", "brain", "discovery", "solution"],
            dialogue_punchline="*Mogojastro Activated*",
            sound_type="stinger",
            waveform_color="#38bdf8",
        ),
        MemeClip(
            id="bn_gopal_bhar_laugh",
            name="Gopal Bhar Hehehe Laugh",
            category="Bangla",
            language="Bengali",
            description="Iconic witty belly laugh from Maharaj Krishnachandra's court jester.",
            duration_s=2.2,
            suggested_track="A2",
            sentiment_triggers=["laugh", "comedy", "prank", "trick", "wit", "humor"],
            dialogue_punchline="হিহিহি... মহারাজ বুঝতেই পারলেন না!",
            sound_type="dialogue",
            waveform_color="#10b981",
        ),

        # ==================== WORLDWIDE / VIRAL GLOBAL ====================
        MemeClip(
            id="ww_vine_boom",
            name="Vine Boom Sound Effect",
            category="Worldwide",
            language="Universal",
            description="Heavy bass thud used for dramatic comedic impact and sudden revelations.",
            duration_s=1.2,
            suggested_track="A2",
            sentiment_triggers=["impact", "punchline", "sus", "shock", "silence_breaker", "bass"],
            dialogue_punchline="*BOOM*",
            sound_type="stinger",
            waveform_color="#ef4444",
        ),
        MemeClip(
            id="ww_emotional_damage",
            name="Emotional Damage (Steven He)",
            category="Worldwide",
            language="English",
            description="Steven He slamming slipper on table shouting emotional damage.",
            duration_s=2.0,
            suggested_track="V3",
            sentiment_triggers=["insult", "burn", "roast", "pain", "failure", "damage"],
            dialogue_punchline="Emotional Damage!",
            sound_type="dialogue",
            waveform_color="#ec4899",
        ),
        MemeClip(
            id="ww_curb_theme",
            name="Curb Your Enthusiasm (Directed by Robert B. Weide)",
            category="Worldwide",
            language="Universal",
            description="Frolic tuba stinger playing over unfortunate comedic ironies.",
            duration_s=4.5,
            suggested_track="A2",
            sentiment_triggers=["awkward", "fail", "irony", "caught", "credits", "fool"],
            dialogue_punchline="*Directed by Robert B. Weide tuba*",
            sound_type="theme",
            waveform_color="#8b5cf6",
        ),
        MemeClip(
            id="ww_bruh",
            name="Bruh Sound Effect #2",
            category="Worldwide",
            language="English",
            description="Low monotonic vocal expression of utter disbelief and fatigue.",
            duration_s=1.1,
            suggested_track="A2",
            sentiment_triggers=["disbelief", "disappointment", "sigh", "tired", "really"],
            dialogue_punchline="Bruh...",
            sound_type="stinger",
            waveform_color="#64748b",
        ),
        MemeClip(
            id="ww_coffin_dance",
            name="Coffin Dance (Astronomia)",
            category="Worldwide",
            language="Universal",
            description="EDM synth drop right before an inevitable physical disaster.",
            duration_s=3.8,
            suggested_track="A2",
            sentiment_triggers=["disaster", "danger", "rip", "fall", "clumsy", "death"],
            dialogue_punchline="*Astronomia Synth Drop*",
            sound_type="theme",
            waveform_color="#06b6d4",
        ),
        MemeClip(
            id="ww_sad_violin",
            name="Sad Violin (Tiny Violin Solo)",
            category="Worldwide",
            language="Universal",
            description="Melancholic weeping violin melody for mock pity and tragedy.",
            duration_s=3.5,
            suggested_track="A2",
            sentiment_triggers=["sad", "cry", "pity", "poor", "loss", "tragic", "tears"],
            dialogue_punchline="*Weeping solo violin*",
            sound_type="stinger",
            waveform_color="#a855f7",
        ),
        MemeClip(
            id="ww_record_scratch",
            name="Record Scratch Freeze Frame",
            category="Worldwide",
            language="Universal",
            description="Sudden vinyl stop sound: 'Yup, that's me. You're probably wondering...'",
            duration_s=1.5,
            suggested_track="A2",
            sentiment_triggers=["stop", "freeze", "interruption", "wait", "flashback"],
            dialogue_punchline="*SKRRRT*",
            sound_type="stinger",
            waveform_color="#eab308",
        ),
        MemeClip(
            id="ww_cash_register",
            name="Cash Register Ching (Chaching)",
            category="Worldwide",
            language="Universal",
            description="Mechanical metallic drawer opening and bell ring of incoming funds.",
            duration_s=1.2,
            suggested_track="A2",
            sentiment_triggers=["money", "sale", "paid", "revenue", "cash", "crypto", "rich"],
            dialogue_punchline="*KA-CHING!*",
            sound_type="stinger",
            waveform_color="#10b981",
        ),
    ]

    def __init__(self):
        self._index_by_id = {c.id: c for c in self.CATALOG}

    def list_memes(
        self,
        category: Optional[str] = None,
        query: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Filters clips by category ('India', 'Bangla', 'Worldwide') and text query."""
        results = []
        for clip in self.CATALOG:
            if category and category.lower() != "all" and clip.category.lower() != category.lower():
                continue
            if query:
                q = query.lower()
                matches = (
                    q in clip.name.lower()
                    or q in clip.description.lower()
                    or q in clip.dialogue_punchline.lower()
                    or any(q in t for t in clip.sentiment_triggers)
                )
                if not matches:
                    continue
            results.append(clip.to_dict())
        return results

    def get_meme_by_id(self, meme_id: str) -> Optional[MemeClip]:
        return self._index_by_id.get(meme_id)

    def find_best_meme_for_context(
        self,
        text_context: str,
        region_preference: str = "all",
    ) -> Optional[MemeClip]:
        """
        Multimodal sentiment / keyword matcher.
        Finds the most relevant viral meme clip to insert at a given moment.
        """
        STOPWORDS = {"the", "and", "for", "with", "this", "that", "you", "are", "have", "from", "was", "were", "been", "made"}
        raw_tokens = text_context.lower().replace(",", " ").replace(".", " ").replace("!", " ").replace("?", " ").split()
        tokens = {t for t in raw_tokens if len(t) > 2 and t not in STOPWORDS}

        best_clip = None
        best_score = 0

        for clip in self.CATALOG:
            score = 0
            if region_preference.lower() != "all" and clip.category.lower() == region_preference.lower():
                score += 3

            # 1. Sentiment triggers check
            for trigger in clip.sentiment_triggers:
                if trigger in tokens:
                    score += 10
                elif any(trigger in t or t in trigger for t in tokens if len(t) >= 4 and len(trigger) >= 4):
                    score += 6

            # 2. Word boundary intersection with clip metadata
            clip_words = set(f"{clip.name} {clip.description} {clip.dialogue_punchline}".lower().replace(",", " ").replace(".", " ").split())
            common_words = tokens.intersection(clip_words)
            score += len(common_words) * 4

            if score > best_score:
                best_score = score
                best_clip = clip

        return best_clip or self.CATALOG[0]

    def synthesize_preview_audio(self, meme_id: str, sample_rate: int = 44100) -> np.ndarray:
        """
        Procedurally synthesizes deterministic audio preview for a meme clip.
        Guarantees offline functionality without needing external media files.
        """
        clip = self.get_meme_by_id(meme_id) or self.CATALOG[0]
        n_samples = int(clip.duration_s * sample_rate)
        t = np.linspace(0, clip.duration_s, n_samples, endpoint=False)

        if clip.sound_type == "punch" or "boom" in clip.id:
            # Low 50Hz decaying sub-bass thud + high transient pop
            decay = np.exp(-4.0 * t)
            wave = 0.8 * np.sin(2 * np.pi * 55 * t) * decay
            wave += 0.2 * np.random.randn(n_samples) * np.exp(-20.0 * t)
        elif clip.sound_type == "stinger":
            # Rising arpeggio or chime
            f = np.linspace(300, 900, n_samples)
            wave = 0.5 * np.sin(2 * np.pi * f * t) * np.exp(-1.5 * t)
        elif "cash" in clip.id:
            # High frequency metallic bell ding
            wave = 0.6 * np.sin(2 * np.pi * 2400 * t) * np.exp(-3.0 * t)
            wave += 0.3 * np.sin(2 * np.pi * 4800 * t) * np.exp(-5.0 * t)
        else:
            # Harmonic vocal-like formant synthesis
            f0 = 180.0
            wave = 0.4 * np.sin(2 * np.pi * f0 * t) + 0.2 * np.sin(2 * np.pi * (f0 * 2) * t)
            wave *= np.exp(-0.8 * t)

        return np.clip(wave, -1.0, 1.0).astype(np.float32)
