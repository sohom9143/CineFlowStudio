"""
CineFlow-AI: Indian Bangla & Multilingual Neural TTS Engine
===========================================================
Zero-VRAM neural speech synthesis powered by Microsoft Edge Neural TTS (`edge-tts`)
with curated presets for Indian Bengali (bn-IN), Hindi (hi-IN), and Indian English (en-IN).
Automatically converts output audio to 16 kHz mono WAV for Lip-Sync and FFmpeg muxing.
"""

from __future__ import annotations

import os
import sys
import time
import asyncio
import logging
import subprocess
import shutil
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("CineFlow.AudioTTS")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    edge_tts = None  # type: ignore
    EDGE_TTS_AVAILABLE = False


INDIAN_VOICE_PRESETS: Dict[str, str] = {
    "None (Silent Video)": "",
    "Bengali (India) - Male (Neel / Bashkar)": "bn-IN-BashkarNeural",
    "Bengali (India) - Female (Meghla / Tanishaa)": "bn-IN-TanishaaNeural",
    "Bengali (Bangladesh) - Male (Pradeep)": "bn-BD-PradeepNeural",
    "Bengali (Bangladesh) - Female (Nabanita)": "bn-BD-NabanitaNeural",
    "Hindi (India) - Male (Madhur)": "hi-IN-MadhurNeural",
    "Hindi (India) - Female (Swara)": "hi-IN-SwaraNeural",
    "English (India) - Male (Prabhat)": "en-IN-PrabhatNeural",
    "English (India) - Female (Neerja)": "en-IN-NeerjaNeural",
}


def list_voice_choices() -> List[str]:
    """Returns UI-friendly voice preset labels."""
    return list(INDIAN_VOICE_PRESETS.keys())


def resolve_voice_id(voice_label_or_id: Optional[str]) -> str:
    """Resolves a UI voice label or raw edge-tts voice ID."""
    if not voice_label_or_id:
        return ""
    val = str(voice_label_or_id).strip()
    if val in INDIAN_VOICE_PRESETS:
        return INDIAN_VOICE_PRESETS[val]
    # Already a voice ID like "bn-IN-BashkarNeural"
    if "-" in val and "Neural" in val:
        return val
    return ""


async def _synthesize_edge_tts_async(text: str, voice_id: str, mp3_path: str) -> None:
    communicate = edge_tts.Communicate(text, voice_id)
    await communicate.save(mp3_path)


def synthesize_dialogue_audio(
    text: str,
    voice_choice: str = "Bengali (India) - Male (Neel / Bashkar)",
    output_dir: str = "outputs/temp_tts",
    sample_rate: int = 16000,
) -> Optional[str]:
    """
    Synthesizes Indian Bangla / Hindi / English dialogue into a 16 kHz mono WAV file.
    Returns the WAV file path on success, or None if silent / text empty.
    """
    clean_text = (text or "").strip()
    voice_id = resolve_voice_id(voice_choice)
    if not clean_text or not voice_id:
        return None

    if not EDGE_TTS_AVAILABLE:
        raise RuntimeError(
            "edge-tts package is not installed. Run `pip install edge-tts` to enable Indian Bangla voice generation."
        )

    os.makedirs(output_dir, exist_ok=True)
    ts = int(time.time() * 1000)
    mp3_path = os.path.abspath(os.path.join(output_dir, f"tts_{voice_id}_{ts}.mp3"))
    wav_path = os.path.abspath(os.path.join(output_dir, f"tts_{voice_id}_{ts}.wav"))

    logger.info(f"Synthesizing TTS Audio | Voice: {voice_id} | Text: '{clean_text[:60]}...'")

    # Safely run async edge-tts even if already inside an active event loop (Gradio/FastAPI)
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(asyncio.run, _synthesize_edge_tts_async(clean_text, voice_id, mp3_path)).result()
    else:
        asyncio.run(_synthesize_edge_tts_async(clean_text, voice_id, mp3_path))

    if not os.path.exists(mp3_path) or os.path.getsize(mp3_path) == 0:
        raise RuntimeError("TTS synthesis produced an empty audio file.")

    # Convert to 16 kHz mono WAV using ffmpeg if available for clean Lip-Sync & muxing
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin:
        cmd = [
            ffmpeg_bin, "-y", "-i", mp3_path,
            "-ar", str(sample_rate), "-ac", "1", "-c:a", "pcm_s16le",
            wav_path,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0 and os.path.exists(wav_path):
            try:
                os.remove(mp3_path)
            except OSError:
                pass
            return wav_path

    return mp3_path
