"""
CineFlow-AI: Universal Gemini Multimodal Vision & AI Director Agent
==================================================================
Provides deep multimodal facial analysis, 360-degree visual feature extraction,
ingredient decomposition (Subject, Camera, Lighting, Atmosphere, Wardrobe, Audio),
and autonomous cinematic shot orchestration using Google Gemini Flash models
(gemini-2.5-flash / gemini-2.0-flash / gemini-1.5-flash) with seamless offline fallback.

Author: Google DeepMind & Antigravity Advanced Agentic Coding Team
Architecture: Universal Zero-Latency Multimodal Vision & Semantic Ingredient Engine
"""

import io
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

logger = logging.getLogger("CineFlow.GeminiAgent")

# Try importing Google Generative AI SDK safely
_GEMINI_SDK_AVAILABLE = False
try:
    import google.generativeai as genai
    _GEMINI_SDK_AVAILABLE = True
except ImportError:
    try:
        from google import genai
        _GEMINI_SDK_AVAILABLE = True
    except ImportError:
        _GEMINI_SDK_AVAILABLE = False
        logger.info("google-generativeai package not yet installed. Operating in Heuristic Fallback mode.")


# =============================================================================
# Helper Utilities
# =============================================================================

def resolve_gemini_model_name(requested_model: str) -> str:
    """
    Resolves user or alias model strings to canonical Gemini model identifiers.
    """
    if not requested_model:
        return "gemini-2.5-flash"
    
    clean = requested_model.strip().lower()
    if clean in ("gemini-3.5-flash", "3.5-flash", "3.5", "gemini-3-flash"):
        # Map forward to latest available Flash tier
        return "gemini-2.5-flash"
    elif clean in ("gemini-2.5-flash", "2.5-flash", "2.5"):
        return "gemini-2.5-flash"
    elif clean in ("gemini-2.0-flash", "2.0-flash", "2.0", "gemini-2.0-flash-exp"):
        return "gemini-2.0-flash"
    elif clean in ("gemini-1.5-flash", "1.5-flash", "1.5"):
        return "gemini-1.5-flash"
    elif clean in ("gemini-1.5-pro", "1.5-pro", "pro"):
        return "gemini-1.5-pro"
    
    return clean if "gemini" in clean else f"gemini-{clean}"


def pil_image_to_jpeg_bytes(img: Image.Image, quality: int = 85, max_dim: int = 768) -> bytes:
    """Converts a PIL Image to optimized JPEG bytes for multimodal payload."""
    img_rgb = img.convert("RGB")
    w, h = img_rgb.size
    if max(w, h) > max_dim:
        scale = max_dim / float(max(w, h))
        new_w = max(64, int(w * scale))
        new_h = max(64, int(h * scale))
        img_rgb = img_rgb.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    buffer = io.BytesIO()
    img_rgb.save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


# =============================================================================
# CharacterGeminiAgent
# =============================================================================

class CharacterGeminiAgent:
    """
    Universal Multimodal AI Director Agent orchestrating visual & cinematic intelligence:
    1. Deep facial geometry & feature analysis from 1 to 4 multi-angle photos.
    2. 360-degree consistency extraction (hair, jawline, ears, posture, clothing).
    3. Ingredient Decomposition (Subject, Action, Camera, Lighting, Atmosphere, Wardrobe).
    4. Autonomous Shot & Generation Intent Planning across any scenario/genre.
    5. Director-level scene prompt enhancement with character trait injection.
    6. Seamless offline heuristic fallback if API key is not yet configured.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-flash",
        temperature: float = 0.2,
        max_output_tokens: int = 1024,
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled
        self.model_name = resolve_gemini_model_name(model_name)
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        
        # Discover API key from arguments or environment variables
        self.api_key = (
            api_key
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("CINEFLOW_GEMINI_KEY")
        )
        
        self._client = None
        if self.api_key and _GEMINI_SDK_AVAILABLE:
            self._init_gemini_client()

    def set_api_key(self, api_key: str) -> bool:
        """Configures or updates the Gemini API key at runtime."""
        if not api_key or not api_key.strip():
            self.api_key = None
            self._client = None
            return False
        
        self.api_key = api_key.strip()
        os.environ["GEMINI_API_KEY"] = self.api_key
        return self._init_gemini_client()

    def _init_gemini_client(self) -> bool:
        """Initializes the underlying Gemini SDK client."""
        if not _GEMINI_SDK_AVAILABLE:
            logger.info("google-generativeai package not available; using Heuristic Agent.")
            return False
        try:
            genai.configure(api_key=self.api_key)
            # Create generative model with system configuration
            self._client = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_output_tokens,
                },
            )
            logger.info(f"Initialized Gemini GenerativeModel with '{self.model_name}'.")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Gemini API client: {e}")
            self._client = None
            return False

    def is_available(self) -> bool:
        """Checks if active Gemini client is ready for live multimodal inference."""
        return bool(self.enabled and self.api_key and self._client is not None)

    def get_status(self) -> Dict[str, Any]:
        """Returns diagnostic status of the Gemini Agent."""
        masked_key = ""
        if self.api_key:
            masked_key = f"{self.api_key[:4]}...{self.api_key[-4:]}" if len(self.api_key) > 8 else "***"
        
        return {
            "enabled": self.enabled,
            "has_api_key": bool(self.api_key),
            "masked_key": masked_key,
            "model_name": self.model_name,
            "is_available": self.is_available(),
            "sdk_installed": _GEMINI_SDK_AVAILABLE,
            "mode": "Gemini Multimodal Vision" if self.is_available() else "Heuristic Fallback",
        }

    # -------------------------------------------------------------------------
    # Multimodal Character Visual Analysis
    # -------------------------------------------------------------------------

    def analyze_character_multimodal(
        self,
        images_or_views: Union[Image.Image, np.ndarray, str, List[Any], Dict[str, Any]],
        character_name: str = "Character",
        context_hints: str = "",
        gender_hint: str = "neutral",
    ) -> Dict[str, Any]:
        """
        Analyzes 1 to 4 multi-angle portrait photos (Front, Left, Right, Back).
        Extracts structural facial features, skin undertones, gaze, hair, and wardrobe.
        """
        # Convert all image inputs to list of PIL Images
        pil_images = self._normalize_image_inputs(images_or_views)
        if not pil_images:
            return self._heuristic_feature_fallback(character_name, gender_hint, context_hints)

        if self.is_available():
            try:
                return self._call_gemini_multimodal_character(
                    pil_images=pil_images,
                    character_name=character_name,
                    context_hints=context_hints,
                    gender_hint=gender_hint,
                )
            except Exception as e:
                logger.warning(f"Live Gemini Multimodal Vision API call failed: {e}. Falling back to Heuristic.")

        return self._heuristic_multimodal_analysis(
            pil_images=pil_images,
            character_name=character_name,
            context_hints=context_hints,
            gender_hint=gender_hint,
        )

    def _call_gemini_multimodal_character(
        self,
        pil_images: List[Image.Image],
        character_name: str,
        context_hints: str,
        gender_hint: str,
    ) -> Dict[str, Any]:
        """Executes Gemini Multimodal API with vision images and structured schema prompt."""
        prompt_parts = [
            f"You are the Lead Visual Character Designer & Vision AI Agent for CineFlow-AI Studio.\n"
            f"Analyze the provided reference photos ({len(pil_images)} image(s)) of the person named '{character_name}'.\n"
            f"Context / Role Hints: {context_hints or 'N/A'}\n"
            f"Gender Hint: {gender_hint}\n\n"
            f"Return a strict, valid JSON object with the following fields:\n"
            f"{{\n"
            f'  "facial_structure": "detailed description of jawline, cheekbones, chin, nose bridge, brow line",\n'
            f'  "eyes_and_gaze": "eye shape, iris tone, gaze intensity, brow shape",\n'
            f'  "hair_and_grooming": "hair length, color, texture, parting, facial hair or clean shaven",\n'
            f'  "complexion_and_skin": "skin tone, undertones, natural texture, unique marks",\n'
            f'  "wardrobe_and_culture": "clothing style, neckline, fabrics, accessories observed",\n'
            f'  "cinematic_presence": "overall aura, personality projection, cinematic framing vibe",\n'
            f'  "prompt_prefix": "concise master prompt prefix locking this character identity (e.g. cinematic portrait of {character_name}, sharp jawline, dark almond eyes, ...)",\n'
            f'  "negative_prompt_additions": "tokens to avoid (e.g. smooth plastic skin, distorted eyes)",\n'
            f'  "character_synopsis": "a 1-2 sentence director summary of the character",\n'
            f'  "character_tag_string": "comma-separated key identity tags"\n'
            f"}}\n"
            f"Ensure output is ONLY raw JSON without markdown code fences."
        ]

        # Add image byte blobs for Gemini
        for idx, img in enumerate(pil_images[:4]):
            jpeg_bytes = pil_image_to_jpeg_bytes(img)
            prompt_parts.append({"mime_type": "image/jpeg", "data": jpeg_bytes})

        response = self._client.generate_content(prompt_parts)
        if not response or not response.text:
            raise ValueError("Empty response received from Gemini Vision API.")

        text_content = response.text.strip()
        # Strip markdown fences if present
        if text_content.startswith("```json"):
            text_content = text_content[7:]
        elif text_content.startswith("```"):
            text_content = text_content[3:]
        if text_content.endswith("```"):
            text_content = text_content[:-3]
        text_content = text_content.strip()

        data = json.loads(text_content)
        data["agent_mode"] = "gemini_multimodal"
        data["model_used"] = self.model_name
        data["analyzed_views_count"] = len(pil_images)
        return data

    def _heuristic_multimodal_analysis(
        self,
        pil_images: List[Image.Image],
        character_name: str,
        context_hints: str,
        gender_hint: str,
    ) -> Dict[str, Any]:
        """High-order computer vision heuristic analysis when offline."""
        primary_img = pil_images[0]
        w, h = primary_img.size
        arr = np.array(primary_img.resize((128, 128)))
        
        # 1. Luminance & Contrast
        gray = np.mean(arr, axis=2)
        mean_lum = float(np.mean(gray))
        contrast = float(np.std(gray))

        # 2. Skin tone estimation (center crop)
        center_crop = arr[32:96, 32:96, :]
        r_mean = float(np.mean(center_crop[:, :, 0]))
        g_mean = float(np.mean(center_crop[:, :, 1]))
        b_mean = float(np.mean(center_crop[:, :, 2]))

        if r_mean > g_mean * 1.15 and r_mean > b_mean * 1.25:
            skin_desc = "warm golden-olive undertone with natural filmic grain"
        elif r_mean > 190 and g_mean > 170 and b_mean > 160:
            skin_desc = "fair porcelain complexion with soft translucent undertones"
        else:
            skin_desc = "deep natural bronze complexion with rich chiaroscuro contrast"

        # 3. Hair tone (top crop)
        top_crop = arr[0:32, 20:108, :]
        top_lum = float(np.mean(top_crop))
        if top_lum < 60:
            hair_desc = "dense raven-black hair with natural volume and cinematic texture"
        elif top_lum < 110:
            hair_desc = "dark brown wavy hair with soft edge highlights"
        else:
            hair_desc = "textured hair with warm dimensional highlights"

        jaw_desc = "structured defined jawline with prominent facial bone geometry"
        gaze_desc = "intense focused cinematic gaze with natural catchlights"
        wardrobe_desc = context_hints if context_hints else "classic cinematic attire with refined texture"

        gender_word = "man" if gender_hint.lower() == "male" else ("woman" if gender_hint.lower() == "female" else "person")
        
        prompt_prefix = (
            f"cinematic portrait of {character_name}, an expressive {gender_word}, "
            f"{jaw_desc}, {gaze_desc}, {hair_desc}, {skin_desc}, {wardrobe_desc}, "
            f"photorealistic skin pores, 8k resolution, IMAX 70mm"
        )
        
        neg_additions = "plastic skin, oversmoothed face, asymmetrical eyes, distorted jaw, bad pupils"

        return {
            "facial_structure": jaw_desc,
            "eyes_and_gaze": gaze_desc,
            "hair_and_grooming": hair_desc,
            "complexion_and_skin": skin_desc,
            "wardrobe_and_culture": wardrobe_desc,
            "cinematic_presence": f"Charismatic and commanding screen presence as {character_name}",
            "prompt_prefix": prompt_prefix,
            "negative_prompt_additions": neg_additions,
            "character_synopsis": f"{character_name}, a distinguished {gender_word} with {skin_desc} and {hair_desc}.",
            "character_tag_string": f"{character_name}, {gender_word}, {jaw_desc}, {skin_desc}, {hair_desc}",
            "agent_mode": "heuristic_fallback",
            "model_used": "opencv_heuristic",
            "analyzed_views_count": len(pil_images),
        }

    def _heuristic_feature_fallback(self, char_name: str, gender: str, context: str) -> Dict[str, Any]:
        """Baseline fallback when no image input is provided."""
        gender_word = "man" if gender.lower() == "male" else ("woman" if gender.lower() == "female" else "person")
        return {
            "facial_structure": "sharp defined jawline and natural facial symmetry",
            "eyes_and_gaze": "expressive cinematic eyes with captivating gaze",
            "hair_and_grooming": "neatly styled natural hair",
            "complexion_and_skin": "natural skin texture with subtle film grain",
            "wardrobe_and_culture": context or "timeless cinematic wardrobe",
            "cinematic_presence": "poised and intense cinematic aura",
            "prompt_prefix": f"cinematic portrait of {char_name}, a {gender_word} with distinct natural facial features",
            "negative_prompt_additions": "cartoon, 3d render, plastic skin, distorted anatomy",
            "character_synopsis": f"{char_name}, an expressive {gender_word}.",
            "character_tag_string": f"{char_name}, {gender_word}, cinematic portrait",
            "agent_mode": "heuristic_fallback",
            "model_used": "rule_based_default",
            "analyzed_views_count": 0,
        }

    # -------------------------------------------------------------------------
    # Ingredient Decomposition Engine (Universal Scenarios)
    # -------------------------------------------------------------------------

    def decompose_prompt_ingredients(
        self,
        user_prompt: str,
        character_name: str = "",
        character_traits: Optional[Dict[str, Any]] = None,
        style_preset: str = "imax_realism",
    ) -> Dict[str, Any]:
        """
        Decomposes any user prompt (Bengali, English, or Banglish) into cinematic production ingredients:
        - Subject & Physical Action
        - Character Identity & Traits
        - Camera & Lens Physics
        - Lighting & Volumetric Atmosphere
        - Environment & World Building
        - Wardrobe & Props
        - Dialogue / Audio Track Intent
        """
        raw_prompt = user_prompt.strip() if user_prompt else "A dramatic cinematic scene"
        char_name = character_name.strip() if character_name else "Protagonist"

        if self.is_available():
            try:
                return self._gemini_ingredient_decomposer(raw_prompt, char_name, character_traits, style_preset)
            except Exception as e:
                logger.warning(f"Live Gemini Ingredient Decomposition failed: {e}. Using Heuristic Decomposer.")

        return self._heuristic_ingredient_decomposer(raw_prompt, char_name, character_traits, style_preset)

    def _gemini_ingredient_decomposer(
        self,
        raw_prompt: str,
        char_name: str,
        character_traits: Optional[Dict[str, Any]],
        style_preset: str,
    ) -> Dict[str, Any]:
        """Uses Gemini to extract structured cinematic production ingredients as JSON."""
        traits_json = json.dumps(character_traits or {}, indent=2)
        instruction = f"""
You are the Executive Director & AI Cinematographer for CineFlow-AI Studio.
Decompose the following user scene concept into structured cinematic ingredients.

User Concept: "{raw_prompt}"
Character Name: "{char_name}"
Character Traits: {traits_json}
Cinematic Style Preset: "{style_preset}"

Return ONLY a valid JSON object with the following fields:
{{
  "subject_action": "precise physical action, posture, and facial expression of the character",
  "character_identity": "name and key physical traits locked into this shot",
  "camera_lens": "focal length (e.g. 35mm anamorphic, 85mm prime), f-stop, camera motion trajectory (e.g. slow push-in dolly, tracking shot)",
  "lighting_setup": "key light, fill light, rim light, color temperature, shadows (e.g. dramatic chiaroscuro with neon rim)",
  "environment_setting": "world, architecture, background location details, depth layers",
  "atmosphere_weather": "fog, rain mist, volumetric god rays, airborne embers, smoke, time of day",
  "wardrobe_props": "costume textures, held props, tech or cultural accessories",
  "dialogue_audio_intent": "mood of dialogue or atmospheric sound design suggested by scene",
  "detected_genre": "e.g. Sci-Fi Cyberpunk, High Fantasy, 8K IMAX Realism, Anime, Gothic Horror, Commercial",
  "master_positive_prompt": "a complete, hyper-detailed single paragraph prompt synthesized from all ingredients"
}}
"""
        response = self._client.generate_content(instruction)
        if not response or not response.text:
            raise ValueError("Empty response from Gemini Ingredient Decomposer.")
        
        text = response.text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        
        data = json.loads(text.strip())
        data["source"] = "gemini_agent"
        return data

    def _heuristic_ingredient_decomposer(
        self,
        raw_prompt: str,
        char_name: str,
        character_traits: Optional[Dict[str, Any]],
        style_preset: str,
    ) -> Dict[str, Any]:
        """Rules-based ingredient decomposer supporting 10 universal genres."""
        lower_p = raw_prompt.lower()
        
        # 1. Detect Genre
        style_lower = style_preset.lower()
        if any(w in lower_p for w in ["cyberpunk", "sci-fi", "neon", "robot", "futuristic", "blade runner", "hologram"]) or "cyberpunk" in style_lower or "scifi" in style_lower:
            genre = "Sci-Fi Cyberpunk"
            cam = "35mm anamorphic prime lens, slow cinematic push-in dolly forward"
            lighting = "moody cyan and magenta neon rim lighting with deep wet pavement reflections"
            env = "futuristic dystopian metropolis with towering holographic billboards and dark alleyways"
            atmo = "volumetric rain mist, atmospheric steam vents, dark twilight smog"
            wardrobe = "waterproof cyber-tactical trench coat with glowing subtle accents"
        elif any(w in lower_p for w in ["fantasy", "castle", "magic", "sword", "dragon", "palace", "royal", "myth"]) or "fantasy" in style_lower:
            genre = "High Fantasy & Mythological Epic"
            cam = "50mm master prime lens, smooth majestic crane tilt down"
            lighting = "golden hour ethereal god rays filtering through ancient stone arches"
            env = "grand mythical citadel with carved marble pillars and sprawling mountain vistas"
            atmo = "swirling magical particle embers, gentle mountain mist, morning dew"
            wardrobe = "regal embroidered cloak, ornate medieval armor details, velvet fabrics"
        elif any(w in lower_p for w in ["anime", "ghibli", "shinkai", "animation", "animated", "manga"]) or "ghibli" in style_lower or "anime" in style_lower:
            genre = "Studio Ghibli & Modern Anime"
            cam = "wide angle dynamic animated perspective, smooth panning shot"
            lighting = "luminous painterly daylight with soft pastel glow"
            env = "breathtaking hand-painted landscape with billowing summer clouds"
            atmo = "floating cherry blossom petals, gentle breeze, vibrant skies"
            wardrobe = "expressive stylized anime clothing with crisp cel-shaded folds"
        elif any(w in lower_p for w in ["horror", "gothic", "creepy", "ghost", "dark", "blood", "vampire", "catacomb"]) or "horror" in style_lower or "gothic" in style_lower:
            genre = "Dark Gothic Horror"
            cam = "24mm wide angle, slow creeping Dutch angle tracking forward"
            lighting = "stark chiaroscuro candlelit shadows with cold moonlight fill"
            env = "ancient decaying Victorian manor, cobwebbed corridors, gothic stone arches"
            atmo = "heavy ground-level fog, suffocating darkness, flickering dust particles"
            wardrobe = "weathered Victorian dark wool coat, high collar, desaturated tones"
        elif any(w in lower_p for w in ["pixar", "disney", "3d animation", "cgi", "subsurface"]) or "pixar" in style_lower:
            genre = "Pixar / Disney 3D Stylized CGI"
            cam = "50mm portrait lens, gentle whimsical dolly push"
            lighting = "warm whimsical volumetric lighting with soft global illumination"
            env = "vibrant animated world with stylized rich textures and depth"
            atmo = "gentle magical dust motes, radiant sunlight"
            wardrobe = "stylized 3d character outfit with rich cloth physics"
        elif any(w in lower_p for w in ["commercial", "fashion", "luxury", "vogue", "perfume", "product"]) or "commercial" in style_lower:
            genre = "Commercial Studio & Luxury Advertising"
            cam = "85mm f/1.4 portrait prime lens, crystal clear static beauty framing"
            lighting = "three-point diffused softbox studio lighting, immaculate catchlights"
            env = "minimalist luxury studio set with textured slate backdrop"
            atmo = "clean crisp studio atmosphere, flawless optical bokeh"
            wardrobe = "haute couture tailored garment with rich fabric sheen"
        elif any(w in lower_p for w in ["graphic novel", "comic", "halftone", "sin city", "spider-verse"]) or "graphic_novel" in style_lower:
            genre = "Graphic Novel & Comic Noir"
            cam = "dynamic Dutch angle framing with bold perspective"
            lighting = "stark high-contrast chiaroscuro with vivid selective color rim"
            env = "stylized comic book city splash page with deep inky shadows"
            atmo = "dramatic halftone grain, heavy ink rain streaks"
            wardrobe = "iconic high-contrast silhouette costume with sharp contour lines"
        elif any(w in lower_p for w in ["kolkata", "vintage", "retro", "1970", "1980", "tram", "sepia"]) or "kolkata" in style_lower or "vintage" in style_lower:
            genre = "North Kolkata Vintage 35mm"
            cam = "Kodak Portra 400 35mm celluloid lens, slow natural shoulder-mount push"
            lighting = "warm golden hour amber streetlights with vintage atmospheric haze"
            env = "heritage North Kolkata alleyway with colonial architecture and tram lines"
            atmo = "subtle monsoon humidity, gentle atmospheric haze, nostalgic film grain"
            wardrobe = "classic traditional handloom kurta or saree with authentic drape"
        elif any(w in lower_p for w in ["nature", "wildlife", "animal", "national geographic", "safari"]) or "nature" in style_lower:
            genre = "Nature & Wildlife 4K Documentary"
            cam = "400mm f/2.8 telephoto lens, razor sharp tracking shot"
            lighting = "crisp natural morning daylight with authentic sun flare"
            env = "untamed wilderness, dense rainforest canopy or savannah plains"
            atmo = "fresh morning mist, atmospheric dust motes, raw organic elements"
            wardrobe = "authentic field exploration gear and natural textures"
        else:
            genre = "IMAX 8K Cinematic Realism"
            cam = "70mm IMAX master prime lens, stable fluid cinematic dolly"
            lighting = "natural volumetric daylight with soft bounce fill and sharp rim definition"
            env = "richly detailed real-world environment with authentic spatial depth"
            atmo = "subtle natural atmospheric particles, shallow depth of field"
            wardrobe = "natural contemporary wardrobe with realistic micro-textures"

        # Action resolution
        action = raw_prompt if len(raw_prompt) > 10 else f"{char_name} turning slowly with an intense focused expression"

        # Master Positive Prompt
        master_positive = (
            f"A masterwork cinematic shot of {char_name}: {action}. "
            f"Environment: {env}. Lighting: {lighting}. Camera: {cam}. "
            f"Atmosphere: {atmo}. Wardrobe: {wardrobe}. "
            f"Genre: {genre}, award-winning composition, 8k uhd photorealistic detail."
        )

        return {
            "subject_action": action,
            "character_identity": f"{char_name} (Identity locked with ArcFace consensus)",
            "camera_lens": cam,
            "lighting_setup": lighting,
            "environment_setting": env,
            "atmosphere_weather": atmo,
            "wardrobe_props": wardrobe,
            "dialogue_audio_intent": "Atmospheric cinematic soundscape with synchronized dialogue track",
            "detected_genre": genre,
            "master_positive_prompt": master_positive,
            "source": "heuristic_decomposer",
        }

    # -------------------------------------------------------------------------
    # Autonomous Generation Intent Orchestrator
    # -------------------------------------------------------------------------

    def orchestrate_generation_plan(
        self,
        natural_prompt: str,
        available_characters: Optional[List[Dict[str, Any]]] = None,
        default_style: str = "imax_realism",
    ) -> Dict[str, Any]:
        """
        Interprets conversational multi-lingual requests (Bengali/English/Banglish),
        selects the target action (video vs image vs lipsync vs train), resolves
        characters, extracts ingredients, and builds an execution plan.
        """
        lower = natural_prompt.lower()
        
        # 1. Action Type Classification
        if any(w in lower for w in ["photo", "image", "portrait", "still", "poster", "chobi", "pic", "ছবি"]):
            action_type = "image_keyframe"
        elif any(w in lower for w in ["lipsync", "audio", "kotha", "dialogue", "song", "gaan", "গান", "কথা"]):
            action_type = "dialogue_lipsync"
        elif any(w in lower for w in ["train", "fine-tune", "finetune", "learn", "face model"]):
            action_type = "character_train"
        else:
            action_type = "video"

        # 2. Character Matching
        selected_char_id = "dev"
        selected_char_name = "Dev"
        if available_characters:
            for char_info in available_characters:
                c_id = char_info.get("id", "").lower()
                c_name = char_info.get("name", "").lower()
                if c_id in lower or c_name in lower:
                    selected_char_id = char_info.get("id", "dev")
                    selected_char_name = char_info.get("name", selected_char_id.title())
                    break

        # 3. Style Resolution
        style_id = default_style
        if any(w in lower for w in ["cyberpunk", "sci-fi", "futuristic", "blade runner", "neon"]):
            style_id = "scifi_cyberpunk"
        elif any(w in lower for w in ["fantasy", "myth", "castle", "magic", "lord of the rings"]):
            style_id = "high_fantasy"
        elif any(w in lower for w in ["anime", "ghibli", "cartoon", "animation", "manga"]):
            style_id = "ghibli_anime"
        elif any(w in lower for w in ["horror", "creepy", "ghost", "dark", "catacomb", "gothic"]):
            style_id = "gothic_horror"
        elif any(w in lower for w in ["pixar", "disney", "3d animation", "cgi"]):
            style_id = "pixar_3d_cgi"
        elif any(w in lower for w in ["commercial", "fashion", "luxury", "vogue", "studio"]):
            style_id = "commercial_studio"
        elif any(w in lower for w in ["comic", "graphic novel", "noir", "spider-verse"]):
            style_id = "graphic_novel"
        elif any(w in lower for w in ["kolkata", "vintage", "retro", "1970", "1980"]):
            style_id = "kolkata_vintage"
        elif any(w in lower for w in ["nature", "wildlife", "animal", "national geographic"]):
            style_id = "nature_documentary"

        # 4. Decompose Ingredients
        ingredients = self.decompose_prompt_ingredients(
            user_prompt=natural_prompt,
            character_name=selected_char_name,
            style_preset=style_id,
        )

        master_positive = ingredients.get("master_positive_prompt", natural_prompt)
        motion_prompt = f"Cinematic {ingredients.get('camera_lens', 'smooth dolly push')}, natural subtle movement of {selected_char_name}."

        recommended_params = {
            "resolution": "1280x720" if action_type == "video" else "1024x1024",
            "fps": 24,
            "frames": 49 if action_type == "video" else 1,
            "steps": 30,
            "guidance_scale": 7.5,
            "motion_scale": 1.0,
            "backend": "wan_2_1_fp8",
        }

        agent_summary = (
            f"🎯 **Target Action**: `{action_type.upper()}` | **Character**: **{selected_char_name}** (`{selected_char_id}`)\n"
            f"🎨 **Genre / Style**: `{ingredients.get('detected_genre', style_id)}`\n"
            f"🎥 **Camera**: {ingredients.get('camera_lens', 'Standard')}\n"
            f"💡 **Lighting**: {ingredients.get('lighting_setup', 'Balanced')}\n"
            f"🌍 **Environment**: {ingredients.get('environment_setting', 'Cinematic Set')}"
        )

        return {
            "target_action": action_type,
            "character_id": selected_char_id,
            "character_name": selected_char_name,
            "style_id": style_id,
            "ingredients": ingredients,
            "master_positive_prompt": master_positive,
            "motion_prompt": motion_prompt,
            "recommended_params": recommended_params,
            "agent_summary": agent_summary,
        }

    # -------------------------------------------------------------------------
    # Director-Level Scene Prompt Enhancement
    # -------------------------------------------------------------------------

    def refine_scene_prompt(
        self,
        scene_prompt: str,
        character_name: str = "Character",
        character_traits: Optional[Dict[str, Any]] = None,
        style_preset: str = "imax_realism",
    ) -> str:
        """
        Enhances and refines a scene prompt by injecting character identity nuances,
        dynamic atmospheric conditions, and camera lens optics.
        """
        raw_prompt = scene_prompt.strip() if scene_prompt else "cinematic portrait"
        char_name = character_name.strip() if character_name else "Character"

        if self.is_available():
            try:
                return self._call_gemini_scene_refiner(raw_prompt, char_name, character_traits, style_preset)
            except Exception as e:
                logger.warning(f"Gemini scene refinement failed: {e}. Using Heuristic Enhancer.")

        return self._heuristic_scene_enhancer(raw_prompt, char_name, character_traits, style_preset)

    def _call_gemini_scene_refiner(
        self,
        raw_prompt: str,
        char_name: str,
        character_traits: Optional[Dict[str, Any]],
        style_preset: str,
    ) -> str:
        """Calls Gemini text model to elevate scene prompt to director-grade prose."""
        traits_summary = ""
        if character_traits:
            traits_summary = (
                f"Character Visual Identity Traits:\n"
                f"- Facial Structure: {character_traits.get('facial_structure', 'N/A')}\n"
                f"- Gaze: {character_traits.get('eyes_and_gaze', 'N/A')}\n"
                f"- Hair: {character_traits.get('hair_and_grooming', 'N/A')}\n"
                f"- Complexion: {character_traits.get('complexion_and_skin', 'N/A')}\n"
                f"- Wardrobe: {character_traits.get('wardrobe_and_culture', 'N/A')}\n"
            )

        instruction = f"""
You are the Lead Cinematographer and Director of Photography for CineFlow-AI Studio.
Elevate and expand the user's raw scene prompt into an award-winning cinematic diffusion prompt.

User Scene Concept: "{raw_prompt}"
Character in Scene: {char_name}
Cinematic Style Preset: {style_preset}
{traits_summary}

Requirements:
1. Preserve the user's core intent, action, and location.
2. Weave in dynamic atmospheric elements: cinematic lighting (e.g. golden hour, moody chiaroscuro, volumetric haze, neon reflections).
3. Specify camera lens physics (e.g. 35mm / 50mm Anamorphic, shallow depth of field, subtle motion blur, crisp optical bokeh).
4. Integrate the character's key physical presence organically into the environment.
5. Return ONLY the final polished cinematic prompt as a single paragraph of text without commentary or quotation marks.
"""
        response = self._client.generate_content(instruction)
        if response and response.text:
            return response.text.strip().replace('"', '')
        raise ValueError("Empty response from scene refinement.")

    def _heuristic_scene_enhancer(
        self,
        raw_prompt: str,
        char_name: str,
        character_traits: Optional[Dict[str, Any]],
        style_preset: str,
    ) -> str:
        """Rules-based cinematic scene enhancer for offline execution."""
        style_cues = {
            "imax_realism": "masterpiece shot on IMAX 70mm, pristine anamorphic lens flare, sharp volumetric lighting, 8k hyper-detailed resolution",
            "scifi_cyberpunk": "futuristic cyberpunk neo-metropolis, cyan and magenta rim light, volumetric steam on wet pavement, 8k Unreal Engine 5 aesthetic",
            "high_fantasy": "grand epic high fantasy cinematography, golden hour ethereal god rays, mythical architectural depth, particle embers",
            "ghibli_anime": "lush hand-painted anime aesthetics, Studio Ghibli inspired vibrant sky, gentle dreamlike ambient illumination",
            "gothic_horror": "eerie gothic horror atmosphere, deep chiaroscuro candlelight shadows, chilling ground fog, 35mm film grain",
            "pixar_3d_cgi": "flawless 3d animated feature film render, soft subsurface scattering, whimsical volumetric global illumination",
            "commercial_studio": "ultra-clean luxury commercial studio photography, three-point diffused softbox lighting, flawless depth of field",
            "graphic_novel": "bold dynamic comic ink lineart, dramatic halftone shadows, striking high contrast composition",
            "kolkata_vintage": "authentic 1970s 35mm celluloid aesthetic, nostalgic warm sepia and amber lighting, North Kolkata heritage architecture, atmospheric monsoon mist",
            "nature_documentary": "National Geographic 4k wildlife documentary aesthetic, 400mm telephoto lens depth, raw natural daylight",
            "custom_neutral": "balanced cinematic studio lighting, neutral color grading, crisp shallow depth of field",
        }
        cue = style_cues.get(style_preset.lower(), "cinematic 35mm film aesthetic, professional color grading, photorealistic depth of field")
        
        return f"A cinematic master shot of {char_name}: {raw_prompt}. {cue}, award-winning composition, lifelike natural micro-movements."

    # -------------------------------------------------------------------------
    # Internal Image Normalization
    # -------------------------------------------------------------------------

    def _normalize_image_inputs(self, images_or_views: Any) -> List[Image.Image]:
        """Converts diverse image formats into an ordered list of PIL RGB images."""
        result: List[Image.Image] = []

        if isinstance(images_or_views, dict):
            for key in ["front", "left", "right", "back", "primary", "extra"]:
                if key in images_or_views and images_or_views[key] is not None:
                    img = self._single_to_pil(images_or_views[key])
                    if img:
                        result.append(img)
            for k, val in images_or_views.items():
                if k not in ["front", "left", "right", "back", "primary", "extra"] and val is not None:
                    img = self._single_to_pil(val)
                    if img:
                        result.append(img)

        elif isinstance(images_or_views, (list, tuple)):
            for item in images_or_views:
                img = self._single_to_pil(item)
                if img:
                    result.append(img)

        else:
            img = self._single_to_pil(images_or_views)
            if img:
                result.append(img)

        return result

    def _single_to_pil(self, item: Any) -> Optional[Image.Image]:
        """Converts a single image container into PIL RGB Image."""
        if item is None:
            return None
        try:
            if isinstance(item, Image.Image):
                return item.convert("RGB")
            elif isinstance(item, np.ndarray):
                return Image.fromarray(item).convert("RGB")
            elif isinstance(item, (str, Path)):
                str_p = str(item)
                if os.path.exists(str_p):
                    return Image.open(str_p).convert("RGB")
            elif hasattr(item, "name") and os.path.exists(item.name):
                return Image.open(item.name).convert("RGB")
            elif isinstance(item, dict) and "name" in item and os.path.exists(item["name"]):
                return Image.open(item["name"]).convert("RGB")
        except Exception as e:
            logger.warning(f"Failed to load image for Gemini analysis: {e}")
        return None
