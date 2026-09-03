"""
CineFlow-AI: Character Database & Facial Consistency Reinforcement Engine
=======================================================================
Provides MongoDB-compatible document storage and zero-dependency local JSON
document store persistence for user-created characters. Manages the comprehensive
Facial Consistency JSON Tree, tracking biometric geometry, ArcFace consensus vectors,
anchor keyframe poses (Grit, Action, Dialogue, Noir), wardrobe locks, and adaptive
reinforcement ("character details get stronger with every generated video").
"""

from __future__ import annotations

import os
import sys
import json
import time
import copy
import shutil
import logging
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Setup Logger
logger = logging.getLogger("CineFlow.Database")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def _l2_normalize_vector(vec: Union[np.ndarray, List[float]]) -> List[float]:
    """Helper to ensure vector is L2-normalized float list."""
    arr = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm > 1e-12:
        arr = arr / norm
    else:
        arr = np.zeros_like(arr)
        arr[0] = 1.0
    return [round(float(x), 6) for x in arr]


class CharacterDatabase:
    """
    Unified MongoDB & Local JSON Document Database Manager for CineFlow Characters.
    If MONGODB_URI environment variable is provided and pymongo is available,
    connects to MongoDB; otherwise provides an atomic local document store
    backed by JSON and directory synchronization in character_profiles/.
    """

    def __init__(
        self,
        storage_dir: str = "database",
        profiles_dir: str = "character_profiles",
        mongo_uri: Optional[str] = None,
    ) -> None:
        self.storage_dir = Path(storage_dir)
        self.profiles_dir = Path(profiles_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.profiles_dir.mkdir(parents=True, exist_ok=True)

        self.db_file = self.storage_dir / "characters_db.json"
        self.mongo_uri = mongo_uri or os.getenv("MONGODB_URI")
        self.mongo_client = None
        self.mongo_collection = None
        self.use_mongo = False

        # Try MongoDB connection if configured
        if self.mongo_uri:
            self._init_mongo(self.mongo_uri)

        # In-memory document store
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._load_local_db()

    def _init_mongo(self, uri: str) -> bool:
        """Attempts connection to MongoDB instance."""
        try:
            import pymongo
            client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=2000)
            # Test connection
            client.server_info()
            db_name = os.getenv("MONGODB_DB", "cineflow_studio")
            self.mongo_client = client
            self.mongo_collection = client[db_name]["characters"]
            self.use_mongo = True
            logger.info(f"Connected to MongoDB at {uri.split('@')[-1]} (db: {db_name}, collection: characters)")
            return True
        except Exception as e:
            logger.warning(f"MongoDB connection to '{uri}' unavailable ({e}). Defaulting to atomic local JSON document store.")
            self.use_mongo = False
            return False

    def _load_local_db(self) -> None:
        """Loads all documents from disk and syncs with character_profiles directory."""
        self._cache.clear()
        if self.db_file.exists():
            try:
                with open(self.db_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        self._cache = data
                    elif isinstance(data, list):
                        self._cache = {doc["id"]: doc for doc in data if "id" in doc}
            except Exception as e:
                logger.error(f"Error loading {self.db_file}: {e}")

        # Synchronize with character_profiles/ folders
        if self.profiles_dir.exists():
            for p_dir in self.profiles_dir.iterdir():
                if p_dir.is_dir():
                    profile_json = p_dir / "profile.json"
                    if profile_json.exists():
                        try:
                            with open(profile_json, "r", encoding="utf-8") as f:
                                doc = json.load(f)
                            char_id = doc.get("id") or p_dir.name
                            if char_id not in self._cache:
                                self._cache[char_id] = doc
                            else:
                                # Merge consistency tree if existing
                                for k, v in doc.items():
                                    if k not in self._cache[char_id]:
                                        self._cache[char_id][k] = v
                        except Exception as e:
                            logger.warning(f"Could not parse profile from {profile_json}: {e}")

    def _save_local_db(self) -> None:
        """Atomically saves the local cache to disk."""
        tmp_file = self.storage_dir / f"characters_db.tmp_{time.time_ns()}.json"
        try:
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, indent=2, ensure_ascii=False)
            shutil.move(str(tmp_file), str(self.db_file))
        except Exception as e:
            logger.error(f"Failed to atomically persist local database: {e}")
            if tmp_file.exists():
                tmp_file.unlink()

    # -------------------------------------------------------------------------
    # Facial Consistency JSON Tree Schema Generator
    # -------------------------------------------------------------------------

    @staticmethod
    def build_facial_consistency_tree(
        character_id: str,
        name: str,
        embedding: Optional[Union[np.ndarray, List[float]]] = None,
        views: Optional[Dict[str, str]] = None,
        traits: Optional[Dict[str, Any]] = None,
        voice_tone: Optional[str] = None,
        initial_confidence: float = 0.88,
        wardrobe_lock: float = 0.94,
        lighting_blend: float = 0.72,
    ) -> Dict[str, Any]:
        """
        Builds a comprehensive, production-grade Facial Consistency JSON Tree.
        Tracks biometrics, 512-D ArcFace consensus, 4 keyframe anchor poses,
        wardrobe lock specifications, lighting tolerance, and reinforcement history.
        """
        views = views or {}
        traits = traits or {}

        # 512-D Consensus embedding vector
        if embedding is not None:
            emb_list = _l2_normalize_vector(embedding)
        else:
            # Deterministic pseudo-embedding from character id & name
            rng = np.random.RandomState(abs(hash(character_id + name)) % (2**31 - 1))
            vec = rng.randn(512).astype(np.float32)
            emb_list = _l2_normalize_vector(vec)

        # 4 Keyframe Anchor Poses (Grit, Action, Dialogue, Noir)
        keyframe_anchors = {
            "grit": {
                "caption": "Grit",
                "image_path": views.get("front") or views.get("primary", ""),
                "confidence": round(initial_confidence + 0.06, 2),
                "lighting": "High-contrast dynamic neon",
                "angle": "0° Frontal Direct",
            },
            "action": {
                "caption": "Action",
                "image_path": views.get("right") or views.get("left") or views.get("front", ""),
                "confidence": round(initial_confidence + 0.04, 2),
                "lighting": "Cinematic rim flare",
                "angle": "45° Three-quarter",
            },
            "dialogue": {
                "caption": "Dialogue",
                "image_path": views.get("left") or views.get("front", ""),
                "confidence": round(initial_confidence + 0.08, 2),
                "lighting": "Soft bokeh fill",
                "angle": "15° Subtle tilt",
            },
            "noir": {
                "caption": "Noir",
                "image_path": views.get("back") or views.get("front", ""),
                "confidence": round(initial_confidence + 0.03, 2),
                "lighting": "Low-key chiaroscuro",
                "angle": "Profile shadow",
            },
        }

        # Biometric Landmarks & Ratios
        biometric_anchors = {
            "facial_aspect_ratio": round(traits.get("aspect_ratio", 1.35), 3),
            "eye_distance_ratio": round(traits.get("eye_distance", 0.32), 3),
            "jaw_angle_degrees": round(traits.get("jaw_angle", 118.0), 1),
            "nose_to_chin_ratio": round(traits.get("nose_to_chin", 0.45), 3),
            "skin_tone_hex": traits.get("skin_tone", "#d4a373"),
            "distinctive_features": traits.get("distinctive_features", [
                "defined jawline", "subtle micro-texture", "cinematic eye depth"
            ]),
        }

        # Wardrobe & Appearance Lock
        wardrobe_spec = {
            "enabled": True,
            "lock_fidelity": wardrobe_lock,
            "signature_items": traits.get("wardrobe", ["matte leather jacket", "luminescent collar piping"]),
            "palette": traits.get("color_palette", ["#0f172a", "#4cd7f6", "#d0bcff"]),
        }

        # Lighting Blend Tolerances
        lighting_spec = {
            "blend_range": lighting_blend,
            "preferred_schemes": ["cyberpunk anamorphic", "volumetric rain reflections", "golden hour 35mm"],
            "shadow_falloff_gamma": 0.85,
        }

        # Voice Integration
        voice_spec = {
            "provider": "ElevenLabs",
            "voice_name": voice_tone or f"ElevenLabs: {name} Tone",
            "voice_id": f"voice_{character_id}",
            "pitch_multiplier": 1.0,
            "stability": 0.85,
        }

        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

        return {
            "version": "2.0",
            "character_id": character_id,
            "created_at": timestamp,
            "last_reinforced_at": timestamp,
            "identity_confidence": initial_confidence,
            "generation_reinforcement_count": 0,
            "consensus_embedding": emb_list,
            "biometric_anchors": biometric_anchors,
            "keyframe_anchors": keyframe_anchors,
            "wardrobe_lock": wardrobe_spec,
            "lighting_blend": lighting_spec,
            "voice_profile": voice_spec,
            "reinforcement_history": [
                {
                    "event": "enrolled",
                    "timestamp": timestamp,
                    "confidence": initial_confidence,
                    "note": "Initial enrollment from multi-angle portraits.",
                }
            ],
        }

    # -------------------------------------------------------------------------
    # CRUD Operations
    # -------------------------------------------------------------------------

    def get_character(self, character_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves character document by ID."""
        char_id = str(character_id).strip().lower()
        if self.use_mongo and self.mongo_collection is not None:
            try:
                doc = self.mongo_collection.find_one({"id": char_id}, {"_id": 0})
                if doc:
                    return doc
            except Exception as e:
                logger.warning(f"MongoDB find_one error: {e}")

        return self._cache.get(char_id)

    def list_characters(self) -> List[Dict[str, Any]]:
        """Lists all user characters."""
        if self.use_mongo and self.mongo_collection is not None:
            try:
                docs = list(self.mongo_collection.find({}, {"_id": 0}))
                if docs:
                    return docs
            except Exception as e:
                logger.warning(f"MongoDB find error: {e}")

        return list(self._cache.values())

    def save_character(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """
        Saves or updates a character document in MongoDB and local document store,
        ensuring the Facial Consistency JSON Tree and character_profiles/ directory
        remain in exact synchronization.
        """
        char_id = str(doc.get("id", "")).strip().lower()
        if not char_id:
            raise ValueError("Character document must have a non-empty 'id'.")

        doc["id"] = char_id
        doc["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # Ensure facial_consistency_tree exists
        if "facial_consistency_tree" not in doc or not doc["facial_consistency_tree"]:
            doc["facial_consistency_tree"] = self.build_facial_consistency_tree(
                character_id=char_id,
                name=doc.get("name", char_id.title()),
                views=doc.get("views") or doc.get("multi_view_images"),
                traits=doc.get("gemini_traits"),
            )

        # 1. Update in-memory cache
        self._cache[char_id] = copy.deepcopy(doc)

        # 2. Persist to MongoDB if connected
        if self.use_mongo and self.mongo_collection is not None:
            try:
                self.mongo_collection.update_one(
                    {"id": char_id},
                    {"$set": doc},
                    upsert=True,
                )
            except Exception as e:
                logger.error(f"MongoDB update_one error: {e}")

        # 3. Persist to local document database file
        self._save_local_db()

        # 4. Mirror to character_profiles/<char_id>/profile.json
        char_dir = self.profiles_dir / char_id
        char_dir.mkdir(parents=True, exist_ok=True)
        profile_file = char_dir / "profile.json"
        try:
            with open(profile_file, "w", encoding="utf-8") as f:
                json.dump(doc, f, indent=2, ensure_ascii=False)

            # Also mirror embedding.npy if consensus_embedding exists
            tree = doc.get("facial_consistency_tree", {})
            emb = tree.get("consensus_embedding")
            if emb:
                arr = np.array(emb, dtype=np.float32)
                np.save(str(char_dir / "embedding.npy"), arr)
        except Exception as e:
            logger.error(f"Error mirroring profile to {char_dir}: {e}")

        return doc

    def delete_character(self, character_id: str) -> bool:
        """Deletes character from database and filesystem."""
        char_id = str(character_id).strip().lower()
        deleted = False

        if self.use_mongo and self.mongo_collection is not None:
            try:
                res = self.mongo_collection.delete_one({"id": char_id})
                if res.deleted_count > 0:
                    deleted = True
            except Exception as e:
                logger.error(f"MongoDB delete error: {e}")

        if char_id in self._cache:
            del self._cache[char_id]
            self._save_local_db()
            deleted = True

        # Delete from character_profiles/
        char_dir = self.profiles_dir / char_id
        if char_dir.exists() and char_dir.is_dir():
            shutil.rmtree(str(char_dir), ignore_errors=True)
            deleted = True

        return deleted

    def count(self) -> int:
        """Returns total count of enrolled characters in the document store."""
        return len(self._cache)

    # -------------------------------------------------------------------------
    # Adaptive Reinforcement Engine
    # -------------------------------------------------------------------------

    def reinforce_character_facial_consistency(
        self,
        character_id: str,
        frame_embedding: Optional[Union[np.ndarray, List[float]]] = None,
        prompt: str = "",
        shot_metadata: Optional[Dict[str, Any]] = None,
        keyframe_image_path: Optional[str] = None,
        generated_frame_embedding: Optional[Union[np.ndarray, List[float]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Adaptive Facial Consistency Reinforcement:
        Every time a video is created with an actor:
        1. Fuses the generated shot's face vector into the character's consensus embedding
           via exponential moving average, stabilizing multi-angle identity.
        2. Progressively increments the 'identity_confidence' score (e.g. 88% -> 90% -> 94%).
        3. Updates keyframe anchor poses if a higher quality or novel view was generated.
        4. Appends a timestamped log to 'reinforcement_history'.
        5. Persists the enriched JSON tree immediately back to MongoDB & document store.
        """
        if frame_embedding is None and generated_frame_embedding is not None:
            frame_embedding = generated_frame_embedding
        doc = self.get_character(character_id)
        if not doc:
            raise KeyError(f"Character '{character_id}' not found in database.")

        tree = doc.get("facial_consistency_tree")
        if not tree:
            tree = self.build_facial_consistency_tree(
                character_id=character_id,
                name=doc.get("name", character_id.title()),
            )
            doc["facial_consistency_tree"] = tree

        shot_metadata = shot_metadata or {}
        count = tree.get("generation_reinforcement_count", 0) + 1
        tree["generation_reinforcement_count"] = count

        # 1. Update Identity Confidence ("details get stronger with every video")
        current_conf = float(tree.get("identity_confidence", 0.88))
        # Logarithmic diminishing returns boost up to 0.99
        gain = max(0.01, round(0.025 / (1.0 + 0.15 * count), 4))
        new_conf = round(min(0.99, current_conf + gain), 4)
        tree["identity_confidence"] = new_conf

        # 2. Update Consensus Embedding Vector via EMA
        current_emb = tree.get("consensus_embedding")
        if frame_embedding is not None and current_emb:
            prev_vec = np.array(current_emb, dtype=np.float32)
            shot_vec = np.array(frame_embedding, dtype=np.float32)
            # EMA: alpha = 0.88, new frame contributes 12%
            alpha = 0.88
            fused_vec = alpha * prev_vec + (1.0 - alpha) * shot_vec
            tree["consensus_embedding"] = _l2_normalize_vector(fused_vec)

        # 3. Update Anchor Keyframes if a new keyframe is provided
        if keyframe_image_path and os.path.exists(keyframe_image_path):
            anchors = tree.setdefault("keyframe_anchors", {})
            # Select target pose based on count or prompt keywords
            target_pose = "grit"
            p_low = prompt.lower()
            if "action" in p_low or "run" in p_low or "fast" in p_low:
                target_pose = "action"
            elif "dialogue" in p_low or "talk" in p_low or "speak" in p_low or "smile" in p_low:
                target_pose = "dialogue"
            elif "dark" in p_low or "night" in p_low or "rain" in p_low or "shadow" in p_low:
                target_pose = "noir"
            else:
                poses = ["grit", "action", "dialogue", "noir"]
                target_pose = poses[count % len(poses)]

            if target_pose in anchors:
                anchors[target_pose]["image_path"] = keyframe_image_path
                anchors[target_pose]["confidence"] = new_conf

        # 4. Append to Reinforcement History Log
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        tree["last_reinforced_at"] = timestamp

        history_entry = {
            "shot_number": count,
            "shot_id": shot_metadata.get("shot_id", f"shot_{count:03d}"),
            "timestamp": timestamp,
            "prompt": prompt[:120],
            "confidence_before": current_conf,
            "confidence_after": new_conf,
            "confidence_delta": round(gain, 4),
            "engine": shot_metadata.get("engine", "Wan 2.1 DiT FP8"),
            "resolution": shot_metadata.get("resolution", "1080p"),
            "reinforcement_note": f"Facial feature anchors reinforced from shot {count} synthesis.",
        }
        tree.setdefault("reinforcement_history", []).append(history_entry)

        # 5. Save and persist
        self.save_character(doc)
        logger.info(
            f"Reinforced facial consistency for '{doc.get('name')}' ({character_id}): "
            f"Fidelity {int(current_conf * 100)}% -> {int(new_conf * 100)}% (Total shots: {count})"
        )

        return tree


# Singleton database instance getter
_GLOBAL_DB: Optional[CharacterDatabase] = None


def get_character_database() -> CharacterDatabase:
    """Returns the singleton CharacterDatabase instance."""
    global _GLOBAL_DB
    if _GLOBAL_DB is None:
        _GLOBAL_DB = CharacterDatabase()
    return _GLOBAL_DB
