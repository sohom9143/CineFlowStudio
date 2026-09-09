"""
CineFlow-AI: Cloud Media Storage & Live Handshake Bridge
=========================================================
Provides unified cloud integration with:
1. Firebase Realtime Database (Zero-Copy auto-handshake & live render feed)
2. Cloudinary Media CDN (Fast H.264 video streaming & thumbnails)
3. Firebase Storage (Persistent cloud storage)
"""

from __future__ import annotations

import os
import sys
import time
import json
import logging
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger("CineFlow.CloudStorage")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Default endpoints from project configuration
FIREBASE_DB_URL = os.getenv(
    "FIREBASE_DATABASE_URL",
    "https://ai-video-studio-ecde9-default-rtdb.firebaseio.com"
).rstrip("/")

FIREBASE_STORAGE_BUCKET = os.getenv(
    "FIREBASE_STORAGE_BUCKET",
    "ai-video-studio-ecde9.firebasestorage.app"
)

CLOUDINARY_CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME", "").strip()
CLOUDINARY_API_KEY = os.getenv("CLOUDINARY_API_KEY", "").strip()
CLOUDINARY_API_SECRET = os.getenv("CLOUDINARY_API_SECRET", "").strip()


# =============================================================================
# 1. Zero-Copy Auto-Handshake Bridge (Firebase Realtime DB)
# =============================================================================

def publish_colab_bridge(
    public_url: str,
    device_name: str = "Tesla T4",
    vram_info: str = "15GB VRAM",
    status: str = "online",
) -> bool:
    """
    Publishes the active Cloudflare Tunnel URL to Firebase Realtime Database.
    Allows the C# Desktop Client to auto-discover Colab without any copy-pasting.
    """
    if not FIREBASE_DB_URL:
        logger.warning("Firebase DB URL is not configured; skipping auto-handshake publish.")
        return False

    endpoint = f"{FIREBASE_DB_URL}/cineflow/bridge.json"
    payload = {
        "url": public_url.rstrip("/"),
        "status": status,
        "device": device_name,
        "vram": vram_info,
        "updated_at": time.time(),
        "human_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    }

    try:
        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=data_bytes,
            headers={"Content-Type": "application/json"},
            method="PUT",
        )
        with urllib.request.urlopen(req, timeout=8.0) as resp:
            if resp.status in (200, 204):
                logger.info(f"✅ Auto-Handshake: Published active Colab GPU URL to Firebase: {public_url}")
                return True
    except Exception as e:
        logger.warning(f"Could not publish bridge to Firebase RTDB: {e}")

    return False


def get_active_colab_bridge() -> Optional[Dict[str, Any]]:
    """
    Fetches the active Colab bridge connection info from Firebase RTDB.
    """
    if not FIREBASE_DB_URL:
        return None

    endpoint = f"{FIREBASE_DB_URL}/cineflow/bridge.json"
    try:
        req = urllib.request.Request(endpoint, headers={"User-Agent": "CineFlow-Client"})
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            if resp.status == 200:
                body = resp.read().decode("utf-8")
                return json.loads(body)
    except Exception as e:
        logger.warning(f"Could not read bridge from Firebase: {e}")

    return None


# =============================================================================
# 2. Cloudinary Media CDN Uploader
# =============================================================================

def is_cloudinary_configured() -> bool:
    """Returns True if Cloudinary Cloud Name, API Key, and API Secret are present."""
    cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME", CLOUDINARY_CLOUD_NAME)
    api_key = os.getenv("CLOUDINARY_API_KEY", CLOUDINARY_API_KEY)
    api_secret = os.getenv("CLOUDINARY_API_SECRET", CLOUDINARY_API_SECRET)
    return bool(cloud_name and api_key and api_secret)


def upload_to_cloudinary(
    video_path: str,
    public_id: Optional[str] = None,
    tags: Optional[list] = None,
) -> Optional[Dict[str, Any]]:
    """
    Uploads a video file to Cloudinary and returns CDN streaming URL + thumbnail URL.
    """
    if not is_cloudinary_configured():
        logger.info("Cloudinary is missing cloud_name or api_key; skipping Cloudinary upload.")
        return None

    cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME", CLOUDINARY_CLOUD_NAME)
    api_key = os.getenv("CLOUDINARY_API_KEY", CLOUDINARY_API_KEY)
    api_secret = os.getenv("CLOUDINARY_API_SECRET", CLOUDINARY_API_SECRET)

    try:
        import hashlib

        timestamp = int(time.time())
        pub_id = public_id or Path(video_path).stem

        # Build signature parameters
        params_to_sign = f"public_id={pub_id}&timestamp={timestamp}{api_secret}"
        signature = hashlib.sha1(params_to_sign.encode("utf-8")).hexdigest()

        upload_url = f"https://api.cloudinary.com/v1_1/{cloud_name}/video/upload"

        # Multi-part or standard form upload
        import mimetypes
        import urllib.request

        with open(video_path, "rb") as f:
            file_data = f.read()

        boundary = "----CineFlowBoundary7MA4YWxkTrZu0gW"
        body = []
        body.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"api_key\"\r\n\r\n{api_key}\r\n".encode())
        body.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"timestamp\"\r\n\r\n{timestamp}\r\n".encode())
        body.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"public_id\"\r\n\r\n{pub_id}\r\n".encode())
        body.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"signature\"\r\n\r\n{signature}\r\n".encode())
        body.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{Path(video_path).name}\"\r\n"
            f"Content-Type: video/mp4\r\n\r\n".encode()
        )
        body.append(file_data)
        body.append(f"\r\n--{boundary}--\r\n".encode())

        req_body = b"".join(body)
        req = urllib.request.Request(
            upload_url,
            data=req_body,
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Content-Length": str(len(req_body)),
            },
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=60.0) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            secure_url = res_data.get("secure_url") or res_data.get("url")
            logger.info(f"✅ Video successfully uploaded to Cloudinary: {secure_url}")
            return {
                "streaming_url": secure_url,
                "thumbnail_url": secure_url.rsplit(".", 1)[0] + ".jpg" if secure_url else "",
                "duration": res_data.get("duration"),
                "format": res_data.get("format"),
                "bytes": res_data.get("bytes"),
            }
    except Exception as e:
        logger.warning(f"Cloudinary upload failed: {e}")
        return None


# =============================================================================
# 3. Master Video Dispatcher & Firebase Live Render Notification
# =============================================================================

def register_master_video(
    local_mp4_path: str,
    character_id: str = "",
    scene_prompt: str = "",
    resolution: str = "720p",
    aspect_ratio: str = "16:9",
    duration_sec: float = 3.375,
    fps: int = 24,
) -> Dict[str, Any]:
    """
    Registers a rendered master video:
    1. Uploads to Cloudinary (if configured)
    2. Publishes record to Firebase Realtime Database for C# live sync
    """
    render_id = Path(local_mp4_path).stem
    result = {
        "render_id": render_id,
        "local_path": local_mp4_path,
        "cdn_url": "",
        "thumbnail_url": "",
        "character_id": character_id,
        "prompt": scene_prompt,
        "resolution": resolution,
        "aspect_ratio": aspect_ratio,
        "fps": fps,
        "created_at": time.time(),
        "human_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    }

    # Attempt Cloudinary upload
    cloudinary_res = upload_to_cloudinary(local_mp4_path, public_id=render_id)
    if cloudinary_res:
        result["cdn_url"] = cloudinary_res.get("streaming_url", "")
        result["thumbnail_url"] = cloudinary_res.get("thumbnail_url", "")

    # Push render notification to Firebase Realtime Database
    if FIREBASE_DB_URL:
        try:
            endpoint = f"{FIREBASE_DB_URL}/cineflow/renders/{render_id}.json"
            data_bytes = json.dumps(result).encode("utf-8")
            req = urllib.request.Request(
                endpoint,
                data=data_bytes,
                headers={"Content-Type": "application/json"},
                method="PUT",
            )
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                if resp.status in (200, 204):
                    logger.info(f"✅ Render metadata synced to Firebase RTDB: {render_id}")
        except Exception as e:
            logger.warning(f"Could not push render metadata to Firebase RTDB: {e}")

    return result
