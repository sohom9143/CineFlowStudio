"""
CineFlow-AI: Local-to-Colab Intelligent Proxy Bridge
====================================================
Runs inside the local Python environment on the user's PC.
Acts as the intermediary between the C# Desktop App (.exe) and Google Colab (Tesla T4 GPU):
[ C# Desktop App ] ---> [ Local Python Engine ] ---> [ Google Colab T4 GPU ]
"""

from __future__ import annotations

import os
import sys
import time
import json
import logging
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from modules.cloud_storage import get_active_colab_bridge

logger = logging.getLogger("CineFlow.ColabProxy")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class ColabProxyBridge:
    """
    Manages communication between the local studio environment and Google Colab T4.
    Auto-discovers the active tunnel via Firebase Realtime Database.
    """

    def __init__(self, fallback_url: Optional[str] = None):
        self._colab_url = fallback_url or ""
        self._last_check_time = 0.0
        self._cached_bridge_info: Optional[Dict[str, Any]] = None

    def get_colab_url(self, force_refresh: bool = False) -> Optional[str]:
        """
        Returns the active Colab URL.
        Auto-refreshes from Firebase RTDB if cache is older than 15 seconds.
        """
        now = time.time()
        if not force_refresh and self._cached_bridge_info and (now - self._last_check_time < 15.0):
            return self._colab_url

        try:
            bridge_info = get_active_colab_bridge()
            if bridge_info and bridge_info.get("url"):
                self._cached_bridge_info = bridge_info
                self._colab_url = bridge_info["url"].rstrip("/")
                self._last_check_time = now
                logger.info(f"Discovered active Colab GPU URL from Firebase: {self._colab_url}")
                return self._colab_url
        except Exception as e:
            logger.warning(f"Could not auto-discover Colab bridge: {e}")

        # Fallback to local configs/colab_connection.json if available
        candidates = [
            Path("configs/colab_connection.json"),
            Path(__file__).resolve().parent.parent / "configs" / "colab_connection.json",
        ]
        for p in candidates:
            if p.exists():
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        url = data.get("colab_url", "").strip().rstrip("/")
                        if url:
                            self._colab_url = url
                            return self._colab_url
                except Exception:
                    pass

        return self._colab_url if self._colab_url else None

    def is_colab_online(self) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Pings Colab /api/health to verify connectivity and GPU telemetry.
        """
        url = self.get_colab_url()
        if not url:
            return False, "No active Colab URL discovered", {}

        health_url = f"{url}/api/health"
        try:
            req = urllib.request.Request(health_url, headers={"User-Agent": "CineFlow-Local-Proxy"})
            with urllib.request.urlopen(req, timeout=4.0) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    device = data.get("device", "Remote GPU")
                    return True, f"Online ({device})", data
        except Exception as e:
            logger.debug(f"Colab ping failed: {e}")

        return False, "Colab endpoint unreachable", {}

    def forward_generation_to_colab(self, payload: Dict[str, Any], timeout_sec: float = 600.0) -> Dict[str, Any]:
        """
        Forwards a scene generation request to Google Colab.
        Returns the Colab response containing video URL and cloud metadata.
        """
        url = self.get_colab_url(force_refresh=True)
        if not url:
            return {
                "status": "error",
                "error": "Google Colab GPU is not connected. Start Cell 6 in Colab to auto-connect.",
            }

        endpoint = f"{url}/api/generate"
        logger.info(f"Forwarding generation request to Colab: {endpoint} ...")

        try:
            req_data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                endpoint,
                data=req_data,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "CineFlow-Local-Proxy",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                if resp.status == 200:
                    result = json.loads(resp.read().decode("utf-8"))
                    logger.info("Successfully received generation result from Colab GPU.")
                    return result
                else:
                    return {
                        "status": "error",
                        "error": f"Colab returned HTTP {resp.status}",
                    }
        except urllib.error.HTTPError as http_err:
            err_body = http_err.read().decode("utf-8")
            logger.error(f"Colab HTTP Error {http_err.code}: {err_body}")
            return {"status": "error", "error": f"Colab Error ({http_err.code}): {err_body}"}
        except Exception as e:
            logger.error(f"Colab forward failed: {e}")
            return {"status": "error", "error": str(e)}


# Global singleton proxy instance
_global_proxy = ColabProxyBridge()

def get_colab_proxy() -> ColabProxyBridge:
    return _global_proxy
