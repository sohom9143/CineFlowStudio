"""
CineFlow-AI: Dynamic Hot-Reloading Daemon
=========================================
Monitors character profiles, cinematic styles, and configuration files in real-time.
Triggers live in-memory cache invalidation, UI state synchronization, and component
refresh without requiring server restarts or causing workflow interruptions.

Author: Google DeepMind & Antigravity Advanced Agentic Coding Team
Architecture: Zero-Downtime Live State Synchronizer & File Watchdog
"""

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger("CineFlow.HotReloader")


class StudioHotReloader:
    """
    Lightweight, thread-safe file system watchdog daemon.
    Monitors target directories and triggers registered reload callbacks when changes occur.
    """

    def __init__(
        self,
        watch_paths: Optional[List[str]] = None,
        poll_interval: float = 1.5,
        enabled: bool = True,
    ) -> None:
        self.poll_interval = max(0.5, poll_interval)
        self.enabled = enabled
        
        default_paths = ["character_profiles", "configs", "modules"]
        self.watch_paths = [Path(p).resolve() for p in (watch_paths or default_paths)]
        
        self._callbacks: Dict[str, List[Callable[..., Any]]] = {
            "profiles": [],
            "styles": [],
            "configs": [],
            "all": [],
        }
        
        self._mtime_cache: Dict[str, float] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._last_reload_time: float = 0.0
        self._total_reload_events: int = 0

        # Prime the cache
        self._mtime_cache = self._scan_mtimes()

    def register_callback(self, event_type: str, callback: Callable[..., Any]) -> None:
        """Registers a callback for specific change events ('profiles', 'styles', 'configs', 'all')."""
        with self._lock:
            evt = event_type.lower()
            if evt not in self._callbacks:
                self._callbacks[evt] = []
            if callback not in self._callbacks[evt]:
                self._callbacks[evt].append(callback)
            logger.debug(f"Registered hot-reload callback for '{evt}' events.")

    def start(self) -> None:
        """Starts the background watchdog monitoring thread."""
        with self._lock:
            if self._running or not self.enabled:
                return
            self._running = True
            self._thread = threading.Thread(target=self._watch_loop, daemon=True, name="CineFlow-HotReloader")
            self._thread.start()
            logger.info(f"StudioHotReloader started (polling every {self.poll_interval:.1f}s).")

    def stop(self) -> None:
        """Stops the background watchdog monitoring thread."""
        with self._lock:
            self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
            logger.info("StudioHotReloader stopped.")

    def check_changes_now(self) -> Dict[str, List[str]]:
        """
        Synchronously checks all monitored directories for modified, added, or deleted files.
        Executes callbacks if changes are detected.
        """
        changes: Dict[str, List[str]] = {"profiles": [], "styles": [], "configs": [], "modules": []}
        current_mtimes = self._scan_mtimes()
        
        with self._lock:
            # Detect modified or added files
            for file_path, current_mtime in current_mtimes.items():
                old_mtime = self._mtime_cache.get(file_path)
                if old_mtime is None or abs(current_mtime - old_mtime) > 0.001:
                    category = self._categorize_path(file_path)
                    changes[category].append(file_path)

            # Detect deleted files
            for file_path in list(self._mtime_cache.keys()):
                if file_path not in current_mtimes:
                    category = self._categorize_path(file_path)
                    changes[category].append(file_path)

            # Update cache
            self._mtime_cache = current_mtimes

        total_changed = sum(len(v) for v in changes.values())
        if total_changed > 0:
            self._last_reload_time = time.time()
            self._total_reload_events += 1
            logger.info(f"Hot-reload detected {total_changed} modified file(s): {changes}")
            self._dispatch_callbacks(changes)

        return changes

    def get_status(self) -> Dict[str, Any]:
        """Returns diagnostic status of the hot reloader."""
        return {
            "running": self._running,
            "enabled": self.enabled,
            "poll_interval": self.poll_interval,
            "monitored_paths": [str(p) for p in self.watch_paths],
            "total_files_watched": len(self._mtime_cache),
            "total_reload_events": self._total_reload_events,
            "last_reload_timestamp": self._last_reload_time,
            "last_reload_formatted": time.strftime("%H:%M:%S", time.localtime(self._last_reload_time)) if self._last_reload_time else "Never",
        }

    # -------------------------------------------------------------------------
    # Internal Watch Loop & Dispatchers
    # -------------------------------------------------------------------------

    def _watch_loop(self) -> None:
        """Continuous background polling loop."""
        while self._running:
            try:
                self.check_changes_now()
            except Exception as e:
                logger.error(f"Error in hot-reload watchdog loop: {e}", exc_info=True)
            time.sleep(self.poll_interval)

    def _scan_mtimes(self) -> Dict[str, float]:
        """Scans all monitored directories and records file modification times."""
        mtimes: Dict[str, float] = {}
        for base_path in self.watch_paths:
            if not base_path.exists():
                continue
            if base_path.is_file():
                try:
                    mtimes[str(base_path)] = base_path.stat().st_mtime
                except Exception:
                    pass
                continue

            for root, _, files in os.walk(base_path):
                # Skip cache directories
                if "__pycache__" in root or ".pytest_cache" in root or ".git" in root:
                    continue
                for f in files:
                    if f.endswith((".json", ".yaml", ".yml", ".py", ".npy", ".npz", ".png", ".jpg")):
                        full_p = os.path.join(root, f)
                        try:
                            mtimes[full_p] = os.path.getmtime(full_p)
                        except Exception:
                            pass
        return mtimes

    def _categorize_path(self, path_str: str) -> str:
        """Determines event category from file path."""
        norm = path_str.lower().replace("\\", "/")
        if "character_profiles" in norm:
            return "profiles"
        elif "styles" in norm:
            return "styles"
        elif "configs" in norm:
            return "configs"
        return "modules"

    def _dispatch_callbacks(self, changes: Dict[str, List[str]]) -> None:
        """Executes registered callback functions safely."""
        with self._lock:
            # Specific category callbacks
            for category, changed_files in changes.items():
                if changed_files and category in self._callbacks:
                    for cb in self._callbacks[category]:
                        try:
                            cb(changed_files)
                        except Exception as e:
                            logger.error(f"Error in hot-reload callback for '{category}': {e}")

            # 'all' category callbacks
            if any(len(v) > 0 for v in changes.values()):
                for cb in self._callbacks.get("all", []):
                    try:
                        cb(changes)
                    except Exception as e:
                        logger.error(f"Error in global hot-reload callback: {e}")
