"""
Unit and Integration Tests for StudioHotReloader Dynamic Daemon
==============================================================
Tests:
1. HotReloader initialization and status diagnostics.
2. File change detection and category classification ('profiles', 'styles', 'configs').
3. Callback registration and execution on file modification.
4. Clean thread startup and shutdown.
"""

import json
import os
import time
import pytest
from pathlib import Path

from modules.hot_reloader import StudioHotReloader


class TestStudioHotReloader:
    @pytest.fixture
    def test_env(self, tmp_path):
        profiles_dir = tmp_path / "character_profiles"
        configs_dir = tmp_path / "configs"
        profiles_dir.mkdir(parents=True, exist_ok=True)
        configs_dir.mkdir(parents=True, exist_ok=True)

        styles_file = configs_dir / "cinematic_styles.json"
        styles_file.write_text(json.dumps({"styles": []}), encoding="utf-8")

        reloader = StudioHotReloader(
            watch_paths=[str(profiles_dir), str(configs_dir)],
            poll_interval=0.5,
            enabled=True,
        )
        return reloader, profiles_dir, configs_dir, styles_file

    def test_hot_reloader_status(self, test_env):
        reloader, _, _, _ = test_env
        status = reloader.get_status()
        assert status["enabled"] is True
        assert status["running"] is False
        assert status["total_files_watched"] >= 1

    def test_file_modification_triggers_callback(self, test_env):
        reloader, _, configs_dir, styles_file = test_env
        
        events_captured = []

        def on_styles_change(files):
            events_captured.extend(files)

        reloader.register_callback("styles", on_styles_change)

        # Modify styles file
        time.sleep(0.1)
        styles_file.write_text(json.dumps({"styles": [{"id": "test"}]}), encoding="utf-8")

        # Synchronous check
        changes = reloader.check_changes_now()
        assert len(changes["styles"]) >= 1
        assert len(events_captured) >= 1
        assert str(styles_file) in [str(Path(p).resolve()) for p in events_captured]

    def test_reloader_thread_lifecycle(self, test_env):
        reloader, _, _, _ = test_env
        reloader.start()
        assert reloader.get_status()["running"] is True
        time.sleep(0.6)
        reloader.stop()
        assert reloader.get_status()["running"] is False
