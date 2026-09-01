"""
Pytest configuration and shared fixtures for CineFlow-AI test suite.
"""

import os
import sys
import pytest

# Ensure workspace root is on sys.path
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from modules.memory_manager import VRAMManager


@pytest.fixture(autouse=True)
def reset_vram_manager_singleton():
    """Reset VRAMManager singleton before and after each test."""
    VRAMManager.reset_instance()
    yield
    VRAMManager.reset_instance()
