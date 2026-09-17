"""
Tests for CineFlow NLE Studio: Regional & Worldwide Viral Meme Vault.
Verifies catalog queries across India, Bangla, Worldwide, sentiment context matching, and procedural audio synthesis.
"""

import numpy as np
import pytest
from modules.meme_vault import MemeVault, MemeClip


def test_meme_vault_catalog_integrity():
    vault = MemeVault()
    all_memes = vault.list_memes()
    assert len(all_memes) >= 15

    categories = {m["category"] for m in all_memes}
    assert "India" in categories
    assert "Bangla" in categories
    assert "Worldwide" in categories


def test_meme_vault_filtering():
    vault = MemeVault()
    india_memes = vault.list_memes(category="India")
    assert all(m["category"] == "India" for m in india_memes)
    assert any("Hera Pheri" in m["name"] for m in india_memes)

    bangla_memes = vault.list_memes(category="Bangla")
    assert all(m["category"] == "Bangla" for m in bangla_memes)
    assert any("খাবার দাবার" in m["name"] for m in bangla_memes)

    worldwide_memes = vault.list_memes(category="Worldwide")
    assert all(m["category"] == "Worldwide" for m in worldwide_memes)
    assert any("Vine Boom" in m["name"] for m in worldwide_memes)


def test_meme_context_matching():
    vault = MemeVault()

    # Match money/wealth prompt
    money_meme = vault.find_best_meme_for_context("We made huge profit and investment return")
    assert money_meme is not None
    assert "paisa" in money_meme.id.lower() or "cash" in money_meme.id.lower()

    # Match Bengali food joke
    bangla_meme = vault.find_best_meme_for_context("lunch fish dinner খাবার দাবার", region_preference="Bangla")
    assert bangla_meme is not None
    assert bangla_meme.category == "Bangla"

    # Match sudden bass shock impact
    impact_meme = vault.find_best_meme_for_context("sudden dramatic bass drop impact", region_preference="Worldwide")
    assert impact_meme is not None
    assert "boom" in impact_meme.id.lower()


def test_meme_procedural_audio_synthesis():
    vault = MemeVault()
    audio = vault.synthesize_preview_audio("ww_vine_boom", sample_rate=44100)
    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float32
    assert audio.size > 0
    # Must have reasonable audio levels
    assert np.max(np.abs(audio)) > 0.1
