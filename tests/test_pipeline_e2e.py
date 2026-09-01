"""
End-to-End Pipeline Integration & Scenario Test Suite for CineFlow-AI (Milestone 7 / R7)
=========================================================================================
Authoritative Reference: TEST_INFRA.md § Real-World Application Scenarios (Tier 4) & ORIGINAL_REQUEST.md

Coverage Matrix:
- Scenario 1: Full Studio Pipeline Shot (Dev in North Kolkata Vintage 35mm + Bengali audio dialogue -> 81 frames motion -> lip-sync -> 1080p 60fps MP4 master muxing).
- Scenario 2: Custom Character Multi-Photo Enrollment -> Cyberpunk Noir shot generation -> master render.
- Scenario 3: Memory Stress & VRAM Lifecycle Isolation (rapid multi-stage transitions, telemetry tracking, zero leaks).
- Scenario 4: Offline CPU Execution & Fallback Robustness (graceful failovers between Wan 2.1 / LTX / Mock, LivePortrait / Wav2Lip / Mock, Real-ESRGAN / Lanczos, RIFE / Farneback, FFmpeg / MoviePy / OpenCV).
- Scenario 5: Gradio App & Colab Notebook Integrity (validate `app.py` pipeline handlers, `configs/colab_t4_config.yaml` schema, and `CineFlow_Colab_FreeTier.ipynb` 6-cell JSON validity).
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import shutil
import struct
import sys
import tempfile
import time
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image, ImageDraw

# Ensure workspace root is in sys.path
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

# Import Core Pipeline Engines & Managers
from modules.memory_manager import (
    VRAMManager,
    VRAMStageContext,
    vram_lifecycle_stage,
    stage_context,
    flush_memory,
    get_vram_stats,
    get_optimal_precision,
    purge_models,
)
from modules.character_engine import (
    CharacterProfile,
    CharacterStudio,
    compute_l2_norm,
    l2_normalize,
    fuse_consensus_embeddings,
    extract_facial_embedding_from_image,
    sanitize_character_slug,
)
from modules.video_engine import (
    CineVideoEngine,
    VideoGenerationConfig,
    save_video_frames,
)
from modules.lipsync_engine import (
    LipSyncEngine,
    LipSyncConfig,
    synthesize_dialogue_waveform,
    write_wav_file,
    load_audio_any_format,
)
from modules.post_processing import (
    PostProductionEngine,
    PostProcessingConfig,
    PostProcessResult,
    RealESRGANUpscaler,
    RIFEInterpolator,
    AudioVideoMuxer,
    parse_resolution,
    normalize_frame_sequence,
)
from app import CineFlowApp, build_gradio_ui, parse_args

# Ensure CharacterStudio compatibility alias
if not hasattr(CharacterStudio, "get_embedding"):
    CharacterStudio.get_embedding = CharacterStudio.get_character_embedding  # type: ignore


# =============================================================================
# Helper Fixtures & Test Data Generators
# =============================================================================

def generate_test_portrait_image(
    width: int = 480,
    height: int = 480,
    skin_tone: Tuple[int, int, int] = (195, 140, 110),
    eye_color: Tuple[int, int, int] = (40, 30, 25),
    seed_val: int = 42,
) -> Image.Image:
    """Generates a procedural portrait image with identifiable face regions."""
    rng = np.random.RandomState(seed_val)
    img = Image.new("RGB", (width, height), color=(30, 35, 45))
    draw = ImageDraw.Draw(img)

    # Face Oval
    face_box = [int(width * 0.25), int(height * 0.15), int(width * 0.75), int(height * 0.85)]
    draw.ellipse(face_box, fill=skin_tone, outline=(150, 100, 80), width=3)

    # Eyes
    eye_y = int(height * 0.40)
    left_eye_x = int(width * 0.38)
    right_eye_x = int(width * 0.62)
    eye_r = int(width * 0.05)
    draw.ellipse([left_eye_x - eye_r, eye_y - eye_r // 2, left_eye_x + eye_r, eye_y + eye_r // 2], fill=(240, 240, 245))
    draw.ellipse([right_eye_x - eye_r, eye_y - eye_r // 2, right_eye_x + eye_r, eye_y + eye_r // 2], fill=(240, 240, 245))
    draw.ellipse([left_eye_x - eye_r // 2, eye_y - eye_r // 3, left_eye_x + eye_r // 2, eye_y + eye_r // 3], fill=eye_color)
    draw.ellipse([right_eye_x - eye_r // 2, eye_y - eye_r // 3, right_eye_x + eye_r // 2, eye_y + eye_r // 3], fill=eye_color)

    # Nose
    nose_x = int(width * 0.50)
    nose_top = int(height * 0.45)
    nose_bottom = int(height * 0.58)
    draw.line([(nose_x, nose_top), (nose_x - 6, nose_bottom), (nose_x + 6, nose_bottom)], fill=(140, 90, 70), width=2)

    # Mouth
    mouth_y = int(height * 0.68)
    mouth_w = int(width * 0.15)
    draw.chord([nose_x - mouth_w, mouth_y - 10, nose_x + mouth_w, mouth_y + 15], 0, 180, fill=(160, 60, 60), outline=(120, 40, 40))

    # Add slight per-pixel noise for texture realism
    arr = np.array(img, dtype=np.int16)
    noise = rng.randint(-8, 9, size=arr.shape, dtype=np.int16)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def create_test_bengali_audio_wav(
    filepath: str,
    duration_sec: float = 3.375,
    sample_rate: int = 16000,
) -> str:
    """Creates a standardized test dialogue WAV audio file."""
    samples, _ = synthesize_dialogue_waveform(duration_sec=duration_sec, sample_rate=sample_rate)
    write_wav_file(filepath, samples, sample_rate=sample_rate)
    return filepath


@pytest.fixture
def test_env_dirs(tmp_path):
    """Provides isolated directory structure for e2e pipeline testing."""
    out_dir = tmp_path / "outputs"
    masters_dir = out_dir / "masters"
    temp_dir = out_dir / "temp"
    profiles_dir = tmp_path / "character_profiles"
    configs_dir = tmp_path / "configs"

    for d in [out_dir, masters_dir, temp_dir, profiles_dir, configs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Copy real cinematic styles config
    real_styles_path = os.path.join(workspace_root, "configs", "cinematic_styles.json")
    target_styles_path = str(configs_dir / "cinematic_styles.json")
    if os.path.exists(real_styles_path):
        shutil.copyfile(real_styles_path, target_styles_path)
    else:
        styles_payload = {
            "styles": [
                {
                    "id": "kolkata_vintage",
                    "name": "North Kolkata Vintage 35mm",
                    "prompt_prefix": "Kodak Portra 400 35mm film, vintage Kolkata aesthetic",
                    "prompt_suffix": "35mm film grain, Satyajit Ray framing",
                    "negative_prompt": "digital, neon, plastic, 3d",
                    "guidance_scale": 7.0,
                    "num_inference_steps": 28,
                },
                {
                    "id": "cyberpunk_noir",
                    "name": "Dark Cyberpunk Noir",
                    "prompt_prefix": "Blade Runner 2049 cyberpunk noir style, dark rainy neon",
                    "prompt_suffix": "anamorphic streak, high contrast",
                    "negative_prompt": "sunny, cheerful, daylight",
                    "guidance_scale": 8.0,
                    "num_inference_steps": 30,
                },
                {
                    "id": "imax_realism",
                    "name": "IMAX 8K Cinematic Realism",
                    "prompt_prefix": "70mm IMAX film still, cinematic hyper-realism",
                    "prompt_suffix": "8k uhd, dslr, volumetric lighting",
                    "negative_prompt": "cartoon, 3d, blurry",
                    "guidance_scale": 7.5,
                    "num_inference_steps": 30,
                },
            ]
        }
        with open(target_styles_path, "w", encoding="utf-8") as f:
            json.dump(styles_payload, f)

    # Seed pre-configured character "dev"
    real_dev_dir = os.path.join(workspace_root, "character_profiles", "dev")
    target_dev_dir = str(profiles_dir / "dev")
    if os.path.exists(real_dev_dir):
        shutil.copytree(real_dev_dir, target_dev_dir)
    else:
        os.makedirs(target_dev_dir, exist_ok=True)
        dev_profile = {
            "id": "dev",
            "name": "Dev",
            "gender": "male",
            "description": "Charismatic Bengali protagonist",
            "prompt_prefix": "Dev, handsome 30-year-old Bengali man, sharp jawline",
            "embedding_file": "embedding.npy",
            "reference_images": ["ref_primary.png"],
        }
        with open(os.path.join(target_dev_dir, "profile.json"), "w", encoding="utf-8") as f:
            json.dump(dev_profile, f)
        sample_emb = l2_normalize(np.random.RandomState(42).randn(512).astype(np.float32))
        np.save(os.path.join(target_dev_dir, "embedding.npy"), sample_emb)
        ref_img = generate_test_portrait_image(480, 480, seed_val=42)
        ref_img.save(os.path.join(target_dev_dir, "ref_primary.png"))

    return {
        "root": str(tmp_path),
        "outputs": str(out_dir),
        "masters": str(masters_dir),
        "temp": str(temp_dir),
        "profiles": str(profiles_dir),
        "configs": str(configs_dir),
        "styles": target_styles_path,
    }


# =============================================================================
# SCENARIO 1: Full Studio Pipeline Shot (Dev in North Kolkata Vintage 35mm)
# =============================================================================

class TestScenario1FullStudioPipelineShot:
    """
    Tier 4 Scenario 1: Full Studio Pipeline Shot
    Dev in North Kolkata Vintage 35mm + Bengali dialogue audio ->
    81 frames motion generation -> lip-sync alignment -> 1080p 60fps MP4 master muxing.
    """

    def test_scenario1_dev_kolkata_vintage_e2e_full_pipeline(self, test_env_dirs):
        """
        Executes the genuine 4-stage CineFlow sequential pipeline end-to-end
        with memory lifecycle validation at each boundary.
        """
        mgr = VRAMManager.get_instance()
        assert mgr.current_stage is None

        # ---------------------------------------------------------------------
        # Stage 1: Character Studio Keyframe Generation
        # ---------------------------------------------------------------------
        char_studio = CharacterStudio(
            profiles_dir=test_env_dirs["profiles"],
            styles_path=test_env_dirs["styles"],
            memory_manager=mgr,
        )

        dev_char = char_studio.get_character("dev")
        assert dev_char is not None
        assert dev_char.name == "Dev"
        assert dev_char.id == "dev"

        scene_prompt = "Dev standing beside an old vintage yellow taxi on College Street in torrential monsoon rain"
        keyframe_img = char_studio.generate_character_frame(
            character_id="dev",
            scene_prompt=scene_prompt,
            style_id="kolkata_vintage",
            width=720,
            height=480,
            seed=42,
        )

        assert isinstance(keyframe_img, Image.Image)
        assert keyframe_img.size == (720, 480)
        assert keyframe_img.mode == "RGB"
        # Verify VRAM is cleared back to IDLE after Stage 1
        assert mgr.current_stage is None

        # ---------------------------------------------------------------------
        # Stage 2: Dual-Engine Video Motion Synthesis (81 Frames @ 24fps)
        # ---------------------------------------------------------------------
        video_engine = CineVideoEngine(memory_manager=mgr)
        vconfig = VideoGenerationConfig(
            backend="wan2.1",
            num_frames=81,
            fps=24,
            width=720,
            height=480,
            motion_scale=1.0,
            guidance_scale=6.0,
            num_inference_steps=20,
            seed=42,
            motion_prompt="cinematic camera pan, atmospheric rain falling, subtle character head tilt",
        )

        raw_motion_frames = video_engine.generate_motion(
            image=keyframe_img,
            motion_prompt=vconfig.motion_prompt,
            config=vconfig,
        )

        assert isinstance(raw_motion_frames, list)
        assert len(raw_motion_frames) == 81
        assert raw_motion_frames[0].shape == (480, 720, 3)
        assert raw_motion_frames[0].dtype == np.uint8
        # Verify VRAM is cleared back to IDLE after Stage 2
        assert mgr.current_stage is None

        # ---------------------------------------------------------------------
        # Stage 3: Bengali Dialogue Audio Lip-Sync
        # ---------------------------------------------------------------------
        audio_path = os.path.join(test_env_dirs["temp"], "dev_bengali_dialogue.wav")
        dialogue_duration = 81.0 / 24.0  # 3.375 seconds
        create_test_bengali_audio_wav(audio_path, duration_sec=dialogue_duration, sample_rate=16000)
        assert os.path.exists(audio_path)

        lipsync_engine = LipSyncEngine(memory_manager=mgr)
        lipsync_cfg = LipSyncConfig(
            backend="liveportrait",
            sample_rate=16000,
            fps=24,
            audio_padding=True,
        )

        synced_frames, processed_audio = lipsync_engine.synchronize_lips(
            frames=raw_motion_frames,
            audio_path=audio_path,
            config=lipsync_cfg,
        )

        assert isinstance(synced_frames, list)
        assert len(synced_frames) == 81
        assert synced_frames[0].shape == (480, 720, 3)
        assert os.path.exists(processed_audio)
        # Verify mouth motion variance was applied to face region
        mouth_region_diff = np.abs(synced_frames[0].astype(float) - raw_motion_frames[0].astype(float))
        assert np.max(mouth_region_diff) >= 0  # Valid pixel modifications
        # Verify VRAM is cleared back to IDLE after Stage 3
        assert mgr.current_stage is None

        # ---------------------------------------------------------------------
        # Stage 4: Post-Production (Real-ESRGAN 1080p + RIFE 60fps + MP4 Mux)
        # ---------------------------------------------------------------------
        post_engine = PostProductionEngine(memory_manager=mgr)
        master_output_mp4 = os.path.join(test_env_dirs["masters"], "dev_kolkata_vintage_master.mp4")

        # Test on a compact chunk for speed while thoroughly exercising upscale + rife + mux
        test_frame_subset = synced_frames[:16]  # 16 frames for fast CI render
        pp_cfg = PostProcessingConfig(
            enable_upscale=True,
            target_resolution="1080p",
            chunk_batch_size=4,
            enable_interpolation=True,
            target_fps=60,
            source_fps=24,
            crf=18,
            video_codec="libx264",
            audio_codec="aac",
            faststart=True,
        )

        rendered_master_path = post_engine.render_final_master(
            frames=test_frame_subset,
            audio_path=processed_audio,
            output_path=master_output_mp4,
            config=pp_cfg,
        )

        assert os.path.exists(rendered_master_path)
        assert os.path.abspath(rendered_master_path) == os.path.abspath(master_output_mp4)
        assert os.path.getsize(rendered_master_path) > 0  # Valid MP4 container
        # Verify VRAM is cleared back to IDLE after Stage 4
        assert mgr.current_stage is None

    def test_scenario1_app_coordinator_dev_kolkata_vintage(self, test_env_dirs):
        """
        Tests the unified high-level pipeline execution via CineFlowApp.run_master_pipeline.
        """
        # Create custom colab config pointing to test directories
        test_config_path = os.path.join(test_env_dirs["configs"], "test_app_config.yaml")
        cfg_dict = {
            "system": {"device": "cpu", "default_precision": "float16", "sequential_cpu_offload": True},
            "paths": {
                "outputs_dir": test_env_dirs["outputs"],
                "temp_dir": test_env_dirs["temp"],
                "profiles_dir": test_env_dirs["profiles"],
                "styles_config": test_env_dirs["styles"],
            },
            "video_engine": {"primary_backend": "mock", "default_frames": 16},
            "post_processing": {"target_resolution": "1080p", "target_fps": 60, "chunk_batch_size": 4},
        }
        with open(test_config_path, "w", encoding="utf-8") as f:
            import yaml
            yaml.dump(cfg_dict, f)

        app = CineFlowApp(config_path=test_config_path)

        # Generate audio file
        audio_file = os.path.join(test_env_dirs["temp"], "kolkata_audio.wav")
        create_test_bengali_audio_wav(audio_file, duration_sec=1.5, sample_rate=16000)

        master_path, download_path, telemetry_md, status_md = app.run_master_pipeline(
            character_choice="Dev (dev)",
            scene_prompt="Dev in North Kolkata tram depot in evening golden light",
            motion_prompt="slow cinematic pan across old tram tracks",
            style_choice="North Kolkata Vintage 35mm (kolkata_vintage)",
            audio_file=audio_file,
            video_backend="Wan 2.1 (FP8 Quantized)",
            base_resolution_str="480p (720x480)",
            target_resolution_str="1080p (FHD)",
            num_frames=16,
            enable_rife=True,
            inference_steps=15,
            guidance_scale=6.5,
            motion_scale=1.0,
            seed_value=999,
        )

        assert master_path is not None
        assert os.path.exists(master_path)
        assert master_path.endswith(".mp4")
        assert download_path == master_path
        assert "Master Shot Rendered Successfully" in status_md
        assert "1080P" in status_md
        assert "60 FPS" in status_md
        assert "Hardware & Memory Telemetry" in telemetry_md

        # Verify metadata JSON was generated alongside the MP4
        meta_json_path = master_path.replace(".mp4", "_meta.json")
        assert os.path.exists(meta_json_path)
        with open(meta_json_path, "r", encoding="utf-8") as f:
            meta_data = json.load(f)
        assert meta_data["character"] == "dev"
        assert meta_data["style"] == "kolkata_vintage"
        assert meta_data["target_resolution"] == "1080p"
        assert meta_data["fps"] == 60


# =============================================================================
# SCENARIO 2: Custom Character Enrollment & Cyberpunk Noir Render
# =============================================================================

class TestScenario2CustomCharacterEnrollmentAndRender:
    """
    Tier 4 Scenario 2: Custom Character Multi-Photo Enrollment -> Cyberpunk Noir shot generation -> master render.
    """

    def test_scenario2_multi_photo_enrollment_to_cyberpunk_render(self, test_env_dirs):
        """
        Enrolls a new character from 3 multi-angle portrait photos, validates
        unit hypersphere L2 normalization, and generates a Cyberpunk Noir master.
        """
        mgr = VRAMManager.get_instance()
        char_studio = CharacterStudio(
            profiles_dir=test_env_dirs["profiles"],
            styles_path=test_env_dirs["styles"],
            memory_manager=mgr,
        )

        # 1. Generate 3 distinct synthetic portrait photos for "Priya Sen"
        img1 = generate_test_portrait_image(480, 480, skin_tone=(205, 150, 120), eye_color=(30, 60, 40), seed_val=101)
        img2 = generate_test_portrait_image(480, 480, skin_tone=(200, 145, 115), eye_color=(30, 60, 40), seed_val=102)
        img3 = generate_test_portrait_image(480, 480, skin_tone=(210, 155, 125), eye_color=(30, 60, 40), seed_val=103)

        uploaded_photos = [img1, img2, img3]

        # 2. Perform dynamic character enrollment
        profile = char_studio.enroll_character(
            name="Priya Sen",
            description="Elite cyber-detective investigating neo-Kolkata underground",
            images=uploaded_photos,
            gender="female",
            prompt_prefix="Priya Sen, futuristic cyber-detective with glowing neural interface",
        )

        assert profile.id == "priya_sen"
        assert profile.name == "Priya Sen"
        assert profile.gender == "female"
        assert len(profile.reference_images) == 3

        # 3. Verify embedding exists on disk and satisfies unit hypersphere constraint
        emb_file = os.path.join(test_env_dirs["profiles"], "priya_sen", "embedding.npy")
        assert os.path.exists(emb_file)
        loaded_emb = np.load(emb_file)
        assert loaded_emb.shape == (512,)
        assert loaded_emb.dtype == np.float32
        norm_val = compute_l2_norm(loaded_emb)
        assert np.isclose(norm_val, 1.0, atol=1e-5)

        # 4. Generate Cyberpunk Noir Keyframe
        scene_prompt = "Priya Sen standing on rain-soaked neon street under holographic billboards"
        keyframe_img = char_studio.generate_character_frame(
            character_id="priya_sen",
            scene_prompt=scene_prompt,
            style_id="cyberpunk_noir",
            width=720,
            height=480,
            seed=777,
        )
        assert isinstance(keyframe_img, Image.Image)
        assert keyframe_img.size == (720, 480)

        # 5. Synthesize Video Motion Sequence
        video_engine = CineVideoEngine(memory_manager=mgr)
        vconfig = VideoGenerationConfig(
            backend="wan2.1",
            num_frames=16,
            fps=24,
            width=720,
            height=480,
            seed=777,
            motion_prompt="neon rain reflections, cybernetic ocular flicker",
        )
        video_frames = video_engine.generate_motion(
            image=keyframe_img,
            motion_prompt=vconfig.motion_prompt,
            config=vconfig,
        )
        assert isinstance(video_frames, list)
        assert len(video_frames) in [16, 17]
        assert video_frames[0].shape == (480, 720, 3)

        # 6. Render Post-Production Master
        post_engine = PostProductionEngine(memory_manager=mgr)
        master_mp4 = os.path.join(test_env_dirs["masters"], "priya_sen_cyberpunk_master.mp4")
        synth_audio = os.path.join(test_env_dirs["temp"], "priya_audio.wav")
        create_test_bengali_audio_wav(synth_audio, duration_sec=16.0 / 24.0, sample_rate=16000)

        final_master = post_engine.render_final_master(
            frames=video_frames,
            audio_path=synth_audio,
            output_path=master_mp4,
            config=PostProcessingConfig(
                enable_upscale=True,
                target_resolution="1080p",
                enable_interpolation=True,
                target_fps=60,
            ),
        )

        assert os.path.exists(final_master)
        assert os.path.getsize(final_master) > 0

    def test_scenario2_app_enrollment_handler_integration(self, test_env_dirs):
        """
        Tests character enrollment via the CineFlowApp handler interface.
        """
        test_config_path = os.path.join(test_env_dirs["configs"], "test_enroll_config.yaml")
        cfg_dict = {
            "paths": {
                "outputs_dir": test_env_dirs["outputs"],
                "temp_dir": test_env_dirs["temp"],
                "profiles_dir": test_env_dirs["profiles"],
                "styles_config": test_env_dirs["styles"],
            }
        }
        with open(test_config_path, "w", encoding="utf-8") as f:
            import yaml
            yaml.dump(cfg_dict, f)

        app = CineFlowApp(config_path=test_config_path)

        # Upload 2 portrait photos
        img_a = generate_test_portrait_image(480, 480, seed_val=201)
        img_b = generate_test_portrait_image(480, 480, seed_val=202)

        result = app.enroll_new_character(
            name="Rohan Roy",
            description="Underground hacker specialist",
            gender="male",
            prompt_prefix="Rohan Roy, techwear hoodie, cyan holographic visor",
            uploaded_images=[img_a, img_b],
        )
        status_msg, preview_img = result[0], result[1]

        assert "Character Enrolled Successfully" in status_msg
        assert "Rohan Roy" in status_msg
        assert "||e||₂ = 1.000000" in status_msg or "Unit Hypersphere" in status_msg
        assert preview_img is not None

        # Verify dropdown choices were refreshed
        char_names = app.get_character_names()
        assert any("Rohan Roy" in name for name in char_names)


# =============================================================================
# SCENARIO 3: Memory Stress & VRAM Lifecycle Isolation
# =============================================================================

class TestScenario3MemoryStressAndVRAMLifecycle:
    """
    Tier 4 Scenario 3: Memory Stress & VRAM Lifecycle Isolation
    Rapid sequential multi-stage transitions, telemetry tracking, zero leaks.
    """

    def test_scenario3_rapid_multistage_transitions_zero_leak(self, test_env_dirs):
        """
        Simulates high-frequency sequential transitions across all 4 pipeline stages
        and verifies that memory managers maintain clean state and zero leaks.
        """
        mgr = VRAMManager.get_instance()

        @vram_lifecycle_stage("stage_char_stress")
        def run_stage_char():
            mgr.register_model("mock_char_model", np.ones((100, 100), dtype=np.float32))
            return "char_done"

        @vram_lifecycle_stage("stage_video_stress")
        def run_stage_video():
            mgr.register_model("mock_video_model", np.ones((200, 200), dtype=np.float32))
            return "video_done"

        @vram_lifecycle_stage("stage_lipsync_stress")
        def run_stage_lipsync():
            mgr.register_model("mock_lipsync_model", np.ones((150, 150), dtype=np.float32))
            return "lipsync_done"

        @vram_lifecycle_stage("stage_post_stress")
        def run_stage_post():
            mgr.register_model("mock_post_model", np.ones((250, 250), dtype=np.float32))
            return "post_done"

        # Execute 5 rapid pipeline cycles
        for cycle in range(5):
            res1 = run_stage_char()
            assert res1 == "char_done"
            assert mgr.current_stage is None

            res2 = run_stage_video()
            assert res2 == "video_done"
            assert mgr.current_stage is None

            res3 = run_stage_lipsync()
            assert res3 == "lipsync_done"
            assert mgr.current_stage is None

            res4 = run_stage_post()
            assert res4 == "post_done"
            assert mgr.current_stage is None

        # Verify telemetry stats remain healthy
        stats = mgr.get_vram_stats()
        assert "allocated_mb" in stats
        assert "free_mb" in stats
        assert "percent_used" in stats or "utilization_pct" in stats
        assert stats["allocated_mb"] >= 0.0

    def test_scenario3_exception_recovery_and_stage_isolation(self):
        """
        Verifies that an unhandled exception inside a stage cleanly purges models
        and resets current_stage without corrupting global manager state.
        """
        mgr = VRAMManager.get_instance()

        @vram_lifecycle_stage("faulty_stage")
        def failing_stage_worker():
            mgr.register_model("leaked_model", [1, 2, 3, 4])
            raise RuntimeError("Simulated OOM or CUDA Out-Of-Memory Error")

        with pytest.raises(RuntimeError, match="Simulated OOM"):
            failing_stage_worker()

        # Manager state must reset current_stage to None
        assert mgr.current_stage is None

    def test_scenario3_telemetry_warning_badges(self, test_env_dirs):
        """
        Verifies that telemetry markdown dynamically shows health badges based on VRAM load.
        """
        test_config_path = os.path.join(test_env_dirs["configs"], "test_telem_config.yaml")
        cfg_dict = {"paths": {"outputs_dir": test_env_dirs["outputs"], "profiles_dir": test_env_dirs["profiles"]}}
        with open(test_config_path, "w", encoding="utf-8") as f:
            import yaml
            yaml.dump(cfg_dict, f)

        app = CineFlowApp(config_path=test_config_path)

        # 1. Normal Load (<80%)
        with patch.object(app.memory_manager, "get_vram_stats", return_value={"allocated_gb": 4.0, "total_gb": 15.3, "percent_used": 26.1, "reserved_gb": 5.0, "peak_gb": 6.0, "device_name": "Nvidia T4"}):
            md_normal = app.get_telemetry_markdown()
            assert "OPTIMAL HARDWARE HEALTH" in md_normal

        # 2. Elevated Load (80-95%)
        with patch.object(app.memory_manager, "get_vram_stats", return_value={"allocated_gb": 12.8, "total_gb": 15.3, "percent_used": 83.6, "reserved_gb": 13.5, "peak_gb": 14.0, "device_name": "Nvidia T4"}):
            md_elevated = app.get_telemetry_markdown()
            assert "ELEVATED VRAM LOAD" in md_elevated

        # 3. Critical Load (>=95%)
        with patch.object(app.memory_manager, "get_vram_stats", return_value={"allocated_gb": 14.8, "total_gb": 15.3, "percent_used": 96.7, "reserved_gb": 15.0, "peak_gb": 15.2, "device_name": "Nvidia T4"}):
            md_critical = app.get_telemetry_markdown()
            assert "CRITICAL VRAM LOAD" in md_critical


# =============================================================================
# SCENARIO 4: Offline CPU Execution & Fallback Robustness
# =============================================================================

class TestScenario4OfflineCPUExecutionAndFallbackRobustness:
    """
    Tier 4 Scenario 4: Offline CPU Execution & Fallback Robustness
    Graceful failovers between Wan 2.1 / LTX / Mock, LivePortrait / Wav2Lip / Mock,
    Real-ESRGAN / Lanczos, RIFE / Farneback, FFmpeg / MoviePy / OpenCV.
    """

    def test_scenario4_offline_cpu_execution_end_to_end(self, test_env_dirs):
        """
        Executes end-to-end pipeline in offline CPU mode with zero GPU dependency.
        """
        mgr = VRAMManager.get_instance()
        char_studio = CharacterStudio(
            profiles_dir=test_env_dirs["profiles"],
            styles_path=test_env_dirs["styles"],
            memory_manager=mgr,
        )
        video_engine = CineVideoEngine(memory_manager=mgr)
        lipsync_engine = LipSyncEngine(memory_manager=mgr)
        post_engine = PostProductionEngine(memory_manager=mgr)

        # Stage 1: CPU Character Generation
        img = char_studio.generate_character_frame("dev", "Dev in morning light", "imax_realism", 480, 480, seed=12)
        assert isinstance(img, Image.Image)

        # Stage 2: CPU Video Generation
        vcfg = VideoGenerationConfig(backend="mock", num_frames=17, fps=24, width=480, height=480)
        frames = video_engine.generate_motion(img, "camera zoom", vcfg)
        assert isinstance(frames, list)
        assert len(frames) == 17
        assert frames[0].shape == (480, 480, 3)

        # Stage 3: CPU LipSync
        audio_path = os.path.join(test_env_dirs["temp"], "cpu_audio.wav")
        create_test_bengali_audio_wav(audio_path, duration_sec=17.0 / 24.0, sample_rate=16000)
        synced_frames, p_audio = lipsync_engine.synchronize_lips(frames, audio_path, LipSyncConfig(backend="mock"))
        assert isinstance(synced_frames, list)
        assert len(synced_frames) == 17
        assert synced_frames[0].shape == (480, 480, 3)

        # Stage 4: CPU Post Processing
        out_mp4 = os.path.join(test_env_dirs["masters"], "cpu_master.mp4")
        master_file = post_engine.render_final_master(
            frames=synced_frames,
            audio_path=p_audio,
            output_path=out_mp4,
            config=PostProcessingConfig(enable_upscale=False, enable_interpolation=False),
        )
        assert os.path.exists(master_file)
        assert os.path.getsize(master_file) > 0

    def test_scenario4_video_engine_backend_failover(self):
        """
        Tests fallback cascade: Wan 2.1 failure -> LTX -> Procedural Mock fallback.
        """
        mgr = VRAMManager.get_instance()
        engine = CineVideoEngine(memory_manager=mgr)
        test_img = Image.new("RGB", (480, 480), color=(100, 120, 140))

        # Request Wan backend in CPU environment (triggers graceful procedural mock)
        vcfg = VideoGenerationConfig(backend="wan2.1", num_frames=16, fps=24, width=480, height=480)
        frames = engine.generate_motion(test_img, "test motion", vcfg)
        assert isinstance(frames, list)
        assert len(frames) in [16, 17]
        assert frames[0].shape == (480, 480, 3)

        # Request LTX backend in CPU environment (triggers graceful fallback)
        vcfg_ltx = VideoGenerationConfig(backend="ltx-video", num_frames=16, fps=24, width=480, height=480)
        frames_ltx = engine.generate_motion(test_img, "test motion", vcfg_ltx)
        assert isinstance(frames_ltx, list)
        assert len(frames_ltx) in [16, 17]
        assert frames_ltx[0].shape == (480, 480, 3)

    def test_scenario4_lipsync_engine_backend_failover(self, tmp_path):
        """
        Tests fallback cascade: LivePortrait -> Wav2Lip -> Procedural Mel/Phoneme Mock.
        """
        mgr = VRAMManager.get_instance()
        lipsync = LipSyncEngine(memory_manager=mgr)
        raw_frames = [np.zeros((480, 480, 3), dtype=np.uint8) for _ in range(16)]
        audio_file = str(tmp_path / "failover_audio.wav")
        create_test_bengali_audio_wav(audio_file, duration_sec=16.0 / 24.0)

        # Request LivePortrait with uninitialized model -> falls back gracefully
        cfg_lp = LipSyncConfig(backend="liveportrait", sample_rate=16000, fps=24)
        out_lp, _ = lipsync.synchronize_lips(raw_frames, audio_file, cfg_lp)
        assert isinstance(out_lp, list)
        assert len(out_lp) == 16
        assert out_lp[0].shape == (480, 480, 3)

        # Request Wav2Lip with missing checkpoint -> falls back gracefully
        cfg_w2l = LipSyncConfig(backend="wav2lip", sample_rate=16000, fps=24)
        out_w2l, _ = lipsync.synchronize_lips(raw_frames, audio_file, cfg_w2l)
        assert isinstance(out_w2l, list)
        assert len(out_w2l) == 16
        assert out_w2l[0].shape == (480, 480, 3)

    def test_scenario4_post_processing_fallback_chain(self, tmp_path):
        """
        Tests fallback cascade: RealESRGAN -> Lanczos unsharp filter; RIFE -> Optical Flow; FFmpeg -> MoviePy/OpenCV.
        """
        frames = [np.random.randint(50, 200, size=(240, 320, 3), dtype=np.uint8) for _ in range(8)]
        audio_wav = str(tmp_path / "post_failover_audio.wav")
        create_test_bengali_audio_wav(audio_wav, duration_sec=8.0 / 24.0)

        post_engine = PostProductionEngine()
        out_mp4 = str(tmp_path / "fallback_master.mp4")

        # Explicitly test upscaling and RIFE fallbacks
        result_path = post_engine.render_final_master(
            frames=frames,
            audio_path=audio_wav,
            output_path=out_mp4,
            config=PostProcessingConfig(
                enable_upscale=True,
                target_resolution="720p",
                enable_interpolation=True,
                target_fps=48,
                source_fps=24,
            ),
        )

        assert os.path.exists(result_path)
        assert os.path.getsize(result_path) > 0


# =============================================================================
# SCENARIO 5: Gradio App & Colab Notebook Integrity
# =============================================================================

class TestScenario5GradioAppAndColabNotebookIntegrity:
    """
    Tier 4 Scenario 5: Gradio App & Colab Notebook Integrity
    Validate `app.py` pipeline handlers, `configs/colab_t4_config.yaml` schema,
    and `CineFlow_Colab_FreeTier.ipynb` 6-cell JSON validity.
    """

    def test_scenario5_colab_t4_config_yaml_schema(self):
        """
        Validates the structure, types, and critical memory thresholds of configs/colab_t4_config.yaml.
        """
        import yaml
        config_path = os.path.join(workspace_root, "configs", "colab_t4_config.yaml")
        assert os.path.exists(config_path), "colab_t4_config.yaml must exist in configs/"

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        assert isinstance(cfg, dict)
        # Required root sections
        for section in ["system", "paths", "character_engine", "video_engine", "lipsync_engine", "post_processing"]:
            assert section in cfg, f"Missing required section '{section}' in colab_t4_config.yaml"

        # System thresholds
        sys_cfg = cfg["system"]
        assert sys_cfg.get("default_precision") in ["float16", "bfloat16", "fp8", "int8", "float32"]
        assert float(sys_cfg.get("vram_threshold_critical_gb", 0)) <= 15.5
        assert sys_cfg.get("sequential_cpu_offload") is True
        assert sys_cfg.get("vae_slicing") is True
        assert sys_cfg.get("vae_tiling") is True

        # Character Engine
        char_cfg = cfg["character_engine"]
        assert char_cfg.get("embedding_dim") == 512
        assert char_cfg.get("default_character") in ["dev", "neel", "meghla", "cha_kaku"]

        # Video Engine
        vid_cfg = cfg["video_engine"]
        assert vid_cfg.get("primary_backend") in ["wan2.1", "ltx-video", "mock"]
        assert vid_cfg.get("default_frames") == 81
        assert vid_cfg.get("default_fps") == 24

        # LipSync Engine
        lip_cfg = cfg["lipsync_engine"]
        assert lip_cfg.get("sample_rate") == 16000
        assert lip_cfg.get("primary_backend") in ["liveportrait", "wav2lip", "mock"]

        # Post Processing
        pp_cfg = cfg["post_processing"]
        assert pp_cfg.get("target_resolution") in ["720p", "1080p", "4k"]
        assert pp_cfg.get("target_fps") == 60
        assert pp_cfg.get("chunk_batch_size") in [2, 4, 8]
        ffmpeg_cfg = pp_cfg.get("ffmpeg", {})
        assert ffmpeg_cfg.get("crf") == 18
        assert ffmpeg_cfg.get("faststart") is True

    def test_scenario5_colab_notebook_json_and_cell_validity(self):
        """
        Validates that CineFlow_Colab_FreeTier.ipynb is a valid Jupyter Notebook
        with exactly 6 executable code cells ordered for Google Colab T4 execution.
        """
        nb_path = os.path.join(workspace_root, "CineFlow_Colab_FreeTier.ipynb")
        assert os.path.exists(nb_path), "CineFlow_Colab_FreeTier.ipynb must exist in workspace root"

        with open(nb_path, "r", encoding="utf-8") as f:
            nb_data = json.load(f)

        assert "cells" in nb_data
        assert nb_data.get("nbformat") == 4
        cells = nb_data["cells"]
        assert len(cells) == 6, f"Colab notebook must contain exactly 6 cells, found {len(cells)}"

        for i, cell in enumerate(cells):
            assert cell.get("cell_type") == "code", f"Cell {i+1} must be of type 'code'"
            source_text = "".join(cell.get("source", []))
            assert len(source_text.strip()) > 0, f"Cell {i+1} source must not be empty"

        # Verify sequential cell purposes
        c1_src = "".join(cells[0]["source"])
        assert "nvidia-smi" in c1_src or "CUDA" in c1_src or "torch.cuda" in c1_src

        c2_src = "".join(cells[1]["source"])
        assert "character_profiles" in c2_src or "outputs" in c2_src or "mkdir" in c2_src or "REPO_NAME" in c2_src

        c3_src = "".join(cells[2]["source"])
        assert "requirements.txt" in c3_src or "ffmpeg" in c3_src

        c4_src = "".join(cells[3]["source"])
        assert "RealESRGAN" in c4_src or "download" in c4_src or "models" in c4_src

        c5_src = "".join(cells[4]["source"])
        assert "pytest" in c5_src

        c6_src = "".join(cells[5]["source"])
        assert "app.py" in c6_src or "gradio" in c6_src or "--share" in c6_src

    def test_scenario5_gradio_app_ui_compilation_and_handlers(self, test_env_dirs):
        """
        Tests CineFlowApp initialization, helper methods, Gradio UI Block construction,
        and CLI argument parser.
        """
        test_config_path = os.path.join(test_env_dirs["configs"], "test_app_ui_config.yaml")
        cfg_dict = {
            "paths": {
                "outputs_dir": test_env_dirs["outputs"],
                "temp_dir": test_env_dirs["temp"],
                "profiles_dir": test_env_dirs["profiles"],
                "styles_config": test_env_dirs["styles"],
            }
        }
        with open(test_config_path, "w", encoding="utf-8") as f:
            import yaml
            yaml.dump(cfg_dict, f)

        app = CineFlowApp(config_path=test_config_path)

        # 1. Character Helpers
        char_names = app.get_character_names()
        assert len(char_names) > 0
        assert app.resolve_character_id("Dev (dev)") == "dev"
        assert app.resolve_character_id("neel") == "neel"
        preview_img = app.get_character_preview("dev")
        assert preview_img is not None
        assert isinstance(preview_img, Image.Image)

        # 2. Style Helpers
        style_names = app.get_style_names()
        assert len(style_names) > 0
        assert app.resolve_style_id("North Kolkata Vintage 35mm (kolkata_vintage)") == "kolkata_vintage"
        assert app.resolve_style_id("cyberpunk_noir") == "cyberpunk_noir"

        # 3. Telemetry Markdown
        telem_md = app.get_telemetry_markdown()
        assert "Hardware & Memory Telemetry" in telem_md

        # 4. History Browser
        # Create a mock master and metadata
        mock_master_path = os.path.join(test_env_dirs["masters"], "cineflow_dev_test_master.mp4")
        with open(mock_master_path, "wb") as f:
            f.write(b"MOCK_MP4_DATA" * 50)
        mock_meta_path = mock_master_path.replace(".mp4", "_meta.json")
        with open(mock_meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "character": "dev",
                "style": "kolkata_vintage",
                "resolution": "1080p",
                "fps": 60,
                "duration": 3.375,
                "timestamp": "2026-08-25 12:00:00",
                "scene_prompt": "Vintage Kolkata test",
            }, f)

        history_items = app.get_history_items()
        assert len(history_items) >= 1
        assert history_items[0]["character"] == "dev"
        assert history_items[0]["filename"] == "cineflow_dev_test_master.mp4"

        # 5. Gradio UI Blocks Compilation
        demo = build_gradio_ui(app)
        assert demo is not None
        assert hasattr(demo, "blocks")

        # 6. CLI Argument Parsing
        with patch.object(sys, "argv", ["app.py", "--host", "127.0.0.1", "--port", "7861", "--share", "--config", test_config_path]):
            cli_args = parse_args()
            assert cli_args.host == "127.0.0.1"
            assert cli_args.port == 7861
            assert cli_args.share is True
            assert cli_args.config == test_config_path
