"""
CineFlow-AI: Modular Cinematic AI Studio
=========================================
Sequential pipeline optimized for Google Colab Free Tier (Nvidia T4 15-16GB VRAM)
and cross-platform CPU/GPU execution.

Package exports:
- Memory & VRAM Engine (M1 / R1)
- Character Studio & Face Bank (M2 / R2)
- Dual-Engine Quantized Video Synthesizer (M3 / R3)
- Bengali Audio Lip-Sync & Phoneme Alignment (M4 / R4)
- Post-Production & Master Rendering Engine (M5 / R5)
"""

from modules.memory_manager import (
    VRAMManager,
    VRAMStageContext,
    vram_lifecycle_stage,
    stage_context,
    flush_memory,
    get_vram_stats,
    get_optimal_precision,
    enable_sequential_cpu_offload,
    enable_vae_optimizations,
    enable_attention_slicing,
    register_model,
    unregister_model,
    purge_models,
)

from modules.character_engine import (
    CharacterProfile,
    CharacterFaceAdapter,
    CharacterStudio,
    compute_l2_norm,
    l2_normalize,
    fuse_consensus_embeddings,
    extract_facial_embedding_from_image,
    sanitize_character_slug,
)

from modules.agent_gemini import (
    CharacterGeminiAgent,
    resolve_gemini_model_name,
)

from modules.hot_reloader import (
    StudioHotReloader,
)

from modules.video_engine import (
    VideoGenerationConfig,
    CineVideoEngine,
    Wan21Backend,
    LTXVideoBackend,
    MockVideoBackend,
    get_valid_dit_frame_counts,
    validate_frame_count,
    save_video_frames,
)

from modules.lipsync_engine import (
    LipSyncConfig,
    AudioAnalysisResult,
    LipSyncResult,
    LipSyncEngine,
    LivePortraitBackend,
    Wav2LipBackend,
    MockLipSyncBackend,
    hz_to_mel,
    mel_to_hz,
    create_mel_filterbank,
    compute_stft,
    extract_log_mel_spectrogram,
    resample_audio_waveform,
    load_audio_any_format,
    write_wav_file,
    synthesize_dialogue_waveform,
    synchronize_dialogue,
)

from modules.post_processing import (
    PostProcessingConfig,
    PostProcessResult,
    PostProductionEngine,
    RESOLUTION_PRESETS,
    parse_resolution,
    normalize_frame_to_numpy,
    normalize_frame_sequence,
)

from modules.nle_timeline import (
    TrackType,
    ClipTransition,
    Clip,
    Track,
    Marker,
    TimecodeConverter,
    NLETimeline,
)

from modules.color_engine import (
    ColorWheelVector,
    ColorGradeSettings,
    CineColorEngine,
)

from modules.audio_engine import (
    EQBand,
    CompressorSettings,
    LimiterSettings,
    NoiseGateSettings,
    AutoDuckingSettings,
    FairlightChannelStrip,
    FairlightAudioEngine,
)

from modules.meme_vault import (
    MemeClip,
    MemeVault,
)

from modules.transitions_engine import (
    TransitionDefinition,
    FontDefinition,
    SubtitleStylePreset,
    TransitionsAndFontEngine,
)

from modules.token_optimizer import (
    TokenBenchmarkResult,
    ASTDeltaTokenOptimizer,
)

from modules.agentic_scripting_engine import (
    PlanStepStatus,
    PlanStep,
    EditingPlan,
    AgenticScriptingEngine,
)

__all__ = [
    # M1 Memory Manager
    "VRAMManager",
    "VRAMStageContext",
    "vram_lifecycle_stage",
    "stage_context",
    "flush_memory",
    "get_vram_stats",
    "get_optimal_precision",
    "enable_sequential_cpu_offload",
    "enable_vae_optimizations",
    "enable_attention_slicing",
    "register_model",
    "unregister_model",
    "purge_models",
    # M2 Character Engine & Face Trainer
    "CharacterProfile",
    "CharacterFaceAdapter",
    "CharacterStudio",
    "compute_l2_norm",
    "l2_normalize",
    "fuse_consensus_embeddings",
    "extract_facial_embedding_from_image",
    "sanitize_character_slug",
    # Gemini Agent & Hot Reloader
    "CharacterGeminiAgent",
    "resolve_gemini_model_name",
    "StudioHotReloader",
    # M3 Video Engine
    "VideoGenerationConfig",
    "CineVideoEngine",
    "Wan21Backend",
    "LTXVideoBackend",
    "MockVideoBackend",
    "get_valid_dit_frame_counts",
    "validate_frame_count",
    "save_video_frames",
    # M4 LipSync Engine
    "LipSyncConfig",
    "AudioAnalysisResult",
    "LipSyncResult",
    "LipSyncEngine",
    "LivePortraitBackend",
    "Wav2LipBackend",
    "MockLipSyncBackend",
    "hz_to_mel",
    "mel_to_hz",
    "create_mel_filterbank",
    "compute_stft",
    "extract_log_mel_spectrogram",
    "resample_audio_waveform",
    "load_audio_any_format",
    "write_wav_file",
    "synthesize_dialogue_waveform",
    "synchronize_dialogue",
    # M5 Post-Processing Engine
    "PostProcessingConfig",
    "PostProcessResult",
    "PostProductionEngine",
    "RESOLUTION_PRESETS",
    "parse_resolution",
    "normalize_frame_to_numpy",
    "normalize_frame_sequence",
    # NLE Studio Core Engines
    "TrackType",
    "ClipTransition",
    "Clip",
    "Track",
    "Marker",
    "TimecodeConverter",
    "NLETimeline",
    "ColorWheelVector",
    "ColorGradeSettings",
    "CineColorEngine",
    "EQBand",
    "CompressorSettings",
    "LimiterSettings",
    "NoiseGateSettings",
    "AutoDuckingSettings",
    "FairlightChannelStrip",
    "FairlightAudioEngine",
    "MemeClip",
    "MemeVault",
    "TransitionDefinition",
    "FontDefinition",
    "SubtitleStylePreset",
    "TransitionsAndFontEngine",
    "TokenBenchmarkResult",
    "ASTDeltaTokenOptimizer",
    "PlanStepStatus",
    "PlanStep",
    "EditingPlan",
    "AgenticScriptingEngine",
]
