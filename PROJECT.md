# Project: CineFlow-AI

## Architecture
CineFlow-AI is a modular cinematic AI studio and sequential pipeline optimized for Google Colab Free Tier (Nvidia T4 15-16GB VRAM / 12.7GB System RAM) and multi-platform local development (Windows/Linux/macOS).

```
+-----------------------------------------------------------------------------------------------+
|                                    Gradio WebUI (app.py)                                      |
|            [Studio Tab]         |      [Character Manager]       |    [History Gallery]       |
+-----------------------------------------------------------------------------------------------+
                                                │
                                                ▼
+-----------------------------------------------------------------------------------------------+
|                                   CineFlow Pipeline Master                                    |
+-----------------------------------------------------------------------------------------------+
       │ Stage 1                       │ Stage 2                  │ Stage 3           │ Stage 4
       ▼                               ▼                          ▼                   ▼
┌──────────────┐               ┌───────────────┐          ┌──────────────┐     ┌──────────────┐
│  Character   │               │ Video Motion  │          │   Lip-Sync   │     │ Post-Process │
│    Engine    │               │  Synthesizer  │          │    Engine    │     │   & Upscale  │
│ (InstantID / │               │ (Wan 2.1 FP8 /│          │(LivePortrait/│     │(Real-ESRGAN /│
│ InsightFace) │               │  LTX-Video)   │          │   Wav2Lip)   │     │ RIFE / Mux)  │
└──────────────┘               └───────────────┘          └──────────────┘     └──────────────┘
       ▲                               ▲                          ▲                   ▲
       └───────────────────────────────┴──────────────────────────┴───────────────────┘
                                                │
                                                ▼
                               ┌─────────────────────────────────┐
                               │     VRAMManager (R1 Engine)     │
                               │  - Aggressive Stage Flushing    │
                               │  - Sequential CPU Offload       │
                               │  - Dynamic VRAM Telemetry       │
                               │  - Auto-Precision Selection     │
                               └─────────────────────────────────┘
```

### Key Hardware & Platform Constraints
1. **Nvidia T4 GPU (Colab Free Tier)**: 15.3 GB VRAM (Turing CC 7.5). bfloat16 fast path unsupported -> float16 / fp8 / int8 quantization.
2. **Sequential Memory Model**: No two heavy neural models reside in VRAM simultaneously. Each stage executes inside `@vram_lifecycle_stage` with full gc, empty_cache, ipc_collect, and sync.
3. **Multi-Platform & Offline CPU Mock Fallback**: All engines must seamlessly fallback to deterministic procedural mock generators when CUDA or model weights are unavailable, enabling 100% test pass rates across CPU/CI.

---

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | VRAMManager Singleton | Centralized memory monitor, telemetry, threshold tracking, and lifecycle coordinator | M1 | R1 (ORIGINAL_REQUEST.md §14-20) |
| 2 | 4-Step Memory Purge | Aggressive stage flushing: gc.collect(), cuda.empty_cache(), ipc_collect(), cuda.synchronize() | M1 | R1 (ORIGINAL_REQUEST.md §16) |
| 3 | Auto-Precision Selector | Precision resolver (bfloat16 for Ampere+, float16 for T4, fp8 for quantized DiT, float32 for CPU) | M1 | R1 (ORIGINAL_REQUEST.md §17) |
| 4 | Stage Lifecycle Decorator | `@vram_lifecycle_stage` decorator ensuring clean memory isolation between pipeline stages | M1 | R1 (ORIGINAL_REQUEST.md §18) |
| 5 | Offload & Slicing Helpers | Sequential CPU offloading helpers and VAE slicing/tiling utilities | M1 | R1 (ORIGINAL_REQUEST.md §19) |
| 6 | InstantID / InsightFace Studio | Character generation engine combining facial embeddings, landmarks, and SDXL/Flux | M2 | R2 (ORIGINAL_REQUEST.md §21-26) |
| 7 | Pre-configured Face Bank | Character profiles for Dev, Neel, Meghla, Cha Kaku with .npy embeddings & metadata | M2 | R2 (ORIGINAL_REQUEST.md §23) |
| 8 | Dynamic Face Enrollment | 1-5 portrait photo enrollment, landmark detection, L2-normalized 512D ArcFace embedding caching | M2 | R2 (ORIGINAL_REQUEST.md §24) |
| 9 | Cinematic Style Presets | Style presets in `configs/cinematic_styles.json` (IMAX 8K, Kolkata 35mm, Ghibli, Cyberpunk, etc.) | M2 | R2 (ORIGINAL_REQUEST.md §25) |
| 10 | Prompt Synthesis Engine | Hierarchical prompt synthesis merging style prefix/suffix, character tokens, and negative prompts | M2 | R2 (ORIGINAL_REQUEST.md §25) |
| 11 | Wan 2.1 FP8 Primary Engine | 1.3B I2V video generator with FP8/4-bit quantization and sequential CPU offloading | M3 | R3 (ORIGINAL_REQUEST.md §27-33) |
| 12 | LTX-Video Fallback Engine | High-speed 0.9.1 quantized video generation backend fallback | M3 | R3 (ORIGINAL_REQUEST.md §30) |
| 13 | Procedural Video CPU Mock | Deterministic mathematical video synthesis (pan/zoom/lighting motion, 81 frames @ 24fps) | M3 | R3 (ORIGINAL_REQUEST.md §32) |
| 14 | Temporal & Camera Controls | Configurable frame count (81 frames), motion bucket, camera guidance scale, resolution (480p/720p) | M3 | R3 (ORIGINAL_REQUEST.md §31) |
| 15 | LivePortrait Primary Sync | Expressive 3D facial retargeting driven by dialogue audio energy and phoneme features | M4 | R4 (ORIGINAL_REQUEST.md §34-39) |
| 16 | Wav2Lip Robust Fallback | Mouth-region neural lip-syncing for extreme head angles and fast speech | M4 | R4 (ORIGINAL_REQUEST.md §37) |
| 17 | Bengali Audio Pipeline | Standard dialogue audio (.wav, .mp3) ingestion, 16kHz resampling, 80-channel log-mel extraction | M4 | R4 (ORIGINAL_REQUEST.md §38) |
| 18 | Procedural LipSync Mock | Energy-reactive mouth deformation and audio synchronization for CPU/test execution | M4 | R4 (ORIGINAL_REQUEST.md §34) |
| 19 | Real-ESRGAN Super-Resolution | 720p -> 1080p / 4K super-resolution with chunked frame batching (N=2-4) to prevent OOM | M5 | R5 (ORIGINAL_REQUEST.md §40-45) |
| 20 | RIFE Frame Interpolation | 24fps -> 60fps smooth cinematic frame interpolation with temporal flow blending | M5 | R5 (ORIGINAL_REQUEST.md §43) |
| 21 | Audio/Video Master Muxer | MoviePy / FFmpeg multiplexing generating H.264/AAC MP4 masters with faststart | M5 | R5 (ORIGINAL_REQUEST.md §44) |
| 22 | Centralized YAML Config | `configs/colab_t4_config.yaml` specifying VRAM limits, precision, model paths, and timeouts | M6 | R6 (ORIGINAL_REQUEST.md §46-53) |
| 23 | Multi-Tab Gradio WebUI | `app.py` with Studio Tab, Character Manager Tab, and Batch Render / History Tab | M6 | R6 (ORIGINAL_REQUEST.md §48-51) |
| 24 | Self-Contained Colab Notebook | `CineFlow_Colab_FreeTier.ipynb` with 6 sequential cells, automated model downloaders, share=True | M6 | R6 (ORIGINAL_REQUEST.md §52) |
| 25 | Packaging & Dependencies | `requirements.txt` and `README.md` with complete installation and architecture guide | M6 | R6 (ORIGINAL_REQUEST.md §46) |
| 26 | Unit & Component Test Suite | Modular test files for all engines in `tests/` passing 100% on CPU and GPU | M7 | R7 (ORIGINAL_REQUEST.md §54-62) |
| 27 | Full E2E Pipeline Test | End-to-end integration test validating Character -> Video -> LipSync -> Upscale -> Master | M7 | R7 (ORIGINAL_REQUEST.md §61) |

---

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| 1 | M1: Memory & VRAM Engine | `modules/memory_manager.py` (VRAMManager, flushing, precision, decorators) | none | DONE |
| 2 | M2: Character Engine & Face Bank | `modules/character_engine.py`, `character_profiles/` (Dev, Neel, Meghla, Cha Kaku), `configs/cinematic_styles.json` | M1 | DONE |
| 3 | M3: Dual Video Engine | `modules/video_engine.py` (Wan 2.1 FP8, LTX fallback, CPU mock, 81 frames @ 24fps) | M1 | DONE |
| 4 | M4: LipSync & Bengali Audio | `modules/lipsync_engine.py` (LivePortrait, Wav2Lip fallback, 16kHz audio sync, CPU mock) | M1, M3 | PLANNED |
| 5 | M5: Post-Processing & Upscaling | `modules/post_processing.py` (Real-ESRGAN chunked, RIFE 24->60fps, MoviePy muxer, CPU mock) | M1, M3, M4 | PLANNED |
| 6 | M6: UI, Notebook & Packaging | `configs/colab_t4_config.yaml`, `app.py` (Gradio 3 tabs), `CineFlow_Colab_FreeTier.ipynb`, `requirements.txt`, `README.md` | M1-M5 | PLANNED |
| 7 | M7: Automated Test Suite & Verification | `tests/test_memory_manager.py`, `tests/test_character_engine.py`, `tests/test_video_engine.py`, `tests/test_lipsync_engine.py`, `tests/test_post_processing.py`, `tests/test_pipeline_e2e.py` | M1-M6 | PLANNED |

---

## Interface Contracts

### 1. `VRAMManager` (`modules/memory_manager.py`)
```python
class VRAMManager:
    @classmethod
    def get_instance(cls) -> "VRAMManager": ...
    def get_vram_stats(self) -> Dict[str, Union[float, str]]: ...
    def flush_memory(self, aggressive: bool = True) -> Dict[str, float]: ...
    def get_optimal_precision(self, model_family: str = "diffusion", quantize: bool = True) -> Any: ...
    def vram_lifecycle_stage(self, stage_name: str): ... # Decorator
    def stage_context(self, stage_name: str): ... # Context manager
    def enable_sequential_cpu_offload(self, pipeline: Any) -> Any: ...
    def enable_vae_optimizations(self, vae: Any) -> Any: ...
```

### 2. `CharacterStudio` (`modules/character_engine.py`)
```python
@dataclass
class CharacterProfile:
    id: str
    name: str
    description: str
    gender: str
    prompt_prefix: str
    negative_prompt: str
    embedding_path: Optional[str]
    reference_images: List[str]

class CharacterStudio:
    def __init__(self, profiles_dir: str = "character_profiles", styles_path: str = "configs/cinematic_styles.json", memory_manager: Optional[VRAMManager] = None): ...
    def list_characters(self) -> List[CharacterProfile]: ...
    def get_character(self, character_id: str) -> Optional[CharacterProfile]: ...
    def list_styles(self) -> List[Dict[str, str]]: ...
    def synthesize_prompt(self, character_id: str, scene_prompt: str, style_id: str, custom_modifiers: str = "") -> Tuple[str, str]: ...
    def enroll_character(self, name: str, description: str, images: List[Union[str, np.ndarray, Image.Image]], gender: str = "neutral", prompt_prefix: str = "") -> CharacterProfile: ...
    def generate_character_frame(self, character_id: str, scene_prompt: str, style_id: str, width: int = 720, height: int = 480, seed: Optional[int] = None) -> Image.Image: ...
```

### 3. `CineVideoEngine` (`modules/video_engine.py`)
```python
@dataclass
class VideoGenerationConfig:
    backend: str = "wan2.1" # "wan2.1", "ltx-video", "mock"
    num_frames: int = 81
    fps: int = 24
    width: int = 720
    height: int = 480
    motion_scale: float = 1.0
    guidance_scale: float = 6.0
    num_inference_steps: int = 30
    seed: Optional[int] = None

class CineVideoEngine:
    def __init__(self, memory_manager: Optional[VRAMManager] = None, config_path: Optional[str] = None): ...
    def generate_motion(self, image: Union[str, Image.Image, np.ndarray], motion_prompt: str, config: Optional[VideoGenerationConfig] = None) -> List[np.ndarray]: ... # Returns frames (N, H, W, 3) RGB uint8
```

### 4. `LipSyncEngine` (`modules/lipsync_engine.py`)
```python
@dataclass
class LipSyncConfig:
    backend: str = "liveportrait" # "liveportrait", "wav2lip", "mock"
    sample_rate: int = 16000
    fps: int = 24
    mel_step_size: int = 16
    temp_dir: str = "outputs/temp_lipsync"

class LipSyncEngine:
    def __init__(self, memory_manager: Optional[VRAMManager] = None, config_path: Optional[str] = None): ...
    def process_audio(self, audio_path: str, target_fps: int = 24) -> Dict[str, Any]: ...
    def synchronize_lips(self, frames: List[np.ndarray], audio_path: str, config: Optional[LipSyncConfig] = None) -> Tuple[List[np.ndarray], str]: ... # Returns (synced_frames, processed_audio_path)
```

### 5. `PostProductionEngine` (`modules/post_processing.py`)
```python
@dataclass
class PostProcessingConfig:
    enable_upscale: bool = True
    target_resolution: str = "1080p" # "1080p" (1920x1080), "4k" (3840x2160), "720p"
    chunk_batch_size: int = 4
    enable_interpolation: bool = True
    target_fps: int = 60 # Interpolate from 24fps -> 60fps
    video_codec: str = "libx264"
    audio_codec: str = "aac"
    crf: int = 18

class PostProductionEngine:
    def __init__(self, memory_manager: Optional[VRAMManager] = None, config_path: Optional[str] = None): ...
    def upscale_frames(self, frames: List[np.ndarray], target_resolution: str = "1080p", chunk_size: int = 4) -> List[np.ndarray]: ...
    def interpolate_fps(self, frames: List[np.ndarray], source_fps: int = 24, target_fps: int = 60) -> List[np.ndarray]: ...
    def mux_video_audio(self, frames: List[np.ndarray], audio_path: Optional[str], output_path: str, fps: int = 24, crf: int = 18) -> str: ...
    def render_final_master(self, frames: List[np.ndarray], audio_path: Optional[str], output_path: str, config: Optional[PostProcessingConfig] = None) -> str: ...
```

---

## Code Layout
```
d:/Antigravity/AI Video Studio/
├── configs/
│   ├── colab_t4_config.yaml        # [M6] System thresholds, VRAM limits, models
│   └── cinematic_styles.json       # [M2] 5 Cinematic style presets
├── modules/
│   ├── __init__.py                 # Export all core engines
│   ├── memory_manager.py           # [M1] VRAMManager, stage decorators, flushing
│   ├── character_engine.py         # [M2] CharacterStudio, InstantID/FaceBank
│   ├── video_engine.py             # [M3] CineVideoEngine (Wan 2.1 + LTX + Mock)
│   ├── lipsync_engine.py           # [M4] LipSyncEngine (LivePortrait + Wav2Lip + Bengali audio)
│   └── post_processing.py          # [M5] PostProductionEngine (Real-ESRGAN, RIFE, Muxer)
├── character_profiles/             # [M2] Preconfigured Character Face Bank
│   ├── dev/
│   │   ├── profile.json
│   │   └── embedding.npy
│   ├── neel/
│   │   ├── profile.json
│   │   └── embedding.npy
│   ├── meghla/
│   │   ├── profile.json
│   │   └── embedding.npy
│   └── cha_kaku/
│       ├── profile.json
│       └── embedding.npy
├── tests/                          # [M7] 100% Passing Automated Test Suite
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_memory_manager.py
│   ├── test_character_engine.py
│   ├── test_video_engine.py
│   ├── test_lipsync_engine.py
│   ├── test_post_processing.py
│   └── test_pipeline_e2e.py
├── app.py                          # [M6] Multi-Tab Gradio WebUI
├── CineFlow_Colab_FreeTier.ipynb   # [M6] Self-Contained Google Colab Free Tier Notebook
├── requirements.txt                # [M6] Dependency definitions
└── README.md                       # [M6] Complete System Documentation
```

### Write Ownership
- **Milestone 1 Worker**: `modules/memory_manager.py`
- **Milestone 2 Worker**: `modules/character_engine.py`, `character_profiles/**`, `configs/cinematic_styles.json`
- **Milestone 3 Worker**: `modules/video_engine.py`
- **Milestone 4 Worker**: `modules/lipsync_engine.py`
- **Milestone 5 Worker**: `modules/post_processing.py`
- **Milestone 6 Worker**: `configs/colab_t4_config.yaml`, `app.py`, `CineFlow_Colab_FreeTier.ipynb`, `requirements.txt`, `README.md`, `modules/__init__.py`
- **Milestone 7 / Test Writer**: `tests/**`
