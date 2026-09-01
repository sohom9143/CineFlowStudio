# 🎬 CineFlow-AI: Modular Cinematic AI Studio

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.2+](https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c.svg)](https://pytorch.org/)
[![Gradio UI](https://img.shields.io/badge/Gradio-4.26%2B-orange.svg)](https://gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Colab Free Tier](https://img.shields.io/badge/Colab-Nvidia%20T4%20Optimized-green.svg)](https://colab.research.google.com/)

> **CineFlow-AI** is a modular cinematic AI studio and sequential pipeline engineered for high-fidelity, identity-consistent video generation. Specially optimized for **Google Colab Free Tier** (Nvidia T4 15-16GB VRAM, 12.7GB System RAM) and multi-platform local development (Windows, Linux, macOS).

---

## 🏛️ System Architecture

CineFlow-AI implements a strict **Sequential Memory Lifecycle Architecture** coordinated by `VRAMManager`. To guarantee zero Out-Of-Memory (OOM) failures on 16GB GPUs, heavy neural networks never co-exist in VRAM simultaneously. Each stage executes inside an isolated `@vram_lifecycle_stage` boundary that aggressively purges tensors, empties CUDA caches, collects IPC handles, and synchronizes CUDA streams between pipeline transitions.

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

---

## 🌟 Key Features & Pipeline Stages

### 🛡️ 1. Sequential VRAM & Memory Lifecycle (`modules/memory_manager.py`)
- **Aggressive 4-Step Purge**: Atomic memory cleanup (`gc.collect()`, `torch.cuda.empty_cache()`, `torch.cuda.ipc_collect()`, `torch.cuda.synchronize()`).
- **Auto-Precision Selector**: Dynamically resolves optimal compute precision (`float16` for Nvidia T4 CC 7.5, `bfloat16` for Ampere/Hopper, `fp8` for quantized DiT, and `float32` on CPU).
- **Stage Lifecycle Decorators**: `@vram_lifecycle_stage` and `stage_context` ensuring 100% memory isolation.
- **Offload & VAE Optimizations**: Sequential CPU offloading, VAE spatial slicing, and VAE 512px tiling.

### 👤 2. Character Studio & Cached Face Bank (`modules/character_engine.py`)
- **Identity Consistency**: InstantID & InsightFace ArcFace 512-dimensional facial embedding engine.
- **Pre-Configured Face Bank**: Built-in character profiles for **Dev**, **Neel**, **Meghla**, and **Cha Kaku**.
- **Dynamic Character Enrollment**: Ingest 1-5 portrait photos, extract unit-hypersphere normalized consensus embeddings ($\|\mathbf{e}\|_2 = 1.0$), and persist profiles to disk.
- **Cinematic Style Presets**: 5 master visual presets in `configs/cinematic_styles.json` (IMAX 8K Realism, North Kolkata Vintage 35mm, Studio Ghibli Anime, Dark Cyberpunk Noir, Satyajit Ray B&W).

### 🎥 3. Dual-Engine Quantized Video Motion Synthesizer (`modules/video_engine.py`)
- **Primary Backend**: Wan 2.1 (1.3B I2V DiT) with FP8 / 4-bit quantization and sequential CPU offload.
- **Fallback Backend**: LTX-Video (0.9.1) high-speed diffusion backend.
- **Procedural CPU Mock**: Deterministic mathematical video synthesis (affine camera motion, lighting breathing, seeded film grain, 81 frames @ 24fps) for offline and CPU development.
- **Temporal DiT Mathematics**: Strict frame count validation adhering to $(4k + 1)$ and $(8k + 1)$ downsampling formulas.

### 🎙️ 4. Bengali Audio Lip-Sync & Phoneme Alignment (`modules/lipsync_engine.py`)
- **Dual-Tier Retargeting**: LivePortrait 3D expressive facial phoneme sync with Wav2Lip lower-face fallback.
- **Standardized Bengali Audio**: Ingests `.wav` and `.mp3` dialogue audio, automatic 16,000 Hz resampling, and 80-channel log-Mel spectrogram feature extraction (16-mel temporal chunking).
- **Procedural Energy Retargeting**: Energy-reactive mouth deformation with contour interpolation for CPU/test execution.

### ✨ 5. Post-Production & Master Rendering Engine (`modules/post_processing.py`)
- **Chunked Super-Resolution**: Real-ESRGAN FP16 upscaling (720p -> 1080p / 4K) using atomic chunk batching ($N=4$ frames) and $512\text{px}$ spatial tiling to bound peak VRAM under $3.8\text{ GB}$.
- **RIFE Temporal Interpolation**: Optical-flow temporal blending converting 24fps cinema footage to ultra-smooth 60fps broadcast video.
- **Broadcast MP4 Multiplexer**: FFmpeg / MoviePy muxing producing web-optimized H.264 / AAC MP4 masters with `+faststart`.

### 🖥️ 6. Gradio WebUI Studio & Colab Packaging (`app.py`, `CineFlow_Colab_FreeTier.ipynb`)
- **Director's Studio Tab**: Full interactive parameter control, character select, scene & motion prompts, Bengali audio upload, live pipeline progress bar, master video player, and VRAM telemetry.
- **Character Manager Tab**: New character registration UI, photo upload, ArcFace embedding extraction, and Face Bank browser.
- **History Gallery Tab**: History of rendered cinematic masters with per-item metadata and instant download links.

---

## ⚡ Google Colab Free Tier Quickstart

Launch CineFlow-AI in Google Colab with a single click using the self-contained notebook `CineFlow_Colab_FreeTier.ipynb`:

1. Open **[Google Colab](https://colab.research.google.com/)**.
2. Upload or open `CineFlow_Colab_FreeTier.ipynb`.
3. Ensure GPU runtime is active: **Runtime** -> **Change runtime type** -> **T4 GPU**.
4. Run the 6 sequential notebook cells:
   - **Cell 1**: Hardware & CUDA diagnostic (`!nvidia-smi`)
   - **Cell 2**: Repository clone & directory initialization
   - **Cell 3**: Dependency installation (`!pip install -r requirements.txt`)
   - **Cell 4**: Model weights downloader
   - **Cell 5**: Diagnostic test verification (`!pytest tests/`)
   - **Cell 6**: Launch Gradio WebUI Studio with public share link (`!python app.py --share`)
5. Click the generated public URL (`https://xxxx.gradio.live`) to open the Director's Studio in your browser!

---

## 💻 Local Installation & Setup

### Prerequisites
- Python 3.10, 3.11, or 3.12 (Python 3.14 compatible)
- FFmpeg installed and available on system `PATH`
- (Optional) Nvidia GPU with CUDA 12.x (Falls back seamlessly to CPU Procedural Mock)

### 1. Clone Repository
```bash
git clone https://github.com/cineflow-ai/cineflow-ai.git
cd "AI Video Studio"
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Launch Gradio Studio WebUI
```bash
python app.py --port 7860
```
Open [http://localhost:7860](http://localhost:7860) in your browser.

#### CLI Launch Options
| Option | Type | Default | Description |
|---|---|---|---|
| `--host` | `str` | `0.0.0.0` | Server host network binding |
| `--port` | `int` | `7860` | Server port number |
| `--share` | `flag` | `False` | Generate public Gradio link |
| `--config` | `str` | `configs/colab_t4_config.yaml` | System configuration YAML path |

---

## ⚙️ Configuration Reference (`configs/colab_t4_config.yaml`)

CineFlow-AI is centrally configured via `configs/colab_t4_config.yaml`:

```yaml
system:
  device: "auto"
  default_precision: "float16"
  vram_threshold_warning_gb: 13.0
  vram_threshold_critical_gb: 14.5
  sequential_cpu_offload: true
  vae_slicing: true
  vae_tiling: true

video_engine:
  primary_backend: "wan2.1"
  default_frames: 81
  default_fps: 24
  quantization: "fp8"

lipsync_engine:
  primary_backend: "liveportrait"
  sample_rate: 16000

post_processing:
  upscaler_model: "RealESRGAN_x4plus"
  default_resolution: "1080p"
  chunk_batch_size: 4
  rife_enabled: true
  rife_target_fps: 60
```

---

## 🧪 Verification & Automated Test Suite

Run the full automated test suite covering unit tests, adversarial stress tests, and end-to-end integration:

```bash
# Run all tests
pytest tests/ -v

# Run specific module tests
pytest tests/test_memory_manager.py -v
pytest tests/test_character_engine.py -v
pytest tests/test_video_engine.py -v
pytest tests/test_lipsync_engine.py -v
pytest tests/test_post_processing.py -v
```

---

## 📂 Project Directory Structure

```
d:/Antigravity/AI Video Studio/
├── configs/
│   ├── colab_t4_config.yaml        # [M6] Centralized VRAM, precision & stage config
│   └── cinematic_styles.json       # [M2] 5 Cinematic style presets
├── modules/
│   ├── __init__.py                 # [M6] Core engine package exports
│   ├── memory_manager.py           # [M1] VRAMManager, stage decorators, flushing
│   ├── character_engine.py         # [M2] CharacterStudio, InstantID/FaceBank
│   ├── video_engine.py             # [M3] CineVideoEngine (Wan 2.1 + LTX + Mock)
│   ├── lipsync_engine.py           # [M4] LipSyncEngine (LivePortrait + Wav2Lip + Bengali audio)
│   └── post_processing.py          # [M5] PostProductionEngine (Real-ESRGAN, RIFE, Muxer)
├── character_profiles/             # [M2] Preconfigured Character Face Bank
│   ├── dev/                        # Dev character profile & ArcFace embeddings
│   ├── neel/                       # Neel character profile & ArcFace embeddings
│   ├── meghla/                     # Meghla character profile & ArcFace embeddings
│   └── cha_kaku/                   # Cha Kaku character profile & ArcFace embeddings
├── outputs/
│   ├── masters/                    # Final rendered MP4 master shots
│   └── temp/                       # Temporary stage buffers
├── tests/                          # Automated Verification Suite
│   ├── conftest.py
│   ├── test_memory_manager.py
│   ├── test_character_engine.py
│   ├── test_video_engine.py
│   ├── test_lipsync_engine.py
│   └── test_post_processing.py
├── app.py                          # [M6] Multi-Tab Gradio WebUI Application
├── CineFlow_Colab_FreeTier.ipynb   # [M6] Self-Contained Google Colab Free Tier Notebook
├── requirements.txt                # [M6] Pinned dependency definitions
└── README.md                       # [M6] Complete System Documentation
```

---

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
