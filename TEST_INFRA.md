# E2E Test Infra: CineFlow-AI

## Test Philosophy
- Opaque-box, requirement-driven testing covering CPU local development, CI environments, and GPU Colab runs.
- Deterministic procedural emulation (mock fallback) ensures 100% test pass rate even without CUDA or gigabytes of remote model weights downloaded.
- Systematic 4-tier testing hierarchy (Tier 1: Feature Isolation, Tier 2: Boundary & Corner Cases, Tier 3: Cross-Module Interactions, Tier 4: Real-World Multi-Shot Cine Workloads).

---

## Feature Inventory & Test Coverage Matrix
| # | Feature | Requirement | Tier 1 (Feature) | Tier 2 (Boundary) | Tier 3 (Cross-Module) | Tier 4 (E2E Scenario) |
|---|---------|-------------|:----------------:|:-----------------:|:---------------------:|:---------------------:|
| 1 | VRAM Tracking & Telemetry | ORIGINAL_REQUEST §15 | 5 | 5 | ✓ | ✓ |
| 2 | 4-Step Memory Flushing | ORIGINAL_REQUEST §16 | 5 | 5 | ✓ | ✓ |
| 3 | Auto-Precision Selection | ORIGINAL_REQUEST §17 | 5 | 5 | ✓ | ✓ |
| 4 | Stage Lifecycle Decorator | ORIGINAL_REQUEST §18 | 5 | 5 | ✓ | ✓ |
| 5 | CPU Offloading & VAE Slicing | ORIGINAL_REQUEST §19 | 5 | 5 | ✓ | ✓ |
| 6 | Character Profile Loading | ORIGINAL_REQUEST §23 | 5 | 5 | ✓ | ✓ |
| 7 | Dynamic Face Enrollment | ORIGINAL_REQUEST §24 | 5 | 5 | ✓ | ✓ |
| 8 | Cinematic Style Presets | ORIGINAL_REQUEST §25 | 5 | 5 | ✓ | ✓ |
| 9 | Prompt Synthesis Engine | ORIGINAL_REQUEST §25 | 5 | 5 | ✓ | ✓ |
| 10 | Wan 2.1 / LTX Video Engine | ORIGINAL_REQUEST §29-30 | 5 | 5 | ✓ | ✓ |
| 11 | Procedural Video Mock | ORIGINAL_REQUEST §32 | 5 | 5 | ✓ | ✓ |
| 12 | Bengali Audio Resampling & Mels | ORIGINAL_REQUEST §38 | 5 | 5 | ✓ | ✓ |
| 13 | LivePortrait / Wav2Lip LipSync | ORIGINAL_REQUEST §36-37 | 5 | 5 | ✓ | ✓ |
| 14 | Real-ESRGAN Chunked Upscaling | ORIGINAL_REQUEST §42 | 5 | 5 | ✓ | ✓ |
| 15 | RIFE 24->60fps Interpolation | ORIGINAL_REQUEST §43 | 5 | 5 | ✓ | ✓ |
| 16 | Audio/Video MP4 Muxer | ORIGINAL_REQUEST §44 | 5 | 5 | ✓ | ✓ |
| 17 | Config & Gradio WebUI Loading | ORIGINAL_REQUEST §47-51 | 5 | 5 | ✓ | ✓ |

---

## Test Architecture & Suite Structure
- **Runner**: `pytest tests/ -v`
- **Configuration**: `tests/conftest.py` with shared fixtures, synthetic audio generators, and mock frame generators.
- **Directory Layout**:
  - `tests/test_memory_manager.py`: Unit and stress tests for `VRAMManager`, flushing, telemetry, decorators.
  - `tests/test_character_engine.py`: Tests for `CharacterStudio`, face profiles (Dev, Neel, Meghla, Cha Kaku), face enrollment, embedding math, style prompts.
  - `tests/test_video_engine.py`: Tests for `CineVideoEngine`, backends (Wan 2.1, LTX, mock), frame synthesis, shape verification (81 frames).
  - `tests/test_lipsync_engine.py`: Tests for `LipSyncEngine`, 16kHz audio processing, phoneme alignment, audio-driven mouth deformation.
  - `tests/test_post_processing.py`: Tests for `PostProductionEngine`, chunked upscaling (720p->1080p), RIFE interpolation (24->60fps), FFmpeg/MoviePy muxing.
  - `tests/test_pipeline_e2e.py`: Complete end-to-end multi-shot pipeline integration testing.

---

## Real-World Application Scenarios (Tier 4)
| # | Scenario | Features Exercised | Expected Outcome |
|---|----------|--------------------|------------------|
| 1 | Full Studio Pipeline Shot: Dev in Kolkata Vintage 35mm with Bengali dialogue | F1-F5, F6, F8-F10, F12-F16 | Identity-consistent character generated, 81 frames motion synthesized, Bengali lip-sync aligned, 1080p 60fps MP4 master muxed |
| 2 | Custom Character Enrollment & Cyberpunk Render: Multi-photo enrollment -> Noir generation | F1-F5, F7, F8-F11, F14-F16 | New character profile saved, L2-normalized embedding stored, stylized cyberpunk video rendered |
| 3 | Memory Stress Test: Rapid sequential multi-stage rendering with VRAM tracking | F1, F2, F4, F10, F13, F14 | Zero memory leaks, peak memory reclaimed after each stage decorator exit |
| 4 | Offline CI CPU Execution: End-to-end pipeline run without GPU or remote weights | F1-F17 | 100% test execution and generation pass without any runtime crashes |
| 5 | Fallback Robustness: LTX fallback when Wan unavailable, Wav2Lip fallback when LivePortrait fails | F10, F11, F13, F14, F16 | Graceful backend failover with log warnings and valid video master output |

---

## Coverage Thresholds
- Tier 1: ≥5 test cases per feature
- Tier 2: ≥5 boundary test cases per feature
- Tier 3: Comprehensive cross-module pairwise integration tests
- Tier 4: 5 realistic application end-to-end scenarios
- Expected Pass Rate: 100%
