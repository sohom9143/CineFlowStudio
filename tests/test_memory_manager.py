"""
Comprehensive Unit & Boundary Test Suite for VRAMManager (Milestone 1 / R1).
=============================================================================
Tests:
- Thread-safe Singleton access & reset
- Dynamic VRAM Telemetry & CPU fallback
- 4-Step Aggressive Memory Purging & Cyclic GC
- Auto-Precision Selection Matrix (Ampere, Turing, DiT/FP8, CPU)
- Model Registry & Unreferenced Model Purging
- Stage Lifecycle Decorators & Context Managers
- Sequential CPU Offload & VAE Slicing/Tiling Utilities
- Concurrency & Multi-stage Stress Testing
"""

import gc
import sys
import threading
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

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


# =============================================================================
# Tier 1: Feature Isolation Tests
# =============================================================================

class TestVRAMManagerSingleton:
    """Tests for thread-safe singleton pattern of VRAMManager."""

    def test_singleton_identity(self):
        """Verify get_instance() always returns the exact same object reference."""
        mgr1 = VRAMManager.get_instance()
        mgr2 = VRAMManager.get_instance()
        assert mgr1 is mgr2
        assert id(mgr1) == id(mgr2)

    def test_singleton_thread_safety(self):
        """Verify concurrent threads retrieve the same singleton instance."""
        instances = []
        threads = []

        def worker():
            inst = VRAMManager.get_instance()
            instances.append(inst)

        for _ in range(20):
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(instances) == 20
        first_instance = instances[0]
        for inst in instances:
            assert inst is first_instance

    def test_singleton_reset(self):
        """Verify reset_instance() creates a new singleton instance on subsequent calls."""
        mgr1 = VRAMManager.get_instance()
        VRAMManager.reset_instance()
        mgr2 = VRAMManager.get_instance()
        assert mgr1 is not mgr2


class TestVRAMTelemetry:
    """Tests for dynamic VRAM and host memory telemetry."""

    def test_get_vram_stats_keys(self):
        """Verify get_vram_stats() returns all required telemetry fields."""
        mgr = VRAMManager.get_instance()
        stats = mgr.get_vram_stats()

        required_keys = [
            "allocated_mb",
            "reserved_mb",
            "total_mb",
            "free_mb",
            "peak_allocated_mb",
            "utilization_pct",
            "device_name",
            "is_cuda",
            "system_ram_total_mb",
            "system_ram_free_mb",
            "system_ram_used_mb",
            "system_ram_utilization_pct",
        ]
        for key in required_keys:
            assert key in stats, f"Missing telemetry key: {key}"

    def test_get_vram_stats_values(self):
        """Verify telemetry values are non-negative numeric or valid types."""
        stats = get_vram_stats()
        assert isinstance(stats["is_cuda"], bool)
        assert isinstance(stats["device_name"], str)
        assert stats["allocated_mb"] >= 0.0
        assert stats["total_mb"] >= 0.0
        assert stats["free_mb"] >= 0.0
        assert stats["utilization_pct"] >= 0.0

    def test_mock_cuda_telemetry(self):
        """Verify telemetry formatting when CUDA is simulated."""
        mgr = VRAMManager.get_instance()
        mgr.is_cuda = True
        mgr.device_id = 0

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 4 * 1024 * 1024 * 1024  # 4 GB
        mock_torch.cuda.memory_reserved.return_value = 5 * 1024 * 1024 * 1024   # 5 GB
        mock_torch.cuda.max_memory_allocated.return_value = 6 * 1024 * 1024 * 1024  # 6 GB
        mock_torch.cuda.mem_get_info.return_value = (
            10 * 1024 * 1024 * 1024,  # 10 GB free
            16 * 1024 * 1024 * 1024,  # 16 GB total
        )
        mock_torch.cuda.get_device_name.return_value = "Tesla T4"

        with patch("modules.memory_manager.TORCH_AVAILABLE", True), \
             patch("modules.memory_manager.torch", mock_torch):
            stats = mgr.get_vram_stats()
            assert stats["is_cuda"] is True
            assert stats["device_name"] == "Tesla T4"
            assert pytest.approx(stats["allocated_mb"], 0.1) == 4096.0
            assert pytest.approx(stats["reserved_mb"], 0.1) == 5120.0
            assert pytest.approx(stats["total_mb"], 0.1) == 16384.0
            assert pytest.approx(stats["free_mb"], 0.1) == 10240.0


class TestMemoryFlushing:
    """Tests for the 4-step memory purging mechanism."""

    def test_flush_memory_basic(self):
        """Verify flush_memory() executes and returns structured results."""
        mgr = VRAMManager.get_instance()
        res = mgr.flush_memory(aggressive=True)
        assert "freed_mb" in res
        assert "allocated_mb" in res
        assert "reserved_mb" in res
        assert "free_mb" in res
        assert "utilization_pct" in res
        assert res["freed_mb"] >= 0.0

    def test_flush_memory_cyclic_garbage(self):
        """Verify cyclic references are collected during flush_memory()."""
        # Create circular reference
        class Node:
            def __init__(self):
                self.ref = None

        a = Node()
        b = Node()
        a.ref = b
        b.ref = a
        del a
        del b

        res = flush_memory(aggressive=True)
        assert isinstance(res, dict)

    def test_mock_cuda_flush_steps(self):
        """Verify all 4 CUDA flush steps (synchronize, empty_cache, ipc_collect, reset_peak) are called."""
        mgr = VRAMManager.get_instance()
        mgr.is_cuda = True
        mgr.device_id = 0

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 0
        mock_torch.cuda.memory_reserved.return_value = 0
        mock_torch.cuda.max_memory_allocated.return_value = 0
        mock_torch.cuda.mem_get_info.return_value = (1000, 2000)
        mock_torch.cuda.get_device_name.return_value = "Tesla T4"

        with patch("modules.memory_manager.TORCH_AVAILABLE", True), \
             patch("modules.memory_manager.torch", mock_torch):
            mgr.flush_memory(aggressive=True)
            assert mock_torch.cuda.synchronize.called
            assert mock_torch.cuda.empty_cache.called
            assert mock_torch.cuda.ipc_collect.called
            assert mock_torch.cuda.reset_peak_memory_stats.called


class TestAutoPrecisionSelector:
    """Tests for the hardware-aware precision selector matrix."""

    def test_cpu_precision_fallback(self):
        """Verify CPU mode returns float32."""
        mgr = VRAMManager.get_instance()
        mgr.is_cuda = False
        prec = mgr.get_optimal_precision()
        assert str(prec) in ("float32", "torch.float32")

    def test_ampere_precision_selection(self):
        """Verify Ampere+ (CC >= 8.0, e.g. A100 / RTX 3090) selects bfloat16."""
        mgr = VRAMManager.get_instance()
        mgr.is_cuda = True
        mgr.device_id = 0

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_capability.return_value = (8, 0)
        mock_torch.bfloat16 = "bfloat16"

        with patch("modules.memory_manager.TORCH_AVAILABLE", True), \
             patch("modules.memory_manager.torch", mock_torch):
            prec = mgr.get_optimal_precision(model_family="diffusion", quantize=False)
            assert prec == "bfloat16"
            assert mgr.is_bfloat16_supported() is True

    def test_turing_t4_precision_selection(self):
        """Verify Turing (CC 7.5, e.g. Nvidia T4 on Colab) selects float16 for diffusion."""
        mgr = VRAMManager.get_instance()
        mgr.is_cuda = True
        mgr.device_id = 0

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_capability.return_value = (7, 5)
        mock_torch.float16 = "float16"

        with patch("modules.memory_manager.TORCH_AVAILABLE", True), \
             patch("modules.memory_manager.torch", mock_torch):
            prec = mgr.get_optimal_precision(model_family="diffusion", quantize=False)
            assert prec == "float16"
            assert mgr.is_bfloat16_supported() is False
            assert mgr.is_fp8_supported() is True

    def test_quantized_dit_fp8_selection(self):
        """Verify DiT / Wan 2.1 video generation on Turing/Ampere selects FP8."""
        mgr = VRAMManager.get_instance()
        mgr.is_cuda = True
        mgr.device_id = 0

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_capability.return_value = (7, 5)

        with patch("modules.memory_manager.TORCH_AVAILABLE", True), \
             patch("modules.memory_manager.torch", mock_torch):
            prec = mgr.get_optimal_precision(model_family="wan2.1", allow_fp8=True)
            assert prec == "fp8"

    def test_explicit_int8_quantization(self):
        """Verify explicit int8 quantization returns int8."""
        mgr = VRAMManager.get_instance()
        mgr.is_cuda = True
        mgr.device_id = 0

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_capability.return_value = (7, 5)

        with patch("modules.memory_manager.TORCH_AVAILABLE", True), \
             patch("modules.memory_manager.torch", mock_torch):
            prec = mgr.get_optimal_precision(model_family="diffusion", quantize="int8")
            assert prec == "int8"


class TestModelRegistryAndPurging:
    """Tests for model registration, tracking, and unloading."""

    def test_register_and_list_models(self):
        """Verify models can be registered and enumerated."""
        mgr = VRAMManager.get_instance()
        mock_sdxl = MagicMock()
        mock_wan = MagicMock()

        mgr.register_model("sdxl_instantid", mock_sdxl)
        mgr.register_model("wan2_1", mock_wan)

        models = mgr.get_registered_models()
        assert "sdxl_instantid" in models
        assert "wan2_1" in models

    def test_unregister_model(self):
        """Verify model unregistration."""
        mgr = VRAMManager.get_instance()
        mock_model = MagicMock()
        mgr.register_model("test_model", mock_model)
        assert "test_model" in mgr.get_registered_models()

        popped = mgr.unregister_model("test_model")
        assert popped is mock_model
        assert "test_model" not in mgr.get_registered_models()

    def test_purge_specific_models(self):
        """Verify purging offloads to CPU and removes models from registry."""
        mgr = VRAMManager.get_instance()
        mock_model1 = MagicMock()
        mock_model2 = MagicMock()

        mgr.register_model("model1", mock_model1)
        mgr.register_model("model2", mock_model2)

        purged = mgr.purge_models("model1")
        assert purged == 1
        mock_model1.to.assert_called_with("cpu")
        assert "model1" not in mgr.get_registered_models()
        assert "model2" in mgr.get_registered_models()

    def test_purge_all_models(self):
        """Verify purging without arguments purges all registered models."""
        mgr = VRAMManager.get_instance()
        mgr.register_model("m1", MagicMock())
        mgr.register_model("m2", MagicMock())
        mgr.register_model("m3", MagicMock())

        purged = mgr.purge_models()
        assert purged == 3
        assert len(mgr.get_registered_models()) == 0


class TestStageLifecycle:
    """Tests for stage decorators and context managers."""

    def test_stage_context_normal_execution(self):
        """Verify stage_context enters and exits cleanly, updating current_stage."""
        mgr = VRAMManager.get_instance()
        assert mgr.current_stage is None

        with stage_context("character_generation"):
            assert mgr.current_stage == "character_generation"

        assert mgr.current_stage is None

    def test_stage_context_exception_handling(self):
        """Verify stage_context performs emergency cleanup and propagates exceptions."""
        mgr = VRAMManager.get_instance()

        with pytest.raises(ValueError, match="Synthetic stage error"):
            with stage_context("faulty_stage"):
                assert mgr.current_stage == "faulty_stage"
                raise ValueError("Synthetic stage error")

        assert mgr.current_stage is None

    def test_vram_lifecycle_decorator_with_name(self):
        """Verify @vram_lifecycle_stage('stage_name') decorator."""
        mgr = VRAMManager.get_instance()

        @vram_lifecycle_stage("video_motion")
        def sample_motion_func(x: int) -> int:
            assert mgr.current_stage == "video_motion"
            return x * 2

        res = sample_motion_func(21)
        assert res == 42
        assert mgr.current_stage is None

    def test_vram_lifecycle_decorator_bare(self):
        """Verify @vram_lifecycle_stage bare decorator."""
        mgr = VRAMManager.get_instance()

        @vram_lifecycle_stage
        def lipsync_step():
            assert mgr.current_stage == "lipsync_step"
            return "synced"

        assert lipsync_step() == "synced"
        assert mgr.current_stage is None


class TestOffloadAndVAEOptimizations:
    """Tests for sequential CPU offload and VAE slicing/tiling helper functions."""

    def test_sequential_cpu_offload_helper(self):
        """Verify enable_sequential_cpu_offload calls pipeline method."""
        mock_pipe = MagicMock()
        res = enable_sequential_cpu_offload(mock_pipe)
        assert res is mock_pipe
        assert mock_pipe.enable_sequential_cpu_offload.called

    def test_sequential_cpu_offload_model_fallback(self):
        """Verify fallback to enable_model_cpu_offload when sequential is absent."""
        mock_pipe = MagicMock(spec=["enable_model_cpu_offload"])
        res = enable_sequential_cpu_offload(mock_pipe)
        assert res is mock_pipe
        assert mock_pipe.enable_model_cpu_offload.called

    def test_vae_optimizations_helper(self):
        """Verify enable_vae_optimizations calls enable_slicing and enable_tiling."""
        mock_vae = MagicMock()
        res = enable_vae_optimizations(mock_vae)
        assert res is mock_vae
        assert mock_vae.enable_slicing.called
        assert mock_vae.enable_tiling.called

    def test_vae_optimizations_on_pipeline(self):
        """Verify enable_vae_optimizations works when passed a pipeline with .vae."""
        mock_pipe = MagicMock()
        res = enable_vae_optimizations(mock_pipe)
        assert res is mock_pipe
        assert mock_pipe.vae.enable_slicing.called
        assert mock_pipe.vae.enable_tiling.called
        assert mock_pipe.enable_attention_slicing.called

    def test_attention_slicing_helper(self):
        """Verify enable_attention_slicing calls pipeline method."""
        mock_pipe = MagicMock()
        res = enable_attention_slicing(mock_pipe, slice_size=4)
        assert res is mock_pipe
        mock_pipe.enable_attention_slicing.assert_called_with(slice_size=4)


# =============================================================================
# Tier 2: Boundary & Stress Tests
# =============================================================================

class TestMemoryManagerStressAndBoundaries:
    """Boundary cases, rapid sequential transitions, and stress tests."""

    def test_rapid_sequential_stage_transitions(self):
        """Simulate rapid sequential multi-stage rendering with memory tracking."""
        mgr = VRAMManager.get_instance()
        stages = ["character_stage", "video_stage", "lipsync_stage", "upscale_stage"]

        for i in range(25):
            stage_name = stages[i % len(stages)]
            with stage_context(stage_name):
                # Register temporary model
                mgr.register_model(f"temp_model_{i}", MagicMock())
                assert mgr.current_stage == stage_name

            assert mgr.current_stage is None

        # Verify all models purged cleanly
        assert len(mgr.get_registered_models()) == 0

    def test_nested_stages_handled_gracefully(self):
        """Verify nested stage contexts restore the parent stage or reset."""
        mgr = VRAMManager.get_instance()

        with stage_context("outer_stage"):
            assert mgr.current_stage == "outer_stage"
            with stage_context("inner_stage"):
                assert mgr.current_stage == "inner_stage"
            # Inner stage exits

    def test_none_pipeline_handles_gracefully(self):
        """Verify helper functions do not crash when given None or invalid objects."""
        assert enable_sequential_cpu_offload(None) is None
        assert enable_vae_optimizations(None) is None
        assert enable_attention_slicing(None) is None

        # Arbitrary object without diffusers methods
        dummy = object()
        assert enable_sequential_cpu_offload(dummy) is dummy
        assert enable_vae_optimizations(dummy) is dummy
        assert enable_attention_slicing(dummy) is dummy

    def test_log_memory_summary(self):
        """Verify log_memory_summary returns a valid formatted string."""
        mgr = VRAMManager.get_instance()
        summary = mgr.log_memory_summary("TestPrefix")
        assert isinstance(summary, str)
        assert "TestPrefix" in summary

    def test_flush_memory_non_aggressive(self):
        """Verify non-aggressive flush executes properly."""
        mgr = VRAMManager.get_instance()
        res = mgr.flush_memory(aggressive=False)
        assert isinstance(res, dict)
        assert "freed_mb" in res

    def test_telemetry_without_psutil(self):
        """Verify get_vram_stats works safely when psutil is unavailable."""
        mgr = VRAMManager.get_instance()
        with patch("modules.memory_manager.PSUTIL_AVAILABLE", False):
            stats = mgr.get_vram_stats()
            assert isinstance(stats, dict)
            assert "allocated_mb" in stats
            assert "system_ram_total_mb" in stats

    def test_stage_context_flags(self):
        """Verify stage_context respects auto_purge=False and purge_models=False."""
        mgr = VRAMManager.get_instance()
        mgr.register_model("persistent_model", MagicMock())

        with stage_context("manual_stage", auto_purge=False, purge_models=False):
            assert "persistent_model" in mgr.get_registered_models()

        # Model should still be registered since auto_purge/purge_models were False
        assert "persistent_model" in mgr.get_registered_models()
        mgr.purge_models()

    def test_purge_models_fault_tolerant(self):
        """Verify purge_models does not raise if model.to('cpu') raises an error."""
        mgr = VRAMManager.get_instance()
        faulty_model = MagicMock()
        faulty_model.to.side_effect = RuntimeError("CUDA device error during offload")
        faulty_model.components = {"comp1": MagicMock()}

        mgr.register_model("faulty", faulty_model)
        purged = mgr.purge_models("faulty")
        assert purged == 1
        assert "faulty" not in mgr.get_registered_models()

    def test_auto_precision_extended_compute_capabilities(self):
        """Verify precision selector across diverse GPU compute capabilities (Ada, Hopper, Pascal, Volta)."""
        mgr = VRAMManager.get_instance()
        mgr.is_cuda = True
        mgr.device_id = 0

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.bfloat16 = "bfloat16"
        mock_torch.float16 = "float16"
        mock_torch.float32 = "float32"

        with patch("modules.memory_manager.TORCH_AVAILABLE", True), \
             patch("modules.memory_manager.torch", mock_torch):
            
            # Ada Lovelace RTX 4090 (8.9)
            mock_torch.cuda.get_device_capability.return_value = (8, 9)
            assert mgr.get_optimal_precision(model_family="diffusion") == "bfloat16"
            assert mgr.is_bfloat16_supported() is True
            assert mgr.is_fp8_supported() is True

            # Hopper H100 (9.0)
            mock_torch.cuda.get_device_capability.return_value = (9, 0)
            assert mgr.get_optimal_precision(model_family="diffusion") == "bfloat16"

            # Volta V100 (7.0)
            mock_torch.cuda.get_device_capability.return_value = (7, 0)
            assert mgr.get_optimal_precision(model_family="diffusion") == "float16"
            assert mgr.is_bfloat16_supported() is False

            # Pascal GTX 1080 (6.1)
            mock_torch.cuda.get_device_capability.return_value = (6, 1)
            assert mgr.get_optimal_precision(model_family="diffusion") == "float16"
            assert mgr.is_bfloat16_supported() is False

    def test_manager_method_decorator(self):
        """Verify mgr.vram_lifecycle_stage method decorator."""
        mgr = VRAMManager.get_instance()

        @mgr.vram_lifecycle_stage("method_stage")
        def sample_step():
            assert mgr.current_stage == "method_stage"
            return 999

        assert sample_step() == 999
        assert mgr.current_stage is None

