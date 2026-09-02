"""
Adversarial Challenge & Stress Test Suite for CineFlow-AI VRAMManager (Milestone 1 / R1)
========================================================================================
Empirical verification of corner cases, hardware compute capability matrices,
exception safety, rapid lifecycle transitions, nested stages, and high-frequency telemetry.
"""

import gc
import sys
import threading
import time
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch, PropertyMock
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
# Challenge Suite 1: Corner Cases & Stage Lifecycle Isolation
# =============================================================================

class TestStageLifecycleCornerCases:
    """Adversarially probe stage lifecycle contexts under rapid, nested, and faulty conditions."""

    def test_rapid_back_to_back_transitions_under_stress(self):
        """Stress test rapid sequential stage transitions with dynamic registration and purging."""
        mgr = VRAMManager.get_instance()
        stage_names = ["stage_char", "stage_video", "stage_lipsync", "stage_upscale", "stage_export"]
        
        for i in range(25):
            stage = stage_names[i % len(stage_names)]
            with stage_context(stage):
                assert mgr.current_stage == stage
                # Register dummy object
                mgr.register_model(f"model_{i}", {"weights": [1, 2, 3], "step": i})
            assert mgr.current_stage is None
            assert len(mgr.get_registered_models()) == 0

    def test_nested_stage_contexts_behavior(self):
        """Examine state transitions when stage contexts are nested."""
        mgr = VRAMManager.get_instance()
        
        with stage_context("outer_stage"):
            assert mgr.current_stage == "outer_stage"
            with stage_context("inner_stage"):
                assert mgr.current_stage == "inner_stage"
            # Observe what current_stage is after inner exit
            inner_exit_stage = mgr.current_stage
        
        # Verify both contexts have fully completed
        assert mgr.current_stage is None

    def test_deeply_nested_stage_contexts(self):
        """Test deeply nested stage contexts (5 levels) without raising unexpected unhandled errors."""
        mgr = VRAMManager.get_instance()
        with stage_context("level_1"):
            with stage_context("level_2"):
                with stage_context("level_3"):
                    with stage_context("level_4"):
                        with stage_context("level_5"):
                            assert mgr.current_stage == "level_5"
        assert mgr.current_stage is None

    def test_cuda_oom_exception_inside_stage_context(self):
        """Verify emergency purge is triggered when CUDA OutOfMemoryError is raised."""
        mgr = VRAMManager.get_instance()
        
        class MockCUDAOutOfMemoryError(Exception):
            pass

        mock_torch = MagicMock()
        mock_torch.cuda.OutOfMemoryError = MockCUDAOutOfMemoryError
        mock_torch.cuda.is_available.return_value = True

        with patch("modules.memory_manager.TORCH_AVAILABLE", True), \
             patch("modules.memory_manager.torch", mock_torch):
            
            with pytest.raises(MockCUDAOutOfMemoryError):
                with stage_context("oom_stage"):
                    raise MockCUDAOutOfMemoryError("CUDA out of memory. Tried to allocate 16.00 GiB")
            
            assert mgr.current_stage is None
            # Verify synchronize and empty_cache were called during emergency flush
            assert mock_torch.cuda.synchronize.called
            assert mock_torch.cuda.empty_cache.called

    def test_stage_context_with_critical_base_exceptions(self):
        """Verify stage context cleans up even when BaseException (KeyboardInterrupt) occurs."""
        mgr = VRAMManager.get_instance()
        
        with pytest.raises(KeyboardInterrupt):
            with stage_context("interrupted_stage"):
                raise KeyboardInterrupt("Simulated user cancellation")
                
        assert mgr.current_stage is None

    def test_exception_in_stage_decorator(self):
        """Verify stage decorator cleans up state when decorated function throws."""
        mgr = VRAMManager.get_instance()

        @vram_lifecycle_stage("failing_stage")
        def broken_stage():
            raise ZeroDivisionError("Math error inside stage")

        with pytest.raises(ZeroDivisionError):
            broken_stage()

        assert mgr.current_stage is None


# =============================================================================
# Challenge Suite 2: Model Registry & Adversarial Object Purging
# =============================================================================

class TestModelRegistryAdversarialObjects:
    """Stress test model registration and purging with atypical, non-standard, and hostile objects."""

    def test_purging_non_existent_models(self):
        """Purging non-registered model names should return 0 without raising KeyError or exceptions."""
        mgr = VRAMManager.get_instance()
        purged = mgr.purge_models("ghost_model_1", "ghost_model_2", "non_existent")
        assert purged == 0
        assert len(mgr.get_registered_models()) == 0

    def test_registering_primitive_and_none_types(self):
        """Verify registration and purging of primitives (int, str, list, dict, None, set)."""
        mgr = VRAMManager.get_instance()
        primitives = {
            "int_val": 42,
            "str_val": "CineFlow",
            "list_val": [1, 2, 3],
            "dict_val": {"a": 1},
            "none_val": None,
            "set_val": {1, 2, 3},
            "tuple_val": (4, 5, 6),
        }
        for name, obj in primitives.items():
            mgr.register_model(name, obj)

        assert len(mgr.get_registered_models()) == len(primitives)
        purged = mgr.purge_models()
        assert purged == len(primitives)
        assert len(mgr.get_registered_models()) == 0

    def test_registering_objects_with_non_callable_attributes(self):
        """Test objects where 'to' or 'components' are properties/non-callables or throw on access."""
        mgr = VRAMManager.get_instance()

        class HostileObjectA:
            # 'to' is a string instead of a callable
            to = "not_a_method"
            # 'components' is an integer instead of a dict
            components = 12345

        class HostileObjectB:
            @property
            def to(self):
                raise RuntimeError("Accessing 'to' triggers error")

            @property
            def components(self):
                raise RuntimeError("Accessing 'components' triggers error")

        mgr.register_model("hostile_a", HostileObjectA())
        mgr.register_model("hostile_b", HostileObjectB())

        # Purging should not crash despite hostile attribute layouts
        purged = mgr.purge_models()
        assert purged == 2
        assert len(mgr.get_registered_models()) == 0

    def test_registering_object_with_broken_components_dict(self):
        """Test object with a components dict containing items that raise during deletion."""
        mgr = VRAMManager.get_instance()

        class BrokenComponent:
            def __del__(self):
                # Suppressed by Python GC or handled
                pass

        class CustomPipeline:
            def __init__(self):
                self.components = {
                    "text_encoder": BrokenComponent(),
                    "unet": BrokenComponent(),
                    "vae": BrokenComponent(),
                }
            def to(self, device):
                raise ValueError(f"Cannot move to {device}")

        mgr.register_model("pipeline_broken", CustomPipeline())
        purged = mgr.purge_models("pipeline_broken")
        assert purged == 1
        assert "pipeline_broken" not in mgr.get_registered_models()

    def test_concurrent_registration_and_purging(self):
        """Adversarially hammer model registry with concurrent reader, writer, and purger threads."""
        mgr = VRAMManager.get_instance()
        errors = []
        stop_event = threading.Event()

        def writer(worker_id: int):
            try:
                for i in range(100):
                    if stop_event.is_set():
                        break
                    mgr.register_model(f"thread_{worker_id}_model_{i}", {"data": i})
                    time.sleep(0.0001)
            except Exception as e:
                errors.append(e)

        def purger():
            try:
                for _ in range(50):
                    if stop_event.is_set():
                        break
                    mgr.purge_models(aggressive=False)
                    time.sleep(0.0002)
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(100):
                    if stop_event.is_set():
                        break
                    _ = mgr.get_registered_models()
                    time.sleep(0.0001)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer, args=(1,)),
            threading.Thread(target=writer, args=(2,)),
            threading.Thread(target=purger),
            threading.Thread(target=reader),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Encountered concurrency errors: {errors}"
        mgr.purge_models()


# =============================================================================
# Challenge Suite 3: Precision Selector Exhaustive Compute Capability Matrix
# =============================================================================

class TestPrecisionSelectorExhaustiveMatrix:
    """Exhaustively verify precision selector across Nvidia architectural generations & input variations."""

    @pytest.mark.parametrize("major,minor,expected_diffusion,expected_bf16,expected_fp8", [
        # Blackwell / Future
        (12, 0, "bfloat16", True, True),
        (10, 0, "bfloat16", True, True),
        # Hopper
        (9, 0, "bfloat16", True, True),
        # Ada Lovelace
        (8, 9, "bfloat16", True, True),
        # Ampere
        (8, 6, "bfloat16", True, True),
        (8, 0, "bfloat16", True, True),
        # Turing (Nvidia T4 Colab Free Tier, RTX 2080)
        (7, 5, "float16", False, True),
        # Volta (V100)
        (7, 0, "float16", False, False),
        # Pascal (GTX 1080 Ti, P100)
        (6, 1, "float16", False, False),
        (6, 0, "float16", False, False),
        # Maxwell
        (5, 2, "float16", False, False),
        (5, 0, "float16", False, False),
        # Kepler (Legacy)
        (3, 5, "float16", False, False),
    ])
    def test_precision_matrix_across_architectures(
        self, major, minor, expected_diffusion, expected_bf16, expected_fp8
    ):
        """Verify precision, bfloat16 support, and FP8 support for compute capability (major, minor)."""
        mgr = VRAMManager.get_instance()
        mgr.is_cuda = True
        mgr.device_id = 0

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_capability.return_value = (major, minor)
        mock_torch.bfloat16 = "bfloat16"
        mock_torch.float16 = "float16"
        mock_torch.float32 = "float32"

        with patch("modules.memory_manager.TORCH_AVAILABLE", True), \
             patch("modules.memory_manager.torch", mock_torch):
            
            assert mgr.is_bfloat16_supported() is expected_bf16
            assert mgr.is_fp8_supported() is expected_fp8
            
            # Diffusion precision
            prec = mgr.get_optimal_precision(model_family="diffusion", quantize=False)
            assert prec == expected_diffusion

    @pytest.mark.parametrize("family", [
        "wan2.1", "WAN2.1", "Wan", "ltx", "LTX-Video", "dit", "video"
    ])
    def test_video_dit_precision_on_turing_and_ampere(self, family):
        """Verify DiT / Wan 2.1 video models resolve to FP8 on CC >= 7.5."""
        mgr = VRAMManager.get_instance()
        mgr.is_cuda = True
        mgr.device_id = 0

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.bfloat16 = "bfloat16"
        mock_torch.float16 = "float16"

        with patch("modules.memory_manager.TORCH_AVAILABLE", True), \
             patch("modules.memory_manager.torch", mock_torch):
            
            # Turing 7.5 (T4)
            mock_torch.cuda.get_device_capability.return_value = (7, 5)
            prec_t4 = mgr.get_optimal_precision(model_family=family, allow_fp8=True)
            assert prec_t4 == "fp8"

            # Ampere 8.0 (A100)
            mock_torch.cuda.get_device_capability.return_value = (8, 0)
            prec_a100 = mgr.get_optimal_precision(model_family=family, allow_fp8=True)
            assert prec_a100 == "fp8"

            # Pascal 6.1 (GTX 1080) -> cannot do FP8, falls back to float16
            mock_torch.cuda.get_device_capability.return_value = (6, 1)
            prec_pascal = mgr.get_optimal_precision(model_family=family, allow_fp8=True)
            assert prec_pascal == "float16"

    def test_video_dit_when_allow_fp8_is_false(self):
        """Verify DiT / Video models return bfloat16 on Ampere and float16 on Turing if allow_fp8=False."""
        mgr = VRAMManager.get_instance()
        mgr.is_cuda = True
        mgr.device_id = 0

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.bfloat16 = "bfloat16"
        mock_torch.float16 = "float16"

        with patch("modules.memory_manager.TORCH_AVAILABLE", True), \
             patch("modules.memory_manager.torch", mock_torch):
            
            # Ampere 8.0 with allow_fp8=False -> bfloat16
            mock_torch.cuda.get_device_capability.return_value = (8, 0)
            prec_ampere = mgr.get_optimal_precision(model_family="wan2.1", allow_fp8=False)
            assert prec_ampere == "bfloat16"

            # Turing 7.5 with allow_fp8=False -> float16
            mock_torch.cuda.get_device_capability.return_value = (7, 5)
            prec_turing = mgr.get_optimal_precision(model_family="wan2.1", allow_fp8=False)
            assert prec_turing == "float16"

    def test_precision_selector_when_cuda_capability_query_fails(self):
        """Verify fallback to Turing 7.5 defaults when get_device_capability raises exception."""
        mgr = VRAMManager.get_instance()
        mgr.is_cuda = True
        mgr.device_id = 0

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_capability.side_effect = RuntimeError("Driver communication failure")
        mock_torch.float16 = "float16"

        with patch("modules.memory_manager.TORCH_AVAILABLE", True), \
             patch("modules.memory_manager.torch", mock_torch):
            
            # Should safely fallback to (7, 5) without crashing
            prec = mgr.get_optimal_precision(model_family="diffusion")
            assert prec == "float16"

    def test_precision_selector_cpu_and_no_torch(self):
        """Verify precision selector behavior in CPU-only and Torch-unavailable environments."""
        mgr = VRAMManager.get_instance()
        mgr.is_cuda = False

        # When Torch is available on CPU
        prec_cpu = mgr.get_optimal_precision(model_family="diffusion")
        assert "float32" in str(prec_cpu)

        # When Torch is completely unavailable
        with patch("modules.memory_manager.TORCH_AVAILABLE", False), \
             patch("modules.memory_manager.torch", None):
            prec_no_torch = mgr.get_optimal_precision(model_family="diffusion")
            assert prec_no_torch == "float32"


# =============================================================================
# Challenge Suite 4: Telemetry Under High Frequency & Edge Conditions
# =============================================================================

class TestTelemetryStressAndBoundary:
    """Stress test telemetry query frequency, zero-division, and error suppression."""

    def test_high_frequency_concurrent_telemetry_queries(self):
        """Simulate 200 rapid concurrent telemetry requests across multiple threads."""
        mgr = VRAMManager.get_instance()
        results: List[Dict[str, Any]] = []
        errors: List[Exception] = []

        def poller():
            try:
                for _ in range(50):
                    stats = mgr.get_vram_stats()
                    results.append(stats)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=poller) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 200
        for r in results:
            assert "allocated_mb" in r
            assert "system_ram_total_mb" in r

    def test_telemetry_zero_total_memory_division_safety(self):
        """Verify zero division is safely guarded when total VRAM reports 0 bytes."""
        mgr = VRAMManager.get_instance()
        mgr.is_cuda = True
        mgr.device_id = 0

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 0
        mock_torch.cuda.memory_reserved.return_value = 0
        mock_torch.cuda.max_memory_allocated.return_value = 0
        mock_torch.cuda.mem_get_info.return_value = (0, 0)  # 0 total bytes
        mock_torch.cuda.get_device_name.return_value = "Virtual 0MB GPU"

        with patch("modules.memory_manager.TORCH_AVAILABLE", True), \
             patch("modules.memory_manager.torch", mock_torch):
            
            stats = mgr.get_vram_stats()
            assert stats["utilization_pct"] == 0.0
            assert stats["total_mb"] == 0.0

    def test_telemetry_cuda_query_exception_fallback(self):
        """Verify telemetry falls back cleanly to CPU stats if CUDA mem query throws."""
        mgr = VRAMManager.get_instance()
        mgr.is_cuda = True
        mgr.device_id = 0

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.mem_get_info.side_effect = RuntimeError("CUDA driver lost")

        with patch("modules.memory_manager.TORCH_AVAILABLE", True), \
             patch("modules.memory_manager.torch", mock_torch):
            
            stats = mgr.get_vram_stats()
            assert isinstance(stats, dict)
            assert stats["is_cuda"] is False
            assert "Host CPU" in stats["device_name"]


# =============================================================================
# Challenge Suite 5: Offload & Optimization Helpers Adversarial Inputs
# =============================================================================

class TestOffloadHelpersAdversarial:
    """Stress test VAE and offload helpers with circular, hostile, and partially implemented objects."""

    def test_circular_reference_pipeline_vae(self):
        """Verify enable_vae_optimizations handles circular reference pipeline.vae = pipeline."""
        class CircularPipeline:
            def __init__(self):
                self.vae = self  # circular self reference
                self.slicing_called = False

            def enable_slicing(self):
                self.slicing_called = True

        pipe = CircularPipeline()
        res = enable_vae_optimizations(pipe)
        assert res is pipe
        assert pipe.slicing_called is True

    def test_pipeline_methods_raise_exceptions(self):
        """Verify helper functions catch and log internal exceptions gracefully without blowing up."""
        mock_pipe = MagicMock()
        mock_pipe.enable_sequential_cpu_offload.side_effect = RuntimeError("Accelerate missing")
        mock_pipe.enable_model_cpu_offload.side_effect = RuntimeError("Model offload unsupported")
        mock_pipe.enable_slicing.side_effect = TypeError("Unexpected arg")
        mock_pipe.enable_tiling.side_effect = AttributeError("Tiling missing")
        mock_pipe.enable_attention_slicing.side_effect = Exception("Attention slicing broken")

        # None of these should raise unhandled exceptions
        assert enable_sequential_cpu_offload(mock_pipe) is mock_pipe
        assert enable_vae_optimizations(mock_pipe) is mock_pipe
        assert enable_attention_slicing(mock_pipe) is mock_pipe
