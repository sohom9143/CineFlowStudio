"""
CineFlow-AI: Sequential VRAM & Memory Lifecycle Engine (Milestone 1 / R1)
========================================================================
Centralized memory management, aggressive stage flushing, hardware telemetry,
auto-precision selection, stage lifecycle isolation, and sequential CPU offloading
optimized for Google Colab Free Tier (Nvidia T4 15-16GB VRAM / 12.7GB System RAM)
and cross-platform local development (Windows/Linux/macOS).
"""

from __future__ import annotations

import gc
import sys
import time
import logging
import threading
import functools
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Optional PyTorch import with graceful fallback
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    TORCH_AVAILABLE = False

# Optional psutil import for host system RAM telemetry
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None  # type: ignore
    PSUTIL_AVAILABLE = False

# Setup dedicated module logger
logger = logging.getLogger("CineFlow.VRAMManager")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class VRAMStageContext:
    """
    Context manager for lifecycle-isolated execution of a pipeline stage.
    
    Guarantees:
    - Pre-stage memory flushing and telemetry capture.
    - Post-stage memory purging, garbage collection, and CUDA cache release.
    - Emergency memory purge and diagnostics on OutOfMemoryError or general exceptions.
    """

    def __init__(
        self,
        stage_name: str,
        manager: Optional["VRAMManager"] = None,
        auto_purge: bool = True,
        purge_models: bool = True,
    ) -> None:
        self.stage_name = stage_name
        self.manager = manager or VRAMManager.get_instance()
        self.auto_purge = auto_purge
        self.purge_models = purge_models
        self.start_stats: Dict[str, Union[float, str, bool]] = {}
        self.start_time: float = 0.0

    def __enter__(self) -> "VRAMStageContext":
        self.start_time = time.time()
        self.manager._current_stage = self.stage_name
        
        # Purge any unreferenced models registered in previous stages if requested
        if self.purge_models:
            self.manager.purge_models(aggressive=True)
        else:
            self.manager.flush_memory(aggressive=True)
            
        self.start_stats = self.manager.get_vram_stats()
        free_mb = float(self.start_stats.get("free_mb", 0.0))
        alloc_mb = float(self.start_stats.get("allocated_mb", 0.0))
        device_name = str(self.start_stats.get("device_name", "Unknown"))
        
        logger.info(
            f"===> [ENTER STAGE: {self.stage_name}] Device: {device_name} | "
            f"Allocated: {alloc_mb:.1f} MB | Free: {free_mb:.1f} MB"
        )
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> bool:
        duration = time.time() - self.start_time
        
        if exc_type is not None:
            # Check for PyTorch CUDA OOM
            is_oom = False
            if TORCH_AVAILABLE and issubclass(exc_type, getattr(torch.cuda, "OutOfMemoryError", Exception)):
                is_oom = True
                logger.error(
                    f"!!! CUDA OUT OF MEMORY in Stage [{self.stage_name}] after {duration:.2f}s: {exc_val}"
                )
            else:
                logger.error(
                    f"!!! Stage [{self.stage_name}] failed with {exc_type.__name__} after {duration:.2f}s: {exc_val}"
                )
            
            # Emergency purge and diagnostic flush
            emergency_stats = self.manager.flush_memory(aggressive=True)
            logger.warning(
                f"Emergency flush performed for [{self.stage_name}]. Reclaimed: {emergency_stats.get('freed_mb', 0.0):.1f} MB"
            )
            self.manager._current_stage = None
            # Re-raise exception
            return False

        # Normal clean exit
        if self.auto_purge:
            if self.purge_models:
                self.manager.purge_models(aggressive=True)
            else:
                self.manager.flush_memory(aggressive=True)
            
        end_stats = self.manager.get_vram_stats()
        free_mb = float(end_stats.get("free_mb", 0.0))
        alloc_mb = float(end_stats.get("allocated_mb", 0.0))
        
        logger.info(
            f"<=== [EXIT STAGE: {self.stage_name}] Duration: {duration:.2f}s | "
            f"Allocated: {alloc_mb:.1f} MB | Free: {free_mb:.1f} MB"
        )
        self.manager._current_stage = None
        return False


class VRAMManager:
    """
    Thread-safe Singleton VRAM & Memory Lifecycle Coordinator.
    
    Provides:
    - 4-step aggressive memory purging (GC, synchronize, empty_cache, ipc_collect).
    - Dynamic hardware telemetry for CUDA GPU and host system memory.
    - Architecture-aware auto-precision resolution (Ampere BF16, Turing FP16, DiT FP8/INT8, CPU FP32).
    - Stage lifecycle decorators and context managers.
    - Model registry and memory unloader utilities.
    - Sequential CPU offload and VAE slicing/tiling helpers.
    """

    _instance: Optional["VRAMManager"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self, device: Optional[Union[str, Any]] = None) -> None:
        # Check if already initialized (singleton safeguard)
        if getattr(self, "_initialized", False):
            return

        self._model_lock = threading.Lock()
        self._registered_models: Dict[str, Any] = {}
        self._current_stage: Optional[str] = None
        
        # Determine CUDA availability
        if TORCH_AVAILABLE and torch.cuda.is_available():
            self.is_cuda: bool = True
            if device is None:
                self.device_id: int = 0
                self.device: Any = torch.device(f"cuda:{self.device_id}")
            elif isinstance(device, str):
                self.device = torch.device(device)
                self.device_id = self.device.index if self.device.index is not None else 0
            else:
                self.device = device
                self.device_id = getattr(device, "index", 0) or 0
        else:
            self.is_cuda = False
            self.device_id = -1
            if TORCH_AVAILABLE:
                self.device = torch.device("cpu")
            else:
                self.device = "cpu"

        self._initialized = True
        logger.info(f"VRAMManager initialized. Device: {self.device} (CUDA: {self.is_cuda})")

    @classmethod
    def get_instance(cls, device: Optional[Union[str, Any]] = None) -> "VRAMManager":
        """
        Thread-safe singleton accessor with double-checked locking.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(device=device)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """
        Reset the singleton instance (primarily for testing and environment resets).
        """
        with cls._lock:
            if cls._instance is not None:
                cls._instance.flush_memory(aggressive=True)
                cls._instance = None

    @property
    def current_stage(self) -> Optional[str]:
        """Returns the currently active pipeline stage name, or None."""
        return self._current_stage

    # -------------------------------------------------------------------------
    # Telemetry & Stats
    # -------------------------------------------------------------------------

    def get_vram_stats(self) -> Dict[str, Union[float, str, bool]]:
        """
        Returns real-time memory telemetry for GPU VRAM and host RAM.
        
        Dict structure:
        - allocated_mb: PyTorch tensor allocated VRAM (or used system RAM on CPU).
        - reserved_mb: PyTorch caching allocator reserved VRAM.
        - total_mb: Total hardware physical VRAM (or total system RAM on CPU).
        - free_mb: Available free physical VRAM (or free system RAM on CPU).
        - peak_allocated_mb: Peak allocated VRAM since last reset.
        - utilization_pct: Percentage of memory utilization.
        - device_name: String descriptor of the GPU (or CPU model/system).
        - is_cuda: Boolean indicating whether CUDA GPU is active.
        - system_ram_total_mb: Total host RAM in MB.
        - system_ram_free_mb: Available host RAM in MB.
        - system_ram_used_mb: Used host RAM in MB.
        - system_ram_utilization_pct: Host RAM utilization percentage.
        """
        sys_ram_total = 0.0
        sys_ram_free = 0.0
        sys_ram_used = 0.0
        sys_ram_pct = 0.0

        if PSUTIL_AVAILABLE:
            try:
                vm = psutil.virtual_memory()
                sys_ram_total = float(vm.total) / (1024 ** 2)
                sys_ram_free = float(vm.available) / (1024 ** 2)
                sys_ram_used = float(vm.used) / (1024 ** 2)
                sys_ram_pct = float(vm.percent)
            except Exception as e:
                logger.debug(f"psutil telemetry error: {e}")

        if self.is_cuda and TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                dev = self.device_id if self.device_id >= 0 else 0
                allocated = torch.cuda.memory_allocated(dev) / (1024 ** 2)
                reserved = torch.cuda.memory_reserved(dev) / (1024 ** 2)
                peak_alloc = torch.cuda.max_memory_allocated(dev) / (1024 ** 2)
                
                free_bytes, total_bytes = torch.cuda.mem_get_info(dev)
                free_mb = free_bytes / (1024 ** 2)
                total_mb = total_bytes / (1024 ** 2)
                
                util_pct = (allocated / total_mb * 100.0) if total_mb > 0 else 0.0
                dev_name = torch.cuda.get_device_name(dev)

                return {
                    "allocated_mb": round(allocated, 2),
                    "reserved_mb": round(reserved, 2),
                    "total_mb": round(total_mb, 2),
                    "free_mb": round(free_mb, 2),
                    "peak_allocated_mb": round(peak_alloc, 2),
                    "utilization_pct": round(util_pct, 2),
                    "device_name": dev_name,
                    "is_cuda": True,
                    "system_ram_total_mb": round(sys_ram_total, 2),
                    "system_ram_free_mb": round(sys_ram_free, 2),
                    "system_ram_used_mb": round(sys_ram_used, 2),
                    "system_ram_utilization_pct": round(sys_ram_pct, 2),
                }
            except Exception as e:
                logger.warning(f"Failed to query CUDA memory stats: {e}")

        # CPU / Non-CUDA Fallback
        return {
            "allocated_mb": round(sys_ram_used, 2),
            "reserved_mb": round(sys_ram_used, 2),
            "total_mb": round(sys_ram_total, 2),
            "free_mb": round(sys_ram_free, 2),
            "peak_allocated_mb": round(sys_ram_used, 2),
            "utilization_pct": round(sys_ram_pct, 2),
            "device_name": "Host CPU (No CUDA)",
            "is_cuda": False,
            "system_ram_total_mb": round(sys_ram_total, 2),
            "system_ram_free_mb": round(sys_ram_free, 2),
            "system_ram_used_mb": round(sys_ram_used, 2),
            "system_ram_utilization_pct": round(sys_ram_pct, 2),
        }

    # -------------------------------------------------------------------------
    # 4-Step Aggressive Memory Purging
    # -------------------------------------------------------------------------

    def flush_memory(self, aggressive: bool = True) -> Dict[str, float]:
        """
        Executes an aggressive multi-step memory purge:
        
        Step 1: Python cyclic garbage collector across all generations (2 and 0/1).
        Step 2: CUDA device stream synchronization to finish pending asynchronous operations.
        Step 3: PyTorch CUDA allocator cache release (empty_cache).
        Step 4: (Aggressive) IPC cache collection and peak memory stat reset.
        
        Returns:
            Dict containing freed_mb, allocated_mb, reserved_mb, free_mb, utilization_pct.
        """
        stats_before = self.get_vram_stats()
        alloc_before = float(stats_before.get("allocated_mb", 0.0))

        # Step 1: Force cyclic garbage collection
        gc.collect(generation=2)
        gc.collect()

        if (self.is_cuda or (TORCH_AVAILABLE and torch.cuda.is_available())) and TORCH_AVAILABLE and torch.cuda.is_available():
            dev = self.device_id if self.device_id >= 0 else 0
            try:
                # Step 2: Stream synchronization
                torch.cuda.synchronize(dev)
                
                # Step 3: Cache release
                torch.cuda.empty_cache()
                
                if aggressive:
                    # Step 4: Inter-Process Communication cache collection
                    try:
                        torch.cuda.ipc_collect()
                    except Exception:
                        pass
                    
                    # Reset peak memory tracker
                    try:
                        torch.cuda.reset_peak_memory_stats(dev)
                    except Exception:
                        pass
            except Exception as e:
                logger.warning(f"CUDA flush encountered warning: {e}")

        # Run another small GC pass to clear allocator debris
        gc.collect()

        stats_after = self.get_vram_stats()
        alloc_after = float(stats_after.get("allocated_mb", 0.0))
        freed_mb = max(0.0, alloc_before - alloc_after)

        result = {
            "freed_mb": round(freed_mb, 2),
            "allocated_mb": float(stats_after.get("allocated_mb", 0.0)),
            "reserved_mb": float(stats_after.get("reserved_mb", 0.0)),
            "free_mb": float(stats_after.get("free_mb", 0.0)),
            "utilization_pct": float(stats_after.get("utilization_pct", 0.0)),
        }
        logger.debug(f"Memory flushed: freed {freed_mb:.1f} MB | Remaining allocated: {result['allocated_mb']:.1f} MB")
        return result

    # -------------------------------------------------------------------------
    # Auto-Precision Selector
    # -------------------------------------------------------------------------

    def is_bfloat16_supported(self) -> bool:
        """
        Checks if native bfloat16 hardware acceleration is supported (Ampere+ Compute Capability >= 8.0).
        """
        if not (self.is_cuda and TORCH_AVAILABLE and torch.cuda.is_available()):
            return False
        try:
            dev = self.device_id if self.device_id >= 0 else 0
            major, _ = torch.cuda.get_device_capability(dev)
            return major >= 8
        except Exception:
            return False

    def is_fp8_supported(self) -> bool:
        """
        Checks if FP8 quantization format is supported on the active CUDA device.
        """
        if not (self.is_cuda and TORCH_AVAILABLE and torch.cuda.is_available()):
            return False
        try:
            dev = self.device_id if self.device_id >= 0 else 0
            major, minor = torch.cuda.get_device_capability(dev)
            # Turing (7.5) and above support FP8 weights in quantized backends
            return (major > 7) or (major == 7 and minor >= 5)
        except Exception:
            return False

    def get_optimal_precision(
        self,
        model_family: str = "diffusion",
        quantize: Union[bool, str] = False,
        allow_fp8: bool = True,
    ) -> Any:
        """
        Auto-selects the optimal PyTorch data type or quantization mode based on GPU compute capability:
        
        - CPU / Non-CUDA: torch.float32 (or "float32").
        - Quantized Video / DiT (Wan 2.1, LTX-Video) with allow_fp8:
            - FP8 ("fp8" or torch.float8_e4m3fn) on Turing / Ampere / Ada.
            - INT8 ("int8") if explicitly requested.
        - Ampere / Ada / Hopper (CC >= 8.0, e.g. A100, RTX 3090/4090, H100):
            - torch.bfloat16 (or "bfloat16") for numerical stability and high speed.
        - Turing / Volta / Pascal (CC 7.5, e.g. Colab Nvidia T4):
            - torch.float16 (or "float16") for native tensor core throughput (avoiding slow emulated bfloat16).
        - Older GPUs / Unknown:
            - torch.float16 or torch.float32.
        """
        if not (self.is_cuda and TORCH_AVAILABLE and torch.cuda.is_available()):
            return torch.float32 if TORCH_AVAILABLE else "float32"

        dev = self.device_id if self.device_id >= 0 else 0
        try:
            major, minor = torch.cuda.get_device_capability(dev)
        except Exception:
            major, minor = (7, 5)  # Default fallback to T4 capability

        norm_family = (model_family or "").lower()

        # Check for explicit or model-family quantization
        if quantize is True or quantize == "fp8" or (norm_family in ("wan2.1", "wan", "ltx", "ltx-video", "dit", "video") and allow_fp8):
            if (major > 7) or (major == 7 and minor >= 5):
                dtype_cls = getattr(torch, "dtype", None)
                if dtype_cls is not None and isinstance(dtype_cls, type):
                    f8 = getattr(torch, "float8_e4m3fn", None)
                    if f8 is not None and isinstance(f8, dtype_cls):
                        return f8
                return "fp8"
            return "int8" if quantize == "int8" else (torch.float16 if TORCH_AVAILABLE else "float16")

        if quantize == "int8":
            return "int8"

        # Ampere (8.0), Ada Lovelace (8.9), Hopper (9.0), Blackwell (10.0+)
        if major >= 8:
            return torch.bfloat16 if TORCH_AVAILABLE else "bfloat16"
        
        # Turing (7.5, Nvidia T4 on Colab), Volta (7.0, V100), Pascal (6.1, GTX 1080)
        return torch.float16 if TORCH_AVAILABLE else "float16"

    # -------------------------------------------------------------------------
    # Model Registry & Purging
    # -------------------------------------------------------------------------

    def register_model(self, name: str, model: Any) -> None:
        """
        Registers an active neural model reference for lifecycle tracking.
        """
        with self._model_lock:
            self._registered_models[name] = model
            logger.debug(f"Model registered: '{name}'")

    def unregister_model(self, name: str) -> Optional[Any]:
        """
        Unregisters a model by name from tracking without immediate memory purge.
        """
        with self._model_lock:
            return self._registered_models.pop(name, None)

    def get_registered_models(self) -> List[str]:
        """
        Returns a list of currently registered model names.
        """
        with self._model_lock:
            return list(self._registered_models.keys())

    def purge_models(self, *names: str, aggressive: bool = True) -> int:
        """
        Explicitly detaches, deletes, and unloads specified models (or all registered models if none specified),
        then invokes aggressive memory flushing.
        
        Returns:
            Number of models successfully purged.
        """
        purged_count = 0
        with self._model_lock:
            target_keys = list(names) if names else list(self._registered_models.keys())
            for key in target_keys:
                if key in self._registered_models:
                    model = self._registered_models.pop(key)
                    purged_count += 1
                    if model is not None:
                        try:
                            # Attempt to move weights to CPU first to immediately free CUDA buffers
                            if hasattr(model, "to") and callable(model.to):
                                model.to("cpu")
                        except Exception:
                            pass
                    
                    try:
                        # Diffusers pipeline components deletion
                        if hasattr(model, "components") and isinstance(model.components, dict):
                            for comp_name in list(model.components.keys()):
                                comp = model.components.pop(comp_name, None)
                                del comp
                    except Exception:
                        pass
                    
                    del model
                    logger.info(f"Purged registered model: '{key}'")

        self.flush_memory(aggressive=aggressive)
        return purged_count

    # -------------------------------------------------------------------------
    # Stage Lifecycle Context & Decorator
    # -------------------------------------------------------------------------

    def stage_context(
        self,
        stage_name: str,
        auto_purge: bool = True,
        purge_models: bool = True,
    ) -> VRAMStageContext:
        """
        Creates a VRAMStageContext context manager for the given stage name.
        """
        return VRAMStageContext(
            stage_name=stage_name,
            manager=self,
            auto_purge=auto_purge,
            purge_models=purge_models,
        )

    def vram_lifecycle_stage(
        self,
        stage_name: Optional[Union[str, Callable]] = None,
        auto_purge: bool = True,
        purge_models: bool = True,
    ) -> Any:
        """
        Method decorator that wraps a function inside a VRAMStageContext.
        """
        return vram_lifecycle_stage(
            stage_name=stage_name,
            auto_purge=auto_purge,
            purge_models=purge_models,
            manager=self,
        )

    # -------------------------------------------------------------------------
    # Sequential CPU Offloading & VAE Optimization Helpers
    # -------------------------------------------------------------------------

    def enable_sequential_cpu_offload(self, pipeline: Any) -> Any:
        """
        Enables sequential model/submodule CPU offloading on a Diffusers or custom pipeline.
        Allows large models (SDXL, Wan 2.1) to run on 15GB VRAM by loading only the active
        submodule into GPU VRAM and keeping inactive components in host RAM.
        """
        if pipeline is None:
            return None

        if hasattr(pipeline, "enable_sequential_cpu_offload") and callable(pipeline.enable_sequential_cpu_offload):
            try:
                pipeline.enable_sequential_cpu_offload()
                logger.info("Enabled sequential CPU offload on pipeline.")
                return pipeline
            except Exception as e:
                logger.warning(f"enable_sequential_cpu_offload failed: {e}. Trying enable_model_cpu_offload.")

        if hasattr(pipeline, "enable_model_cpu_offload") and callable(pipeline.enable_model_cpu_offload):
            try:
                pipeline.enable_model_cpu_offload()
                logger.info("Enabled model CPU offload on pipeline.")
                return pipeline
            except Exception as e:
                logger.warning(f"enable_model_cpu_offload failed: {e}.")

        return pipeline

    def enable_vae_optimizations(self, vae_or_pipeline: Any) -> Any:
        """
        Enables VAE spatial slicing and spatial tiling optimizations to prevent VAE decoder OOMs
        when generating or reconstructing 720p / 1080p frames.
        """
        if vae_or_pipeline is None:
            return None

        target = vae_or_pipeline

        # 1. Check if target is a VAE object directly with enable_slicing / enable_tiling
        if hasattr(target, "enable_slicing") and callable(target.enable_slicing):
            try:
                target.enable_slicing()
                logger.info("Enabled VAE slicing optimization.")
            except Exception as e:
                logger.debug(f"VAE slicing not applied: {e}")

        if hasattr(target, "enable_tiling") and callable(target.enable_tiling):
            try:
                target.enable_tiling()
                logger.info("Enabled VAE tiling optimization.")
            except Exception as e:
                logger.debug(f"VAE tiling not applied: {e}")

        # 2. Check if target is a pipeline with a distinct .vae attribute
        vae = getattr(target, "vae", None)
        if vae is not None and vae is not target:
            if hasattr(vae, "enable_slicing") and callable(vae.enable_slicing):
                try:
                    vae.enable_slicing()
                    logger.info("Enabled VAE slicing optimization on pipeline.vae.")
                except Exception as e:
                    logger.debug(f"Pipeline VAE slicing not applied: {e}")

            if hasattr(vae, "enable_tiling") and callable(vae.enable_tiling):
                try:
                    vae.enable_tiling()
                    logger.info("Enabled VAE tiling optimization on pipeline.vae.")
                except Exception as e:
                    logger.debug(f"Pipeline VAE tiling not applied: {e}")

        # 3. If target is a pipeline, also enable attention slicing if available
        if hasattr(target, "enable_attention_slicing") and callable(target.enable_attention_slicing):
            try:
                target.enable_attention_slicing(slice_size="auto")
                logger.info("Enabled attention slicing on pipeline.")
            except Exception as e:
                logger.debug(f"Attention slicing not applied: {e}")

        return target

    def enable_attention_slicing(self, pipeline: Any, slice_size: str = "auto") -> Any:
        """
        Enables attention slicing on a diffusion or transformer pipeline to minimize peak activation memory.
        """
        if pipeline is None:
            return None
            
        if hasattr(pipeline, "enable_attention_slicing") and callable(pipeline.enable_attention_slicing):
            try:
                pipeline.enable_attention_slicing(slice_size=slice_size)
                logger.info(f"Enabled attention slicing with slice_size='{slice_size}'.")
            except Exception as e:
                logger.debug(f"Attention slicing failed: {e}")
                
        return pipeline

    def log_memory_summary(self, prefix: str = "") -> str:
        """
        Returns and logs a formatted memory status summary string.
        """
        stats = self.get_vram_stats()
        p = f"[{prefix}] " if prefix else ""
        if stats["is_cuda"]:
            summary = (
                f"{p}GPU: {stats['device_name']} | Alloc: {stats['allocated_mb']:.1f} MB | "
                f"Reserved: {stats['reserved_mb']:.1f} MB | Free: {stats['free_mb']:.1f} MB | "
                f"Total: {stats['total_mb']:.1f} MB | Util: {stats['utilization_pct']:.1f}% | "
                f"Host RAM Used: {stats['system_ram_used_mb']:.1f}/{stats['system_ram_total_mb']:.1f} MB"
            )
        else:
            summary = (
                f"{p}Device: {stats['device_name']} | "
                f"Host RAM Used: {stats['system_ram_used_mb']:.1f}/{stats['system_ram_total_mb']:.1f} MB "
                f"({stats['system_ram_utilization_pct']:.1f}%)"
            )
        logger.info(summary)
        return summary


# -----------------------------------------------------------------------------
# Standalone Decorator & Convenience Module Functions
# -----------------------------------------------------------------------------

def vram_lifecycle_stage(
    stage_name: Optional[Union[str, Callable]] = None,
    auto_purge: bool = True,
    purge_models: bool = True,
    manager: Optional[VRAMManager] = None,
) -> Any:
    """
    Decorator for wrapping functions in a clean VRAM lifecycle stage.
    
    Supports:
    - @vram_lifecycle_stage
    - @vram_lifecycle_stage("stage_name")
    - @vram_lifecycle_stage(stage_name="stage_name", auto_purge=True)
    """
    # Case 1: Used as bare decorator @vram_lifecycle_stage
    if callable(stage_name):
        fn = stage_name
        resolved_name = fn.__name__
        mgr = manager or VRAMManager.get_instance()

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with VRAMStageContext(resolved_name, manager=mgr, auto_purge=auto_purge, purge_models=purge_models):
                return fn(*args, **kwargs)

        return wrapper

    # Case 2: Used with arguments @vram_lifecycle_stage("stage_name", ...)
    def decorator(fn: Callable) -> Callable:
        resolved_name = stage_name or fn.__name__
        mgr = manager or VRAMManager.get_instance()

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with VRAMStageContext(resolved_name, manager=mgr, auto_purge=auto_purge, purge_models=purge_models):
                return fn(*args, **kwargs)

        return wrapper

    return decorator


def flush_memory(aggressive: bool = True) -> Dict[str, float]:
    """Module-level convenience wrapper for VRAMManager.get_instance().flush_memory()"""
    return VRAMManager.get_instance().flush_memory(aggressive=aggressive)


def get_vram_stats() -> Dict[str, Union[float, str, bool]]:
    """Module-level convenience wrapper for VRAMManager.get_instance().get_vram_stats()"""
    return VRAMManager.get_instance().get_vram_stats()


def get_optimal_precision(
    model_family: str = "diffusion",
    quantize: Union[bool, str] = False,
    allow_fp8: bool = True,
) -> Any:
    """Module-level convenience wrapper for VRAMManager.get_instance().get_optimal_precision()"""
    return VRAMManager.get_instance().get_optimal_precision(
        model_family=model_family, quantize=quantize, allow_fp8=allow_fp8
    )


def stage_context(
    stage_name: str,
    auto_purge: bool = True,
    purge_models: bool = True,
) -> VRAMStageContext:
    """Module-level convenience wrapper for VRAMManager.get_instance().stage_context()"""
    return VRAMManager.get_instance().stage_context(
        stage_name=stage_name, auto_purge=auto_purge, purge_models=purge_models
    )


def enable_sequential_cpu_offload(pipeline: Any) -> Any:
    """Module-level convenience wrapper for VRAMManager.get_instance().enable_sequential_cpu_offload()"""
    return VRAMManager.get_instance().enable_sequential_cpu_offload(pipeline)


def enable_vae_optimizations(vae_or_pipeline: Any) -> Any:
    """Module-level convenience wrapper for VRAMManager.get_instance().enable_vae_optimizations()"""
    return VRAMManager.get_instance().enable_vae_optimizations(vae_or_pipeline)


def enable_attention_slicing(pipeline: Any, slice_size: str = "auto") -> Any:
    """Module-level convenience wrapper for VRAMManager.get_instance().enable_attention_slicing()"""
    return VRAMManager.get_instance().enable_attention_slicing(pipeline, slice_size=slice_size)


def register_model(name: str, model: Any) -> None:
    """Module-level convenience wrapper for VRAMManager.get_instance().register_model()"""
    VRAMManager.get_instance().register_model(name, model)


def unregister_model(name: str) -> Optional[Any]:
    """Module-level convenience wrapper for VRAMManager.get_instance().unregister_model()"""
    return VRAMManager.get_instance().unregister_model(name)


def purge_models(*names: str, aggressive: bool = True) -> int:
    """Module-level convenience wrapper for VRAMManager.get_instance().purge_models()"""
    return VRAMManager.get_instance().purge_models(*names, aggressive=aggressive)
