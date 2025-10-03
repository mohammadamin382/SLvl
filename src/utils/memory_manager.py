"""
Advanced Memory Management
Prevents OOM errors through intelligent monitoring and optimization
"""

import torch
import gc
from typing import Optional, Dict, Any
import logging
import psutil
import os


logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Intelligent memory management to prevent OOM errors

    Features:
    - Real-time memory monitoring
    - Automatic garbage collection
    - Dynamic batch size adjustment
    - Memory profiling
    """

    def __init__(self, device: torch.device, config: Dict[str, Any]):
        self.device = device
        self.config = config

        self.max_memory_allocated = 0
        self.memory_history = []

        # Thresholds
        self.gc_threshold = config.get('garbage_collection_threshold', 0.8)
        self.empty_cache_frequency = config.get('empty_cache_frequency', 100)
        self.reserved_memory_ratio = config.get('reserved_memory_ratio', 0.15)

        self.step_count = 0

        if torch.cuda.is_available():
            self.total_memory = torch.cuda.get_device_properties(device).total_memory
            logger.info(f"GPU Memory: {self.total_memory / (1024**3):.2f} GB")
        else:
            self.total_memory = psutil.virtual_memory().total
            logger.info(f"CPU Memory: {self.total_memory / (1024**3):.2f} GB")

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(self.device)
            reserved = torch.cuda.memory_reserved(self.device)
            free = self.total_memory - reserved

            return {
                'allocated_gb': allocated / (1024**3),
                'reserved_gb': reserved / (1024**3),
                'free_gb': free / (1024**3),
                'utilization': allocated / self.total_memory,
            }
        else:
            mem = psutil.virtual_memory()
            return {
                'allocated_gb': mem.used / (1024**3),
                'reserved_gb': mem.used / (1024**3),
                'free_gb': mem.available / (1024**3),
                'utilization': mem.percent / 100.0,
            }

    def check_memory_pressure(self) -> bool:
        """Check if memory pressure is high"""
        usage = self.get_memory_usage()
        return usage['utilization'] > self.gc_threshold

    def optimize_memory(self, force: bool = False):
        """Optimize memory usage"""
        self.step_count += 1

        # Periodic cache clearing
        if force or (self.step_count % self.empty_cache_frequency == 0):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Garbage collection if memory pressure is high
        if force or self.check_memory_pressure():
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            usage = self.get_memory_usage()
            logger.debug(f"Memory optimized - Usage: {usage['utilization']*100:.1f}%")

    def log_memory_stats(self, prefix: str = ""):
        """Log current memory statistics"""
        usage = self.get_memory_usage()
        logger.info(
            f"{prefix}Memory - "
            f"Allocated: {usage['allocated_gb']:.2f}GB, "
            f"Reserved: {usage['reserved_gb']:.2f}GB, "
            f"Free: {usage['free_gb']:.2f}GB, "
            f"Utilization: {usage['utilization']*100:.1f}%"
        )

    def get_safe_batch_size(self, current_batch_size: int) -> int:
        """
        Calculate safe batch size based on current memory usage

        Returns adjusted batch size to prevent OOM
        """
        usage = self.get_memory_usage()

        # If memory usage is high, reduce batch size
        if usage['utilization'] > 0.85:
            new_batch = max(1, int(current_batch_size * 0.75))
            logger.warning(
                f"High memory usage ({usage['utilization']*100:.1f}%), "
                f"reducing batch size from {current_batch_size} to {new_batch}"
            )
            return new_batch

        return current_batch_size

    def estimate_batch_size(
        self,
        model: torch.nn.Module,
        sample_input: Dict[str, torch.Tensor],
        max_batch_size: int = 256
    ) -> int:
        """
        Estimate optimal batch size through binary search

        This is a safe way to find the maximum batch size without OOM
        """
        logger.info("Estimating optimal batch size...")

        def test_batch_size(batch_size: int) -> bool:
            """Test if a batch size fits in memory"""
            try:
                # Create dummy batch
                dummy_batch = {
                    k: v.repeat(batch_size, *([1] * (v.ndim - 1)))
                    for k, v in sample_input.items()
                }

                # Forward pass
                with torch.cuda.amp.autocast(enabled=True):
                    _ = model(**dummy_batch)

                # Simulate backward pass memory
                torch.cuda.empty_cache()

                return True

            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    return False
                raise

        # Binary search for optimal batch size
        low, high = 1, max_batch_size
        optimal = 1

        while low <= high:
            mid = (low + high) // 2

            if test_batch_size(mid):
                optimal = mid
                low = mid + 1
            else:
                high = mid - 1

            torch.cuda.empty_cache()

        logger.info(f"Estimated optimal batch size: {optimal}")
        return optimal

    def create_memory_snapshot(self, filename: str):
        """Create a memory snapshot for debugging"""
        if torch.cuda.is_available():
            snapshot = torch.cuda.memory_snapshot()
            torch.save(snapshot, filename)
            logger.info(f"Memory snapshot saved to {filename}")

    def reset_peak_stats(self):
        """Reset peak memory statistics"""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
            self.max_memory_allocated = 0


class GradientAccumulator:
    """
    Gradient accumulation for effective larger batch sizes
    without increasing memory usage
    """

    def __init__(self, accumulation_steps: int):
        self.accumulation_steps = accumulation_steps
        self.current_step = 0

    def should_step(self) -> bool:
        """Check if optimizer should step"""
        self.current_step += 1
        if self.current_step >= self.accumulation_steps:
            self.current_step = 0
            return True
        return False

    def get_loss_scale(self) -> float:
        """Get loss scaling factor for accumulation"""
        return 1.0 / self.accumulation_steps


class MixedPrecisionManager:
    """
    Advanced mixed precision training manager
    Handles different precision modes based on GPU capability
    """

    def __init__(self, gpu_config: Dict[str, Any], enabled: bool = True):
        self.enabled = enabled
        self.gpu_config = gpu_config

        # Determine precision mode
        if gpu_config.get('use_fp8', False) and enabled:
            self.precision = 'fp8'
        elif gpu_config.get('use_bf16', False) and enabled:
            self.precision = 'bf16'
        elif gpu_config.get('use_fp16', False) and enabled:
            self.precision = 'fp16'
        else:
            self.precision = 'fp32'

        logger.info(f"Using precision: {self.precision}")

        # Create GradScaler for FP16
        self.scaler = None
        if self.precision == 'fp16':
            self.scaler = torch.cuda.amp.GradScaler()

    def get_autocast_context(self):
        """Get autocast context manager"""
        if not self.enabled or self.precision == 'fp32':
            return torch.cuda.amp.autocast(enabled=False)

        dtype = None
        if self.precision == 'bf16':
            dtype = torch.bfloat16
        elif self.precision == 'fp16':
            dtype = torch.float16

        return torch.cuda.amp.autocast(enabled=True, dtype=dtype)

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for mixed precision"""
        if self.scaler is not None:
            return self.scaler.scale(loss)
        return loss

    def step_optimizer(self, optimizer, scaler_update: bool = True):
        """Step optimizer with gradient scaling"""
        if self.scaler is not None:
            self.scaler.step(optimizer)
            if scaler_update:
                self.scaler.update()
        else:
            optimizer.step()

    def unscale_gradients(self, optimizer):
        """Unscale gradients for gradient clipping"""
        if self.scaler is not None:
            self.scaler.unscale_(optimizer)
