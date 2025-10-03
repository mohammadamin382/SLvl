"""
GPU-Specific Optimizations Module
Automatically detects GPU and loads appropriate optimizations
"""

import torch
import subprocess
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod


class GPUOptimizer(ABC):
    """Base class for GPU-specific optimizations"""

    def __init__(self):
        self.name = "Base"
        self.compute_capability = None
        self.memory_gb = 0

    @abstractmethod
    def get_optimal_batch_size(self, model_params: int, sequence_length: int) -> int:
        """Calculate optimal batch size based on model size and GPU memory"""
        pass

    @abstractmethod
    def get_tensor_core_config(self) -> Dict[str, Any]:
        """Get tensor core specific configurations"""
        pass

    @abstractmethod
    def get_memory_config(self) -> Dict[str, Any]:
        """Get memory optimization configurations"""
        pass

    @abstractmethod
    def get_compile_config(self) -> Dict[str, Any]:
        """Get torch.compile configurations"""
        pass

    def apply_optimizations(self, model: torch.nn.Module) -> torch.nn.Module:
        """Apply GPU-specific optimizations to model"""
        return model


def detect_gpu() -> Optional[str]:
    """Detect the current GPU model"""
    if not torch.cuda.is_available():
        return None

    try:
        # Get GPU name
        gpu_name = torch.cuda.get_device_name(0).upper()

        # Map GPU names to optimizer classes
        if "A100" in gpu_name:
            return "A100"
        elif "H100" in gpu_name:
            return "H100"
        elif "H200" in gpu_name:
            return "H200"
        elif "V100" in gpu_name:
            return "V100"
        elif "P100" in gpu_name:
            return "P100"
        elif "RTX 5090" in gpu_name or "5090" in gpu_name:
            return "RTX5090"
        else:
            return "GENERIC"
    except Exception as e:
        print(f"Warning: Could not detect GPU: {e}")
        return None


def get_gpu_optimizer() -> GPUOptimizer:
    """Get the appropriate GPU optimizer for the current hardware"""
    gpu_type = detect_gpu()

    if gpu_type == "A100":
        from .a100 import A100Optimizer
        return A100Optimizer()
    elif gpu_type == "H100":
        from .h100 import H100Optimizer
        return H100Optimizer()
    elif gpu_type == "H200":
        from .h200 import H200Optimizer
        return H200Optimizer()
    elif gpu_type == "V100":
        from .v100 import V100Optimizer
        return V100Optimizer()
    elif gpu_type == "P100":
        from .p100 import P100Optimizer
        return P100Optimizer()
    elif gpu_type == "RTX5090":
        from .rtx5090 import RTX5090Optimizer
        return RTX5090Optimizer()
    else:
        from .generic import GenericOptimizer
        return GenericOptimizer()


__all__ = ['GPUOptimizer', 'get_gpu_optimizer', 'detect_gpu']
