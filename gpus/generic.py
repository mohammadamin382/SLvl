"""
Generic GPU Optimizations
Fallback for unrecognized GPUs
Conservative settings that work on most hardware
"""

import torch
from typing import Dict, Any
from . import GPUOptimizer


class GenericOptimizer(GPUOptimizer):
    """Generic GPU optimization strategies"""

    def __init__(self):
        super().__init__()
        self.name = "Generic"

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            self.memory_gb = props.total_memory / (1024**3)
            self.compute_capability = props.major + props.minor * 0.1
        else:
            self.memory_gb = 8
            self.compute_capability = 7.0

        # Detect capabilities
        self.supports_tf32 = self.compute_capability >= 8.0
        self.supports_bf16 = self.compute_capability >= 8.0
        self.supports_fp16 = self.compute_capability >= 7.0

    def get_optimal_batch_size(self, model_params: int, sequence_length: int) -> int:
        """Conservative batch size calculation"""
        model_memory = (model_params * 2) / (1024**3)
        activation_per_sample = model_memory * 5.0 * (sequence_length / 512)

        available_memory = self.memory_gb * 0.65
        usable_memory = available_memory - model_memory

        batch_size = int(usable_memory / activation_per_sample)
        batch_size = max(4, (batch_size // 4) * 4)

        return min(batch_size, 32)

    def get_tensor_core_config(self) -> Dict[str, Any]:
        """Generic tensor core configuration"""
        return {
            'use_tf32': self.supports_tf32,
            'use_bf16': self.supports_bf16,
            'use_fp16': self.supports_fp16,
            'allow_tf32_matmul': self.supports_tf32,
            'allow_tf32_conv': self.supports_tf32,
            'matmul_precision': 'high' if self.supports_tf32 else 'highest',
        }

    def get_memory_config(self) -> Dict[str, Any]:
        """Generic memory configuration"""
        return {
            'max_split_size_mb': 256,
            'garbage_collection_threshold': 0.75,
            'empty_cache_frequency': 50,
            'gradient_checkpointing': True,
            'activation_checkpointing_ratio': 0.6,
            'pin_memory': True,
            'non_blocking_transfer': True,
            'reserved_memory_ratio': 0.20,
        }

    def get_compile_config(self) -> Dict[str, Any]:
        """Generic compile configuration"""
        return {
            'mode': 'default',
            'fullgraph': False,
            'dynamic': False,
            'backend': 'inductor',
            'options': {}
        }

    def apply_optimizations(self, model: torch.nn.Module) -> torch.nn.Module:
        """Apply generic optimizations"""
        if self.supports_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        torch.backends.cudnn.benchmark = True

        if self.supports_tf32:
            torch.set_float32_matmul_precision('high')
        else:
            torch.set_float32_matmul_precision('highest')

        return model

    def get_dataloader_config(self) -> Dict[str, Any]:
        """Generic dataloader settings"""
        return {
            'num_workers': 4,
            'pin_memory': True,
            'persistent_workers': True,
            'prefetch_factor': 2,
        }

    def get_optimizer_config(self) -> Dict[str, Any]:
        """Generic optimizer settings"""
        return {
            'fused': False,
            'foreach': True,
            'capturable': False,
        }
