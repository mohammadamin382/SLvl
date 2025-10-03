"""
NVIDIA P100 Specific Optimizations
- 12GB/16GB VRAM variants
- Pascal architecture (NO Tensor Cores)
- Compute Capability 6.0
- FP16 support but no Tensor Cores
"""

import torch
from typing import Dict, Any
from . import GPUOptimizer


class P100Optimizer(GPUOptimizer):
    """P100-specific optimization strategies"""

    def __init__(self):
        super().__init__()
        self.name = "P100"
        self.compute_capability = 6.0

        props = torch.cuda.get_device_properties(0)
        self.memory_gb = props.total_memory / (1024**3)
        self.is_16gb = self.memory_gb > 14

        # P100 limitations
        self.supports_tf32 = False
        self.supports_bf16 = False
        self.supports_fp16 = True
        self.tensor_core_gen = 0  # No tensor cores

    def get_optimal_batch_size(self, model_params: int, sequence_length: int) -> int:
        """
        P100 very conservative - limited memory and no tensor cores
        """
        model_memory = (model_params * 2) / (1024**3)

        # No tensor cores means higher memory overhead
        activation_per_sample = model_memory * 5.0 * (sequence_length / 512)

        available_memory = self.memory_gb * 0.65  # Very conservative
        usable_memory = available_memory - model_memory

        batch_size = int(usable_memory / activation_per_sample)
        batch_size = max(4, (batch_size // 4) * 4)

        max_batch = 32 if self.is_16gb else 16
        return min(batch_size, max_batch)

    def get_tensor_core_config(self) -> Dict[str, Any]:
        """P100 has no tensor cores"""
        return {
            'use_tf32': False,
            'use_bf16': False,
            'use_fp16': True,  # Can use FP16 but without tensor core acceleration
            'allow_tf32_matmul': False,
            'allow_tf32_conv': False,
            'matmul_precision': 'highest',
        }

    def get_memory_config(self) -> Dict[str, Any]:
        """P100 very aggressive memory optimization needed"""
        return {
            'max_split_size_mb': 128,
            'garbage_collection_threshold': 0.70,
            'empty_cache_frequency': 25,
            'gradient_checkpointing': True,
            'activation_checkpointing_ratio': 0.8,  # Very aggressive
            'pin_memory': True,
            'non_blocking_transfer': True,
            'reserved_memory_ratio': 0.25,
        }

    def get_compile_config(self) -> Dict[str, Any]:
        """torch.compile minimal for P100"""
        return {
            'mode': 'default',  # Conservative mode
            'fullgraph': False,
            'dynamic': False,
            'backend': 'inductor',
            'options': {
                'triton.cudagraphs': False,
                'max_autotune': False,
                'epilogue_fusion': False,
                'shape_padding': False,
            }
        }

    def apply_optimizations(self, model: torch.nn.Module) -> torch.nn.Module:
        """Apply P100-specific optimizations"""
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        # Benchmarking still helps on P100
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

        torch.set_float32_matmul_precision('highest')

        return model

    def get_dataloader_config(self) -> Dict[str, Any]:
        """Optimal dataloader settings for P100"""
        return {
            'num_workers': 2,
            'pin_memory': True,
            'persistent_workers': True,
            'prefetch_factor': 2,
        }

    def get_optimizer_config(self) -> Dict[str, Any]:
        """Optimizer settings for P100"""
        return {
            'fused': False,
            'foreach': False,  # Less efficient on P100
            'capturable': False,
        }
