"""
NVIDIA V100 Specific Optimizations
- 16GB/32GB VRAM variants
- Volta architecture with 1st gen Tensor Cores
- Compute Capability 7.0
- Optimized for FP16 (no BF16 or TF32 support)
"""

import torch
from typing import Dict, Any
from . import GPUOptimizer


class V100Optimizer(GPUOptimizer):
    """V100-specific optimization strategies"""

    def __init__(self):
        super().__init__()
        self.name = "V100"
        self.compute_capability = 7.0

        props = torch.cuda.get_device_properties(0)
        self.memory_gb = props.total_memory / (1024**3)
        self.is_32gb = self.memory_gb > 20

        # V100 specific features
        self.supports_tf32 = False  # Volta doesn't have TF32
        self.supports_bf16 = False  # No BF16 support
        self.supports_fp16 = True
        self.tensor_core_gen = 1

    def get_optimal_batch_size(self, model_params: int, sequence_length: int) -> int:
        """
        V100 has less memory, need conservative batch sizes
        """
        model_memory = (model_params * 2) / (1024**3)  # FP16

        activation_per_sample = model_memory * 4.5 * (sequence_length / 512)

        # More conservative memory usage
        available_memory = self.memory_gb * 0.70
        usable_memory = available_memory - model_memory

        batch_size = int(usable_memory / activation_per_sample)

        # V100 works well with multiples of 8
        batch_size = max(8, (batch_size // 8) * 8)

        max_batch = 48 if self.is_32gb else 24
        return min(batch_size, max_batch)

    def get_tensor_core_config(self) -> Dict[str, Any]:
        """V100 1st gen Tensor Cores (FP16 only)"""
        return {
            'use_tf32': False,  # Not supported on Volta
            'use_bf16': False,  # Not supported on Volta
            'use_fp16': True,   # Primary precision for V100
            'allow_tf32_matmul': False,
            'allow_tf32_conv': False,
            'matmul_precision': 'highest',  # Use FP32 accumulation
        }

    def get_memory_config(self) -> Dict[str, Any]:
        """V100 conservative memory settings"""
        return {
            'max_split_size_mb': 256,
            'garbage_collection_threshold': 0.75,
            'empty_cache_frequency': 50,
            'gradient_checkpointing': True,  # Essential for V100
            'activation_checkpointing_ratio': 0.7,  # Aggressive checkpointing
            'pin_memory': True,
            'non_blocking_transfer': True,
            'reserved_memory_ratio': 0.20,
        }

    def get_compile_config(self) -> Dict[str, Any]:
        """torch.compile with V100 limitations"""
        return {
            'mode': 'reduce-overhead',  # Less aggressive on older hardware
            'fullgraph': False,  # More flexible for V100
            'dynamic': False,
            'backend': 'inductor',
            'options': {
                'triton.cudagraphs': False,  # Can be unstable on V100
                'max_autotune': False,
                'epilogue_fusion': True,
                'shape_padding': False,
            }
        }

    def apply_optimizations(self, model: torch.nn.Module) -> torch.nn.Module:
        """Apply V100-specific optimizations"""
        # No TF32 on V100
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

        # V100 uses FP32 for highest precision
        torch.set_float32_matmul_precision('highest')

        return model

    def get_dataloader_config(self) -> Dict[str, Any]:
        """Optimal dataloader settings for V100"""
        return {
            'num_workers': 4,
            'pin_memory': True,
            'persistent_workers': True,
            'prefetch_factor': 2,
        }

    def get_optimizer_config(self) -> Dict[str, Any]:
        """Optimizer settings for V100"""
        return {
            'fused': False,  # Fused optimizers less stable on V100
            'foreach': True,
            'capturable': False,
        }
