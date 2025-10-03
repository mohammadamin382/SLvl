"""
NVIDIA H100 Specific Optimizations
- 80GB VRAM
- Hopper architecture with 4th gen Tensor Cores
- Compute Capability 9.0
- FP8 Transformer Engine support
- 2x faster than A100
"""

import torch
from typing import Dict, Any
from . import GPUOptimizer


class H100Optimizer(GPUOptimizer):
    """H100-specific optimization strategies"""

    def __init__(self):
        super().__init__()
        self.name = "H100"
        self.compute_capability = 9.0

        props = torch.cuda.get_device_properties(0)
        self.memory_gb = props.total_memory / (1024**3)

        # H100 specific features
        self.supports_tf32 = True
        self.supports_bf16 = True
        self.supports_fp8 = True  # Major H100 advantage
        self.tensor_core_gen = 4
        self.nvlink_enabled = torch.cuda.device_count() > 1

    def get_optimal_batch_size(self, model_params: int, sequence_length: int) -> int:
        """
        H100 can handle larger batches due to FP8 and better memory bandwidth
        """
        model_memory = (model_params * 2) / (1024**3)

        # FP8 reduces memory footprint significantly
        if self.supports_fp8:
            model_memory *= 0.5  # FP8 is half of FP16

        activation_per_sample = model_memory * 3.5 * (sequence_length / 512)

        available_memory = self.memory_gb * 0.80  # H100 better memory management
        usable_memory = available_memory - model_memory

        batch_size = int(usable_memory / activation_per_sample)

        # H100 performs best with multiples of 16
        batch_size = max(16, (batch_size // 16) * 16)

        return min(batch_size, 256)  # H100 can handle much larger batches

    def get_tensor_core_config(self) -> Dict[str, Any]:
        """H100 4th gen Tensor Core with FP8 support"""
        return {
            'use_tf32': True,
            'use_bf16': True,
            'use_fp8': True,  # Unique to H100/H200
            'use_fp16': False,
            'allow_tf32_matmul': True,
            'allow_tf32_conv': True,
            'matmul_precision': 'high',
            'fp8_recipe': {
                'margin': 0,
                'fp8_format': 'HYBRID',  # E4M3 for forward, E5M2 for backward
                'amax_history_len': 1024,
                'amax_compute_algo': 'max',
            }
        }

    def get_memory_config(self) -> Dict[str, Any]:
        """H100 advanced memory optimization"""
        return {
            'max_split_size_mb': 1024,  # H100 can handle larger splits
            'garbage_collection_threshold': 0.85,
            'empty_cache_frequency': 150,
            'gradient_checkpointing': False,  # Less needed with 80GB
            'activation_checkpointing_ratio': 0.3,
            'pin_memory': True,
            'non_blocking_transfer': True,
            'reserved_memory_ratio': 0.10,
            'use_flash_attention': True,  # H100 optimized
        }

    def get_compile_config(self) -> Dict[str, Any]:
        """torch.compile with H100 optimizations"""
        return {
            'mode': 'max-autotune-no-cudagraphs',
            'fullgraph': True,
            'dynamic': False,
            'backend': 'inductor',
            'options': {
                'triton.cudagraphs': True,
                'triton.cudagraph_trees': True,
                'max_autotune': True,
                'epilogue_fusion': True,
                'shape_padding': True,
                'coordinate_descent_tuning': True,
                'coordinate_descent_check_all_directions': True,
            }
        }

    def apply_optimizations(self, model: torch.nn.Module) -> torch.nn.Module:
        """Apply H100-specific optimizations"""
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

        # H100 specific: enable flash attention
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)

        torch.set_float32_matmul_precision('high')

        model = model.to(memory_format=torch.channels_last)

        return model

    def get_dataloader_config(self) -> Dict[str, Any]:
        """Optimal dataloader settings for H100"""
        return {
            'num_workers': 12,
            'pin_memory': True,
            'persistent_workers': True,
            'prefetch_factor': 6,
        }

    def get_optimizer_config(self) -> Dict[str, Any]:
        """Optimizer settings for H100"""
        return {
            'fused': True,
            'foreach': True,
            'capturable': True,
            'differentiable': False,
        }
