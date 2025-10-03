"""
NVIDIA H200 Specific Optimizations
- 141GB VRAM (massive!)
- Hopper architecture with 4th gen Tensor Cores
- Compute Capability 9.0
- FP8 Transformer Engine support
- Improved memory bandwidth over H100
"""

import torch
from typing import Dict, Any
from . import GPUOptimizer


class H200Optimizer(GPUOptimizer):
    """H200-specific optimization strategies"""

    def __init__(self):
        super().__init__()
        self.name = "H200"
        self.compute_capability = 9.0

        props = torch.cuda.get_device_properties(0)
        self.memory_gb = props.total_memory / (1024**3)

        # H200 specific features (same as H100 but more memory)
        self.supports_tf32 = True
        self.supports_bf16 = True
        self.supports_fp8 = True
        self.tensor_core_gen = 4
        self.nvlink_enabled = torch.cuda.device_count() > 1
        self.hbm3_memory = True  # Faster memory than H100

    def get_optimal_batch_size(self, model_params: int, sequence_length: int) -> int:
        """
        H200 with 141GB can handle massive batches
        """
        model_memory = (model_params * 2) / (1024**3)

        if self.supports_fp8:
            model_memory *= 0.5

        activation_per_sample = model_memory * 3.5 * (sequence_length / 512)

        # Can use even more memory due to massive VRAM
        available_memory = self.memory_gb * 0.85
        usable_memory = available_memory - model_memory

        batch_size = int(usable_memory / activation_per_sample)
        batch_size = max(16, (batch_size // 16) * 16)

        return min(batch_size, 512)  # H200 can handle very large batches

    def get_tensor_core_config(self) -> Dict[str, Any]:
        """Same as H100 but with more aggressive settings"""
        return {
            'use_tf32': True,
            'use_bf16': True,
            'use_fp8': True,
            'use_fp16': False,
            'allow_tf32_matmul': True,
            'allow_tf32_conv': True,
            'matmul_precision': 'high',
            'fp8_recipe': {
                'margin': 0,
                'fp8_format': 'HYBRID',
                'amax_history_len': 2048,  # Longer history due to more memory
                'amax_compute_algo': 'max',
            }
        }

    def get_memory_config(self) -> Dict[str, Any]:
        """H200 can be very aggressive with memory"""
        return {
            'max_split_size_mb': 2048,  # Much larger splits possible
            'garbage_collection_threshold': 0.90,
            'empty_cache_frequency': 200,
            'gradient_checkpointing': False,  # Not needed with 141GB
            'activation_checkpointing_ratio': 0.0,  # Disable for speed
            'pin_memory': True,
            'non_blocking_transfer': True,
            'reserved_memory_ratio': 0.08,
            'use_flash_attention': True,
        }

    def get_compile_config(self) -> Dict[str, Any]:
        """torch.compile with maximum optimization"""
        return {
            'mode': 'max-autotune',
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
        """Apply H200-specific optimizations"""
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)  # Use optimized versions only

        torch.set_float32_matmul_precision('high')

        model = model.to(memory_format=torch.channels_last)

        return model

    def get_dataloader_config(self) -> Dict[str, Any]:
        """Optimal dataloader settings for H200"""
        return {
            'num_workers': 16,  # Can handle more workers
            'pin_memory': True,
            'persistent_workers': True,
            'prefetch_factor': 8,
        }

    def get_optimizer_config(self) -> Dict[str, Any]:
        """Optimizer settings for H200"""
        return {
            'fused': True,
            'foreach': True,
            'capturable': True,
            'differentiable': False,
        }
