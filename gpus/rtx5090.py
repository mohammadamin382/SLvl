"""
NVIDIA RTX 5090 Specific Optimizations
- 32GB VRAM (expected)
- Blackwell architecture with 5th gen Tensor Cores (expected)
- Compute Capability 10.0 (expected)
- Consumer flagship with professional features
"""

import torch
from typing import Dict, Any
from . import GPUOptimizer


class RTX5090Optimizer(GPUOptimizer):
    """RTX 5090-specific optimization strategies"""

    def __init__(self):
        super().__init__()
        self.name = "RTX5090"
        self.compute_capability = 10.0  # Expected for Blackwell

        props = torch.cuda.get_device_properties(0)
        self.memory_gb = props.total_memory / (1024**3)

        # RTX 5090 expected features
        self.supports_tf32 = True
        self.supports_bf16 = True
        self.supports_fp8 = True  # Expected in Blackwell
        self.tensor_core_gen = 5

    def get_optimal_batch_size(self, model_params: int, sequence_length: int) -> int:
        """
        RTX 5090 expected to have good memory bandwidth
        """
        model_memory = (model_params * 2) / (1024**3)

        if self.supports_fp8:
            model_memory *= 0.5

        activation_per_sample = model_memory * 3.8 * (sequence_length / 512)

        # Consumer card, slightly more conservative than datacenter
        available_memory = self.memory_gb * 0.75
        usable_memory = available_memory - model_memory

        batch_size = int(usable_memory / activation_per_sample)
        batch_size = max(8, (batch_size // 8) * 8)

        return min(batch_size, 128)

    def get_tensor_core_config(self) -> Dict[str, Any]:
        """RTX 5090 5th gen Tensor Cores (expected)"""
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
                'amax_history_len': 1024,
                'amax_compute_algo': 'max',
            }
        }

    def get_memory_config(self) -> Dict[str, Any]:
        """RTX 5090 memory optimization"""
        return {
            'max_split_size_mb': 512,
            'garbage_collection_threshold': 0.80,
            'empty_cache_frequency': 100,
            'gradient_checkpointing': False,  # 32GB should be sufficient
            'activation_checkpointing_ratio': 0.4,
            'pin_memory': True,
            'non_blocking_transfer': True,
            'reserved_memory_ratio': 0.15,
            'use_flash_attention': True,
        }

    def get_compile_config(self) -> Dict[str, Any]:
        """torch.compile for RTX 5090"""
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
            }
        }

    def apply_optimizations(self, model: torch.nn.Module) -> torch.nn.Module:
        """Apply RTX 5090-specific optimizations"""
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

        # Enable flash attention
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)

        torch.set_float32_matmul_precision('high')

        model = model.to(memory_format=torch.channels_last)

        return model

    def get_dataloader_config(self) -> Dict[str, Any]:
        """Optimal dataloader settings for RTX 5090"""
        return {
            'num_workers': 8,
            'pin_memory': True,
            'persistent_workers': True,
            'prefetch_factor': 4,
        }

    def get_optimizer_config(self) -> Dict[str, Any]:
        """Optimizer settings for RTX 5090"""
        return {
            'fused': True,
            'foreach': True,
            'capturable': True,
            'differentiable': False,
        }
