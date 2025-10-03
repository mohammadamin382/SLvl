"""
NVIDIA A100 Specific Optimizations
- 40GB/80GB VRAM variants
- Ampere architecture with 3rd gen Tensor Cores
- Compute Capability 8.0
- Optimized for FP16, TF32, and BF16
"""

import torch
from typing import Dict, Any
from . import GPUOptimizer


class A100Optimizer(GPUOptimizer):
    """A100-specific optimization strategies"""

    def __init__(self):
        super().__init__()
        self.name = "A100"
        self.compute_capability = 8.0

        # Detect 40GB vs 80GB variant
        props = torch.cuda.get_device_properties(0)
        self.memory_gb = props.total_memory / (1024**3)
        self.is_80gb = self.memory_gb > 50

        # A100 specific features
        self.supports_tf32 = True
        self.supports_bf16 = True
        self.tensor_core_gen = 3
        self.nvlink_enabled = torch.cuda.device_count() > 1

    def get_optimal_batch_size(self, model_params: int, sequence_length: int) -> int:
        """
        Calculate optimal batch size for A100
        Conservative estimation to prevent OOM
        """
        # Base memory for model parameters (in GB)
        model_memory = (model_params * 4) / (1024**3)  # 4 bytes per param (FP32)

        # With mixed precision, effective memory is ~2 bytes per param
        model_memory_mixed = (model_params * 2) / (1024**3)

        # Activation memory scales with batch size and sequence length
        # Conservative estimate: 4x model size for activations per sample
        activation_per_sample = model_memory_mixed * 4 * (sequence_length / 512)

        # Reserve 20% memory for CUDA operations and gradients
        available_memory = self.memory_gb * 0.75
        usable_memory = available_memory - model_memory_mixed

        # Calculate batch size
        batch_size = int(usable_memory / activation_per_sample)

        # A100 performs best with batch sizes that are multiples of 8
        batch_size = max(8, (batch_size // 8) * 8)

        # Cap based on variant
        max_batch = 128 if self.is_80gb else 64
        return min(batch_size, max_batch)

    def get_tensor_core_config(self) -> Dict[str, Any]:
        """A100 3rd gen Tensor Core optimizations"""
        return {
            'use_tf32': True,  # TF32 for matmuls (automatic speedup)
            'use_bf16': True,  # BFloat16 for mixed precision
            'use_fp16': False,  # BF16 preferred over FP16 on A100
            'allow_tf32_matmul': True,
            'allow_tf32_conv': True,
            'matmul_precision': 'high',  # Use TF32 for accuracy
        }

    def get_memory_config(self) -> Dict[str, Any]:
        """A100 memory optimization settings"""
        return {
            'max_split_size_mb': 512,
            'garbage_collection_threshold': 0.8,
            'empty_cache_frequency': 100,
            'gradient_checkpointing': True,
            'activation_checkpointing_ratio': 0.5,
            'pin_memory': True,
            'non_blocking_transfer': True,
            'reserved_memory_ratio': 0.15,
        }

    def get_compile_config(self) -> Dict[str, Any]:
        """torch.compile optimization for A100"""
        return {
            'mode': 'max-autotune',  # Aggressive optimization
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
        """Apply A100-specific optimizations"""
        # Enable TF32 for matmuls
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Enable cudnn benchmarking for optimal kernels
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

        # Set matmul precision
        torch.set_float32_matmul_precision('high')

        # Apply memory format optimization
        model = model.to(memory_format=torch.channels_last)

        return model

    def get_dataloader_config(self) -> Dict[str, Any]:
        """Optimal dataloader settings for A100"""
        return {
            'num_workers': 8 if self.is_80gb else 4,
            'pin_memory': True,
            'persistent_workers': True,
            'prefetch_factor': 4,
        }

    def get_optimizer_config(self) -> Dict[str, Any]:
        """Optimizer settings optimized for A100"""
        return {
            'fused': True,  # Use fused optimizer kernels
            'foreach': True,  # Multi-tensor operations
            'capturable': True,  # CUDA graph compatible
        }
