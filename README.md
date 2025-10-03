# Chess Supervised Learning (SLvl)

Advanced, state-of-the-art chess model training system with GPU-specific optimizations and OOM prevention.

## Features

- **State-of-the-Art Architecture**: Advanced Transformer with RoPE, SwiGLU, RMSNorm
- **GPU-Specific Optimizations**: Auto-detects and optimizes for A100, H100, H200, V100, P100, RTX 5090
- **OOM Prevention**: Intelligent memory management prevents out-of-memory errors
- **DRY RUN Mode**: Quick syntax validation without full training
- **Highly Configurable**: YAML-based configuration for all parameters
- **PGN Auto-Detection**: Automatically detects and parses various PGN formats

## Supported GPUs

The system automatically detects your GPU and applies optimized settings:

- **NVIDIA A100** (40GB/80GB): TF32, BF16, 3rd gen Tensor Cores
- **NVIDIA H100** (80GB): FP8, BF16, 4th gen Tensor Cores, Flash Attention
- **NVIDIA H200** (141GB): FP8, BF16, massive memory capacity
- **NVIDIA V100** (16GB/32GB): FP16, 1st gen Tensor Cores
- **NVIDIA P100** (12GB/16GB): FP16, no Tensor Cores
- **NVIDIA RTX 5090** (32GB): FP8, BF16, 5th gen Tensor Cores (expected)

## Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data

Place your PGN files in `data/pgn/`:

```bash
mkdir -p data/pgn
# Copy your .pgn files to data/pgn/
```

### 3. Configure Training

Edit `settings.yaml` to configure your training:

```yaml
model:
  dim: 512
  depth: 12
  num_heads: 8

data:
  min_elo: 2000
  batch_size: 64
```

Edit `data_settings_format.yaml` to configure PGN parsing (or use auto-detection).

### 4. Dry Run (Recommended)

Test your setup with a quick dry run:

```bash
python train.py --dry-run
```

This will:
- Load configurations
- Detect GPU
- Initialize model
- Load a small sample of data
- Run a few training steps
- Verify everything works

### 5. Full Training

Start full training:

```bash
python train.py
```

## Configuration Files

### settings.yaml

Main training configuration:
- Model architecture (dim, depth, heads)
- Optimizer settings (learning rate, weight decay)
- Training parameters (steps, batch size)
- Data filtering (ELO, game length)
- GPU overrides

### data_settings_format.yaml

PGN data format configuration:
- Tag name mappings (WhiteElo, BlackElo, etc.)
- Quality filters
- Time control categories
- Move filtering
- Data augmentation

## Architecture Details

### Model Architecture

**ChessTransformer** - State-of-the-art transformer for chess:

- **Input Encoding**: 64 squares + metadata (turn, castling, en passant)
- **Positional Encoding**: Rotary Position Embeddings (RoPE)
- **Attention**: Multi-head self-attention with optional Flash Attention
- **FFN**: SwiGLU activation (superior to ReLU/GELU)
- **Normalization**: RMSNorm (more stable than LayerNorm)
- **Output**: Dual heads for policy (move prediction) and value (position evaluation)

### Training Features

- **Mixed Precision**: Automatic FP16/BF16/FP8 based on GPU
- **Gradient Checkpointing**: Reduce memory usage for large models
- **Gradient Accumulation**: Simulate larger batches
- **EMA**: Exponential moving average for better generalization
- **Cosine Annealing**: Learning rate schedule with warmup
- **Gradient Clipping**: Prevent exploding gradients

### Memory Management

Intelligent memory management prevents OOM:

- Real-time memory monitoring
- Automatic garbage collection
- Dynamic batch size adjustment
- Memory profiling and snapshots
- GPU-specific optimization

## Directory Structure

```
SLvl/
├── train.py                      # Main training script
├── settings.yaml                 # Main configuration
├── data_settings_format.yaml     # Data format configuration
├── requirements.txt              # Dependencies
│
├── gpus/                         # GPU-specific optimizations
│   ├── __init__.py              # GPU detection and loader
│   ├── a100.py                  # A100 optimizations
│   ├── h100.py                  # H100 optimizations
│   ├── h200.py                  # H200 optimizations
│   ├── v100.py                  # V100 optimizations
│   ├── p100.py                  # P100 optimizations
│   ├── rtx5090.py              # RTX 5090 optimizations
│   └── generic.py               # Fallback for other GPUs
│
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   └── chess_transformer.py # Model architecture
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   └── pgn_loader.py        # PGN data loader
│   │
│   └── utils/
│       ├── __init__.py
│       ├── memory_manager.py    # Memory management
│       └── training_utils.py    # Training utilities
│
├── data/
│   └── pgn/                     # Place your PGN files here
│
├── checkpoints/                 # Model checkpoints (auto-created)
└── logs/                        # Training logs (auto-created)
```

## Training Output

During training, you'll see:

- GPU detection and configuration
- Model parameter count
- Memory usage statistics
- Training metrics (loss, accuracy)
- Checkpoints saved periodically

Example output:

```
Detected GPU: H100
Using GPU optimizer: H100
GPU Memory: 80.00 GB
Model created with 50,234,880 parameters
GPU-optimized batch size: 128

[Step 100] Loss: 2.3456, Policy: 2.1234, Value: 0.2222, Accuracy: 0.342
[Step 100] Memory - Allocated: 12.34GB, Utilization: 15.4%
```

## Advanced Usage

### Custom GPU Configuration

Override GPU settings in `settings.yaml`:

```yaml
gpu:
  force_gpu_type: 'H100'  # Force specific GPU type
  max_split_size_mb: 1024
  enable_compile: true    # Use torch.compile
```

### Resume Training

```bash
python train.py --resume checkpoints/checkpoint_step_50000.pt
```

### Gradient Accumulation

Simulate larger batches:

```yaml
training:
  batch_size: 32
  gradient_accumulation_steps: 8  # Effective batch = 32 * 8 = 256
```

## Performance Tips

1. **Use appropriate ELO filtering**: Higher ELO = better quality games
2. **Enable mixed precision**: Automatic on supported GPUs
3. **Adjust batch size**: System auto-calculates optimal size
4. **Use gradient checkpointing**: For large models on smaller GPUs
5. **Monitor memory**: Check logs for memory statistics

## Troubleshooting

### Out of Memory

If you encounter OOM despite protections:

1. Reduce `batch_size` in settings.yaml
2. Enable `gradient_checkpointing`
3. Reduce model size (`dim`, `depth`)
4. Increase `gradient_accumulation_steps`

### Slow Training

1. Check `num_workers` in dataloader settings
2. Enable `torch.compile` (experimental)
3. Verify GPU utilization
4. Use larger batch size if memory allows

### Data Loading Issues

1. Check PGN file format
2. Verify tag mappings in data_settings_format.yaml
3. Enable `auto_detection` for tag names
4. Check file encoding settings

## Citation

If you use this code, please cite:

```bibtex
@software{slvl_chess_training,
  title={SLvl: Advanced Chess Supervised Learning System},
  author={Your Name},
  year={2025}
}
```

## License

MIT License - See LICENSE file for details

## Acknowledgments

- Chess library: [python-chess](https://python-chess.readthedocs.io/)
- Inspired by: AlphaZero, Leela Chess Zero, Stockfish NNUE
