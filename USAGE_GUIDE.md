# SLvl - Usage Guide

Complete guide for using the Chess Supervised Learning system.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Configuration](#configuration)
4. [Running Training](#running-training)
5. [Advanced Usage](#advanced-usage)
6. [Troubleshooting](#troubleshooting)

## Installation

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- `torch>=2.1.0` - PyTorch with CUDA support
- `python-chess>=1.999` - Chess library
- `pyyaml>=6.0` - YAML configuration
- `numpy>=1.24.0` - Numerical computing
- `psutil>=5.9.0` - System monitoring

### Step 2: Verify Installation

```bash
python validate_setup.py
```

This will check:
- All dependencies installed
- CUDA availability and GPU detection
- All modules can be imported
- Configuration files present

## Quick Start

### 1. Prepare Your Data

Create data directory and add PGN files:

```bash
mkdir -p data/pgn
# Copy your .pgn files to data/pgn/
cp /path/to/your/games.pgn data/pgn/
```

### 2. Configure Training

**settings.yaml** - Main configuration:

```yaml
# Minimal configuration
model:
  dim: 512         # Model size
  depth: 12        # Number of layers
  num_heads: 8     # Attention heads

data:
  min_elo: 2000    # Filter games by ELO
  batch_size: 64   # Will be auto-adjusted

training:
  total_steps: 100000
  learning_rate: 3.0e-4
```

**data_settings_format.yaml** - Data format:

```yaml
# Auto-detection enabled by default
auto_detection:
  enabled: true

quality_filters:
  require_elo: true
  required_tags:
    - 'WhiteElo'
    - 'BlackElo'
    - 'Result'
```

### 3. Dry Run (IMPORTANT!)

Always test with dry run first:

```bash
python train.py --dry-run
```

This will:
- ✓ Load configurations
- ✓ Detect GPU and apply optimizations
- ✓ Initialize model (you'll see parameter count)
- ✓ Load sample data
- ✓ Run 5 training steps
- ✓ Verify memory management
- ✓ Check for syntax errors

Expected output:
```
================================================================================
DRY RUN MODE - Quick validation only
================================================================================
Detected GPU: A100
Using GPU optimizer: A100
GPU Memory: 40.00 GB
Model created with 50,234,880 parameters
DRY RUN: Limited to 1 file, 10 games
[Step 1] Loss: 8.4523, Policy: 8.1234, Value: 0.3289, Accuracy: 0.001
[Step 5] Loss: 7.8234, Policy: 7.5123, Value: 0.3111, Accuracy: 0.023
Training Complete!
```

### 4. Start Full Training

```bash
python train.py
```

Monitor the output:
```
[Step 100] Loss: 2.3456, Policy: 2.1234, Value: 0.2222, Accuracy: 0.342
[Step 100] Memory - Allocated: 12.34GB, Utilization: 30.8%
```

## Configuration

### Model Configuration

```yaml
model:
  # Model dimension (embedding size)
  # Larger = more capacity but slower
  # Must be divisible by num_heads
  dim: 512          # Options: 256, 512, 768, 1024

  # Transformer depth (number of layers)
  # Deeper = more capacity but slower
  depth: 12         # Options: 6, 12, 18, 24

  # Number of attention heads
  # More heads = more parallel attention
  num_heads: 8      # Options: 4, 8, 16

  # Dropout for regularization
  dropout: 0.1      # Range: 0.0 - 0.3
```

**Model Size Examples:**

| Config | Dim | Depth | Heads | Parameters | GPU Memory |
|--------|-----|-------|-------|------------|------------|
| Small  | 256 | 6     | 4     | ~12M       | ~4GB       |
| Medium | 512 | 12    | 8     | ~50M       | ~12GB      |
| Large  | 768 | 18    | 12    | ~150M      | ~30GB      |
| XLarge | 1024| 24    | 16    | ~350M      | ~60GB      |

### Data Configuration

```yaml
data:
  # PGN file location
  pgn_directory: "data/pgn"

  # Batch size (auto-adjusted by GPU optimizer)
  batch_size: 64

  # ELO filtering
  min_elo: 2000     # null = no minimum
  max_elo: null     # null = no maximum

  # Game quality
  min_moves: 10     # Skip very short games
  max_moves: 300    # Skip very long games

  # Result filtering
  result_filter: ['1-0', '0-1', '1/2-1/2']  # Exclude unfinished
```

**Data Quality Recommendations:**

| Use Case | min_elo | max_elo | Description |
|----------|---------|---------|-------------|
| General  | 1800    | null    | Amateur to master games |
| Strong   | 2200    | null    | Strong player games |
| Elite    | 2500    | null    | Super-GM level |
| Specific | 2000    | 2400    | Narrow skill range |

### Training Configuration

```yaml
training:
  # Total training steps
  total_steps: 100000

  # Learning rate
  learning_rate: 3.0e-4

  # Warmup steps (gradual LR increase)
  warmup_steps: 2000

  # Gradient accumulation
  # Effective batch = batch_size * accumulation_steps
  gradient_accumulation_steps: 4

  # Gradient clipping (prevent explosions)
  max_grad_norm: 1.0

  # Loss weights
  policy_weight: 1.0   # Move prediction
  value_weight: 0.5    # Position evaluation

  # Exponential Moving Average
  use_ema: true

  # Checkpointing
  checkpoint_interval: 5000
  keep_checkpoints: 5
```

## Running Training

### Standard Training

```bash
# Default configuration
python train.py

# Custom config file
python train.py --config my_settings.yaml
```

### Resume Training

```bash
# Resume from latest checkpoint
python train.py --resume

# Resume from specific checkpoint
python train.py --resume checkpoints/checkpoint_step_50000.pt
```

### Distributed Training (Multi-GPU)

```bash
# Coming soon
```

## Advanced Usage

### GPU-Specific Optimization

The system automatically detects your GPU and applies optimal settings:

**A100 (40GB/80GB)**
- Precision: TF32 + BF16
- Batch size: 64-128
- Features: 3rd gen Tensor Cores
- Recommended: Large models (512-1024 dim)

**H100 (80GB)**
- Precision: FP8 + BF16
- Batch size: 128-256
- Features: 4th gen Tensor Cores, Flash Attention
- Recommended: XLarge models (1024 dim)

**H200 (141GB)**
- Precision: FP8 + BF16
- Batch size: 256-512
- Features: Massive memory, HBM3
- Recommended: Largest models possible

**V100 (16GB/32GB)**
- Precision: FP16
- Batch size: 24-48
- Features: 1st gen Tensor Cores
- Recommended: Small-Medium models (256-512 dim)

**P100 (12GB/16GB)**
- Precision: FP16
- Batch size: 16-32
- Features: No Tensor Cores
- Recommended: Small models (256 dim)

**RTX 5090 (32GB)**
- Precision: FP8 + BF16
- Batch size: 64-128
- Features: 5th gen Tensor Cores
- Recommended: Large models

### Memory Optimization

If you encounter memory issues:

**1. Reduce Batch Size**
```yaml
data:
  batch_size: 32  # or 16
```

**2. Enable Gradient Checkpointing**
```yaml
# Automatically enabled on GPUs with less memory
# Trades compute for memory
```

**3. Use Gradient Accumulation**
```yaml
training:
  batch_size: 16
  gradient_accumulation_steps: 8
  # Effective batch = 16 * 8 = 128
```

**4. Reduce Model Size**
```yaml
model:
  dim: 256   # Instead of 512
  depth: 6   # Instead of 12
```

### Custom Data Filtering

**Example 1: Opening-focused training**
```yaml
# data_settings_format.yaml
move_filters:
  max_move_for_training: 20  # Only first 20 moves
```

**Example 2: Endgame-focused training**
```yaml
move_filters:
  min_move_for_training: 40  # Only moves after 40
```

**Example 3: Blitz games only**
```yaml
data:
  time_control_filter: "blitz"
```

### Monitoring Training

**1. Watch logs**
```bash
tail -f training.log
```

**2. Monitor GPU**
```bash
watch -n 1 nvidia-smi
```

**3. View metrics**
```bash
cat logs/metrics.json | jq .
```

## Troubleshooting

### Problem: Out of Memory (OOM)

**Solutions:**
1. Reduce batch size in settings.yaml
2. Reduce model size (dim, depth)
3. Increase gradient_accumulation_steps
4. Close other GPU programs

**Check memory usage:**
```python
# In training output:
[Step 100] Memory - Allocated: 12.34GB, Utilization: 85.2%
```

### Problem: Slow Training

**Solutions:**
1. Check GPU utilization (`nvidia-smi`)
2. Increase num_workers for data loading
3. Reduce logging frequency
4. Use larger batch size if memory allows

**Check data loading:**
```yaml
# In settings.yaml
advanced:
  num_workers: 8  # Increase for faster data loading
```

### Problem: Poor Accuracy

**Solutions:**
1. Increase model size
2. Train longer (more steps)
3. Use higher quality data (higher min_elo)
4. Reduce learning rate
5. Check data quality filters

### Problem: Training Diverges (Loss → NaN)

**Solutions:**
1. Reduce learning rate
2. Increase warmup steps
3. Check gradient clipping
4. Verify data quality

**Fix:**
```yaml
training:
  learning_rate: 1.0e-4  # Reduce from 3.0e-4
  warmup_steps: 5000     # Increase from 2000
  max_grad_norm: 0.5     # Reduce from 1.0
```

### Problem: No PGN Files Found

**Error:**
```
ValueError: No PGN files found in data/pgn
```

**Solution:**
```bash
# Check directory exists
ls -la data/pgn/

# Verify files have .pgn extension
mv games.txt games.pgn

# Check path in settings.yaml
data:
  pgn_directory: "data/pgn"  # Correct path
```

### Problem: Invalid PGN Format

**Solutions:**
1. Enable auto-detection
2. Check tag mappings
3. Verify file encoding

**Fix:**
```yaml
# data_settings_format.yaml
auto_detection:
  enabled: true

encoding:
  file_encoding: 'utf-8'
  encoding_errors: 'ignore'
```

## Performance Tips

1. **Use Dry Run First**: Always test with `--dry-run`
2. **Start Small**: Begin with small model, increase if needed
3. **Monitor Memory**: Watch for OOM warnings
4. **Quality > Quantity**: Higher ELO games train better
5. **Checkpoints**: Save frequently, disk is cheap
6. **Batch Size**: Let GPU optimizer choose optimal size
7. **Learning Rate**: Use default unless you know better
8. **Gradient Accumulation**: Simulate larger batches

## Best Practices

### For Beginners

1. Use default settings
2. Start with min_elo=2000
3. Run dry-run first
4. Use medium model (512 dim, 12 depth)
5. Monitor for first 1000 steps

### For Advanced Users

1. Tune learning rate
2. Experiment with model size
3. Use custom data filters
4. Enable torch.compile
5. Multi-GPU training

### For Production

1. Use EMA for final model
2. Save all checkpoints
3. Log to wandb/tensorboard
4. Validate on separate data
5. Multiple training runs

## Next Steps

1. ✓ Validate setup: `python validate_setup.py`
2. ✓ Dry run: `python train.py --dry-run`
3. ✓ Start training: `python train.py`
4. Monitor training progress
5. Evaluate checkpoints
6. Deploy best model

## Support

For issues:
1. Check this guide
2. Read error messages carefully
3. Verify configuration files
4. Check GPU memory and utilization
5. Review training.log

Happy training! 🚀
