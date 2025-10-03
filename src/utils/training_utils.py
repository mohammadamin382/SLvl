"""
Advanced Training Utilities
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import logging
import time
from pathlib import Path
import json
import numpy as np


logger = logging.getLogger(__name__)


class EMAModel:
    """
    Exponential Moving Average of model parameters
    Improves model generalization
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """Update EMA parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """Apply EMA parameters to model (for evaluation)"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """Restore original parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


class LearningRateScheduler:
    """
    Advanced learning rate scheduling
    Combines warmup + cosine decay with restarts
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-6,
        num_cycles: float = 0.5
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.num_cycles = num_cycles
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

        self.step_count = 0

    def step(self):
        """Update learning rate"""
        self.step_count += 1

        if self.step_count < self.warmup_steps:
            # Linear warmup
            lr_scale = self.step_count / self.warmup_steps
        else:
            # Cosine decay
            progress = (self.step_count - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr_scale = max(0.0, 0.5 * (1.0 + np.cos(np.pi * self.num_cycles * 2.0 * progress)))

        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group['lr'] = self.min_lr + (base_lr - self.min_lr) * lr_scale

    def get_last_lr(self):
        """Get current learning rate"""
        return [group['lr'] for group in self.optimizer.param_groups]


class MetricsTracker:
    """Track and log training metrics"""

    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.metrics = {
            'train_loss': [],
            'policy_loss': [],
            'value_loss': [],
            'policy_accuracy': [],
            'learning_rate': [],
            'step': [],
            'time': [],
        }

        self.step_count = 0
        self.start_time = time.time()

    def update(self, metrics: Dict[str, float]):
        """Update metrics"""
        self.step_count += 1
        self.metrics['step'].append(self.step_count)
        self.metrics['time'].append(time.time() - self.start_time)

        for key, value in metrics.items():
            if key in self.metrics:
                self.metrics[key].append(value)

    def get_average(self, key: str, last_n: int = 100) -> float:
        """Get average of last N values"""
        if key not in self.metrics or len(self.metrics[key]) == 0:
            return 0.0

        values = self.metrics[key][-last_n:]
        return sum(values) / len(values)

    def save(self):
        """Save metrics to file"""
        metrics_file = self.log_dir / 'metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)

    def log_summary(self, prefix: str = ""):
        """Log summary of recent metrics"""
        avg_loss = self.get_average('train_loss', 100)
        avg_policy_loss = self.get_average('policy_loss', 100)
        avg_value_loss = self.get_average('value_loss', 100)
        avg_policy_acc = self.get_average('policy_accuracy', 100)

        logger.info(
            f"{prefix}Step {self.step_count} - "
            f"Loss: {avg_loss:.4f}, "
            f"Policy: {avg_policy_loss:.4f}, "
            f"Value: {avg_value_loss:.4f}, "
            f"Accuracy: {avg_policy_acc:.3f}"
        )


class CheckpointManager:
    """Manage model checkpoints"""

    def __init__(self, checkpoint_dir: str, keep_last_n: int = 5):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n

        self.checkpoints = []

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        step: int,
        metrics: Dict[str, float],
        ema_model: Optional[EMAModel] = None
    ):
        """Save a checkpoint"""
        checkpoint_path = self.checkpoint_dir / f'checkpoint_step_{step}.pt'

        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.__dict__ if scheduler else None,
            'step': step,
            'metrics': metrics,
        }

        if ema_model is not None:
            checkpoint['ema_shadow'] = ema_model.shadow

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

        # Track checkpoints
        self.checkpoints.append((step, checkpoint_path))

        # Remove old checkpoints
        if len(self.checkpoints) > self.keep_last_n:
            old_step, old_path = self.checkpoints.pop(0)
            if old_path.exists():
                old_path.unlink()
                logger.info(f"Removed old checkpoint: {old_path}")

    def load_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        checkpoint_path: Optional[str] = None
    ) -> int:
        """Load a checkpoint"""
        if checkpoint_path is None:
            # Find latest checkpoint
            checkpoints = sorted(self.checkpoint_dir.glob('checkpoint_step_*.pt'))
            if not checkpoints:
                logger.warning("No checkpoints found")
                return 0
            checkpoint_path = checkpoints[-1]

        logger.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.__dict__.update(checkpoint['scheduler_state_dict'])

        step = checkpoint.get('step', 0)
        logger.info(f"Checkpoint loaded from step {step}")

        return step


class GradientClipper:
    """Advanced gradient clipping"""

    def __init__(self, max_grad_norm: float = 1.0, clip_mode: str = 'norm'):
        self.max_grad_norm = max_grad_norm
        self.clip_mode = clip_mode

    def clip_gradients(self, model: nn.Module) -> float:
        """Clip gradients and return total norm"""
        if self.clip_mode == 'norm':
            total_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                self.max_grad_norm
            )
        elif self.clip_mode == 'value':
            total_norm = torch.nn.utils.clip_grad_value_(
                model.parameters(),
                self.max_grad_norm
            )
        else:
            # No clipping
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5

        return total_norm
