#!/usr/bin/env python3
"""
Advanced Chess Supervised Learning Training Script

Features:
- DRY RUN mode for quick validation
- GPU-specific optimizations with auto-detection
- OOM prevention with intelligent memory management
- Configurable via settings.yaml
- State-of-the-art architecture and training techniques
"""

import os
import sys
import argparse
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models import ChessTransformer
from src.data import PGNDataset, GameFilter, MoveConverter, collate_fn
from src.utils import (
    MemoryManager, GradientAccumulator, MixedPrecisionManager,
    EMAModel, LearningRateScheduler, MetricsTracker,
    CheckpointManager, GradientClipper
)
from gpus import get_gpu_optimizer, detect_gpu

#batch_size force
force_batch = True
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)


class ChessTrainer:
    """Advanced Chess Model Trainer"""

    def __init__(self, config: Dict[str, Any], dry_run: bool = False, resume_checkpoint: Optional[str] = None):
        self.config = config
        self.dry_run = dry_run
        self.resume_checkpoint = resume_checkpoint

        # Set random seed for reproducibility
        self.set_seed()

        # Setup device and GPU optimizer
        self.setup_device()

        # Load data configuration
        self.load_data_config()

        # Initialize components
        self.move_converter = MoveConverter()
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.memory_manager = None
        self.precision_manager = None
        self.metrics_tracker = None
        self.checkpoint_manager = None

        logger.info(f"Initialized trainer (DRY_RUN={dry_run})")

    def set_seed(self):
        """Set random seed for reproducibility"""
        seed = self.config.get('advanced', {}).get('seed')
        if seed is not None:
            import random
            import numpy as np

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            # Make cudnn deterministic if requested
            if self.config.get('advanced', {}).get('deterministic', False):
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

            logger.info(f"Random seed set to: {seed}")

    def setup_device(self):
        """Setup device and GPU-specific optimizations"""
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, using CPU")
            self.device = torch.device('cpu')
            self.gpu_optimizer = None
            return

        self.device = torch.device('cuda:0')

        # Detect and configure GPU
        gpu_type = detect_gpu()
        logger.info(f"Detected GPU: {gpu_type}")

        self.gpu_optimizer = get_gpu_optimizer()
        logger.info(f"Using GPU optimizer: {self.gpu_optimizer.name}")
        logger.info(f"GPU Memory: {self.gpu_optimizer.memory_gb:.2f} GB")

    def load_data_config(self):
        """Load data settings configuration"""
        data_config_path = Path('data_settings_format.yaml')

        if data_config_path.exists():
            with open(data_config_path, 'r') as f:
                self.data_config = yaml.safe_load(f)
            logger.info("Loaded data settings from data_settings_format.yaml")
        else:
            logger.warning("data_settings_format.yaml not found, using defaults")
            self.data_config = {}

    def create_model(self) -> nn.Module:
        """Create model with configuration"""
        model_config = self.config['model']

        # Use flash attention if supported by GPU
        use_flash = False
        if self.gpu_optimizer is not None:
            memory_config = self.gpu_optimizer.get_memory_config()
            use_flash = memory_config.get('use_flash_attention', False)

        model = ChessTransformer(
            dim=model_config.get('dim', 512),
            depth=model_config.get('depth', 12),
            num_heads=model_config.get('num_heads', 8),
            dropout=model_config.get('dropout', 0.1),
            num_moves=len(self.move_converter.move_to_idx),
            use_flash_attention=use_flash,
            use_gradient_checkpointing=False  # Will enable if needed
        )

        model = model.to(self.device)

        # Apply GPU-specific optimizations
        if self.gpu_optimizer is not None:
            model = self.gpu_optimizer.apply_optimizations(model)

        # Enable gradient checkpointing if configured
        if self.gpu_optimizer is not None:
            memory_config = self.gpu_optimizer.get_memory_config()
            if memory_config.get('gradient_checkpointing', False):
                model.enable_gradient_checkpointing()
                logger.info("Gradient checkpointing enabled")

        param_count = model.count_parameters()
        logger.info(f"Model created with {param_count:,} parameters")

        # Compile model if enabled
        if self.config.get('gpu', {}).get('enable_compile', False) and not self.dry_run:
            compile_mode = self.config.get('gpu', {}).get('compile_mode', 'default')
            logger.info(f"Compiling model with mode: {compile_mode}")
            try:
                model = torch.compile(model, mode=compile_mode)
                logger.info("Model compiled successfully")
            except Exception as e:
                logger.warning(f"Model compilation failed: {e}")

        return model

    def create_optimizer(self, model: nn.Module):
        """Create optimizer with GPU-specific settings"""
        optimizer_config = self.config['optimizer']

        lr = optimizer_config.get('learning_rate', 3e-4)
        weight_decay = optimizer_config.get('weight_decay', 0.01)

        # Get GPU-specific optimizer settings
        gpu_opt_config = {}
        if self.gpu_optimizer is not None:
            gpu_opt_config = self.gpu_optimizer.get_optimizer_config()
        # Patch for PyTorch constraint: fused and foreach cannot be True together
        if gpu_opt_config.get("fused", False) and gpu_opt_config.get("foreach", False):
            logger.warning("Both `fused` and `foreach` were True - disabling `foreach` to avoid crash.")
            gpu_opt_config["foreach"] = False
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.95),
            weight_decay=weight_decay,
            **gpu_opt_config
        )

        logger.info(f"Optimizer created: AdamW (lr={lr}, wd={weight_decay})")
        logger.info(f"GPU optimizer settings: {gpu_opt_config}")

        return optimizer

    def create_dataloader(self) -> DataLoader:
        """Create dataloader with configuration"""
        data_config = self.config['data']

        # Create game filter
        game_filter = GameFilter(
            min_elo=data_config.get('min_elo'),
            max_elo=data_config.get('max_elo'),
            min_moves=data_config.get('min_moves', 10),
            max_moves=data_config.get('max_moves', 300),
            result_filter=data_config.get('result_filter')
        )

        # Get PGN paths
        pgn_dir = Path(data_config.get('pgn_directory', 'data/pgn'))

        # Create directory if it doesn't exist
        pgn_dir.mkdir(parents=True, exist_ok=True)

        pgn_paths = list(pgn_dir.glob('*.pgn'))

        if not pgn_paths:
            raise ValueError(f"No PGN files found in {pgn_dir}. Please add .pgn files to this directory.")

        logger.info(f"Found {len(pgn_paths)} PGN files")

        # In dry run, limit to 1 file and few games
        if self.dry_run:
            pgn_paths = pgn_paths[:1]
            max_games = 10
            logger.info("DRY RUN: Limited to 1 file, 10 games")
        else:
            max_games = data_config.get('max_games')

        # Create dataset
        dataset = PGNDataset(
            pgn_paths=[str(p) for p in pgn_paths],
            data_config=self.data_config,
            game_filter=game_filter,
            move_converter=self.move_converter,
            max_games=max_games
        )

        # Get dataloader config from GPU optimizer
        dataloader_config = {'num_workers': 4, 'pin_memory': True}
        if self.gpu_optimizer is not None:
            dataloader_config = self.gpu_optimizer.get_dataloader_config()

        # Override with advanced settings if provided
        advanced_config = self.config.get('advanced', {})
        if advanced_config.get('num_workers') is not None:
            dataloader_config['num_workers'] = advanced_config['num_workers']
        if advanced_config.get('pin_memory') is not None:
            dataloader_config['pin_memory'] = advanced_config['pin_memory']

        # In dry run, use simpler settings
        if self.dry_run:
            dataloader_config = {'num_workers': 0, 'pin_memory': False}

        batch_size = data_config.get('batch_size', 32)
        if self.gpu_optimizer is not None and not self.dry_run:
            try:
                model_params = self.model.count_parameters() if self.model else 50_000_000
                sequence_length = 68
                optimal_batch = self.gpu_optimizer.get_optimal_batch_size(model_params, sequence_length)

                if isinstance(optimal_batch, int) and optimal_batch > 0:
                    if force_batch:
                        batch_size = batch_size
                    else:
                        batch_size = min(batch_size, optimal_batch)
                        logger.info(f"GPU-optimized batch size: {batch_size}")
                else:
                    logger.warning("Optimal batch size not valid, using default batch size.")
            except Exception as e:
                logger.warning(f"Error determining optimal batch size: {e} -- using default batch size.")
        elif self.dry_run:
            batch_size = 2

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            **dataloader_config
        )

        return dataloader

    def compute_loss(
        self,
        policy_logits: torch.Tensor,
        value_pred: torch.Tensor,
        target_move: torch.Tensor,
        target_value: torch.Tensor
    ):
        """Compute combined policy and value loss"""
        # Policy loss (cross entropy)
        policy_loss = F.cross_entropy(policy_logits, target_move)

        # Value loss (MSE)
        value_loss = F.mse_loss(value_pred.squeeze(-1), target_value)

        # Compute accuracy
        pred_moves = policy_logits.argmax(dim=-1)
        policy_accuracy = (pred_moves == target_move).float().mean()

        # Combined loss
        policy_weight = self.config['training'].get('policy_weight', 1.0)
        value_weight = self.config['training'].get('value_weight', 0.5)

        total_loss = policy_weight * policy_loss + value_weight * value_loss

        return total_loss, policy_loss, value_loss, policy_accuracy

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        grad_accumulator: GradientAccumulator,
        grad_clipper: GradientClipper
    ) -> Dict[str, float]:
        """Single training step"""
        # Move batch to device
        board = batch['board'].to(self.device, non_blocking=True)
        metadata = batch['metadata'].to(self.device, non_blocking=True)
        target_move = batch['move'].to(self.device, non_blocking=True)
        target_value = batch['value'].to(self.device, non_blocking=True)

        # Forward pass with mixed precision
        with self.precision_manager.get_autocast_context():
            policy_logits, value_pred = self.model(board, metadata)

            # Compute loss
            loss, policy_loss, value_loss, accuracy = self.compute_loss(
                policy_logits, value_pred, target_move, target_value
            )

            # Scale loss for gradient accumulation
            loss = loss * grad_accumulator.get_loss_scale()

        # Backward pass
        if self.precision_manager.scaler is not None:
            self.precision_manager.scaler.scale(loss).backward()
        else:
            loss.backward()

        # Optimizer step (if accumulated enough)
        if grad_accumulator.should_step():
            # Unscale gradients for clipping
            if self.precision_manager.scaler is not None:
                self.precision_manager.scaler.unscale_(self.optimizer)

            # Clip gradients
            grad_norm = grad_clipper.clip_gradients(self.model)

            # Optimizer step
            self.precision_manager.step_optimizer(self.optimizer)
            self.optimizer.zero_grad(set_to_none=True)

            # Scheduler step
            if self.scheduler is not None:
                self.scheduler.step()

        return {
            'train_loss': loss.item() / grad_accumulator.get_loss_scale(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'policy_accuracy': accuracy.item(),
        }

    def resume_from_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Resume training from a checkpoint
        
        Returns:
            Dictionary containing resumed state information
        """
        logger.info("=" * 80)
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        logger.info("=" * 80)
        
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device,weights_only=False)
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise
        
        # Extract checkpoint information
        resume_step = checkpoint.get('step', 0)
        resume_epoch = checkpoint.get('epoch', 0)
        resume_metrics = checkpoint.get('metrics', {})
        
        logger.info(f"Checkpoint was saved at step: {resume_step}")
        logger.info(f"Checkpoint was saved at epoch: {resume_epoch}")
        logger.info(f"Checkpoint metrics: {resume_metrics}")
        
        # Validate checkpoint contents
        required_keys = ['model_state_dict', 'step']
        missing_keys = [key for key in required_keys if key not in checkpoint]
        if missing_keys:
            logger.warning(f"Checkpoint missing keys: {missing_keys}")
        
        # Return checkpoint data for later use
        return {
            'checkpoint': checkpoint,
            'step': resume_step,
            'epoch': resume_epoch,
            'metrics': resume_metrics
        }

    def train(self):
        """Main training loop"""
        logger.info("=" * 80)
        if self.resume_checkpoint:
            logger.info("Resuming Training from Checkpoint")
        else:
            logger.info("Starting Training")
        logger.info("=" * 80)

        # Enable anomaly detection if requested
        if self.config.get('advanced', {}).get('detect_anomaly', False):
            torch.autograd.set_detect_anomaly(True)
            logger.info("Anomaly detection enabled")

        # Create model
        logger.info("Creating model...")
        self.model = self.create_model()

        # Create optimizer
        logger.info("Creating optimizer...")
        self.optimizer = self.create_optimizer(self.model)

        # Create dataloader
        logger.info("Creating dataloader...")
        dataloader = self.create_dataloader()

        # Training configuration
        training_config = self.config['training']
        total_steps = training_config.get('total_steps', 100000)
        if self.dry_run:
            total_steps = 5  # Just a few steps for dry run

        warmup_steps = training_config.get('warmup_steps', 1000)
        if self.dry_run:
            warmup_steps = 1

        # Create scheduler
        self.scheduler = LearningRateScheduler(
            self.optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr=training_config.get('min_lr', 1e-6)
        )

        # Handle checkpoint resuming
        resume_data = None
        start_epoch = 0
        if self.resume_checkpoint:
            try:
                resume_data = self.resume_from_checkpoint(self.resume_checkpoint)
                
                # Load model state
                self.model.load_state_dict(resume_data['checkpoint']['model_state_dict'])
                logger.info("✓ Model state loaded from checkpoint")
                
                # Load optimizer state
                if 'optimizer_state_dict' in resume_data['checkpoint']:
                    try:
                        self.optimizer.load_state_dict(resume_data['checkpoint']['optimizer_state_dict'])
                        logger.info("✓ Optimizer state loaded from checkpoint")
                    except Exception as e:
                        logger.warning(f"Failed to load optimizer state: {e}. Starting with fresh optimizer.")
                
                # Load scheduler state properly
                if 'scheduler_state_dict' in resume_data['checkpoint'] and resume_data['checkpoint']['scheduler_state_dict']:
                    try:
                        scheduler_state = resume_data['checkpoint']['scheduler_state_dict']
                        if isinstance(scheduler_state, dict):
                            for key, value in scheduler_state.items():
                                if hasattr(self.scheduler, key):
                                    setattr(self.scheduler, key, value)
                        logger.info("✓ Scheduler state loaded from checkpoint")
                    except Exception as e:
                        logger.warning(f"Failed to load scheduler state: {e}. Scheduler will continue from current state.")
                
                # Extract epoch to resume from
                start_epoch = resume_data.get('epoch', 0)
                
            except Exception as e:
                logger.error(f"Failed to resume from checkpoint: {e}")
                logger.info("Starting training from scratch...")
                resume_data = None
                start_epoch = 0

        # Create memory manager
        if self.gpu_optimizer is not None:
            memory_config = self.gpu_optimizer.get_memory_config()
        else:
            memory_config = {}

        self.memory_manager = MemoryManager(self.device, memory_config)

        # Create precision manager
        if self.gpu_optimizer is not None:
            tensor_config = self.gpu_optimizer.get_tensor_core_config()
        else:
            tensor_config = {}

        self.precision_manager = MixedPrecisionManager(
            tensor_config,
            enabled=not self.dry_run  # Disable in dry run for simplicity
        )

        # Gradient accumulation
        accumulation_steps = training_config.get('gradient_accumulation_steps', 1)
        if self.dry_run:
            accumulation_steps = 1

        grad_accumulator = GradientAccumulator(accumulation_steps)

        # Gradient clipping
        grad_clipper = GradientClipper(
            max_grad_norm=training_config.get('max_grad_norm', 1.0)
        )

        # Metrics tracker
        self.metrics_tracker = MetricsTracker(log_dir='logs')
        
        # Restore metrics if resuming
        if resume_data:
            if 'metrics_history' in resume_data['checkpoint']:
                # Restore full metrics history
                self.metrics_tracker.metrics = resume_data['checkpoint']['metrics_history']
                logger.info("✓ Metrics history restored from checkpoint")
            
            # Update step count to resume point
            resume_step = resume_data.get('step', 0)
            self.metrics_tracker.step_count = resume_step
            logger.info(f"✓ Metrics tracker initialized at step {resume_step}")

        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir='checkpoints',
            keep_last_n=training_config.get('keep_checkpoints', 5)
        )

        # EMA model (disabled in dry run)
        ema_model = None
        if not self.dry_run and training_config.get('use_ema', True):
            ema_model = EMAModel(self.model, decay=0.999)
            
            # Restore EMA state if resuming
            if resume_data and 'ema_shadow' in resume_data['checkpoint']:
                try:
                    ema_model.shadow = resume_data['checkpoint']['ema_shadow']
                    logger.info("✓ EMA state loaded from checkpoint")
                except Exception as e:
                    logger.warning(f"Failed to load EMA state: {e}. EMA will start fresh.")

        # Restore gradient accumulator state if resuming
        if resume_data and 'grad_accumulator_state' in resume_data['checkpoint']:
            try:
                grad_acc_state = resume_data['checkpoint']['grad_accumulator_state']
                grad_accumulator.accumulation_count = grad_acc_state.get('accumulation_count', 0)
                logger.info("✓ Gradient accumulator state loaded from checkpoint")
            except Exception as e:
                logger.warning(f"Failed to load gradient accumulator state: {e}")

        # Initialize step counter
        step = 0
        if resume_data:
            step = resume_data['step']
            logger.info(f"✓ Resuming from step: {step}")

        # Training loop
        logger.info(f"Starting training for {total_steps} steps (current: {step})...")
        self.memory_manager.log_memory_stats("Initial ")

        # Start from correct epoch
        epoch = start_epoch

        try:
            while step < total_steps:
                epoch += 1
                logger.info(f"Epoch {epoch} (Step {step}/{total_steps})")

                for batch in dataloader:
                    # Train step
                    metrics = self.train_step(batch, grad_accumulator, grad_clipper)

                    # Update metrics
                    metrics['learning_rate'] = self.scheduler.get_last_lr()[0]
                    self.metrics_tracker.update(metrics)

                    # Update EMA
                    if ema_model is not None:
                        ema_model.update()

                    step += 1

                    # Logging
                    if step % training_config.get('log_interval', 100) == 0:
                        self.metrics_tracker.log_summary(f"[Step {step}] ")
                        self.memory_manager.log_memory_stats(f"[Step {step}] ")

                    # Memory optimization
                    if step % 50 == 0:
                        self.memory_manager.optimize_memory()

                    # Checkpointing
                    if not self.dry_run and step % training_config.get('checkpoint_interval', 5000) == 0:
                        # Save checkpoint with all state information
                        checkpoint_extras = {
                            'epoch': epoch,
                            'metrics_history': self.metrics_tracker.metrics,
                            'grad_accumulator_state': {
                                'accumulation_count': grad_accumulator.accumulation_count
                            }
                        }
                        self.checkpoint_manager.save_checkpoint(
                            self.model,
                            self.optimizer,
                            self.scheduler,
                            step,
                            metrics,
                            ema_model,
                            extras=checkpoint_extras
                        )

                    # Check if done
                    if step >= total_steps:
                        break

                    # In dry run, break after a few steps
                    if self.dry_run and step >= 5:
                        break

        except Exception as e:
            logger.error(f"Training error: {e}", exc_info=True)
            raise

        finally:
            # Final checkpoint
            if not self.dry_run:
                logger.info("Saving final checkpoint...")
                checkpoint_extras = {
                    'epoch': epoch,
                    'metrics_history': self.metrics_tracker.metrics,
                    'grad_accumulator_state': {
                        'accumulation_count': grad_accumulator.accumulation_count
                    },
                    'training_complete': True
                }
                self.checkpoint_manager.save_checkpoint(
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    step,
                    metrics,
                    ema_model,
                    extras=checkpoint_extras
                )

            # Save metrics
            self.metrics_tracker.save()

            # Final memory stats
            self.memory_manager.log_memory_stats("Final ")

        logger.info("=" * 80)
        logger.info("Training Complete!")
        logger.info("=" * 80)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Chess Supervised Learning Training')
    parser.add_argument(
        '--config',
        type=str,
        default='settings.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run in dry-run mode (minimal execution for validation)'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint file to resume training from'
    )

    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded configuration from {config_path}")

    if args.dry_run:
        logger.info("=" * 80)
        logger.info("DRY RUN MODE - Quick validation only")
        logger.info("=" * 80)

    # Validate resume checkpoint if provided
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            logger.error(f"Checkpoint file not found: {resume_path}")
            sys.exit(1)
        logger.info(f"Will resume from checkpoint: {resume_path}")

    # Create trainer
    trainer = ChessTrainer(config, dry_run=args.dry_run, resume_checkpoint=args.resume)

    # Start training
    trainer.train()


if __name__ == '__main__':
    main()
