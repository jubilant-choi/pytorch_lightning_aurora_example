"""
Simple PyTorch Lightning Example for Intel Aurora

This is a minimal working example demonstrating how to use PyTorch Lightning
on Aurora with Intel XPUs. It uses a simple synthetic dataset and a basic
neural network for demonstration purposes.

Based on SwiFT v2 neuroimaging project patterns.
"""

import os
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from deepspeed.ops.adam import FusedAdam, DeepSpeedCPUAdam
from torch.utils.data import DataLoader, TensorDataset, random_split
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.plugins.precision import MixedPrecisionPlugin, PrecisionPlugin
from torchmetrics import Accuracy

# Import Aurora-specific utilities
from aurora_utils.ddp_intel import MPIDDPStrategy, MPIEnvironment
from aurora_utils.deepspeed_intel import XPUDeepSpeedStrategy


class SimpleClassifier(pl.LightningModule):
    """
    A simple feedforward classifier for demonstration.

    This model shows the basic structure needed for PyTorch Lightning on Aurora:
    - Standard forward/training/validation steps
    - XPU-compatible optimizer configuration
    - Logging compatible with distributed training
    """

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 256,
        output_dim: int = 10,
        learning_rate: float = 1e-3,
        optimizer: str = 'AdamW',
        weight_decay: float = 1e-1,
        lr_scaling: str = 'none',
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters() # this line saves all arguments in the above __init__ in self.hparams

        # Simple feedforward architecture
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )
        # Track metrics
        self.train_acc = Accuracy(task='multiclass', num_classes=output_dim)
        self.val_acc = Accuracy(task='multiclass', num_classes=output_dim)
        self.test_acc = Accuracy(task='multiclass', num_classes=output_dim)

    def forward(self, x):
        """Forward pass through the network."""
        return self.network(x)

    def training_step(self, batch, batch_idx):
        """Training step - called for each batch during training."""
        x, y = batch

        # Forward pass
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = self.train_acc(preds, y)

        # Log metrics
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        self.log('train_acc', acc, prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step - called for each batch during validation."""
        x, y = batch

        # Forward pass
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = self.val_acc(preds, y)

        # Log metrics
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        self.log('val_acc', acc, prog_bar=True, sync_dist=True)

        return loss

    def test_step(self, batch, batch_idx):
        """Test step - called for each batch during validation."""
        x, y = batch

        # Forward pass
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = self.test_acc(preds, y)

        # Log metrics
        self.log('test_loss', loss, prog_bar=True, sync_dist=True)
        self.log('test_acc', acc, prog_bar=True, sync_dist=True)

        return loss
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        if self.hparams.optimizer == 'AdamW':
            optim_class = torch.optim.AdamW
        elif self.hparams.optimizer == 'Adam':
            optim_class = torch.optim.Adam
        elif self.hparams.optimizer == 'SGD':
            optim_class = torch.optim.SGD
        elif self.hparams.optimizer == 'FusedAdam':
            optim_class = FusedAdam
        elif self.hparams.optimizer == 'DeepSpeedCPUAdam':
            optim_class = DeepSpeedCPUAdam

        orig_lr = self.hparams.learning_rate
        world_size = int(os.environ.get('SLURM_NTASKS', 384)) if not os.environ.get('WORLD_SIZE') else int(os.environ.get('WORLD_SIZE'))
        global_bsz = self.trainer.datamodule.hparams.batch_size * world_size
        if self.hparams.lr_scaling == 'square': lr_batchsize_multiplier = np.sqrt(global_bsz / 384.0)
        elif self.hparams.lr_scaling == 'linear': lr_batchsize_multiplier = global_bsz / 384.0
        else: lr_batchsize_multiplier = 1

        effective_lr = self.hparams.learning_rate * lr_batchsize_multiplier
        print(f"Global Bsz: {global_bsz}. LR scaling: {self.hparams.lr_scaling} ({lr_batchsize_multiplier:.2f}x). Base LR: {orig_lr:.2e}. Effective LR: {effective_lr:.2e}")
        self.hparams.learning_rate = effective_lr # Update hparams for scheduler

        optimizer = optim_class(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )

        # Optional: Add learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=10
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }


class SyntheticDataModule(pl.LightningDataModule):
    """
    A simple synthetic data module for demonstration.

    In a real application, replace this with your actual dataset.
    Shows the pattern for DataLoader configuration on Aurora.
    """

    def __init__(
        self,
        data_size: int = 10000,
        input_dim: int = 784,
        num_classes: int = 10,
        batch_size: int = 32,
        num_workers: int = 4,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters() 
        self.data_size = data_size
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        """Create train/val/test datasets."""
        # Generate synthetic data
        X = torch.randn(self.data_size, self.input_dim)
        y = torch.randint(0, self.num_classes, (self.data_size,))

        # Create dataset
        full_dataset = TensorDataset(X, y)

        # Split into train/val/test (80/10/10)
        train_size = int(0.8 * self.data_size)
        val_size = int(0.1 * self.data_size)
        test_size = self.data_size - train_size - val_size

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        """Return training dataloader with optimized settings."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,        # Important for fast host-to-device transfers
            persistent_workers=True, # Reuse workers across epochs
            prefetch_factor=2        # Prefetch 2 batches per worker
        )

    def val_dataloader(self):
        """Return validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )

    def test_dataloader(self):
        """Return test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )


def setup_aurora_environment(args, make_rank0_env=False):
    """
    Setup Aurora-specific environment for training.

    This function initializes:
    - MPI environment
    - Process group with xccl backend (https://pytorch.org/blog/pytorch-2-8-brings-native-xccl-support-to-intel-gpus-case-studies-from-argonne-national-laboratory/)
    - Strategy (DDP or DeepSpeed)
    - Precision plugin

    Returns:
        strategy: PyTorch Lightning strategy for distributed training
        env: MPI environment object
    """
    # Create MPI environment
    env = MPIEnvironment(use_only_rank0 = make_rank0_env)

    # Seed everything for reproducibility
    pl.seed_everything(args.seed)
    torch.xpu.manual_seed_all(args.seed)  # XPU-specific seeding

    # Set precision for matrix multiplication
    if "16" in str(args.precision):
        torch.set_float32_matmul_precision("medium") # https://docs.pytorch.org/docs/2.8/generated/torch.set_float32_matmul_precision.html

    # Select strategy based on args
    if 'deepspeed' in args.strategy:
        # DeepSpeed strategy for memory-efficient training
        from pytorch_lightning.plugins.precision import DeepSpeedPrecisionPlugin

        precision_plugin = DeepSpeedPrecisionPlugin(precision=args.precision)

        # Extract stage number
        stage = int(args.strategy.split('stage_')[-1][0]) if 'stage' in args.strategy else 1

        # CPU offload configuration (for Stage 2/3)
        offload_kwargs = {}
        if 'offload' in args.strategy and stage >= 2:
            offload_kwargs = {
                'offload_optimizer': True,
                'offload_parameters': True if stage == 3 else False,
            }

        strategy = XPUDeepSpeedStrategy(
            accelerator="xpu",
            cluster_environment=env,
            precision_plugin=precision_plugin,
            process_group_backend='xccl',
            stage=stage,
            logging_batch_size_per_gpu=args.batch_size,
            **offload_kwargs
        )

    elif 'ddp' in args.strategy:
        # Standard DDP strategy
        torch.distributed.init_process_group(
            "xccl",  # Use xccl backend (not NCCL!)
            init_method=f"env://",
            world_size=env.world_size(),
            rank=env.global_rank()
        )

        # Setup precision plugin
        if str(args.precision) == '32':
            precision_plugin = PrecisionPlugin()
        elif '16' in str(args.precision):
            precision_plugin = MixedPrecisionPlugin(
                precision=str(args.precision),
                device='xpu',
                scaler=None if str(args.precision) == 'bf16' else torch.amp.GradScaler(device='xpu')
            )
        else:
            raise ValueError(f"Unsupported precision: {args.precision}")

        strategy = MPIDDPStrategy(
            accelerator="xpu",
            cluster_environment=env,
            precision_plugin=precision_plugin,
            find_unused_parameters=False
        )
    else:
        raise ValueError(f"Unknown strategy: {args.strategy}")

    return strategy, env


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description='Simple PyTorch Lightning example for Aurora'
    )

    # Training arguments
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--max_epochs', type=int, default=10,
                       help='Maximum number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size per GPU')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-1,
                       help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='AdamW', choices=['AdamW', 'Adam', 'SGD', 'FusedAdam', 'DeepSpeedCPUAdam'],
                       help='Optimizer class')
    parser.add_argument('--lr_scaling', type=str, default='none', choices=['square', 'linear', 'none'],
                       help='Learning rate scaling factor')
    
    # Model arguments
    parser.add_argument('--input_dim', type=int, default=784,
                       help='Input dimension')
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='Hidden layer dimension')
    parser.add_argument('--output_dim', type=int, default=10,
                       help='Output dimension (number of classes)')

    # Data arguments
    parser.add_argument('--data_size', type=int, default=10000,
                       help='Size of synthetic dataset')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of DataLoader workers')

    # Distributed training arguments
    parser.add_argument('--num_nodes', type=int, default=1,
                       help='Number of nodes')
    parser.add_argument('--devices', type=int, default=12,
                       help='Number of devices (GPUs) per node')
    parser.add_argument('--strategy', type=str, default='ddp',
                       choices=['ddp', 'deepspeed_stage_1', 'deepspeed_stage_2', 'deepspeed_stage_3'],
                       help='Training strategy')
    parser.add_argument('--precision', type=str, default='bf16',
                       choices=['32', '16', 'bf16'],
                       help='Training precision')

    # Checkpoint and logging
    parser.add_argument('--output_dir', type=str, default='./output',
                       help='Output directory for checkpoints and logs')
    parser.add_argument('--save_every_n_epochs', type=int, default=1,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--resume_ckpt_path', type=str, default=None,
                        help='Path to resume from checkpoint')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup Aurora environment
    strategy, env = setup_aurora_environment(args)

    # Print configuration (only rank 0)
    if env.global_rank() == 0:
        print("="*80)
        print("PyTorch Lightning on Aurora - Simple Example")
        print("="*80)
        print(f"World size: {env.world_size()}")
        print(f"Num nodes: {args.num_nodes}")
        print(f"Devices per node: {args.devices}")
        print(f"Strategy: {args.strategy}")
        print(f"Precision: {args.precision}")
        print(f"Batch size per GPU: {args.batch_size}")
        print(f"Global batch size: {args.batch_size * env.world_size()}")
        print("="*80)

    # Create data module
    data_module = SyntheticDataModule(
        data_size=args.data_size,
        input_dim=args.input_dim,
        num_classes=args.output_dim,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # Create model
    model = SimpleClassifier(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        learning_rate=args.learning_rate
    )

    # Setup callbacks
    callbacks = [
        # Save checkpoints
        ModelCheckpoint(
            dirpath=args.output_dir,
            filename='checkpoint-{epoch:02d}-{val_loss:.4f}',
            monitor='val_loss',
            mode='min',
            save_top_k=3,
            save_last=True,
            every_n_epochs=args.save_every_n_epochs
        ),
        # Log learning rate
        LearningRateMonitor(logging_interval='epoch')
    ]

    # Create trainer
    trainer = pl.Trainer(
        strategy=strategy,
        devices=args.devices,
        num_nodes=args.num_nodes,
        precision=args.precision,
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        log_every_n_steps=10,
        enable_progress_bar=True if env.global_rank() == 0 else False,
        enable_model_summary=True if env.global_rank() == 0 else False,
        default_root_dir=args.output_dir
    )

    # Train the model
    trainer.fit(model, data_module, ckpt_path=args.resume_ckpt_path)

    # Test the model (https://github.com/Lightning-AI/pytorch-lightning/issues/8375#issuecomment-879678629)
    if env.global_rank() == 0:
        torch.distributed.destroy_process_group()
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        torch.distributed.init_process_group("xccl", init_method=f"env://", world_size=1, rank=0)
        strategy, env = setup_aurora_environment(args, make_rank0_env=True)
        trainer = pl.Trainer(
            strategy=strategy,
            devices=1,
            num_nodes=1,
            precision=args.precision,
            callbacks=callbacks,
        )
                    
        trainer.test(model, data_module, ckpt_path='best')

if __name__ == '__main__':
    main()