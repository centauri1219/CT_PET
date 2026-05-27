"""
Helper Utilities
================
General utility functions for configuration, checkpointing, and reproducibility.
"""

import os
import yaml
import torch
import random
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config.yaml
    
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], save_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save config.yaml
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def set_seed(seed: int):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Make cudnn deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Random seed set to {seed}")


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
    
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    save_path: str,
    scheduler: Optional[Any] = None,
    metrics: Optional[Dict[str, float]] = None,
    global_step: int = 0
):
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss
        save_path: Path to save checkpoint
        scheduler: Optional learning rate scheduler
        metrics: Optional metrics dictionary
        global_step: Current global training step
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'global_step': global_step,
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    if metrics is not None:
        checkpoint['metrics'] = metrics
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: str = 'cuda'
) -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: PyTorch model
        optimizer: Optional optimizer
        scheduler: Optional learning rate scheduler
        device: Device to load model to
    
    Returns:
        Dictionary with epoch, loss, and metrics
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"Checkpoint loaded from {checkpoint_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Loss: {checkpoint.get('loss', 'N/A'):.4f}")
    
    if 'metrics' in checkpoint:
        print(f"  Metrics: {checkpoint['metrics']}")
    
    return {
        'epoch': checkpoint.get('epoch', 0),
        'loss': checkpoint.get('loss', float('inf')),
        'metrics': checkpoint.get('metrics', {})
    }


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """
    Get current learning rate from optimizer.
    
    Args:
        optimizer: PyTorch optimizer
    
    Returns:
        Current learning rate
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable string.
    
    Args:
        seconds: Time in seconds
    
    Returns:
        Formatted time string (e.g., "1h 23m 45s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self, name: str = ''):
        self.name = name
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        return f"{self.name}: {self.avg:.4f}"


def create_exp_dir(base_dir: str, exp_name: str) -> Path:
    """
    Create experiment directory with subdirectories.
    
    Args:
        base_dir: Base directory for experiments
        exp_name: Experiment name
    
    Returns:
        Path to experiment directory
    """
    exp_dir = Path(base_dir) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (exp_dir / 'checkpoints').mkdir(exist_ok=True)
    (exp_dir / 'logs').mkdir(exist_ok=True)
    (exp_dir / 'images').mkdir(exist_ok=True)
    
    return exp_dir


if __name__ == "__main__":
    # Test utilities
    print("Testing utility functions...")
    
    # Test seed setting
    set_seed(42)
    
    # Test parameter counting
    model = torch.nn.Linear(100, 10)
    print(f"Model parameters: {count_parameters(model)}")
    
    # Test AverageMeter
    meter = AverageMeter('Loss')
    for i in range(10):
        meter.update(1.0 / (i + 1))
    print(meter)
    
    # Test time formatting
    print(f"Formatted time: {format_time(3725)}")
    
    print("All tests passed!")
