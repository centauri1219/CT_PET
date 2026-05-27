"""
Visualization Utilities
=======================
Functions for visualizing training progress and results.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Optional
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments


def denormalize(tensor: torch.Tensor, vmin: float = -1.0, vmax: float = 1.0) -> np.ndarray:
    """
    Denormalize tensor from [-1, 1] to [0, 1] for visualization.
    
    Args:
        tensor: Input tensor
        vmin: Minimum value of input range
        vmax: Maximum value of input range
    
    Returns:
        Denormalized numpy array
    """
    arr = tensor.detach().cpu().numpy()
    arr = (arr - vmin) / (vmax - vmin)
    arr = np.clip(arr, 0, 1)
    return arr


def save_comparison_images(
    ct: torch.Tensor,
    pet_real: torch.Tensor,
    pet_recon: torch.Tensor,
    save_path: str,
    num_samples: int = 4,
    title: Optional[str] = None
):
    """
    Save comparison images showing CT, Real PET, and Reconstructed PET.
    
    Args:
        ct: CT images [B, 1, H, W]
        pet_real: Real PET images [B, 1, H, W]
        pet_recon: Reconstructed PET images [B, 1, H, W]
        save_path: Path to save the figure
        num_samples: Number of samples to display
        title: Optional title for the figure
    """
    num_samples = min(num_samples, ct.size(0))
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # CT
        ct_img = denormalize(ct[i, 0])
        axes[i, 0].imshow(ct_img, cmap='gray')
        axes[i, 0].set_title('CT Input')
        axes[i, 0].axis('off')
        
        # Real PET
        pet_real_img = denormalize(pet_real[i, 0])
        axes[i, 1].imshow(pet_real_img, cmap='hot')
        axes[i, 1].set_title('Real PET')
        axes[i, 1].axis('off')
        
        # Reconstructed PET
        pet_recon_img = denormalize(pet_recon[i, 0])
        axes[i, 2].imshow(pet_recon_img, cmap='hot')
        axes[i, 2].set_title('Reconstructed PET')
        axes[i, 2].axis('off')
    
    if title:
        fig.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_vqvae_reconstruction(
    pet_real: torch.Tensor,
    pet_recon: torch.Tensor,
    save_path: str,
    num_samples: int = 4,
    title: Optional[str] = None
):
    """
    Save VQ-VAE reconstruction comparison (Real PET vs Reconstructed PET).
    
    Args:
        pet_real: Real PET images [B, 1, H, W]
        pet_recon: Reconstructed PET images [B, 1, H, W]
        save_path: Path to save the figure
        num_samples: Number of samples to display
        title: Optional title for the figure
    """
    num_samples = min(num_samples, pet_real.size(0))
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Real PET
        pet_real_img = denormalize(pet_real[i, 0])
        axes[i, 0].imshow(pet_real_img, cmap='hot')
        axes[i, 0].set_title('Real PET')
        axes[i, 0].axis('off')
        
        # Reconstructed PET
        pet_recon_img = denormalize(pet_recon[i, 0])
        axes[i, 1].imshow(pet_recon_img, cmap='hot')
        axes[i, 1].set_title('Reconstructed PET')
        axes[i, 1].axis('off')
        
        # Difference map
        diff = np.abs(pet_real_img - pet_recon_img)
        axes[i, 2].imshow(diff, cmap='viridis')
        axes[i, 2].set_title('Absolute Difference')
        axes[i, 2].axis('off')
    
    if title:
        fig.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    save_path: str,
    metrics: Optional[dict] = None,
    title: str = "Training Curves"
):
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        save_path: Path to save the figure
        metrics: Optional dict of metric lists (e.g., {'PSNR': [...], 'SSIM': [...]})
        title: Title for the plot
    """
    num_plots = 1 if metrics is None else 2
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 5))
    
    if num_plots == 1:
        axes = [axes]
    
    # Plot losses
    epochs = range(1, len(train_losses) + 1)
    axes[0].plot(epochs, train_losses, label='Train Loss', marker='o', markersize=3)
    axes[0].plot(epochs, val_losses, label='Val Loss', marker='s', markersize=3)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot metrics if provided
    if metrics is not None and num_plots == 2:
        for metric_name, metric_values in metrics.items():
            if len(metric_values) > 0:
                metric_epochs = range(1, len(metric_values) + 1)
                axes[1].plot(metric_epochs, metric_values, label=metric_name, marker='o', markersize=3)
        
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Metric Value')
        axes[1].set_title('Evaluation Metrics')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_codebook_usage(indices: torch.Tensor, codebook_size: int, save_path: str):
    """
    Visualize codebook usage distribution.
    
    Args:
        indices: Codebook indices [B, H, W]
        codebook_size: Total number of codebook entries
        save_path: Path to save the figure
    """
    indices_flat = indices.flatten().cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Histogram
    counts, bins, patches = ax.hist(indices_flat, bins=codebook_size, 
                                     range=(0, codebook_size), 
                                     edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Codebook Index')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Codebook Usage Distribution (Total entries: {codebook_size})')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add statistics
    unique_codes = len(np.unique(indices_flat))
    usage_percent = (unique_codes / codebook_size) * 100
    ax.text(0.02, 0.98, f'Unique codes used: {unique_codes}/{codebook_size} ({usage_percent:.1f}%)',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    # Test visualization functions
    ct = torch.randn(4, 1, 256, 256)
    pet_real = torch.randn(4, 1, 256, 256)
    pet_recon = pet_real + 0.1 * torch.randn_like(pet_real)
    
    # Test comparison images
    save_comparison_images(ct, pet_real, pet_recon, 'test_comparison.png', num_samples=2)
    print("Saved test_comparison.png")
    
    # Test training curves
    train_losses = [1.0, 0.8, 0.6, 0.5, 0.4]
    val_losses = [1.1, 0.85, 0.65, 0.55, 0.45]
    metrics = {
        'PSNR': [20, 22, 24, 25, 26],
        'SSIM': [0.7, 0.75, 0.8, 0.82, 0.85]
    }
    plot_training_curves(train_losses, val_losses, 'test_curves.png', metrics=metrics)
    print("Saved test_curves.png")
    
    # Test codebook usage
    indices = torch.randint(0, 512, (8, 64, 64))
    visualize_codebook_usage(indices, 512, 'test_codebook.png')
    print("Saved test_codebook.png")
