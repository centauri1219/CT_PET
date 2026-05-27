"""
Evaluation Script
=================
Comprehensive evaluation of the trained CT-to-PET translation model.

Metrics:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- MAE (Mean Absolute Error)

Usage:
    python evaluate.py --config configs/config.yaml --split test
"""

import os
import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from models import VQVAE2, UNetTranslator
from datasets import create_dataloaders_npy
from utils import (
    load_config, set_seed, load_checkpoint,
    MetricsTracker, save_comparison_images
)


def load_models(config, device):
    """Load trained models."""
    
    # VQ-VAE
    vqvae = VQVAE2(
        in_channels=config['vqvae']['image_channels'],
        base_channels=config['vqvae']['base_channels'],
        num_res_blocks=config['vqvae']['num_res_blocks'],
        codebook_size=config['vqvae']['codebook_size'],
        embedding_dim=config['vqvae']['embedding_dim'],
        commitment_cost=config['vqvae']['commitment_cost']
    ).to(device)
    
    load_checkpoint(config['inference']['vqvae_checkpoint'], vqvae, device=device)
    vqvae.eval()
    
    # Translator
    translator = UNetTranslator(
        in_channels=config['translator']['in_channels'],
        num_classes=config['vqvae']['codebook_size'],
        base_channels=config['translator']['base_channels'],
        dropout=config['translator']['dropout']
    ).to(device)
    
    translator_checkpoint = config['inference'].get('translator_checkpoint',
                                                     'checkpoints/translator_best.pth')
    load_checkpoint(translator_checkpoint, translator, device=device)
    translator.eval()
    
    return vqvae, translator


def evaluate(vqvae, translator, dataloader, device, save_dir=None):
    """
    Evaluate model on a dataset.
    
    Returns:
        Dictionary with metrics and per-sample results
    """
    tracker = MetricsTracker()
    per_sample_results = []
    
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc='Evaluating')):
            ct = batch['ct'].to(device)
            pet_real = batch['pet'].to(device)
            patient_ids = batch['patient_id']
            
            # Generate PET
            predictions = translator.predict_indices(ct)
            pet_pred = vqvae.decode_indices(
                predictions['top_indices'],
                predictions['bottom_indices']
            )
            
            # Calculate metrics for each sample in batch
            for i in range(ct.size(0)):
                from utils import calculate_all_metrics
                metrics = calculate_all_metrics(pet_pred[i:i+1], pet_real[i:i+1])
                
                tracker.update(pet_pred[i:i+1], pet_real[i:i+1])
                
                per_sample_results.append({
                    'patient_id': patient_ids[i],
                    'PSNR': metrics['PSNR'],
                    'SSIM': metrics['SSIM'],
                    'MAE': metrics['MAE']
                })
            
            # Save sample images
            if save_dir and batch_idx < 10:
                save_comparison_images(
                    ct[:4],
                    pet_real[:4],
                    pet_pred[:4],
                    save_dir / f'sample_{batch_idx}.png',
                    num_samples=min(4, ct.size(0)),
                    title=f'Sample {batch_idx}'
                )
    
    # Get average metrics
    avg_metrics = tracker.get_average()
    std_metrics = tracker.get_std()
    
    return {
        'average': avg_metrics,
        'std': std_metrics,
        'per_sample': per_sample_results
    }


def plot_metrics_distribution(results, save_path):
    """Plot distribution of metrics across samples."""
    df = pd.DataFrame(results['per_sample'])
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # PSNR
    axes[0].hist(df['PSNR'], bins=30, edgecolor='black', alpha=0.7)
    axes[0].axvline(results['average']['PSNR'], color='r', linestyle='--', 
                    label=f'Mean: {results["average"]["PSNR"]:.2f}')
    axes[0].set_xlabel('PSNR (dB)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('PSNR Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # SSIM
    axes[1].hist(df['SSIM'], bins=30, edgecolor='black', alpha=0.7)
    axes[1].axvline(results['average']['SSIM'], color='r', linestyle='--',
                    label=f'Mean: {results["average"]["SSIM"]:.4f}')
    axes[1].set_xlabel('SSIM')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('SSIM Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # MAE
    axes[2].hist(df['MAE'], bins=30, edgecolor='black', alpha=0.7)
    axes[2].axvline(results['average']['MAE'], color='r', linestyle='--',
                    label=f'Mean: {results["average"]["MAE"]:.4f}')
    axes[2].set_xlabel('MAE')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('MAE Distribution')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Metrics distribution saved to {save_path}")


def save_results(results, save_dir):
    """Save evaluation results to files."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save summary
    with open(save_dir / 'summary.txt', 'w') as f:
        f.write("CT to PET Translation - Evaluation Results\n")
        f.write("=" * 60 + "\n\n")
        f.write("Average Metrics:\n")
        f.write(f"  PSNR: {results['average']['PSNR']:.2f} ± {results['std']['PSNR_std']:.2f} dB\n")
        f.write(f"  SSIM: {results['average']['SSIM']:.4f} ± {results['std']['SSIM_std']:.4f}\n")
        f.write(f"  MAE:  {results['average']['MAE']:.4f} ± {results['std']['MAE_std']:.4f}\n")
        f.write(f"\nTotal samples: {len(results['per_sample'])}\n")
    
    # Save per-sample results to CSV
    df = pd.DataFrame(results['per_sample'])
    df.to_csv(save_dir / 'per_sample_results.csv', index=False)
    print(f"Per-sample results saved to {save_dir / 'per_sample_results.csv'}")
    
    # Save statistics
    stats_df = df[['PSNR', 'SSIM', 'MAE']].describe()
    stats_df.to_csv(save_dir / 'statistics.csv')
    print(f"Statistics saved to {save_dir / 'statistics.csv'}")
    
    # Plot distributions
    plot_metrics_distribution(results, save_dir / 'metrics_distribution.png')
    
    print(f"\nResults saved to {save_dir}")


def main(args):
    # Load config
    config = load_config(args.config)
    
    # Set seed
    set_seed(config['seed'])
    
    # Device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create dataloaders
    print('\n=== Loading Dataset ===')
    train_loader, val_loader, test_loader = create_dataloaders_npy(
        config
    )
    
    # Select dataloader based on split
    if args.split == 'train':
        dataloader = train_loader
    elif args.split == 'val':
        dataloader = val_loader
    else:
        dataloader = test_loader
    
    print(f"Evaluating on {args.split} set ({len(dataloader.dataset)} samples)")
    
    # Load models
    print('\n=== Loading Models ===')
    vqvae, translator = load_models(config, device)
    print('Models loaded successfully')
    
    # Evaluate
    print('\n=== Starting Evaluation ===')
    save_dir = Path(args.output_dir) / args.split
    results = evaluate(vqvae, translator, dataloader, device, save_dir=save_dir)
    
    # Print results
    print('\n=== Evaluation Results ===')
    print(f"Average PSNR: {results['average']['PSNR']:.2f} ± {results['std']['PSNR_std']:.2f} dB")
    print(f"Average SSIM: {results['average']['SSIM']:.4f} ± {results['std']['SSIM_std']:.4f}")
    print(f"Average MAE:  {results['average']['MAE']:.4f} ± {results['std']['MAE_std']:.4f}")
    
    # Save results
    save_results(results, save_dir)
    
    print('\n=== Evaluation Complete ===')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate CT to PET Translation')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'],
                       help='Dataset split to evaluate on')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    main(args)
