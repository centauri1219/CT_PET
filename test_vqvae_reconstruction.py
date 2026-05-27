"""
Test VQ-VAE Reconstruction Quality
This script tests the reconstruction quality of the VQ-VAE by passing real PET images
through Encoder -> Quantizer -> Decoder (no translator involved).

Uses PET images from the processed_data folder:
    data_root/
        train/
            A/  (CT .npy files)
            B/  (PET .npy files)  <-- PET images loaded from here
        val/
            A/
            B/
        test/
            A/
            B/
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

from models.vqvae2 import VQVAE2
from datasets.ct_pet_dataset_npy import CTPETDatasetNPY
from utils.metrics import calculate_psnr, calculate_ssim, calculate_mae, calculate_lpips


def load_vqvae(checkpoint_path, device):
    """Load the trained VQ-VAE-2 model."""
    print(f"Loading VQ-VAE-2 from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model configuration from checkpoint
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        # Default config for VQVAE2 - adjust if needed
        config = {
            'in_channels': 1,
            'base_channels': 128,
            'num_res_blocks': 2,
            'codebook_size': 512,
            'embedding_dim': 64,
            'commitment_cost': 0.25
        }
    
    # Create and load model
    model = VQVAE2(
        in_channels=config.get('in_channels', 1),
        base_channels=config.get('base_channels', 128),
        num_res_blocks=config.get('num_res_blocks', 2),
        codebook_size=config.get('codebook_size', 512),
        embedding_dim=config.get('embedding_dim', 64),
        commitment_cost=config.get('commitment_cost', 0.25)
    )
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    print("VQ-VAE-2 loaded successfully")
    return model, config


def save_comparison_image(original, reconstructed, save_path, title='VQ-VAE Reconstruction'):
    """Save a side-by-side comparison of original and reconstructed images."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Convert to numpy and remove batch/channel dims
    orig_np = original.squeeze().cpu().numpy()
    recon_np = reconstructed.squeeze().cpu().numpy()
    
    axes[0].imshow(orig_np, cmap='hot')
    axes[0].set_title('Original PET')
    axes[0].axis('off')
    
    axes[1].imshow(recon_np, cmap='hot')
    axes[1].set_title('VQ-VAE Reconstructed')
    axes[1].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def test_vqvae_reconstruction(model, dataloader, device, output_dir, max_samples=None, num_save_images=15):
    """
    Test VQ-VAE reconstruction on real PET images.
    
    Args:
        model: Trained VQ-VAE model
        dataloader: DataLoader with PET images (returns dict with 'ct', 'pet', 'patient_id')
        device: torch device
        output_dir: Directory to save results
        max_samples: Maximum number of samples to test (None for all)
        num_save_images: Number of random comparison images to save
    """
    import random
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'overall': {'psnr': [], 'ssim': [], 'mae': [], 'lpips': []},
        'per_patient': {}
    }
    
    # Store all images for random selection later
    all_comparisons = []
    
    print("\nTesting VQ-VAE-2 reconstruction quality...")
    print("This tests: Real PET -> Encoder -> Quantizer -> Decoder -> Reconstructed PET")
    print("=" * 60)
    
    sample_count = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
            if max_samples and sample_count >= max_samples:
                break
            
            # Unpack batch - CTPETDatasetNPY returns dict with 'ct', 'pet', 'patient_id'
            pet_images = batch['pet'].to(device)
            patient_ids = batch['patient_id']  # List of patient IDs
            
            # Pass through VQ-VAE-2: Encoder -> Quantizer -> Decoder
            # VQVAE2 returns a dict with 'recon', 'vq_loss', etc.
            output = model(pet_images)
            if isinstance(output, dict):
                reconstructed = output['recon']
            elif isinstance(output, tuple):
                reconstructed = output[0]
            else:
                reconstructed = output
            
            # Move to CPU for metric calculation
            pet_images_cpu = pet_images.cpu()
            reconstructed_cpu = reconstructed.cpu()
            
            # Calculate metrics for each image in batch
            batch_size = pet_images.shape[0]
            for i in range(batch_size):
                if max_samples and sample_count >= max_samples:
                    break
                
                original = pet_images_cpu[i:i+1]
                recon = reconstructed_cpu[i:i+1]
                
                # Calculate metrics
                psnr = calculate_psnr(original, recon)
                ssim = calculate_ssim(original, recon)
                mae = calculate_mae(original, recon)
                lpips_val = calculate_lpips(original, recon)
                
                # Store results
                results['overall']['psnr'].append(psnr)
                results['overall']['ssim'].append(ssim)
                results['overall']['mae'].append(mae)
                results['overall']['lpips'].append(lpips_val)
                
                # Per-patient results
                patient_id = patient_ids[i]
                if patient_id not in results['per_patient']:
                    results['per_patient'][patient_id] = {
                        'psnr': [], 'ssim': [], 'mae': [], 'lpips': [], 'count': 0
                    }
                
                results['per_patient'][patient_id]['psnr'].append(psnr)
                results['per_patient'][patient_id]['ssim'].append(ssim)
                results['per_patient'][patient_id]['mae'].append(mae)
                results['per_patient'][patient_id]['lpips'].append(lpips_val)
                results['per_patient'][patient_id]['count'] += 1
                
                # Store for random selection later
                all_comparisons.append({
                    'original': original.clone(),
                    'recon': recon.clone(),
                    'patient_id': patient_id,
                    'psnr': psnr,
                    'ssim': ssim,
                    'idx': sample_count
                })
                
                sample_count += 1
    
    # Randomly select images to save
    print(f"\nSaving {min(num_save_images, len(all_comparisons))} random comparison images...")
    if len(all_comparisons) > 0:
        random.seed(42)  # For reproducibility
        selected_samples = random.sample(all_comparisons, min(num_save_images, len(all_comparisons)))
        
        for i, sample in enumerate(selected_samples):
            save_comparison_image(
                sample['original'], sample['recon'], 
                output_dir / f"reconstruction_{i:03d}_{sample['patient_id']}.png",
                title=f"VQ-VAE Recon - {sample['patient_id']} (PSNR: {sample['psnr']:.2f}, SSIM: {sample['ssim']:.3f})"
            )
    
    # Calculate summary statistics
    print("\n" + "=" * 60)
    print("VQ-VAE-2 RECONSTRUCTION TEST RESULTS")
    print("=" * 60)
    print(f"\nTotal images processed: {sample_count}")
    print(f"\nOverall Average Metrics:")
    print("-" * 60)
    print(f"PSNR:  {np.mean(results['overall']['psnr']):.4f} ± {np.std(results['overall']['psnr']):.4f} dB")
    print(f"SSIM:  {np.mean(results['overall']['ssim']):.4f} ± {np.std(results['overall']['ssim']):.4f}")
    print(f"MAE:   {np.mean(results['overall']['mae']):.4f} ± {np.std(results['overall']['mae']):.4f}")
    print(f"LPIPS: {np.mean(results['overall']['lpips']):.4f} ± {np.std(results['overall']['lpips']):.4f}")
    
    # Save detailed results
    summary = {
        'total_images': sample_count,
        'overall_metrics': {
            'psnr': {
                'mean': float(np.mean(results['overall']['psnr'])),
                'std': float(np.std(results['overall']['psnr'])),
                'min': float(np.min(results['overall']['psnr'])),
                'max': float(np.max(results['overall']['psnr']))
            },
            'ssim': {
                'mean': float(np.mean(results['overall']['ssim'])),
                'std': float(np.std(results['overall']['ssim'])),
                'min': float(np.min(results['overall']['ssim'])),
                'max': float(np.max(results['overall']['ssim']))
            },
            'mae': {
                'mean': float(np.mean(results['overall']['mae'])),
                'std': float(np.std(results['overall']['mae'])),
                'min': float(np.min(results['overall']['mae'])),
                'max': float(np.max(results['overall']['mae']))
            },
            'lpips': {
                'mean': float(np.mean(results['overall']['lpips'])),
                'std': float(np.std(results['overall']['lpips'])),
                'min': float(np.min(results['overall']['lpips'])),
                'max': float(np.max(results['overall']['lpips']))
            }
        },
        'per_patient_metrics': {}
    }
    
    # Per-patient summary
    print("\n" + "=" * 60)
    print("PER-PATIENT RESULTS:")
    print("=" * 60)
    for patient_id, metrics in results['per_patient'].items():
        print(f"\n{patient_id} ({metrics['count']} images):")
        print(f"  PSNR:  {np.mean(metrics['psnr']):.4f} dB")
        print(f"  SSIM:  {np.mean(metrics['ssim']):.4f}")
        print(f"  MAE:   {np.mean(metrics['mae']):.4f}")
        print(f"  LPIPS: {np.mean(metrics['lpips']):.4f}")
        
        summary['per_patient_metrics'][patient_id] = {
            'count': metrics['count'],
            'psnr': float(np.mean(metrics['psnr'])),
            'ssim': float(np.mean(metrics['ssim'])),
            'mae': float(np.mean(metrics['mae'])),
            'lpips': float(np.mean(metrics['lpips']))
        }
    
    # Save JSON results
    with open(output_dir / 'vqvae_reconstruction_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save text summary
    with open(output_dir / 'vqvae_reconstruction_summary.txt', 'w') as f:
        f.write("VQ-VAE Reconstruction Test Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total images processed: {sample_count}\n\n")
        f.write("Overall Average Metrics:\n")
        f.write("-" * 60 + "\n")
        f.write(f"PSNR:  {np.mean(results['overall']['psnr']):.4f} ± {np.std(results['overall']['psnr']):.4f} dB\n")
        f.write(f"SSIM:  {np.mean(results['overall']['ssim']):.4f} ± {np.std(results['overall']['ssim']):.4f}\n")
        f.write(f"MAE:   {np.mean(results['overall']['mae']):.4f} ± {np.std(results['overall']['mae']):.4f}\n")
        f.write(f"LPIPS: {np.mean(results['overall']['lpips']):.4f} ± {np.std(results['overall']['lpips']):.4f}\n")
        f.write("\n" + "=" * 60 + "\n")
        f.write("Interpretation:\n")
        f.write("-" * 60 + "\n")
        f.write("If VQ-VAE reconstruction metrics are MUCH better than translation metrics,\n")
        f.write("then the problem is likely in the Translator (CT -> Latent mapping).\n\n")
        f.write("If VQ-VAE reconstruction metrics are similar to translation metrics,\n")
        f.write("then the VQ-VAE itself may need improvement (more capacity, better training).\n")
    
    print(f"\nResults saved to {output_dir}")
    print("=" * 60)
    
    return summary


def main():
    parser = argparse.ArgumentParser(description='Test VQ-VAE reconstruction quality')
    parser.add_argument('--vqvae_checkpoint', type=str, required=True,
                        help='Path to VQ-VAE checkpoint')
    parser.add_argument('--data_path', type=str, 
                        default='/scratch/b24cs1085/CPDM/processed_data',
                        help='Path to dataset')
    parser.add_argument('--output_dir', type=str,
                        default='./outputs/vqvae_reconstruction_test',
                        help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for testing')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to test (default: all)')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to use')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load VQ-VAE model
    vqvae_model, config = load_vqvae(args.vqvae_checkpoint, device)
    
    # Load dataset - using CTPETDatasetNPY which loads from:
    # data_path/split/B/ for PET images (as .npy files)
    print(f"\nLoading PET images from {args.data_path}/{args.split}/B/")
    dataset = CTPETDatasetNPY(
        data_root=args.data_path,
        split=args.split,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"Dataset size: {len(dataset)} paired CT-PET slices")
    print(f"Testing VQ-VAE reconstruction using real PET images\n")
    
    # Run reconstruction test
    results = test_vqvae_reconstruction(
        vqvae_model,
        dataloader,
        device,
        args.output_dir,
        max_samples=args.max_samples
    )
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Compare these metrics with your CT->PET translation metrics")
    print("2. If VQ-VAE reconstruction is much better, focus on improving the Translator")
    print("3. If VQ-VAE reconstruction is similar, consider:")
    print("   - Training VQ-VAE longer")
    print("   - Increasing VQ-VAE capacity")
    print("   - Adjusting VQ-VAE hyperparameters")


if __name__ == '__main__':
    main()
