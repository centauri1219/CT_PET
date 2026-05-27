"""
Inference: Generate PET from CT
================================
This script performs CT-to-PET translation using trained models.

Pipeline:
1. Load CT scan
2. Translator predicts codebook indices
3. VQ-VAE decoder generates PET from indices

Usage:
    python inference.py --config configs/config.yaml --ct_path path/to/ct --output_dir outputs
"""

import os
import argparse
import torch
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

from models import VQVAE2, UNetTranslator
from datasets import CTPETDatasetNPY
from utils import (
    load_config, set_seed, load_checkpoint,
    save_comparison_images, calculate_all_metrics
)


def load_models(config, device):
    """Load trained VQ-VAE and Translator models."""
    
    # Load VQ-VAE
    vqvae = VQVAE2(
        in_channels=config['vqvae']['image_channels'],
        base_channels=config['vqvae']['base_channels'],
        num_res_blocks=config['vqvae']['num_res_blocks'],
        codebook_size=config['vqvae']['codebook_size'],
        embedding_dim=config['vqvae']['embedding_dim'],
        commitment_cost=config['vqvae']['commitment_cost']
    ).to(device)
    
    vqvae_checkpoint = config['inference']['vqvae_checkpoint']
    if not os.path.exists(vqvae_checkpoint):
        raise FileNotFoundError(f"VQ-VAE checkpoint not found: {vqvae_checkpoint}")
    
    load_checkpoint(vqvae_checkpoint, vqvae, device=device)
    vqvae.eval()
    
    # Load Translator
    translator = UNetTranslator(
        in_channels=config['translator']['in_channels'],
        num_classes=config['vqvae']['codebook_size'],
        base_channels=config['translator']['base_channels'],
        dropout=config['translator']['dropout']
    ).to(device)
    
    translator_checkpoint = config['inference'].get('translator_checkpoint', 
                                                     'checkpoints/translator_best.pth')
    if not os.path.exists(translator_checkpoint):
        raise FileNotFoundError(f"Translator checkpoint not found: {translator_checkpoint}")
    
    load_checkpoint(translator_checkpoint, translator, device=device)
    translator.eval()
    
    return vqvae, translator


def generate_pet_from_ct(ct_tensor, translator, vqvae, device, 
                         sampling_mode='argmax', temperature=0.9, top_k=50):
    """
    Generate PET scan from CT scan.
    
    Args:
        ct_tensor: CT image [1, 1, H, W] or [B, 1, H, W]
        translator: Trained translator model
        vqvae: Trained VQ-VAE model
        device: Device
        sampling_mode: 'argmax' (deterministic) or 'topk' (stochastic)
        temperature: Temperature for top-k sampling (0.8-0.9 is sweet spot)
        top_k: Number of candidates for sampling (50-100 is sweet spot)
    
    Returns:
        Generated PET image [B, 1, H, W]
    """
    with torch.no_grad():
        # Predict codebook indices
        predictions = translator.predict_indices(
            ct_tensor,
            sampling_mode=sampling_mode,
            temperature=temperature,
            top_k=top_k
        )
        
        # DEBUG: Count unique codes to check for mode collapse
        top_indices = predictions['top_indices']
        bottom_indices = predictions['bottom_indices']
        unique_top = torch.unique(top_indices).numel()
        unique_bottom = torch.unique(bottom_indices).numel()
        print(f"DEBUG: Unique Codes Used - Top: {unique_top}/512, Bottom: {unique_bottom}/512")
        
        # Decode indices to PET image
        pet_pred = vqvae.decode_indices(
            predictions['top_indices'],
            predictions['bottom_indices']
        )
    
    return pet_pred


def save_nifti(data, save_path, affine=None):
    """Save data as NIfTI file."""
    if affine is None:
        affine = np.eye(4)
    
    nifti_img = nib.Nifti1Image(data, affine)
    nib.save(nifti_img, save_path)
    print(f"Saved: {save_path}")


def inference_single_file(ct_path, vqvae, translator, config, device, output_dir, pet_path=None,
                          sampling_mode='argmax', temperature=1.0, top_k=50):
    """
    Perform inference on a single CT file.
    
    Args:
        ct_path: Path to CT NIfTI file
        vqvae: VQ-VAE model
        translator: Translator model
        config: Configuration dict
        device: Device
        output_dir: Output directory
        pet_path: Optional path to ground truth PET for comparison
        sampling_mode: 'argmax' (deterministic), 'sample' (softmax sampling), or 'topk' (top-k sampling)
        temperature: Temperature for sampling (lower=sharper)
        top_k: Number of top candidates for top-k sampling
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load CT
    print(f"Loading CT from {ct_path}")
    ct_img = nib.load(ct_path)
    ct_data = ct_img.get_fdata().squeeze()
    affine = ct_img.affine
    
    # Normalize CT
    dataset_helper = CTPETDataset.__new__(CTPETDataset)
    dataset_helper.ct_window = (config['data']['ct_window_min'], config['data']['ct_window_max'])
    ct_norm = dataset_helper.normalize_ct(ct_data)
    
    # Convert to tensor
    ct_tensor = torch.from_numpy(ct_norm).unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, H, W]
    
    # Generate PET with optional sampling
    print(f"Generating PET (mode={sampling_mode}, temp={temperature}, top_k={top_k})...")
    pet_pred = generate_pet_from_ct(
        ct_tensor, translator, vqvae, device,
        sampling_mode=sampling_mode,
        temperature=temperature,
        top_k=top_k
    )
    
    # Denormalize and save
    pet_pred_np = pet_pred.squeeze().cpu().numpy()
    
    # Save as NIfTI
    save_nifti(pet_pred_np, output_dir / 'predicted_pet.nii.gz', affine)
    
    # If ground truth PET is provided, compare
    if pet_path and os.path.exists(pet_path):
        print(f"Loading ground truth PET from {pet_path}")
        pet_img = nib.load(pet_path)
        pet_data = pet_img.get_fdata().squeeze()
        
        # Normalize PET
        dataset_helper.pet_log_scale = config['data']['pet_log_scale']
        dataset_helper.pet_max_log_value = config['data'].get('pet_max_log_value', 10.0)
        pet_norm = dataset_helper.normalize_pet(pet_data)
        pet_tensor = torch.from_numpy(pet_norm).unsqueeze(0).unsqueeze(0).to(device)
        
        # Calculate metrics
        metrics = calculate_all_metrics(pet_pred, pet_tensor, include_lpips=True)
        print(f"\nMetrics vs Ground Truth:")
        print(f"  PSNR: {metrics['PSNR']:.2f} dB")
        print(f"  SSIM: {metrics['SSIM']:.4f}")
        print(f"  MAE: {metrics['MAE']:.4f}")
        if 'LPIPS' in metrics:
            print(f"  LPIPS: {metrics['LPIPS']:.4f}")
        
        # Save comparison image
        save_comparison_images(
            ct_tensor,
            pet_tensor,
            pet_pred,
            output_dir / 'comparison.png',
            num_samples=1,
            title='CT to PET Translation'
        )
        
        # Save metrics
        with open(output_dir / 'metrics.txt', 'w') as f:
            f.write(f"PSNR: {metrics['PSNR']:.2f} dB\n")
            f.write(f"SSIM: {metrics['SSIM']:.4f}\n")
            f.write(f"MAE: {metrics['MAE']:.4f}\n")
            if 'LPIPS' in metrics:
                f.write(f"LPIPS: {metrics['LPIPS']:.4f}\n")
    else:
        # Save visualization without ground truth
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        ct_vis = (ct_norm + 1) / 2  # Denormalize for visualization
        pet_vis = (pet_pred_np + 1) / 2
        
        axes[0].imshow(ct_vis, cmap='gray')
        axes[0].set_title('CT Input')
        axes[0].axis('off')
        
        axes[1].imshow(pet_vis, cmap='hot')
        axes[1].set_title('Generated PET')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'result.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"\nResults saved to {output_dir}")


def inference_directory(data_dir, vqvae, translator, config, device, output_dir, 
                        num_samples=100, save_images_per_patient=5,
                        sampling_mode='argmax', temperature=0.9, top_k=50):
    """
    Perform inference on NPY slice files from a directory.
    Groups results by patient ID extracted from filenames.
    
    Args:
        data_dir: Directory containing train/val/test folders with A/ and B/ subdirs
        vqvae: VQ-VAE model
        translator: Translator model
        config: Configuration dict
        device: Device
        output_dir: Output directory
        num_samples: Number of random samples to process (default: 100, use large number for all)
        save_images_per_patient: Number of slices to save as images per patient (default: 5)
        sampling_mode: 'argmax' (deterministic) or 'topk' (stochastic with texture variety)
        temperature: Temperature for top-k sampling
            - 0.1-0.5: Conservative (smooth, like argmax)
            - 0.8-0.9: Sweet spot (sharp structure, good variance)
            - 1.0+: Noisy (may look like "snow")
        top_k: Number of candidates for top-k sampling
            - 1: Same as argmax
            - 50-100: Sweet spot (texture variety without wrong codes)
            - 512: Pure random (risky)
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Log sampling settings
    print(f"\n=== Sampling Settings ===")
    print(f"Mode: {sampling_mode}")
    if sampling_mode == 'topk':
        print(f"Temperature: {temperature}")
        print(f"Top-K: {top_k}")
    
    # Create dataset for test split
    print(f"\nLoading dataset from {data_dir}")
    test_dataset = CTPETDatasetNPY(
        data_root=str(data_dir),
        split='test',
        modality_ct=config['data']['modality_ct'],
        modality_pet=config['data']['modality_pet'],
        ct_window=(config['data']['ct_window_min'], config['data']['ct_window_max']),
        pet_log_scale=config['data']['pet_log_scale'],
        pet_max_log_value=config['data'].get('pet_max_log_value')
    )
    
    print(f"Found {len(test_dataset)} test slices")
    
    # Sample random indices if dataset is large
    if len(test_dataset) > num_samples:
        print(f"Sampling {num_samples} random slices for evaluation")
        indices = np.random.choice(len(test_dataset), num_samples, replace=False)
    else:
        indices = range(len(test_dataset))
    
    # Dictionary to group metrics by patient
    patient_metrics = {}
    
    # Process samples
    for idx in tqdm(indices, desc="Processing slices"):
        sample = test_dataset[idx]
        ct = sample['ct'].unsqueeze(0).to(device)  # [1, 1, H, W]
        pet = sample['pet'].unsqueeze(0).to(device)  # [1, 1, H, W]
        
        # Extract patient ID and verify GT matching from filenames
        # Format: "Lung_Dx-A0167_slice000.npy" or "CT_Lung_Dx-A0167_slice000.npy"
        ct_file, pet_file = test_dataset.paired_files[idx]
        ct_filename = ct_file.name
        pet_filename = pet_file.name
        
        # Verify CT and PET are from same patient/slice
        assert ct_filename == pet_filename or ct_filename == f"CT_{pet_filename}", \
            f"Mismatch: CT={ct_filename}, PET={pet_filename}"
        
        # Remove CT_ prefix if present and extract patient ID
        if ct_filename.startswith('CT_'):
            ct_filename = ct_filename[3:]
        patient_id = '_'.join(ct_filename.split('_')[:-1])  # Remove "_sliceXXX.npy"
        slice_num = ct_filename.split('_')[-1].replace('.npy', '')  # Get "sliceXXX"
        
        # Generate PET with optional sampling
        with torch.no_grad():
            pet_pred = generate_pet_from_ct(
                ct, translator, vqvae, device,
                sampling_mode=sampling_mode,
                temperature=temperature,
                top_k=top_k
            )
        
        # Calculate metrics
        metrics = calculate_all_metrics(pet_pred, pet, include_lpips=True)
        
        # Store metrics by patient
        if patient_id not in patient_metrics:
            patient_metrics[patient_id] = {
                'metrics': [],
                'slices_saved': 0
            }
        patient_metrics[patient_id]['metrics'].append(metrics)
        
        # Save first N slices per patient for visualization
        if patient_metrics[patient_id]['slices_saved'] < save_images_per_patient:
            patient_output_dir = output_dir / patient_id
            patient_output_dir.mkdir(exist_ok=True)
            
            slice_output_dir = patient_output_dir / slice_num
            slice_output_dir.mkdir(exist_ok=True)
            
            # Save as NPY
            np.save(slice_output_dir / 'predicted_pet.npy', pet_pred.squeeze().cpu().numpy())
            
            # Save comparison image
            save_comparison_images(
                ct,
                pet,
                pet_pred,
                slice_output_dir / 'comparison.png',
                num_samples=1,
                title=f'{patient_id} - {slice_num}'
            )
            
            # Save metrics
            with open(slice_output_dir / 'metrics.txt', 'w') as f:
                for key, value in metrics.items():
                    f.write(f"{key}: {value:.4f}\n")
            
            patient_metrics[patient_id]['slices_saved'] += 1
    
    # Calculate per-patient average metrics
    patient_averages = {}
    for patient_id, data in patient_metrics.items():
        avg_metrics = {}
        for key in data['metrics'][0].keys():
            values = [m[key] for m in data['metrics'] if key in m and m[key] >= 0]
            if values:
                avg_metrics[key] = np.mean(values)
        patient_averages[patient_id] = {
            'avg_metrics': avg_metrics,
            'num_slices': len(data['metrics'])
        }
        
        # Save patient summary
        patient_output_dir = output_dir / patient_id
        if patient_output_dir.exists():
            with open(patient_output_dir / 'patient_summary.txt', 'w') as f:
                f.write(f"Patient: {patient_id}\n")
                f.write(f"Number of slices: {len(data['metrics'])}\n")
                f.write("=" * 40 + "\n\n")
                f.write("Average Metrics Across Slices:\n")
                f.write("-" * 40 + "\n")
                for key, value in avg_metrics.items():
                    f.write(f"{key}: {value:.4f}\n")
    
    # Calculate overall average across all patients
    all_patient_avgs = {}
    for key in patient_averages[list(patient_averages.keys())[0]]['avg_metrics'].keys():
        values = [p['avg_metrics'][key] for p in patient_averages.values() if key in p['avg_metrics']]
        if values:
            all_patient_avgs[key] = np.mean(values)
    
    # Save overall summary
    summary_file = output_dir / 'summary.txt'
    with open(summary_file, 'w') as f:
        f.write("CT to PET Translation - Test Set Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total slices evaluated: {sum(len(d['metrics']) for d in patient_metrics.values())}\n")
        f.write(f"Number of patients: {len(patient_metrics)}\n")
        f.write(f"Dataset: {data_dir}\n\n")
        
        f.write("Overall Average Metrics (averaged across patients):\n")
        f.write("-" * 60 + "\n")
        for key, value in all_patient_avgs.items():
            f.write(f"{key}: {value:.4f}\n")
        f.write("\n")
        
        f.write("Per-Patient Results:\n")
        f.write("-" * 60 + "\n")
        for patient_id, data in sorted(patient_averages.items()):
            f.write(f"\n{patient_id} ({data['num_slices']} slices):\n")
            for key, value in data['avg_metrics'].items():
                f.write(f"  {key}: {value:.4f}\n")
        
        f.write("\n\nMetric Guidelines:\n")
        f.write("-" * 60 + "\n")
        f.write("PSNR > 25 dB: Good reconstruction\n")
        f.write("SSIM > 0.80: High structural similarity\n")
        f.write("MAE < 0.05: Low pixel-wise error\n")
        f.write("LPIPS < 0.20: Good perceptual similarity\n")
    
    print(f"\n{'='*60}")
    print("Test Set Results:")
    print(f"{'='*60}")
    print(f"Patients evaluated: {len(patient_metrics)}")
    print(f"Total slices: {sum(len(d['metrics']) for d in patient_metrics.values())}")
    print(f"\nOverall Average Metrics (across patients):")
    print("-" * 60)
    for key, value in all_patient_avgs.items():
        print(f"{key:12s}: {value:.4f}")
    print(f"{'='*60}")
    print(f"\nDetailed results saved to {output_dir}")
    print(f"Summary saved to {summary_file}")


def main(args):
    # Load config
    config = load_config(args.config)
    
    # Set seed
    set_seed(config['seed'])
    
    # Device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load models
    print('\n=== Loading Models ===')
    vqvae, translator = load_models(config, device)
    print('Models loaded successfully')
    
    # Perform inference
    print('\n=== Starting Inference ===')
    
    if args.ct_path:
        # Single file inference
        inference_single_file(
            args.ct_path,
            vqvae,
            translator,
            config,
            device,
            args.output_dir,
            pet_path=args.pet_path,
            sampling_mode=args.sampling_mode,
            temperature=args.temperature,
            top_k=args.top_k
        )
    elif args.data_dir:
        # Directory inference on NPY slices
        inference_directory(
            args.data_dir,
            vqvae,
            translator,
            config,
            device,
            args.output_dir,
            num_samples=args.num_samples,
            save_images_per_patient=args.save_images_per_patient,
            sampling_mode=args.sampling_mode,
            temperature=args.temperature,
            top_k=args.top_k
        )
    else:
        print("Error: Please specify either --ct_path or --data_dir")
        return
    
    print('\n=== Inference Complete ===')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CT to PET Translation Inference')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--ct_path', type=str, default=None,
                       help='Path to single CT NIfTI file')
    parser.add_argument('--pet_path', type=str, default=None,
                       help='Path to ground truth PET (for comparison)')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Directory containing train/val/test folders with NPY slices')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Number of random samples to evaluate (default: 100)')
    parser.add_argument('--save_images_per_patient', type=int, default=5,
                       help='Number of slices to save as images per patient (default: 5)')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Output directory')
    # Sampling arguments
    parser.add_argument('--sampling_mode', type=str, default='argmax',
                       choices=['argmax', 'sample', 'topk'],
                       help='Index prediction mode: argmax (deterministic), sample (softmax sampling), topk (top-k sampling)')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Temperature for sampling (lower=sharper, higher=more random). Default: 1.0')
    parser.add_argument('--top_k', type=int, default=50,
                       help='Number of top candidates for top-k sampling. Default: 50')
    
    args = parser.parse_args()
    main(args)
