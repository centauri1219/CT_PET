"""
Phase 2: Train Translator (CT to PET Codes)
==========================================
This script trains two U-Nets to translate CT scans to PET codebook indices.

Training Objective:
- Learn CT → PET Top Level Codes (64x64)
- Learn CT → PET Bottom Level Codes (128x128)
- Loss = Cross-Entropy (classification of codebook indices)

Prerequisites:
- Trained VQ-VAE-2 model from Phase 1

Usage:
    python train_translator.py --config configs/config.yaml
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time
from pathlib import Path

from models import VQVAE2, UNetTranslator
from datasets import create_dataloaders_npy
from utils import (
    load_config, set_seed, save_checkpoint, load_checkpoint,
    count_parameters, translator_combined_loss,
    FocalLoss, focal_translator_combined_loss,
    save_comparison_images, plot_training_curves,
    AverageMeter, format_time, get_lr, calculate_all_metrics
)


def extract_codes(vqvae, pet, device):
    """Extract codebook indices from PET images using VQ-VAE."""
    with torch.no_grad():
        enc_output = vqvae.encode(pet)
        top_indices = enc_output['top_indices']
        bottom_indices = enc_output['bottom_indices']
    return top_indices, bottom_indices


def train_one_epoch(translator, vqvae, train_loader, optimizer, scaler, device, epoch, config, writer, global_step, focal_loss_fn=None):
    """Train for one epoch using Focal Loss."""
    translator.train()
    vqvae.eval()
    
    loss_meter = AverageMeter('Loss')
    top_loss_meter = AverageMeter('Top_Loss')
    bottom_loss_meter = AverageMeter('Bottom_Loss')
    top_pt_meter = AverageMeter('Top_Pt')  # Monitor p_t for top level
    bottom_pt_meter = AverageMeter('Bottom_Pt')  # Monitor p_t for bottom level
    
    # Gradient accumulation settings
    accumulation_steps = config['train_translator'].get('gradient_accumulation_steps', 1)
    
    # Use Focal Loss if provided, otherwise fall back to Cross-Entropy
    use_focal_loss = focal_loss_fn is not None
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, batch in enumerate(pbar):
        ct = batch['ct'].to(device)
        pet = batch['pet'].to(device)
        batch_size = ct.size(0)
        
        # Extract ground truth codes from PET
        top_indices, bottom_indices = extract_codes(vqvae, pet, device)
        
        # Forward pass with mixed precision
        with torch.cuda.amp.autocast(enabled=config['train_translator']['use_amp']):
            outputs = translator(ct)
            
            if use_focal_loss:
                # Use Focal Loss
                # Compute p_t stats every 50 batches to avoid overhead
                compute_stats = (batch_idx % 50 == 0)
                
                losses = focal_translator_combined_loss(
                    outputs['top_logits'],
                    outputs['bottom_logits'],
                    top_indices,
                    bottom_indices,
                    focal_loss_fn=focal_loss_fn,
                    top_size=(64, 64),
                    bottom_size=(128, 128),
                    compute_pt_stats=compute_stats
                )
            else:
                # Fallback to standard Cross-Entropy
                losses = translator_combined_loss(
                    outputs['top_logits'],
                    outputs['bottom_logits'],
                    top_indices,
                    bottom_indices,
                    top_size=(64, 64),
                    bottom_size=(128, 128)
                )
            
            # Scale loss for gradient accumulation
            loss = losses['total_loss'] / accumulation_steps
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Update weights every accumulation_steps
        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        # Save loss values before cleanup
        total_loss_val = losses['total_loss'].item()
        top_loss_val = losses['top_loss'].item()
        bottom_loss_val = losses['bottom_loss'].item()
        
        # Update meters (use original unscaled loss)
        loss_meter.update(total_loss_val, batch_size)
        top_loss_meter.update(top_loss_val, batch_size)
        bottom_loss_meter.update(bottom_loss_val, batch_size)
        
        # Update p_t meters if stats were computed
        if use_focal_loss and 'top_pt_stats' in losses:
            top_pt_meter.update(losses['top_pt_stats']['mean_pt'], batch_size)
            bottom_pt_meter.update(losses['bottom_pt_stats']['mean_pt'], batch_size)
        
        # Free up memory
        del ct, pet, top_indices, bottom_indices, outputs, losses, loss
        if (batch_idx + 1) % 10 == 0:
            torch.cuda.empty_cache()
        
        # Update progress bar
        postfix = {
            'loss': f'{loss_meter.avg:.4f}',
            'top': f'{top_loss_meter.avg:.4f}',
            'bottom': f'{bottom_loss_meter.avg:.4f}'
        }
        if use_focal_loss and top_pt_meter.count > 0:
            postfix['top_pt'] = f'{top_pt_meter.avg:.3f}'
            postfix['btm_pt'] = f'{bottom_pt_meter.avg:.3f}'
        pbar.set_postfix(postfix)
        
        # Log to TensorBoard
        if batch_idx % config['logging']['print_freq'] == 0:
            writer.add_scalar('Train/Loss', total_loss_val, global_step)
            writer.add_scalar('Train/Top_Loss', top_loss_val, global_step)
            writer.add_scalar('Train/Bottom_Loss', bottom_loss_val, global_step)
            writer.add_scalar('Train/LR', get_lr(optimizer), global_step)
            
            # Log p_t statistics if using focal loss
            if use_focal_loss and top_pt_meter.count > 0:
                writer.add_scalar('Train/Top_Pt_Mean', top_pt_meter.avg, global_step)
                writer.add_scalar('Train/Bottom_Pt_Mean', bottom_pt_meter.avg, global_step)
        
        global_step += 1
    
    # Print epoch summary with p_t stats
    if use_focal_loss and top_pt_meter.count > 0:
        print(f"  => Epoch {epoch} p_t stats: Top={top_pt_meter.avg:.4f}, Bottom={bottom_pt_meter.avg:.4f}")
        print(f"     (p_t < 0.1 = guessing, p_t > 0.5 = confident)")
    
    return loss_meter.avg, global_step


def validate(translator, vqvae, val_loader, device, epoch, config, writer):
    """Validate the translator."""
    translator.eval()
    vqvae.eval()
    
    loss_meter = AverageMeter('Val_Loss')
    top_loss_meter = AverageMeter('Top_Loss')
    bottom_loss_meter = AverageMeter('Bottom_Loss')
    
    psnr_values = []
    ssim_values = []
    mae_values = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc='Validation')):
            ct = batch['ct'].to(device)
            pet = batch['pet'].to(device)
            
            # Extract ground truth codes
            top_indices_gt, bottom_indices_gt = extract_codes(vqvae, pet, device)
            
            # Forward pass
            outputs = translator(ct)
            
            losses = translator_combined_loss(
                outputs['top_logits'],
                outputs['bottom_logits'],
                top_indices_gt,
                bottom_indices_gt,
                top_size=(64, 64),
                bottom_size=(128, 128)
            )
            
            # Update meters
            loss_meter.update(losses['total_loss'].item(), ct.size(0))
            top_loss_meter.update(losses['top_loss'].item(), ct.size(0))
            bottom_loss_meter.update(losses['bottom_loss'].item(), ct.size(0))
            
            # Generate PET predictions
            predictions = translator.predict_indices(ct)
            pet_pred = vqvae.decode_indices(
                predictions['top_indices'],
                predictions['bottom_indices']
            )
            
            # Calculate metrics
            metrics = calculate_all_metrics(pet_pred, pet)
            psnr_values.append(metrics['PSNR'])
            ssim_values.append(metrics['SSIM'])
            mae_values.append(metrics['MAE'])
            
            # Save images periodically
            if batch_idx == 0 and config['logging']['log_images']:
                save_dir = Path(config['logging']['tensorboard_dir']).parent / 'images'
                save_dir.mkdir(exist_ok=True, parents=True)
                save_comparison_images(
                    ct[:4],
                    pet[:4],
                    pet_pred[:4],
                    save_dir / f'epoch_{epoch}_translation.png',
                    num_samples=4,
                    title=f'Epoch {epoch} - CT to PET Translation'
                )
    
    # Average metrics
    avg_psnr = sum(psnr_values) / len(psnr_values)
    avg_ssim = sum(ssim_values) / len(ssim_values)
    avg_mae = sum(mae_values) / len(mae_values)
    
    # Log to TensorBoard
    writer.add_scalar('Val/Loss', loss_meter.avg, epoch)
    writer.add_scalar('Val/Top_Loss', top_loss_meter.avg, epoch)
    writer.add_scalar('Val/Bottom_Loss', bottom_loss_meter.avg, epoch)
    writer.add_scalar('Val/PSNR', avg_psnr, epoch)
    writer.add_scalar('Val/SSIM', avg_ssim, epoch)
    writer.add_scalar('Val/MAE', avg_mae, epoch)
    
    print(f'\nValidation Results:')
    print(f'  Loss: {loss_meter.avg:.4f}')
    print(f'  PSNR: {avg_psnr:.2f} dB')
    print(f'  SSIM: {avg_ssim:.4f}')
    print(f'  MAE: {avg_mae:.4f}')
    
    return loss_meter.avg, {'PSNR': avg_psnr, 'SSIM': avg_ssim, 'MAE': avg_mae}


def main(args):
    # Load config
    config = load_config(args.config)
    
    # Set seed
    set_seed(config['seed'])
    
    # Device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Memory optimization: disable cudnn benchmark for deterministic memory usage
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    
    # Create dataloaders
    print('\n=== Loading Dataset ===')
    train_loader, val_loader, test_loader = create_dataloaders_npy(
        config,
        batch_size=config['train_translator']['batch_size']
    )
    
    # Load pretrained VQ-VAE
    print('\n=== Loading Pretrained VQ-VAE ===')
    vqvae = VQVAE2(
        in_channels=config['vqvae']['image_channels'],
        base_channels=config['vqvae']['base_channels'],
        num_res_blocks=config['vqvae']['num_res_blocks'],
        codebook_size=config['vqvae']['codebook_size'],
        embedding_dim=config['vqvae']['embedding_dim'],
        commitment_cost=config['vqvae']['commitment_cost']
    ).to(device)
    
    vqvae_checkpoint = config['train_translator']['vqvae_checkpoint']
    if not os.path.exists(vqvae_checkpoint):
        raise FileNotFoundError(
            f"VQ-VAE checkpoint not found: {vqvae_checkpoint}\n"
            "Please train Phase 1 (VQ-VAE) first using train_vqvae.py"
        )
    
    load_checkpoint(vqvae_checkpoint, vqvae, device=device)
    
    if config['train_translator']['freeze_vqvae']:
        for param in vqvae.parameters():
            param.requires_grad = False
        vqvae.eval()
        print('VQ-VAE frozen (not trainable)')
    
    # Create translator model
    print('\n=== Creating Translator ===')
    translator = UNetTranslator(
        in_channels=config['translator']['in_channels'],
        num_classes=config['vqvae']['codebook_size'],
        base_channels=config['translator']['base_channels'],
        dropout=config['translator']['dropout']
    ).to(device)
    
    print(f'Translator parameters: {count_parameters(translator) / 1e6:.2f}M')
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        translator.parameters(),
        lr=config['train_translator']['learning_rate'],
        weight_decay=config['train_translator']['weight_decay']
    )
    
    # Learning rate scheduler
    if config['train_translator']['lr_scheduler'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['train_translator']['num_epochs'],
            eta_min=config['train_translator']['lr_min']
        )
    else:
        scheduler = None
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler(enabled=config['train_translator']['use_amp'])
    
    # TensorBoard writer
    log_dir = Path(config['logging']['tensorboard_dir']) / 'translator'
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    # Auto-resume from latest checkpoint if exists
    start_epoch = 1
    best_loss = float('inf')
    global_step = 0
    
    # Check for existing checkpoints
    checkpoint_dir = Path('checkpoints')
    if args.resume:
        # Manual resume from specified checkpoint
        checkpoint_info = load_checkpoint(args.resume, translator, optimizer, scheduler, device)
        start_epoch = checkpoint_info['epoch'] + 1
        best_loss = checkpoint_info['loss']
        global_step = checkpoint_info.get('global_step', 0)
        print(f"Resumed from {args.resume}")
    elif checkpoint_dir.exists():
        # Auto-resume from latest epoch checkpoint
        epoch_checkpoints = list(checkpoint_dir.glob('translator_epoch_*.pth'))
        if epoch_checkpoints:
            # Find the latest epoch checkpoint
            latest_checkpoint = max(epoch_checkpoints, key=lambda p: int(p.stem.split('_')[-1]))
            print(f"\nFound existing checkpoint: {latest_checkpoint.name}")
            print(f"Resuming training from epoch {latest_checkpoint.stem.split('_')[-1]}...")
            
            checkpoint_info = load_checkpoint(str(latest_checkpoint), translator, optimizer, scheduler, device)
            start_epoch = checkpoint_info['epoch'] + 1
            global_step = checkpoint_info.get('global_step', 0)
            
            # Load best loss from translator_best.pth if it exists
            best_checkpoint = checkpoint_dir / 'translator_best.pth'
            if best_checkpoint.exists():
                best_info = torch.load(str(best_checkpoint), map_location=device)
                best_loss = best_info['loss']
                print(f"Best validation loss so far: {best_loss:.4f}")
    
    # Training loop
    print('\n=== Starting Training ===')
    if start_epoch > 1:
        print(f'Continuing from epoch {start_epoch}')
    
    # Initialize Focal Loss
    # Parameters: alpha=1.0 (balanced), gamma=2.0 (focus on hard examples)
    focal_gamma = config['train_translator'].get('focal_gamma', 2.0)
    focal_alpha = config['train_translator'].get('focal_alpha', 1.0)
    use_focal_loss = config['train_translator'].get('use_focal_loss', True)
    
    if use_focal_loss:
        focal_loss_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction='mean')
        print(f'\n🎯 Using Focal Loss with gamma={focal_gamma}, alpha={focal_alpha}')
        print(f'   Focal loss focuses on hard examples by down-weighting easy ones')
        print(f'   Formula: Loss = -α(1-p_t)^γ * log(p_t)')
    else:
        focal_loss_fn = None
        print('\n📋 Using standard Cross-Entropy Loss')
    
    train_losses = []
    val_losses = []
    psnr_history = []
    ssim_history = []
    
    start_time = time.time()
    
    for epoch in range(start_epoch, config['train_translator']['num_epochs'] + 1):
        print(f'\n{"="*60}')
        print(f'Epoch {epoch}/{config["train_translator"]["num_epochs"]}')
        print(f'{"="*60}')
        
        # Train with Focal Loss
        train_loss, global_step = train_one_epoch(
            translator, vqvae, train_loader, optimizer, scaler, device,
            epoch, config, writer, global_step, focal_loss_fn=focal_loss_fn
        )
        train_losses.append(train_loss)
        
        # Validate
        if epoch % config['train_translator']['val_every'] == 0:
            val_loss, metrics = validate(translator, vqvae, val_loader, device, epoch, config, writer)
            val_losses.append(val_loss)
            psnr_history.append(metrics['PSNR'])
            ssim_history.append(metrics['SSIM'])
            
            # Save best model
            if val_loss < best_loss:
                best_loss = val_loss
                save_checkpoint(
                    translator, optimizer, epoch, val_loss,
                    'checkpoints/translator_best.pth',
                    scheduler, metrics, global_step
                )
                print(f'✓ Best model saved (loss: {best_loss:.4f})')
        
        # Save checkpoint periodically
        if epoch % config['train_translator']['save_every'] == 0:
            save_checkpoint(
                translator, optimizer, epoch, train_loss,
                f'checkpoints/translator_epoch_{epoch}.pth',
                scheduler, None, global_step
            )
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
        
        # Time estimate
        elapsed = time.time() - start_time
        eta = elapsed / epoch * (config['train_translator']['num_epochs'] - epoch)
        print(f'Time: {format_time(elapsed)} | ETA: {format_time(eta)}')
    
    # Save final model
    save_checkpoint(
        translator, optimizer, config['train_translator']['num_epochs'], train_losses[-1],
        'checkpoints/translator_final.pth',
        scheduler, None, global_step
    )
    
    # Plot training curves
    if len(val_losses) > 0:
        plot_training_curves(
            train_losses[::config['train_translator']['val_every']],
            val_losses,
            'logs/translator_training_curves.png',
            metrics={'PSNR': psnr_history, 'SSIM': ssim_history},
            title='Translator Training'
        )
    
    writer.close()
    print('\n=== Training Complete ===')
    print(f'Total time: {format_time(time.time() - start_time)}')
    print(f'Best validation loss: {best_loss:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Translator for CT to PET translation')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    main(args)
