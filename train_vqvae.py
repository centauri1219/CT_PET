"""
Phase 1: Train VQ-VAE-2 on PET Scans
====================================
This script trains the VQ-VAE-2 model to learn high-quality PET reconstruction
from discrete codebook entries.

Training Objective:
- Learn to compress PET scans into discrete codes
- Reconstruct high-quality PET images from codes
- Loss = Reconstruction Loss (L1) + VQ Loss (commitment + codebook)

Usage:
    python train_vqvae.py --config configs/config.yaml
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time
from pathlib import Path

from models import VQVAE2
from datasets import create_dataloaders_npy
from utils import (
    load_config, set_seed, save_checkpoint, load_checkpoint,
    count_parameters, vqvae_loss, calculate_all_metrics,
    save_vqvae_reconstruction, plot_training_curves,
    AverageMeter, format_time, get_lr
)


def train_one_epoch(model, train_loader, optimizer, scaler, device, epoch, config, writer, global_step):
    """Train for one epoch."""
    model.train()
    
    loss_meter = AverageMeter('Loss')
    recon_meter = AverageMeter('Recon')
    vq_meter = AverageMeter('VQ')
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, batch in enumerate(pbar):
        pet = batch['pet'].to(device)
        
        # Forward pass with mixed precision
        with torch.cuda.amp.autocast(enabled=config['train_vqvae']['use_amp']):
            output = model(pet)
            losses = vqvae_loss(
                output['recon'],
                pet,
                output['vq_loss'],
                recon_weight=config['train_vqvae']['recon_loss_weight']
            )
        
        # Backward pass
        optimizer.zero_grad()
        scaler.scale(losses['total_loss']).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Update meters
        loss_meter.update(losses['total_loss'].item(), pet.size(0))
        recon_meter.update(losses['recon_loss'].item(), pet.size(0))
        vq_meter.update(losses['vq_loss'].item(), pet.size(0))
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'recon': f'{recon_meter.avg:.4f}',
            'vq': f'{vq_meter.avg:.4f}'
        })
        
        # Log to TensorBoard
        if batch_idx % config['logging']['print_freq'] == 0:
            writer.add_scalar('Train/Loss', losses['total_loss'].item(), global_step)
            writer.add_scalar('Train/Recon_Loss', losses['recon_loss'].item(), global_step)
            writer.add_scalar('Train/VQ_Loss', losses['vq_loss'].item(), global_step)
            writer.add_scalar('Train/LR', get_lr(optimizer), global_step)
        
        global_step += 1
    
    return loss_meter.avg, global_step


def validate(model, val_loader, device, epoch, config, writer):
    """Validate the model."""
    model.eval()
    
    loss_meter = AverageMeter('Val_Loss')
    recon_meter = AverageMeter('Val_Recon')
    vq_meter = AverageMeter('Val_VQ')
    
    psnr_values = []
    ssim_values = []
    mae_values = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc='Validation')):
            pet = batch['pet'].to(device)
            
            # Forward pass
            output = model(pet)
            losses = vqvae_loss(
                output['recon'],
                pet,
                output['vq_loss'],
                recon_weight=config['train_vqvae']['recon_loss_weight']
            )
            
            # Update meters
            loss_meter.update(losses['total_loss'].item(), pet.size(0))
            recon_meter.update(losses['recon_loss'].item(), pet.size(0))
            vq_meter.update(losses['vq_loss'].item(), pet.size(0))
            
            # Calculate metrics
            metrics = calculate_all_metrics(output['recon'], pet)
            psnr_values.append(metrics['PSNR'])
            ssim_values.append(metrics['SSIM'])
            mae_values.append(metrics['MAE'])
            
            # Save images periodically
            if batch_idx == 0 and config['logging']['log_images']:
                save_dir = Path(config['logging']['tensorboard_dir']).parent / 'images'
                save_dir.mkdir(exist_ok=True, parents=True)
                save_vqvae_reconstruction(
                    pet[:4],
                    output['recon'][:4],
                    save_dir / f'epoch_{epoch}_recon.png',
                    num_samples=4,
                    title=f'Epoch {epoch} - VQ-VAE Reconstruction'
                )
    
    # Average metrics
    avg_psnr = sum(psnr_values) / len(psnr_values)
    avg_ssim = sum(ssim_values) / len(ssim_values)
    avg_mae = sum(mae_values) / len(mae_values)
    
    # Log to TensorBoard
    writer.add_scalar('Val/Loss', loss_meter.avg, epoch)
    writer.add_scalar('Val/Recon_Loss', recon_meter.avg, epoch)
    writer.add_scalar('Val/VQ_Loss', vq_meter.avg, epoch)
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
    
    # Create dataloaders
    print('\n=== Loading Dataset ===')
    train_loader, val_loader, test_loader = create_dataloaders_npy(
        config
    )
    
    # Create model
    print('\n=== Creating Model ===')
    model = VQVAE2(
        in_channels=config['vqvae']['image_channels'],
        base_channels=config['vqvae']['base_channels'],
        num_res_blocks=config['vqvae']['num_res_blocks'],
        codebook_size=config['vqvae']['codebook_size'],
        embedding_dim=config['vqvae']['embedding_dim'],
        commitment_cost=config['vqvae']['commitment_cost']
    ).to(device)
    
    print(f'Model parameters: {count_parameters(model) / 1e6:.2f}M')
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['train_vqvae']['learning_rate'],
        weight_decay=config['train_vqvae']['weight_decay']
    )
    
    # Learning rate scheduler
    if config['train_vqvae']['lr_scheduler'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['train_vqvae']['num_epochs'],
            eta_min=config['train_vqvae']['lr_min']
        )
    else:
        scheduler = None
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler(enabled=config['train_vqvae']['use_amp'])
    
    # TensorBoard writer
    log_dir = Path(config['logging']['tensorboard_dir']) / 'vqvae'
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
        checkpoint_info = load_checkpoint(args.resume, model, optimizer, scheduler, device)
        start_epoch = checkpoint_info['epoch'] + 1
        best_loss = checkpoint_info['loss']
        global_step = checkpoint_info.get('global_step', 0)
        print(f"Resumed from {args.resume}")
    elif checkpoint_dir.exists():
        # Auto-resume from latest epoch checkpoint
        epoch_checkpoints = list(checkpoint_dir.glob('vqvae_epoch_*.pth'))
        if epoch_checkpoints:
            # Find the latest epoch checkpoint
            latest_checkpoint = max(epoch_checkpoints, key=lambda p: int(p.stem.split('_')[-1]))
            print(f"\n🔄 Found existing checkpoint: {latest_checkpoint.name}")
            print(f"Resuming training from epoch {latest_checkpoint.stem.split('_')[-1]}...")
            
            checkpoint_info = load_checkpoint(str(latest_checkpoint), model, optimizer, scheduler, device)
            start_epoch = checkpoint_info['epoch'] + 1
            global_step = checkpoint_info.get('global_step', 0)
            
            # Load best loss from vqvae_best.pth if it exists
            best_checkpoint = checkpoint_dir / 'vqvae_best.pth'
            if best_checkpoint.exists():
                best_info = torch.load(str(best_checkpoint), map_location=device)
                best_loss = best_info['loss']
                print(f"Best validation loss so far: {best_loss:.4f}")
    
    # Training loop
    print('\n=== Starting Training ===')
    if start_epoch > 1:
        print(f'Continuing from epoch {start_epoch}')
    
    train_losses = []
    val_losses = []
    psnr_history = []
    ssim_history = []
    
    start_time = time.time()
    
    for epoch in range(start_epoch, config['train_vqvae']['num_epochs'] + 1):
        print(f'\n{"="*60}')
        print(f'Epoch {epoch}/{config["train_vqvae"]["num_epochs"]}')
        print(f'{"="*60}')
        
        # Train
        train_loss, global_step = train_one_epoch(
            model, train_loader, optimizer, scaler, device,
            epoch, config, writer, global_step
        )
        train_losses.append(train_loss)
        
        # Validate
        if epoch % config['train_vqvae']['val_every'] == 0:
            val_loss, metrics = validate(model, val_loader, device, epoch, config, writer)
            val_losses.append(val_loss)
            psnr_history.append(metrics['PSNR'])
            ssim_history.append(metrics['SSIM'])
            
            # Save best model
            if val_loss < best_loss:
                best_loss = val_loss
                save_checkpoint(
                    model, optimizer, epoch, val_loss,
                    'checkpoints/vqvae_best.pth',
                    scheduler, metrics, global_step
                )
                print(f'✓ Best model saved (loss: {best_loss:.4f})')
        
        # Save checkpoint periodically
        if epoch % config['train_vqvae']['save_every'] == 0:
            save_checkpoint(
                model, optimizer, epoch, train_loss,
                f'checkpoints/vqvae_epoch_{epoch}.pth',
                scheduler, None, global_step
            )
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
        
        # Time estimate
        elapsed = time.time() - start_time
        eta = elapsed / epoch * (config['train_vqvae']['num_epochs'] - epoch)
        print(f'Time: {format_time(elapsed)} | ETA: {format_time(eta)}')
    
    # Save final model
    save_checkpoint(
        model, optimizer, config['train_vqvae']['num_epochs'], train_losses[-1],
        'checkpoints/vqvae_final.pth',
        scheduler, None, global_step
    )
    
    # Plot training curves
    if len(val_losses) > 0:
        plot_training_curves(
            train_losses[::config['train_vqvae']['val_every']],
            val_losses,
            'logs/vqvae_training_curves.png',
            metrics={'PSNR': psnr_history, 'SSIM': ssim_history},
            title='VQ-VAE-2 Training'
        )
    
    writer.close()
    print('\n=== Training Complete ===')
    print(f'Total time: {format_time(time.time() - start_time)}')
    print(f'Best validation loss: {best_loss:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train VQ-VAE-2 for PET reconstruction')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    main(args)
