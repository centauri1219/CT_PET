"""
Loss Functions for VQ-VAE-2 and Translator Training
====================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for Multi-Class Classification
    
    Focuses learning on hard examples by down-weighting easy examples.
    Formula: Loss = -α * (1 - p_t)^γ * log(p_t)
    
    Where p_t is the probability of the correct class.
    
    Args:
        alpha: Balancing factor (default=1.0, can also be a tensor of per-class weights)
        gamma: Focusing parameter. Higher gamma = more focus on hard examples (default=2.0)
        reduction: 'mean', 'sum', or 'none' (default='mean')
        ignore_index: Index to ignore in loss calculation (default=-100)
    
    Input:
        logits: [B, C, H, W] where C is the number of classes (e.g., 512 codebook indices)
        targets: [B, H, W] with class indices
    """
    
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean', ignore_index=-100):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
    
    def forward(self, logits, targets):
        """
        Args:
            logits: [B, C, H, W] - Raw logits from the model
            targets: [B, H, W] - Ground truth class indices
        
        Returns:
            Focal loss (scalar if reduction='mean' or 'sum')
        """
        B, C, H, W = logits.shape
        
        # Ensure logits and targets have matching spatial dimensions
        if targets.shape[1:] != (H, W):
            targets = F.interpolate(
                targets.unsqueeze(1).float(),
                size=(H, W),
                mode='nearest'
            ).squeeze(1).long()
        
        # Use log_softmax for numerical stability
        # log_softmax: log(softmax(x)) computed in a numerically stable way
        log_probs = F.log_softmax(logits, dim=1)  # [B, C, H, W]
        
        # Get probabilities (for focal weight computation)
        probs = torch.exp(log_probs)  # [B, C, H, W]
        
        # Flatten spatial dimensions for easier indexing
        log_probs_flat = log_probs.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]
        probs_flat = probs.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]
        targets_flat = targets.reshape(-1)  # [B*H*W]
        
        # Create mask for valid targets (ignore index)
        valid_mask = targets_flat != self.ignore_index
        
        # Get p_t (probability of correct class) for valid targets
        # Gather the probability at the target index
        targets_valid = targets_flat.clone()
        targets_valid[~valid_mask] = 0  # Temporary placeholder for invalid indices
        
        p_t = probs_flat.gather(1, targets_valid.unsqueeze(1)).squeeze(1)  # [B*H*W]
        log_p_t = log_probs_flat.gather(1, targets_valid.unsqueeze(1)).squeeze(1)  # [B*H*W]
        
        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        
        # Focal loss: -alpha * (1 - p_t)^gamma * log(p_t)
        focal_loss = -self.alpha * focal_weight * log_p_t
        
        # Apply mask for ignore_index
        focal_loss = focal_loss * valid_mask.float()
        
        # Apply reduction
        if self.reduction == 'mean':
            # Mean over valid elements only
            return focal_loss.sum() / valid_mask.float().sum().clamp(min=1)
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss.reshape(B, H, W)
    
    def compute_pt_stats(self, logits, targets):
        """
        Compute statistics about p_t (probability of correct class).
        Useful for monitoring model confidence.
        
        Returns:
            dict with mean_pt, std_pt, min_pt, max_pt
        """
        with torch.no_grad():
            B, C, H, W = logits.shape
            
            # Resize targets if needed
            if targets.shape[1:] != (H, W):
                targets = F.interpolate(
                    targets.unsqueeze(1).float(),
                    size=(H, W),
                    mode='nearest'
                ).squeeze(1).long()
            
            probs = F.softmax(logits, dim=1)
            probs_flat = probs.permute(0, 2, 3, 1).reshape(-1, C)
            targets_flat = targets.reshape(-1)
            
            p_t = probs_flat.gather(1, targets_flat.unsqueeze(1)).squeeze(1)
            
            return {
                'mean_pt': p_t.mean().item(),
                'std_pt': p_t.std().item(),
                'min_pt': p_t.min().item(),
                'max_pt': p_t.max().item(),
                'median_pt': p_t.median().item()
            }


def focal_translator_loss(logits, target_indices, target_size, focal_loss_fn):
    """
    Translator Loss using Focal Loss instead of Cross-Entropy.
    
    Args:
        logits: Predicted logits [B, num_classes, H, W]
        target_indices: Ground truth codebook indices [B, H', W']
        target_size: Target spatial size (H', W')
        focal_loss_fn: FocalLoss instance
    
    Returns:
        loss: Focal loss value
    """
    # Resize logits to match target size
    if logits.shape[2:] != target_size:
        logits = F.interpolate(
            logits,
            size=target_size,
            mode='bilinear',
            align_corners=False
        )
    
    return focal_loss_fn(logits, target_indices)


def focal_translator_combined_loss(top_logits, bottom_logits, top_indices, bottom_indices,
                                   focal_loss_fn,
                                   top_size=(64, 64), bottom_size=(128, 128),
                                   top_weight=1.0, bottom_weight=1.0,
                                   compute_pt_stats=False):
    """
    Combined Focal Loss for training both top and bottom translators.
    
    Args:
        top_logits: Top level logits [B, num_classes, H, W]
        bottom_logits: Bottom level logits [B, num_classes, H, W]
        top_indices: Top level target indices [B, H_top, W_top]
        bottom_indices: Bottom level target indices [B, H_bottom, W_bottom]
        focal_loss_fn: FocalLoss instance
        top_size: Target size for top level
        bottom_size: Target size for bottom level
        top_weight: Weight for top level loss
        bottom_weight: Weight for bottom level loss
        compute_pt_stats: Whether to compute p_t statistics for monitoring
    
    Returns:
        dict with:
            - total_loss: Combined focal loss
            - top_loss: Top level focal loss
            - bottom_loss: Bottom level focal loss
            - top_pt_stats: (optional) p_t statistics for top level
            - bottom_pt_stats: (optional) p_t statistics for bottom level
    """
    # Resize logits to target sizes
    if top_logits.shape[2:] != top_size:
        top_logits_resized = F.interpolate(top_logits, size=top_size, mode='bilinear', align_corners=False)
    else:
        top_logits_resized = top_logits
    
    if bottom_logits.shape[2:] != bottom_size:
        bottom_logits_resized = F.interpolate(bottom_logits, size=bottom_size, mode='bilinear', align_corners=False)
    else:
        bottom_logits_resized = bottom_logits
    
    # Compute focal losses
    top_loss = focal_loss_fn(top_logits_resized, top_indices)
    bottom_loss = focal_loss_fn(bottom_logits_resized, bottom_indices)
    
    total_loss = top_weight * top_loss + bottom_weight * bottom_loss
    
    result = {
        'total_loss': total_loss,
        'top_loss': top_loss,
        'bottom_loss': bottom_loss
    }
    
    # Optionally compute p_t statistics for monitoring
    if compute_pt_stats:
        result['top_pt_stats'] = focal_loss_fn.compute_pt_stats(top_logits_resized, top_indices)
        result['bottom_pt_stats'] = focal_loss_fn.compute_pt_stats(bottom_logits_resized, bottom_indices)
    
    return result


def vqvae_loss(recon, target, vq_loss, recon_weight=1.0):
    """
    VQ-VAE-2 Loss: Reconstruction Loss + VQ Loss
    
    Args:
        recon: Reconstructed image [B, C, H, W]
        target: Target image [B, C, H, W]
        vq_loss: Vector quantization loss (scalar)
        recon_weight: Weight for reconstruction loss
    
    Returns:
        dict with:
            - total_loss: Combined loss
            - recon_loss: Reconstruction loss (L1)
            - vq_loss: VQ commitment + codebook loss
    """
    # Reconstruction loss (L1 for sharper boundaries)
    recon_loss = F.l1_loss(recon, target)
    
    # Total loss
    total_loss = recon_weight * recon_loss + vq_loss
    
    return {
        'total_loss': total_loss,
        'recon_loss': recon_loss,
        'vq_loss': vq_loss
    }


def translator_loss(logits, target_indices, target_size):
    """
    Translator Loss: Cross-Entropy for codebook index classification
    
    Args:
        logits: Predicted logits [B, num_classes, H, W]
        target_indices: Ground truth codebook indices [B, H', W']
        target_size: Target spatial size (H', W')
    
    Returns:
        loss: Cross-entropy loss
    """
    # Downsample logits to match target size
    if logits.shape[2:] != target_size:
        logits = F.interpolate(
            logits,
            size=target_size,
            mode='bilinear',
            align_corners=False
        )
    
    # Cross-entropy loss
    # logits: [B, C, H, W], target_indices: [B, H, W]
    loss = F.cross_entropy(logits, target_indices)
    
    return loss


def translator_combined_loss(top_logits, bottom_logits, top_indices, bottom_indices,
                             top_size=(64, 64), bottom_size=(128, 128),
                             top_weight=1.0, bottom_weight=1.0):
    """
    Combined loss for training both top and bottom translators.
    
    Args:
        top_logits: Top level logits [B, num_classes, H, W]
        bottom_logits: Bottom level logits [B, num_classes, H, W]
        top_indices: Top level target indices [B, H_top, W_top]
        bottom_indices: Bottom level target indices [B, H_bottom, W_bottom]
        top_size: Target size for top level
        bottom_size: Target size for bottom level
        top_weight: Weight for top level loss
        bottom_weight: Weight for bottom level loss
    
    Returns:
        dict with:
            - total_loss: Combined loss
            - top_loss: Top level cross-entropy loss
            - bottom_loss: Bottom level cross-entropy loss
    """
    top_loss = translator_loss(top_logits, top_indices, top_size)
    bottom_loss = translator_loss(bottom_logits, bottom_indices, bottom_size)
    
    total_loss = top_weight * top_loss + bottom_weight * bottom_loss
    
    return {
        'total_loss': total_loss,
        'top_loss': top_loss,
        'bottom_loss': bottom_loss
    }


class PerceptualLoss(nn.Module):
    """
    Optional: Perceptual Loss using VGG features
    (Can be added if reconstructions are too blurry)
    """
    
    def __init__(self):
        super().__init__()
        # Import only if needed
        try:
            import torchvision.models as models
            vgg = models.vgg16(pretrained=True).features
            self.blocks = nn.ModuleList([
                vgg[:4],   # relu1_2
                vgg[4:9],  # relu2_2
                vgg[9:16]  # relu3_3
            ])
            for block in self.blocks:
                for p in block.parameters():
                    p.requires_grad = False
            self.blocks.eval()
        except:
            print("Warning: Could not load VGG for perceptual loss")
            self.blocks = None
    
    def forward(self, input, target):
        if self.blocks is None:
            return torch.tensor(0.0, device=input.device)
        
        # Convert grayscale to RGB
        if input.shape[1] == 1:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        
        loss = 0.0
        x = input
        y = target
        
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += F.l1_loss(x, y)
        
        return loss


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Loss Functions")
    print("=" * 60)
    
    # Test VQ-VAE loss
    recon = torch.randn(4, 1, 256, 256)
    target = torch.randn(4, 1, 256, 256)
    vq_loss_val = torch.tensor(0.5)
    
    loss_dict = vqvae_loss(recon, target, vq_loss_val)
    print("\n1. VQ-VAE Loss Test:")
    print(f"   Total Loss: {loss_dict['total_loss'].item():.4f}")
    print(f"   Recon Loss: {loss_dict['recon_loss'].item():.4f}")
    print(f"   VQ Loss: {loss_dict['vq_loss'].item():.4f}")
    
    # Test Translator loss (Cross-Entropy)
    logits = torch.randn(4, 512, 256, 256)
    indices = torch.randint(0, 512, (4, 64, 64))
    
    loss = translator_loss(logits, indices, target_size=(64, 64))
    print(f"\n2. Translator CE Loss Test: {loss.item():.4f}")
    
    # Test Focal Loss
    print("\n3. Focal Loss Test:")
    focal_loss_fn = FocalLoss(alpha=1.0, gamma=2.0, reduction='mean')
    
    # Test with random logits
    test_logits = torch.randn(4, 512, 64, 64)
    test_indices = torch.randint(0, 512, (4, 64, 64))
    
    focal_loss_val = focal_loss_fn(test_logits, test_indices)
    print(f"   Focal Loss (random): {focal_loss_val.item():.4f}")
    
    # Test p_t statistics
    pt_stats = focal_loss_fn.compute_pt_stats(test_logits, test_indices)
    print(f"\n   p_t Statistics (random predictions):")
    print(f"     Mean p_t: {pt_stats['mean_pt']:.4f}")
    print(f"     Std p_t: {pt_stats['std_pt']:.4f}")
    print(f"     Min p_t: {pt_stats['min_pt']:.4f}")
    print(f"     Max p_t: {pt_stats['max_pt']:.4f}")
    print(f"     Median p_t: {pt_stats['median_pt']:.4f}")
    
    # Test with perfect predictions (logits strongly favor correct class)
    print("\n   Testing with confident predictions:")
    confident_logits = torch.zeros(4, 512, 64, 64)
    for b in range(4):
        for h in range(64):
            for w in range(64):
                confident_logits[b, test_indices[b, h, w], h, w] = 10.0  # High logit for correct class
    
    confident_focal_loss = focal_loss_fn(confident_logits, test_indices)
    confident_pt_stats = focal_loss_fn.compute_pt_stats(confident_logits, test_indices)
    print(f"   Focal Loss (confident): {confident_focal_loss.item():.6f}")
    print(f"   Mean p_t (confident): {confident_pt_stats['mean_pt']:.4f}")
    
    # Compare with Cross-Entropy
    ce_loss_random = F.cross_entropy(test_logits, test_indices)
    ce_loss_confident = F.cross_entropy(confident_logits, test_indices)
    print(f"\n   Comparison with Cross-Entropy:")
    print(f"     CE Loss (random): {ce_loss_random.item():.4f}")
    print(f"     CE Loss (confident): {ce_loss_confident.item():.6f}")
    print(f"     Focal Loss reduction factor (random): {focal_loss_val.item() / ce_loss_random.item():.4f}")
    print(f"     Focal Loss reduction factor (confident): {confident_focal_loss.item() / ce_loss_confident.item():.4f}")
    
    # Test combined focal loss
    print("\n4. Combined Focal Translator Loss Test:")
    top_logits = torch.randn(2, 512, 64, 64)
    bottom_logits = torch.randn(2, 512, 128, 128)
    top_indices = torch.randint(0, 512, (2, 64, 64))
    bottom_indices = torch.randint(0, 512, (2, 128, 128))
    
    combined = focal_translator_combined_loss(
        top_logits, bottom_logits,
        top_indices, bottom_indices,
        focal_loss_fn=focal_loss_fn,
        compute_pt_stats=True
    )
    
    print(f"   Total Loss: {combined['total_loss'].item():.4f}")
    print(f"   Top Loss: {combined['top_loss'].item():.4f}")
    print(f"   Bottom Loss: {combined['bottom_loss'].item():.4f}")
    print(f"   Top p_t mean: {combined['top_pt_stats']['mean_pt']:.4f}")
    print(f"   Bottom p_t mean: {combined['bottom_pt_stats']['mean_pt']:.4f}")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
    print("\nInterpretation of p_t:")
    print("  - p_t < 0.01: Model is guessing (expected for 512 classes)")
    print("  - p_t ≈ 0.1-0.3: Model is learning")
    print("  - p_t > 0.5: Model is confident")
    print("  - p_t > 0.9: Model is very confident (may overfit)")
