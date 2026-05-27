
"""
Evaluation Metrics for CT-to-PET Translation
=============================================
Includes PSNR, SSIM, MAE, and LPIPS for quantitative evaluation.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Union

# Try to import LPIPS
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Warning: lpips not installed. Install with: pip install lpips")


def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor, max_val: float = 2.0) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR).
    
    Args:
        img1: First image [B, C, H, W] or [C, H, W]
        img2: Second image [B, C, H, W] or [C, H, W]
        max_val: Maximum possible pixel value (2.0 for [-1, 1] range)
    
    Returns:
        PSNR value in dB
    """
    mse = F.mse_loss(img1, img2)
    
    if mse == 0:
        return float('inf')
    
    psnr = 20 * torch.log10(torch.tensor(max_val)) - 10 * torch.log10(mse)
    return psnr.item()


def calculate_ssim(img1: torch.Tensor, img2: torch.Tensor, 
                   window_size: int = 11, max_val: float = 2.0) -> float:
    """
    Calculate Structural Similarity Index (SSIM).
    
    Args:
        img1: First image [B, C, H, W] or [C, H, W]
        img2: Second image [B, C, H, W] or [C, H, W]
        window_size: Size of the Gaussian window
        max_val: Maximum possible pixel value
    
    Returns:
        SSIM value (between -1 and 1, higher is better)
    """
    # Ensure 4D tensors
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
    if img2.dim() == 3:
        img2 = img2.unsqueeze(0)
    
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2
    
    # Create Gaussian window
    def create_window(window_size, channel):
        def gaussian(window_size, sigma):
            gauss = torch.Tensor([
                np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
                for x in range(window_size)
            ])
            return gauss / gauss.sum()
        
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    channel = img1.size(1)
    window = create_window(window_size, channel).to(img1.device)
    
    # Calculate means
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    # Calculate variances
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    
    # SSIM formula
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean().item()


def calculate_mae(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """
    Calculate Mean Absolute Error (MAE).
    
    Args:
        img1: First image [B, C, H, W] or [C, H, W]
        img2: Second image [B, C, H, W] or [C, H, W]
    
    Returns:
        MAE value
    """
    mae = F.l1_loss(img1, img2)
    return mae.item()


def calculate_lpips(img1: torch.Tensor, img2: torch.Tensor, net: str = 'alex') -> float:
    """
    Calculate Learned Perceptual Image Patch Similarity (LPIPS).
    
    Args:
        img1: First image [B, C, H, W] or [C, H, W] in range [-1, 1]
        img2: Second image [B, C, H, W] or [C, H, W] in range [-1, 1]
        net: Network type ('alex', 'vgg', or 'squeeze')
    
    Returns:
        LPIPS value (lower is better, typically 0-1)
    """
    if not LPIPS_AVAILABLE:
        return -1.0  # Return -1 to indicate not available
    
    # Ensure 4D tensors
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
    if img2.dim() == 3:
        img2 = img2.unsqueeze(0)
    
    # Convert grayscale to RGB if needed (LPIPS expects 3 channels)
    if img1.size(1) == 1:
        img1 = img1.repeat(1, 3, 1, 1)
    if img2.size(1) == 1:
        img2 = img2.repeat(1, 3, 1, 1)
    
    # Initialize LPIPS model (cached)
    if not hasattr(calculate_lpips, 'loss_fn'):
        calculate_lpips.loss_fn = lpips.LPIPS(net=net).to(img1.device)
        calculate_lpips.loss_fn.eval()
    
    with torch.no_grad():
        lpips_val = calculate_lpips.loss_fn(img1, img2)
    
    return lpips_val.mean().item()


def calculate_all_metrics(pred: torch.Tensor, target: torch.Tensor, include_lpips: bool = True) -> dict:
    """
    Calculate all metrics at once.
    
    Args:
        pred: Predicted image [B, C, H, W]
        target: Target image [B, C, H, W]
        include_lpips: Whether to calculate LPIPS (requires lpips package)
    
    Returns:
        dict with PSNR, SSIM, MAE, and optionally LPIPS
    """
    with torch.no_grad():
        psnr = calculate_psnr(pred, target)
        ssim = calculate_ssim(pred, target)
        mae = calculate_mae(pred, target)
        
        metrics = {
            'PSNR': psnr,
            'SSIM': ssim,
            'MAE': mae
        }
        
        if include_lpips and LPIPS_AVAILABLE:
            lpips_val = calculate_lpips(pred, target)
            metrics['LPIPS'] = lpips_val
    
    return metrics


class MetricsTracker:
    """Track metrics over multiple batches."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.psnr_values = []
        self.ssim_values = []
        self.mae_values = []
    
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """Update metrics with a new batch."""
        metrics = calculate_all_metrics(pred, target)
        self.psnr_values.append(metrics['PSNR'])
        self.ssim_values.append(metrics['SSIM'])
        self.mae_values.append(metrics['MAE'])
    
    def get_average(self) -> dict:
        """Get average metrics."""
        return {
            'PSNR': np.mean(self.psnr_values) if self.psnr_values else 0.0,
            'SSIM': np.mean(self.ssim_values) if self.ssim_values else 0.0,
            'MAE': np.mean(self.mae_values) if self.mae_values else 0.0
        }
    
    def get_std(self) -> dict:
        """Get standard deviation of metrics."""
        return {
            'PSNR_std': np.std(self.psnr_values) if self.psnr_values else 0.0,
            'SSIM_std': np.std(self.ssim_values) if self.ssim_values else 0.0,
            'MAE_std': np.std(self.mae_values) if self.mae_values else 0.0
        }


if __name__ == "__main__":
    # Test metrics
    img1 = torch.randn(4, 1, 256, 256)
    img2 = img1 + 0.1 * torch.randn_like(img1)  # Add noise
    
    print("Testing Metrics:")
    print(f"PSNR: {calculate_psnr(img1, img2):.2f} dB")
    print(f"SSIM: {calculate_ssim(img1, img2):.4f}")
    print(f"MAE: {calculate_mae(img1, img2):.4f}")
    
    # Test tracker
    tracker = MetricsTracker()
    for _ in range(5):
        img1 = torch.randn(4, 1, 256, 256)
        img2 = img1 + 0.1 * torch.randn_like(img1)
        tracker.update(img1, img2)
    
    print("\nAverage metrics over 5 batches:")
    print(tracker.get_average())
