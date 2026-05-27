"""
U-Net Translator for CT to PET Code Translation
================================================
This module implements two independent U-Net architectures that translate
CT scans to PET codebook indices (Top and Bottom levels).

Instead of generating pixel values, these networks perform classification,
predicting which codebook entry should be used at each spatial location.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Double convolution block: Conv -> GroupNorm -> ReLU -> Conv -> GroupNorm -> ReLU."""
    
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        mid_channels = mid_channels or out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups=32, num_channels=mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups=32, num_channels=out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv."""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv."""
    
    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Handle spatial size mismatch
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate skip connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net architecture for medical image segmentation/translation.
    
    Modified for classification: outputs logits for codebook indices.
    """
    
    def __init__(
        self,
        in_channels=1,
        out_channels=512,  # Number of codebook classes
        base_channels=64,
        bilinear=False,
        dropout=0.1
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        
        # Encoder
        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_channels * 8, base_channels * 16 // factor)
        
        # Dropout for regularization
        self.dropout = nn.Dropout2d(dropout)
        
        # Decoder
        self.up1 = Up(base_channels * 16, base_channels * 8 // factor, bilinear)
        self.up2 = Up(base_channels * 8, base_channels * 4 // factor, bilinear)
        self.up3 = Up(base_channels * 4, base_channels * 2 // factor, bilinear)
        self.up4 = Up(base_channels * 2, base_channels, bilinear)
        
        # Output layer: Classification head
        self.outc = nn.Conv2d(base_channels, out_channels, 1)
    
    def forward(self, x):
        """
        Args:
            x: Input CT image [B, 1, H, W]
        
        Returns:
            logits: Classification logits [B, num_classes, H', W']
                   where H' and W' depend on the target quantization level
        """
        # Encoder with skip connections
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x5 = self.dropout(x5)
        
        # Decoder with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Classification logits
        logits = self.outc(x)
        
        return logits


class UNetTranslator(nn.Module):
    """
    Complete CT-to-PET Translator using two independent U-Nets.
    
    - top_unet: Predicts Top Level codebook indices (64x64)
    - bottom_unet: Predicts Bottom Level codebook indices (128x128)
    """
    
    def __init__(
        self,
        in_channels=1,
        num_classes=512,  # Codebook size
        base_channels=64,
        bilinear=False,
        dropout=0.1
    ):
        super().__init__()
        
        # Two independent U-Nets
        self.top_unet = UNet(
            in_channels=in_channels,
            out_channels=num_classes,
            base_channels=base_channels,
            bilinear=bilinear,
            dropout=dropout
        )
        
        self.bottom_unet = UNet(
            in_channels=in_channels,
            out_channels=num_classes,
            base_channels=base_channels,
            bilinear=bilinear,
            dropout=dropout
        )
    
    def forward(self, ct_image):
        """
        Args:
            ct_image: CT scan [B, 1, 256, 256]
        
        Returns:
            dict with:
                - top_logits: [B, 512, 256, 256] (will be downsampled to 64x64)
                - bottom_logits: [B, 512, 256, 256] (will be downsampled to 128x128)
        """
        top_logits = self.top_unet(ct_image)
        bottom_logits = self.bottom_unet(ct_image)
        
        return {
            'top_logits': top_logits,
            'bottom_logits': bottom_logits
        }
    
    def _topk_sample(self, logits, temperature=1.0, top_k=50):
        """
        Apply Top-K sampling with temperature to logits.
        
        Args:
            logits: [B, C, H, W] - Raw logits from model
            temperature: Controls sharpness of distribution
                - 0.1-0.5: Very conservative (like argmax, smooth blobs)
                - 0.8-0.9: Sweet spot (sharp structure, good variance)
                - 1.0+: Very noisy (might look like "snow")
            top_k: Number of top candidates to sample from
                - 1: Identical to argmax
                - 50-100: Sweet spot (valid texture variations)
                - 512: Pure random sampling (risky)
        
        Returns:
            indices: [B, H, W] - Sampled indices
        """
        B, C, H, W = logits.shape
        
        # Reshape for easier processing: [B, C, H, W] -> [B*H*W, C]
        logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]
        
        # Apply temperature scaling
        scaled_logits = logits_flat / temperature
        
        # Get top-k values and indices
        top_k = min(top_k, C)  # Ensure top_k doesn't exceed num classes
        topk_values, topk_indices = torch.topk(scaled_logits, top_k, dim=-1)  # [B*H*W, top_k]
        
        # Convert to probabilities (only over top-k)
        topk_probs = F.softmax(topk_values, dim=-1)  # [B*H*W, top_k]
        
        # Sample from top-k distribution
        sampled_topk_idx = torch.multinomial(topk_probs, num_samples=1).squeeze(-1)  # [B*H*W]
        
        # Map back to original codebook indices
        sampled_indices = topk_indices.gather(1, sampled_topk_idx.unsqueeze(-1)).squeeze(-1)  # [B*H*W]
        
        # Reshape back to spatial dimensions
        indices = sampled_indices.reshape(B, H, W)
        
        return indices
    
    def predict_indices(self, ct_image, top_target_size=(64, 64), bottom_target_size=(128, 128),
                        sampling_mode='argmax', temperature=0.9, top_k=50):
        """
        Predict codebook indices for a CT image.
        
        Args:
            ct_image: CT scan [B, 1, 256, 256]
            top_target_size: Target spatial size for top level
            bottom_target_size: Target spatial size for bottom level
            sampling_mode: 'argmax' or 'topk'
                - 'argmax': Deterministic, always picks highest probability
                - 'topk': Stochastic, samples from top-k candidates with temperature
            temperature: Temperature for top-k sampling (default: 0.9)
                - 0.1-0.5: Conservative (smooth, like argmax)
                - 0.8-0.9: Sweet spot (sharp structure, good variance)
                - 1.0+: Noisy (may look like "snow")
            top_k: Number of candidates for top-k sampling (default: 50)
                - 1: Same as argmax
                - 50-100: Sweet spot (texture variety without wrong codes)
                - 512: Pure random (risky)
        
        Returns:
            dict with:
                - top_indices: [B, H_top, W_top]
                - bottom_indices: [B, H_bottom, W_bottom]
        """
        with torch.no_grad():
            outputs = self.forward(ct_image)
            
            # Downsample logits to target sizes
            top_logits = F.interpolate(
                outputs['top_logits'],
                size=top_target_size,
                mode='bilinear',
                align_corners=False
            )
            bottom_logits = F.interpolate(
                outputs['bottom_logits'],
                size=bottom_target_size,
                mode='bilinear',
                align_corners=False
            )
            
            # Get predicted indices based on sampling mode
            if sampling_mode == 'topk':
                top_indices = self._topk_sample(top_logits, temperature, top_k)
                bottom_indices = self._topk_sample(bottom_logits, temperature, top_k)
            else:  # argmax (default, deterministic)
                top_indices = torch.argmax(top_logits, dim=1)  # [B, H, W]
                bottom_indices = torch.argmax(bottom_logits, dim=1)  # [B, H, W]
            
            return {
                'top_indices': top_indices,
                'bottom_indices': bottom_indices
            }


if __name__ == "__main__":
    # Test the translator
    translator = UNetTranslator(
        in_channels=1,
        num_classes=512,
        base_channels=64,
        dropout=0.1
    )
    
    # Test input
    ct = torch.randn(4, 1, 256, 256)
    outputs = translator(ct)
    
    print("CT Input shape:", ct.shape)
    print("Top logits shape:", outputs['top_logits'].shape)
    print("Bottom logits shape:", outputs['bottom_logits'].shape)
    
    # Test prediction with argmax
    predictions_argmax = translator.predict_indices(ct, sampling_mode='argmax')
    print("\n=== Argmax Sampling ===")
    print("Top indices shape:", predictions_argmax['top_indices'].shape)
    print("Bottom indices shape:", predictions_argmax['bottom_indices'].shape)
    
    # Test prediction with top-k sampling
    print("\n=== Top-K Sampling (temp=0.9, k=50) ===")
    predictions_topk = translator.predict_indices(ct, sampling_mode='topk', temperature=0.9, top_k=50)
    print("Top indices shape:", predictions_topk['top_indices'].shape)
    print("Bottom indices shape:", predictions_topk['bottom_indices'].shape)
    
    # Compare argmax vs top-k
    argmax_top = predictions_argmax['top_indices']
    topk_top = predictions_topk['top_indices']
    agreement = (argmax_top == topk_top).float().mean() * 100
    print(f"\nArgmax vs Top-K agreement: {agreement:.1f}%")
    print("(Lower agreement = more texture variety)")
    
    # Test different temperatures
    print("\n=== Temperature Comparison ===")
    for temp in [0.3, 0.5, 0.9, 1.2]:
        pred = translator.predict_indices(ct, sampling_mode='topk', temperature=temp, top_k=50)
        agreement = (argmax_top == pred['top_indices']).float().mean() * 100
        print(f"  temp={temp}: {agreement:.1f}% agreement with argmax")
    
    # Test different top_k
    print("\n=== Top-K Comparison ===")
    for k in [1, 10, 50, 100, 512]:
        pred = translator.predict_indices(ct, sampling_mode='topk', temperature=0.9, top_k=k)
        agreement = (argmax_top == pred['top_indices']).float().mean() * 100
        print(f"  top_k={k}: {agreement:.1f}% agreement with argmax")
    
    print("\nTranslator parameters:", sum(p.numel() for p in translator.parameters()) / 1e6, "M")
