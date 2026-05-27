"""
VQ-VAE-2 Model for PET Scan Generation
========================================
This implements a hierarchical Vector Quantized Variational Autoencoder
with two levels (Top and Bottom) for high-quality PET image reconstruction.

Architecture:
- Encoder: Downsamples images hierarchically (Top: 4x, Bottom: 2x)
- Quantizer: Maps continuous features to discrete codebook vectors
- Decoder: Reconstructs images from quantized codes hierarchically
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block with GroupNorm and ReLU activation."""
    
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        out_channels = out_channels or in_channels
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        
        # Projection shortcut if channels differ
        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
    
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = F.relu(out, inplace=True)
        
        out = self.conv2(out)
        out = self.norm2(out)
        out = out + residual
        out = F.relu(out, inplace=True)
        
        return out


class Encoder(nn.Module):
    """Hierarchical Encoder for VQ-VAE-2."""
    
    def __init__(self, in_channels, base_channels, num_res_blocks, embedding_dim):
        super().__init__()
        
        # Bottom Encoder (256 -> 128)
        self.bottom_conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.bottom_conv_down = nn.Conv2d(base_channels, base_channels, 4, stride=2, padding=1)
        self.bottom_res_blocks = nn.Sequential(
            *[ResidualBlock(base_channels) for _ in range(num_res_blocks)]
        )
        self.bottom_conv_out = nn.Conv2d(base_channels, embedding_dim, 1)
        
        # Top Encoder (128 -> 64)
        self.top_conv_down = nn.Conv2d(base_channels, base_channels * 2, 4, stride=2, padding=1)
        self.top_res_blocks = nn.Sequential(
            *[ResidualBlock(base_channels * 2) for _ in range(num_res_blocks)]
        )
        self.top_conv_out = nn.Conv2d(base_channels * 2, embedding_dim, 1)
    
    def forward(self, x):
        # Bottom level encoding
        h = self.bottom_conv_in(x)
        h = self.bottom_conv_down(h)
        h = self.bottom_res_blocks(h)
        bottom_enc = self.bottom_conv_out(h)
        
        # Top level encoding
        h = self.top_conv_down(h)
        h = self.top_res_blocks(h)
        top_enc = self.top_conv_out(h)
        
        return top_enc, bottom_enc


class Decoder(nn.Module):
    """Hierarchical Decoder for VQ-VAE-2."""
    
    def __init__(self, out_channels, base_channels, num_res_blocks, embedding_dim):
        super().__init__()
        
        # Top Decoder (64 -> 128)
        self.top_conv_in = nn.Conv2d(embedding_dim, base_channels * 2, 3, padding=1)
        self.top_res_blocks = nn.Sequential(
            *[ResidualBlock(base_channels * 2) for _ in range(num_res_blocks)]
        )
        self.top_conv_up = nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1)
        
        # Bottom Decoder (128 -> 256)
        # Combine top features with bottom quantized codes
        self.bottom_conv_in = nn.Conv2d(embedding_dim + base_channels, base_channels, 3, padding=1)
        self.bottom_res_blocks = nn.Sequential(
            *[ResidualBlock(base_channels) for _ in range(num_res_blocks)]
        )
        self.bottom_conv_up = nn.ConvTranspose2d(base_channels, base_channels, 4, stride=2, padding=1)
        self.bottom_conv_out = nn.Conv2d(base_channels, out_channels, 3, padding=1)
    
    def forward(self, top_quant, bottom_quant):
        # Top level decoding
        h_top = self.top_conv_in(top_quant)
        h_top = self.top_res_blocks(h_top)
        h_top = self.top_conv_up(h_top)
        
        # Bottom level decoding (combine with top features)
        h = torch.cat([bottom_quant, h_top], dim=1)
        h = self.bottom_conv_in(h)
        h = self.bottom_res_blocks(h)
        h = self.bottom_conv_up(h)
        recon = self.bottom_conv_out(h)
        
        return recon


class VectorQuantizer(nn.Module):
    """
    Vector Quantization layer.
    Maps continuous embeddings to discrete codebook entries.
    """
    
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # Initialize codebook
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
    
    def forward(self, z):
        """
        Args:
            z: Input tensor [B, C, H, W]
        Returns:
            z_q: Quantized tensor [B, C, H, W]
            loss: VQ loss (commitment + codebook)
            indices: Codebook indices [B, H, W]
        """
        # Flatten spatial dimensions
        z_flattened = z.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        B, H, W, C = z_flattened.shape
        z_flattened = z_flattened.view(-1, C)  # [B*H*W, C]
        
        # Calculate distances to codebook entries
        # ||z - e||^2 = ||z||^2 + ||e||^2 - 2*z*e
        distances = (
            torch.sum(z_flattened ** 2, dim=1, keepdim=True) +
            torch.sum(self.embedding.weight ** 2, dim=1) -
            2 * torch.matmul(z_flattened, self.embedding.weight.t())
        )  # [B*H*W, num_embeddings]
        
        # Find nearest codebook entry
        encoding_indices = torch.argmin(distances, dim=1)  # [B*H*W]
        
        # Quantize
        z_q = self.embedding(encoding_indices).view(B, H, W, C)  # [B, H, W, C]
        
        # VQ Loss
        # Codebook loss: move embeddings towards encoder outputs
        codebook_loss = F.mse_loss(z_q.detach(), z_flattened.view(B, H, W, C))
        # Commitment loss: encourage encoder to commit to embeddings
        commitment_loss = F.mse_loss(z_q, z_flattened.view(B, H, W, C).detach())
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss
        
        # Restore shape to match input format [B, C, H, W]
        z_q = z_q.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
        
        # Straight-through estimator: copy gradients from z_q to z
        z_q = z + (z_q - z).detach()
        
        indices = encoding_indices.view(B, H, W)  # [B, H, W]
        
        return z_q, vq_loss, indices
    
    def quantize_indices(self, indices):
        """Convert indices to quantized vectors."""
        z_q = self.embedding(indices)  # [B, H, W, C]
        z_q = z_q.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
        return z_q


class VQVAE2(nn.Module):
    """
    VQ-VAE-2: Hierarchical Vector Quantized Variational Autoencoder
    
    This model learns to compress PET images into discrete codes at two
    hierarchical levels and reconstruct high-quality PET scans.
    """
    
    def __init__(
        self,
        in_channels=1,
        base_channels=128,
        num_res_blocks=2,
        codebook_size=512,
        embedding_dim=64,
        commitment_cost=0.25
    ):
        super().__init__()
        
        self.encoder = Encoder(in_channels, base_channels, num_res_blocks, embedding_dim)
        self.decoder = Decoder(in_channels, base_channels, num_res_blocks, embedding_dim)
        
        self.quantizer_top = VectorQuantizer(codebook_size, embedding_dim, commitment_cost)
        self.quantizer_bottom = VectorQuantizer(codebook_size, embedding_dim, commitment_cost)
    
    def encode(self, x):
        """Encode image to discrete codes."""
        top_enc, bottom_enc = self.encoder(x)
        
        top_quant, top_loss, top_indices = self.quantizer_top(top_enc)
        bottom_quant, bottom_loss, bottom_indices = self.quantizer_bottom(bottom_enc)
        
        return {
            'top_quant': top_quant,
            'bottom_quant': bottom_quant,
            'top_indices': top_indices,
            'bottom_indices': bottom_indices,
            'vq_loss': top_loss + bottom_loss
        }
    
    def decode(self, top_quant, bottom_quant):
        """Decode quantized codes to image."""
        return self.decoder(top_quant, bottom_quant)
    
    def decode_indices(self, top_indices, bottom_indices):
        """Decode from discrete indices."""
        top_quant = self.quantizer_top.quantize_indices(top_indices)
        bottom_quant = self.quantizer_bottom.quantize_indices(bottom_indices)
        return self.decode(top_quant, bottom_quant)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input PET image [B, 1, 256, 256]
        
        Returns:
            dict with:
                - recon: Reconstructed image
                - vq_loss: Vector quantization loss
                - top_indices: Top level codebook indices
                - bottom_indices: Bottom level codebook indices
        """
        enc_output = self.encode(x)
        recon = self.decode(enc_output['top_quant'], enc_output['bottom_quant'])
        
        return {
            'recon': recon,
            'vq_loss': enc_output['vq_loss'],
            'top_indices': enc_output['top_indices'],
            'bottom_indices': enc_output['bottom_indices']
        }


if __name__ == "__main__":
    # Test the model
    model = VQVAE2(
        in_channels=1,
        base_channels=128,
        num_res_blocks=2,
        codebook_size=512,
        embedding_dim=64,
        commitment_cost=0.25
    )
    
    # Test input
    x = torch.randn(4, 1, 256, 256)
    output = model(x)
    
    print("Input shape:", x.shape)
    print("Reconstruction shape:", output['recon'].shape)
    print("Top indices shape:", output['top_indices'].shape)
    print("Bottom indices shape:", output['bottom_indices'].shape)
    print("VQ Loss:", output['vq_loss'].item())
    print("\nModel parameters:", sum(p.numel() for p in model.parameters()) / 1e6, "M")
