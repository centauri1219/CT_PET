# CT to PET Translation using VQ-VAE-2

A complete implementation of CT-to-PET scan translation using Vector Quantized Variational Autoencoder version 2 (VQ-VAE-2) with U-Net based translator networks.

## 📋 Overview

This project implements a **3-phase pipeline** for translating CT scans to PET scans:

1. **Phase 1**: Train VQ-VAE-2 to learn high-quality PET reconstruction from discrete codes
2. **Phase 2**: Train U-Net translator to map CT scans to PET codebook indices
3. **Phase 3**: Inference - Generate synthetic PET scans from CT scans

### Key Features

- ✅ **Hierarchical Vector Quantization**: Two-level (Top: 64×64, Bottom: 128×128) discrete representation
- ✅ **On-the-fly Normalization**: CT windowing and PET log-scaling during training
- ✅ **Mixed Precision Training**: FP16 support for faster training on A30 GPU
- ✅ **TensorBoard Logging**: Real-time monitoring of training progress
- ✅ **Comprehensive Metrics**: PSNR, SSIM, MAE evaluation
- ✅ **Modular Design**: Clean separation of models, datasets, and utilities

## 🏗️ Architecture

### VQ-VAE-2 (Phase 1)
```
PET Image (256×256) 
    → Encoder → Quantizer (512 codes, 64 dim)
    → Decoder → Reconstructed PET (256×256)

Hierarchy:
- Top Level: 64×64 spatial resolution
- Bottom Level: 128×128 spatial resolution
```

### U-Net Translator (Phase 2)
```
CT Image (256×256)
    → U-Net (Top) → Top Indices (64×64)
    → U-Net (Bottom) → Bottom Indices (128×128)
    
Frozen VQ-VAE Decoder → Synthetic PET (256×256)
```

## 📂 Project Structure

```
CT2PET_VQVAE2/
├── configs/
│   └── config.yaml              # Configuration file
├── models/
│   ├── vqvae2.py               # VQ-VAE-2 implementation
│   └── translator.py           # U-Net translator
├── datasets/
│   └── ct_pet_dataset.py       # Dataset loader with normalization
├── utils/
│   ├── losses.py               # Loss functions
│   ├── metrics.py              # PSNR, SSIM, MAE
│   ├── visualization.py        # Plotting utilities
│   └── helpers.py              # General utilities
├── train_vqvae.py              # Phase 1: Train VQ-VAE
├── train_translator.py         # Phase 2: Train Translator
├── inference.py                # Phase 3: Generate PET from CT
├── evaluate.py                 # Comprehensive evaluation
├── checkpoints/                # Model checkpoints
├── logs/                       # Training logs and images
└── README.md                   # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Create conda environment (recommended)
conda create -n ct2pet python=3.9
conda activate ct2pet

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install nibabel pyyaml tensorboard matplotlib seaborn pandas tqdm
```

### 2. Data Preparation

Your data should be organized as:
```
processed_data/
├── patient_001/
│   ├── A.nii.gz  (CT scan)
│   └── B.nii.gz  (PET scan)
├── patient_002/
│   ├── A.nii.gz
│   └── B.nii.gz
...
```

All scans should be:
- **Format**: NIfTI (.nii.gz)
- **Resolution**: 256×256 2D slices
- **CT Range**: Hounsfield Units (-1024 to ~1400)
- **PET Range**: SUV values (0 to ~16000)

### 3. Configuration

Edit `configs/config.yaml` to match your setup:

```yaml
data:
  data_root: "/path/to/processed_data"
  modality_ct: "A"
  modality_pet: "B"
```

### 4. Training

#### Phase 1: Train VQ-VAE-2 (100 epochs, ~10-12 hours on A30)

```bash
python train_vqvae.py --config configs/config.yaml
```

**What it does:**
- Learns to compress PET scans into discrete codes
- Trains reconstruction to be as sharp as possible
- Loss = L1 Reconstruction + VQ Commitment

**Expected output:**
- Checkpoints in `checkpoints/vqvae_*.pth`
- Training curves in `logs/vqvae_training_curves.png`
- Sample reconstructions in `logs/images/`

#### Phase 2: Train Translator (200 epochs, ~20-24 hours on A30)

```bash
python train_translator.py --config configs/config.yaml
```

**What it does:**
- Learns CT → PET code mapping
- Two U-Nets predict Top and Bottom level indices
- Loss = Cross-Entropy (classification)

**Expected output:**
- Checkpoints in `checkpoints/translator_*.pth`
- Training curves in `logs/translator_training_curves.png`
- CT→PET translations in `logs/images/`

### 5. Inference

#### Single CT scan:
```bash
python inference.py \
    --config configs/config.yaml \
    --ct_path path/to/ct.nii.gz \
    --pet_path path/to/pet.nii.gz \  # Optional, for comparison
    --output_dir outputs/sample1
```

#### Batch inference on directory:
```bash
python inference.py \
    --config configs/config.yaml \
    --data_dir /path/to/processed_data \
    --output_dir outputs/batch_results
```

### 6. Evaluation

```bash
python evaluate.py \
    --config configs/config.yaml \
    --split test \
    --output_dir evaluation_results
```

**Outputs:**
- `summary.txt`: Average metrics
- `per_sample_results.csv`: Metrics for each sample
- `metrics_distribution.png`: Distribution plots
- Sample comparison images

## 📊 Monitoring Training

### TensorBoard
```bash
tensorboard --logdir logs/tensorboard
```

Navigate to `http://localhost:6006` to view:
- Training/validation loss curves
- PSNR, SSIM, MAE metrics
- Sample reconstructions/translations

## ⚙️ Configuration Details

### Key Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| **VQ-VAE** |
| Codebook Size | 512 | Number of discrete codes |
| Embedding Dim | 64 | Dimension of each code |
| Base Channels | 128 | Starting channel count |
| Num Res Blocks | 2 | Residual blocks per level |
| **Translator** |
| Base Channels | 64 | U-Net base channels |
| Dropout | 0.1 | Regularization |
| **Training** |
| Batch Size | 64 | Samples per batch |
| Learning Rate | 3e-4 | Adam/AdamW LR |
| Mixed Precision | True | FP16 training |
| **Normalization** |
| CT Window | [-1000, 1000] | Hounsfield clipping |
| PET Log Scale | True | Log-transform SUV |

## 📈 Expected Results

Based on similar medical imaging translation tasks:

| Metric | Expected Range | Best Case |
|--------|---------------|-----------|
| PSNR | 25-32 dB | >30 dB |
| SSIM | 0.75-0.90 | >0.85 |
| MAE | 0.10-0.20 | <0.15 |

*Note: Actual performance depends on dataset quality and training duration*

## 🔧 Troubleshooting

### GPU Out of Memory
- Reduce batch size in `config.yaml`
- Enable gradient checkpointing (modify models)
- Use smaller codebook size (e.g., 256)

### Blurry Reconstructions
- Increase codebook size to 1024
- Add perceptual loss (set `perceptual_loss_weight: 0.1`)
- Train for more epochs

### Poor CT→PET Translation
- Check CT normalization (windowing)
- Verify PET log-scaling is working
- Increase translator training epochs
- Ensure VQ-VAE is well-trained first

### NaN Loss
- Lower learning rate (1e-4)
- Check data normalization
- Clip gradients (add to optimizer)

## 📚 References

1. **VQ-VAE-2**: Razavi et al., "Generating Diverse High-Fidelity Images with VQ-VAE-2", NeurIPS 2019
2. **U-Net**: Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation", MICCAI 2015
3. **Medical Image Translation**: Ben-Cohen et al., "Cross-Modality Synthesis from CT to PET using FCN and GAN Networks for Improved Automated Lesion Detection", 2018

## 🤝 Contributing

This is a research implementation. Feel free to:
- Report issues
- Suggest improvements
- Extend to 3D volumes
- Add attention mechanisms
- Implement conditional generation

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{ct2pet_vqvae2,
  title={CT to PET Translation using VQ-VAE-2},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/ct2pet_vqvae2}
}
```

## 📄 License

This project is provided for research and educational purposes. Please respect the licenses of all dependencies.

## 🙏 Acknowledgments

- VQ-VAE-2 architecture inspired by DeepMind's work
- U-Net implementation adapted for medical imaging
- Dataset preprocessing follows medical imaging standards

## 📧 Contact

For questions or collaborations, please open an issue or contact [your-email@example.com]

---

**Happy Training! 🚀**
