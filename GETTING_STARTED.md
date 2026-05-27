# CT2PET VQ-VAE-2 Quick Start Guide

## ✅ Project Setup Complete!

All components have been successfully created and tested. The pipeline is ready for training.

## 📁 Project Structure

```
CT2PET_VQVAE2/
├── configs/config.yaml          ✅ Configuration file
├── models/
│   ├── vqvae2.py               ✅ VQ-VAE-2 (7.94M params)
│   └── translator.py           ✅ U-Net Translator (62.14M params)
├── datasets/ct_pet_dataset.py  ✅ Dataset loader with normalization
├── utils/                      ✅ Losses, metrics, visualization
├── train_vqvae.py             ✅ Phase 1 training
├── train_translator.py        ✅ Phase 2 training
├── inference.py               ✅ CT→PET generation
├── evaluate.py                ✅ Comprehensive evaluation
├── test_pipeline.py           ✅ Pipeline verification (PASSED!)
└── scripts/                    ✅ SLURM job scripts
```

## 🚀 Training Workflow

### Step 1: Verify Your Data

Your data should be in: `/scratch/b24cs1085/CPDM/processed_data`

Expected structure:
```
processed_data/
├── patient_001/
│   ├── A.nii.gz  (CT)
│   └── B.nii.gz  (PET)
├── patient_002/
│   ├── A.nii.gz
│   └── B.nii.gz
...
```

### Step 2: Phase 1 - Train VQ-VAE-2 (100 epochs, ~10-12 hours)

**Interactive:**
```bash
cd /scratch/b24cs1085/CT2PET_VQVAE2
python train_vqvae.py --config configs/config.yaml
```

**SLURM Job:**
```bash
cd /scratch/b24cs1085/CT2PET_VQVAE2
sbatch scripts/job_train_vqvae.sh
```

**What it does:**
- Learns to compress PET scans into 512 discrete codes
- Reconstructs high-quality PET from codes
- Saves best model to `checkpoints/vqvae_best.pth`

**Monitor progress:**
```bash
tensorboard --logdir logs/tensorboard
```

### Step 3: Phase 2 - Train Translator (200 epochs, ~20-24 hours)

**After Phase 1 completes:**

**Interactive:**
```bash
python train_translator.py --config configs/config.yaml
```

**SLURM Job:**
```bash
sbatch scripts/job_train_translator.sh
```

**What it does:**
- Learns CT → PET code mapping
- Two U-Nets predict Top (64×64) and Bottom (128×128) indices
- Saves best model to `checkpoints/translator_best.pth`

### Step 4: Inference - Generate PET from CT

**Single CT scan:**
```bash
python inference.py \
    --config configs/config.yaml \
    --ct_path path/to/ct.nii.gz \
    --pet_path path/to/pet.nii.gz \  # Optional, for comparison
    --output_dir outputs/sample1
```

**Batch processing:**
```bash
python inference.py \
    --config configs/config.yaml \
    --data_dir /scratch/b24cs1085/CPDM/processed_data \
    --output_dir outputs/all_patients
```

**SLURM Job:**
```bash
sbatch scripts/job_inference.sh
```

### Step 5: Evaluation

```bash
python evaluate.py \
    --config configs/config.yaml \
    --split test \
    --output_dir evaluation_results
```

**Outputs:**
- `summary.txt`: Average PSNR, SSIM, MAE
- `per_sample_results.csv`: Per-patient metrics
- `metrics_distribution.png`: Distribution plots

## 📊 Expected Training Time (on A30 GPU)

| Phase | Duration | Epochs | Output |
|-------|----------|--------|--------|
| Phase 1: VQ-VAE | 10-12 hours | 100 | `vqvae_best.pth` |
| Phase 2: Translator | 20-24 hours | 200 | `translator_best.pth` |
| **Total** | **~30-36 hours** | - | - |

## 🔍 Monitoring Training

### TensorBoard
```bash
tensorboard --logdir logs/tensorboard --port 6006
```

Then open: `http://localhost:6006` (or forward port if on remote server)

### Check Training Logs
```bash
# VQ-VAE training
tail -f logs/vqvae_train_*.out

# Translator training
tail -f logs/translator_train_*.out
```

### View Checkpoints
```bash
ls -lh checkpoints/
```

## ⚙️ Configuration Adjustments

Edit `configs/config.yaml` to customize:

**For faster training (lower quality):**
```yaml
train_vqvae:
  batch_size: 128  # Increase batch size
  num_epochs: 50   # Reduce epochs

train_translator:
  num_epochs: 100  # Reduce epochs
```

**For better quality (slower training):**
```yaml
vqvae:
  codebook_size: 1024  # More codes
  embedding_dim: 128   # Larger embeddings

train_vqvae:
  num_epochs: 150      # More epochs
```

**For GPU memory issues:**
```yaml
train_vqvae:
  batch_size: 32  # Reduce batch size

translator:
  base_channels: 32  # Smaller translator
```

## 🎯 Quick Commands Reference

```bash
# Test pipeline
python test_pipeline.py

# Train Phase 1
python train_vqvae.py --config configs/config.yaml

# Train Phase 2  
python train_translator.py --config configs/config.yaml

# Inference
python inference.py --ct_path ct.nii.gz --output_dir outputs

# Evaluate
python evaluate.py --split test

# Monitor training
tensorboard --logdir logs/tensorboard

# Submit SLURM jobs
sbatch scripts/job_train_vqvae.sh
sbatch scripts/job_train_translator.sh
sbatch scripts/job_inference.sh
```

---

**Good luck with your training! 🚀**
