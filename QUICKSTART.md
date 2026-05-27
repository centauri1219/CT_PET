# Quick Start Guide - CT2PET VQ-VAE-2

## 🎯 Complete Setup in 5 Steps

### Step 1: Test the Pipeline (2 minutes)

```bash
cd /scratch/b24cs1085/CT2PET_VQVAE2
python test_pipeline.py
```

This will verify:
- ✅ All imports work
- ✅ Models can be created
- ✅ Forward pass works
- ✅ Loss computation works
- ✅ Dataset loads correctly

**Expected output:** "✓ ALL TESTS PASSED!"

---

### Step 2: Install Dependencies (if needed)

```bash
# If you don't have the packages installed yet:
pip install nibabel pyyaml tensorboard matplotlib seaborn pandas tqdm

# Or use requirements.txt:
pip install -r requirements.txt
```

---

### Step 3: Verify Your Data

Your data should be in `/scratch/b24cs1085/CPDM/processed_data` with structure:
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

Quick check:
```bash
ls -lh /scratch/b24cs1085/CPDM/processed_data/ | head -10
```

---

### Step 4: Train Phase 1 (VQ-VAE) - 10-12 hours

**Option A: Interactive (for testing)**
```bash
cd /scratch/b24cs1085/CT2PET_VQVAE2
python train_vqvae.py --config configs/config.yaml
```

**Option B: SLURM Job (recommended)**
```bash
cd /scratch/b24cs1085/CT2PET_VQVAE2
chmod +x scripts/job_train_vqvae.sh
sbatch scripts/job_train_vqvae.sh

# Monitor progress:
tail -f logs/vqvae_train_*.out
```

**What to expect:**
- Training for 100 epochs
- Checkpoints saved every 10 epochs in `checkpoints/`
- Validation every 2 epochs
- Best model: `checkpoints/vqvae_best.pth`
- TensorBoard logs: `tensorboard --logdir logs/tensorboard/vqvae`

**Success indicators:**
- ✅ Validation loss decreasing
- ✅ PSNR > 30 dB after ~50 epochs
- ✅ SSIM > 0.85 after ~80 epochs
- ✅ Reconstructed PET images look sharp (check `logs/images/`)

---

### Step 5: Train Phase 2 (Translator) - 20-24 hours

**IMPORTANT:** Only start after Phase 1 is complete!

**Option A: Interactive**
```bash
cd /scratch/b24cs1085/CT2PET_VQVAE2
python train_translator.py --config configs/config.yaml
```

**Option B: SLURM Job (recommended)**
```bash
cd /scratch/b24cs1085/CT2PET_VQVAE2
chmod +x scripts/job_train_translator.sh
sbatch scripts/job_train_translator.sh

# Monitor progress:
tail -f logs/translator_train_*.out
```

**What to expect:**
- Training for 200 epochs
- Uses frozen VQ-VAE from Phase 1
- Checkpoints saved every 20 epochs
- Validation every 5 epochs
- Best model: `checkpoints/translator_best.pth`
- TensorBoard logs: `tensorboard --logdir logs/tensorboard/translator`

**Success indicators:**
- ✅ Cross-entropy loss < 2.0 after 100 epochs
- ✅ PSNR > 25 dB for CT→PET translation
- ✅ SSIM > 0.75
- ✅ Generated PET images capture tumor locations

---

## 🚀 Quick Inference Test

After both phases are trained:

```bash
# Test on a single patient
python inference.py \
    --config configs/config.yaml \
    --data_dir /scratch/b24cs1085/CPDM/processed_data \
    --output_dir outputs/test_inference

# Check results
ls -lh outputs/test_inference/patient_001/
```

You should see:
- `predicted_pet.nii.gz` - Generated PET scan
- `comparison.png` - Visual comparison
- `metrics.txt` - Quantitative metrics

---

## 📊 Monitoring Training

### View TensorBoard (Real-time)

**On compute node:**
```bash
# Start TensorBoard
tensorboard --logdir logs/tensorboard --port 6006 --bind_all

# Note the URL, then forward the port using SSH tunnel from your laptop:
# ssh -L 6006:compute-node:6006 username@server
```

**On your laptop:**
Navigate to `http://localhost:6006`

### Check Training Progress (Without TensorBoard)

```bash
# View training curves
ls -lh logs/*.png

# View sample images
ls -lh logs/images/

# Check latest metrics
tail -n 50 logs/vqvae_train_*.out
tail -n 50 logs/translator_train_*.out
```

---

## 🔧 Quick Fixes

### If Phase 1 fails with OOM (Out of Memory):
Edit `configs/config.yaml`:
```yaml
train_vqvae:
  batch_size: 32  # Reduce from 64
```

### If reconstructions are blurry:
Edit `configs/config.yaml`:
```yaml
vqvae:
  codebook_size: 1024  # Increase from 512
```

### If Phase 2 never learns anything:
1. Check Phase 1 is complete: `ls -lh checkpoints/vqvae_best.pth`
2. Verify VQ-VAE checkpoint path in config
3. Lower learning rate: `learning_rate: 1.0e-4`

---

## 📈 Expected Timeline

| Phase | Duration | GPU Hours |
|-------|----------|-----------|
| Phase 1: VQ-VAE | 10-12 hours | ~12 |
| Phase 2: Translator | 20-24 hours | ~24 |
| **Total** | **30-36 hours** | **~36** |

*With NVIDIA A30 24GB GPU*

---

## ✅ Validation Checklist

Before moving to Phase 2:
- [ ] Phase 1 completed 100 epochs
- [ ] `checkpoints/vqvae_best.pth` exists
- [ ] Validation PSNR > 28 dB
- [ ] Validation SSIM > 0.80
- [ ] Sample reconstructions look good

Before running inference:
- [ ] Phase 2 completed 200 epochs
- [ ] `checkpoints/translator_best.pth` exists
- [ ] CT→PET PSNR > 23 dB
- [ ] Generated PET images are not blank/noisy

---

## 🆘 Getting Help

1. **Check logs first:**
   ```bash
   cat logs/vqvae_train_*.err
   cat logs/translator_train_*.err
   ```

2. **Run test pipeline:**
   ```bash
   python test_pipeline.py
   ```

3. **Common issues:** See README.md "Troubleshooting" section

4. **Still stuck?** Check the error messages for:
   - File paths (data_root in config)
   - CUDA/GPU availability
   - Checkpoint existence

---

## 🎓 Learning Resources

**Understanding VQ-VAE-2:**
- Paper: https://arxiv.org/abs/1906.00446
- Visualizations: Check `logs/images/` during training

**Medical Image Translation:**
- CT values: Hounsfield Units (-1000 to 1000)
- PET values: Standardized Uptake Values (SUV)
- Why log-scaling: SUV has long-tail distribution

---

**Ready to start? Run:**
```bash
python test_pipeline.py && echo "All systems go! 🚀"
```

Good luck! 🎉
