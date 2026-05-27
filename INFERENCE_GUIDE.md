# CT2PET Inference Guide

## Quick Start

### 1. Install LPIPS (Optional but Recommended)
```bash
conda activate CPDM
pip install lpips
```

### 2. Run Inference

#### Option A: Batch Processing Test Set (Recommended)
```bash
python inference.py \
    --config configs/config.yaml \
    --data_dir /scratch/b24cs1085/CPDM/processed_data \
    --num_samples 500 \
    --save_images_per_patient 10 \
    --output_dir outputs/test_results
```

**Parameters:**
- `--num_samples 500`: Evaluate 500 random slices (ALL contribute to metrics)
- `--save_images_per_patient 10`: Save first 10 slices per patient as images (for visualization)

**Note:** ALL sampled slices are evaluated for metrics, but only `save_images_per_patient` slices are saved as PNG/NPY to conserve disk space.

#### Option B: Process All Test Slices
```bash
python inference.py \
    --config configs/config.yaml \
    --data_dir /scratch/b24cs1085/CPDM/processed_data \
    --num_samples 99999 \
    --output_dir outputs/full_test
```

#### Option C: Using Job Script (For Large-Scale Evaluation)
```bash
cd /scratch/b24cs1085/CT2PET_VQVAE2/scripts
sbatch job_inference.sh
```

**Note**: Your data is stored as **2D slice NPY files** in:
- `processed_data/test/A/` - CT slices
- `processed_data/test/B/` - PET slices

---

## Output Structure

### Test Set Evaluation Output
```
outputs/test_results/
├── Lung_Dx-A0167/              # Patient 1
│   ├── slice000/
│   │   ├── predicted_pet.npy   # Generated PET (saved for first N slices)
│   │   ├── comparison.png      # Visual comparison
│   │   └── metrics.txt         # Per-slice metrics
│   ├── slice001/
│   ├── ...                     # Up to --save_images_per_patient slices
│   └── patient_summary.txt     # ⭐ Avg metrics across ALL patient's slices
├── Lung_Dx-A0168/              # Patient 2
│   └── ...
└── summary.txt                 # ⭐⭐ OVERALL RESULTS (all patients)
```

**Important:**
- **ALL evaluated slices** contribute to metrics (not just saved ones)
- Images saved: First `--save_images_per_patient` slices per patient (default: 5)
- Metrics computed: ALL `--num_samples` slices (default: 100)
- `patient_summary.txt`: Average across **all slices** from that patient
- `summary.txt`: Average across **all patients**

**Ground Truth Matching:**
- CT and PET files matched by exact filename (e.g., `Lung_Dx-A0167_slice000.npy`)
- Script verifies matching before evaluation (will error if mismatch detected)

---

## Metrics Explained

### metrics.txt Format
```
PSNR: 28.45 dB        # Peak Signal-to-Noise Ratio (higher is better, >25 is good)
SSIM: 0.8234          # Structural Similarity (0-1, higher is better, >0.8 is good)
MAE: 0.0342           # Mean Absolute Error (lower is better)
LPIPS: 0.1245         # Learned Perceptual Similarity (lower is better, <0.2 is good)
```

### Metric Guidelines
- **PSNR > 25 dB**: Good reconstruction quality
- **SSIM > 0.80**: High structural similarity
- **MAE < 0.05**: Low pixel-wise error
- **LPIPS < 0.20**: Good perceptual similarity

---

## Using Different Checkpoints

### Use Best Model (Default)
```bash
python inference.py \
    --config configs/config.yaml \
    --ct_path input.nii.gz \
    --output_dir outputs
```

### Use Specific Epoch Checkpoint
Edit `configs/config.yaml`:
```yaml
inference:
  vqvae_checkpoint: "checkpoints/vqvae_best.pth"
  translator_checkpoint: "checkpoints/translator_epoch_100.pth"  # Change this
```

Or create a custom config:
```bash
cp configs/config.yaml configs/config_epoch100.yaml
# Edit the checkpoint paths
python inference.py --config configs/config_epoch100.yaml --ct_path input.nii.gz
```

---

## Visualizing Results

### 1. View Images
```bash
# Install visualization tool if needed
pip install nibabel matplotlib

# View generated PET
python -c "
import nibabel as nib
import matplotlib.pyplot as plt
img = nib.load('outputs/single_patient/predicted_pet.nii.gz')
plt.imshow(img.get_fdata()[:, :, img.shape[2]//2], cmap='hot')
plt.title('Generated PET (Middle Slice)')
plt.colorbar()
plt.savefig('pet_preview.png')
"
```

### 2. View Comparison Images
The `comparison.png` files show:
- **Left**: Input CT scan
- **Middle**: Ground truth PET (if provided)
- **Right**: Generated PET

### 3. TensorBoard (Training Metrics)
```bash
cd /scratch/b24cs1085/CT2PET_VQVAE2
tensorboard --logdir=logs/tensorboard --port=6006
```

---

## Test Set Evaluation

To evaluate on the official test set:

```bash
python inference.py \
    --config configs/config.yaml \
    --data_dir /scratch/b24cs1085/CPDM/processed_data/test \
    --output_dir outputs/test_results
```

Check `outputs/test_results/summary.txt` for aggregated metrics across all test patients.

---

## Troubleshooting

### Issue: "Checkpoint not found"
**Solution**: Check if training completed successfully
```bash
ls -lh checkpoints/translator_best.pth
```

### Issue: "CUDA out of memory during inference"
**Solution**: Reduce batch size in config
```yaml
inference:
  batch_size: 4  # Reduce from 16
```

Or process one patient at a time using `--ct_path` instead of `--data_dir`.

### Issue: "LPIPS not available"
**Solution**: Install lpips
```bash
conda activate CPDM
pip install lpips
```

### Issue: Wrong output format
**Solution**: Outputs are in NIfTI (.nii.gz) format. Use medical imaging tools:
- **ITK-SNAP**: Visual inspection
- **3D Slicer**: Advanced visualization
- **nibabel**: Python library for reading

---

## Performance Tips

1. **GPU Inference**: Much faster than CPU
   ```bash
   # Check GPU availability
   nvidia-smi
   ```

2. **Batch Processing**: Use `--data_dir` for multiple patients
   - Processes all patients automatically
   - Generates summary statistics

3. **Parallel Processing**: Edit job script for multi-GPU
   ```bash
   #SBATCH --gres=gpu:2  # Use 2 GPUs
   ```

---

## Next Steps

1. ✅ **Run test inference** on a single patient
2. ✅ **Check metrics.txt** - ensure PSNR > 25, SSIM > 0.8
3. ✅ **View comparison.png** - visual quality check
4. ✅ **Batch process** entire test set
5. ✅ **Analyze summary.txt** - average metrics across all patients

---

## Example Workflow

```bash
# 1. Activate environment
conda activate CPDM

# 2. Install LPIPS
pip install lpips

# 3. Run inference on test set (200 random slices)
cd /scratch/b24cs1085/CT2PET_VQVAE2
python inference.py \
    --config configs/config.yaml \
    --data_dir /scratch/b24cs1085/CPDM/processed_data \
    --num_samples 200 \
    --output_dir outputs/test_200

# 4. Check results
cat outputs/test_200/summary.txt

# 5. View sample images
ls outputs/test_200/slice_*/comparison.png

# 6. If results look good, evaluate on more samples or submit job
sbatch scripts/job_inference.sh

# 7. Monitor job
squeue -u $(whoami)
tail -f logs/inference_*.out
```

## Understanding Your Data

Your dataset structure:
```
processed_data/
├── train/
│   ├── A/  # CT slices: Lung_Dx-A0167_slice000.npy, ..., Lung_Dx-A0168_slice000.npy, ...
│   └── B/  # PET slices: Lung_Dx-A0167_slice000.npy, ..., Lung_Dx-A0168_slice000.npy, ...
├── val/
│   ├── A/
│   └── B/
└── test/
    ├── A/  # ~6740 CT slices from multiple patients
    └── B/  # ~6740 PET slices (ground truth)
```

**File naming convention:**
- Format: `PatientID_sliceXXX.npy` (e.g., `Lung_Dx-A0167_slice000.npy`)
- Each patient has multiple slices (typically 20-100 slices per patient)
- All patient slices are stored flat in A/ and B/ folders
- Inference groups results by patient ID automatically

**Metrics calculation:**
1. Evaluate each slice individually
2. Average metrics per patient (across their slices)
3. Overall average across all patients

---

## Citation & References

If LPIPS is used:
```
@inproceedings{zhang2018perceptual,
  title={The Unreasonable Effectiveness of Deep Features as a Perceptual Metric},
  author={Zhang, Richard and Isola, Phillip and Efros, Alexei A and Shechtman, Eli and Wang, Oliver},
  booktitle={CVPR},
  year={2018}
}
```
