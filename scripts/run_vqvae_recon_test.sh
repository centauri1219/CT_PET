#!/bin/bash
# Quick test script for VQ-VAE reconstruction (without SLURM)

cd /scratch/b24cs1085/CT2PET_VQVAE2

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate /scratch/b24cs1085/envs/CPDM

echo "=========================================="
echo "VQ-VAE Reconstruction Test"
echo "Start Time: $(date)"
echo "=========================================="

# Set paths - ADJUST THESE IF NEEDED
VQVAE_CHECKPOINT="checkpoints/vqvae_best.pth"
DATA_PATH="/scratch/b24cs1085/CPDM/processed_data"
OUTPUT_DIR="outputs/vqvae_reconstruction_test"

# Check if checkpoint exists
if [ ! -f "$VQVAE_CHECKPOINT" ]; then
    echo "ERROR: VQ-VAE checkpoint not found at $VQVAE_CHECKPOINT"
    echo "Please update the VQVAE_CHECKPOINT path in this script"
    echo ""
    echo "Available checkpoints:"
    find checkpoints/ -name "*.pth" -o -name "*.ckpt" 2>/dev/null
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run reconstruction test
python test_vqvae_reconstruction.py \
    --vqvae_checkpoint "$VQVAE_CHECKPOINT" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 16 \
    --num_workers 4 \
    --split test \
    --max_samples 500

echo "=========================================="
echo "Test completed!"
echo "End Time: $(date)"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="
