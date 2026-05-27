#!/bin/bash
#SBATCH --job-name=vqvae_recon_test
#SBATCH --partition=btech
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --output=logs/vqvae_recon_test_%j.out
#SBATCH --error=logs/vqvae_recon_test_%j.err

# Activate environment
cd /scratch/b24cs1085/CT2PET_VQVAE2
source activate_env.sh

echo "=========================================="
echo "VQ-VAE Reconstruction Test"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start Time: $(date)"
echo "=========================================="

# Set paths
VQVAE_CHECKPOINT="checkpoints/vqvae_best.pth"
DATA_PATH="/scratch/b24cs1085/CPDM/processed_data"
OUTPUT_DIR="outputs/vqvae_reconstruction_test"

# Create output directory
mkdir -p logs
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
