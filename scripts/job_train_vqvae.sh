#!/bin/bash
#SBATCH --job-name=vqvae_train
#SBATCH --output=logs/vqvae_train_%j.out
#SBATCH --error=logs/vqvae_train_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=btech
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Activate environment
conda activate CPDM # or: conda activate ct2pet

# Print job information
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
nvidia-smi

# Navigate to project directory
cd /scratch/b24cs1085/CT2PET_VQVAE2

# Run Phase 1: Train VQ-VAE
echo "Starting VQ-VAE training..."
python train_vqvae.py --config configs/config.yaml

echo "Job completed at: $(date)"
