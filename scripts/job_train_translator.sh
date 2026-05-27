#!/bin/bash
#SBATCH --job-name=translator_train
#SBATCH --output=logs/translator_train_%j.out
#SBATCH --error=logs/translator_train_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=btech
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Activate environment
conda activate CPDM  # or: conda activate ct2pet

# Set PyTorch memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Print job information
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
nvidia-smi

# Navigate to project directory
cd /scratch/b24cs1085/CT2PET_VQVAE2

# Run Phase 2: Train Translator
echo "Starting Translator training..."
python train_translator.py --config configs/config.yaml

echo "Job completed at: $(date)"
