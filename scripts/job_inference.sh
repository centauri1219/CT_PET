#!/bin/bash
#SBATCH --job-name=inference
#SBATCH --output=logs/inference_%j.out
#SBATCH --error=logs/inference_%j.err
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

# Navigate to project directory
cd /scratch/b24cs1085/CT2PET_VQVAE2

# Run inference on test data (NPY slices)
echo "Starting inference on test set..."
python inference.py \
    --config configs/config.yaml \
    --data_dir /scratch/b24cs1085/CPDM/processed_data \
    --num_samples 500 \
    --save_images_per_patient 1 \
    --output_dir outputs/test_results2

echo "Job completed at: $(date)"
