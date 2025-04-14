#!/bin/bash
#SBATCH --ntasks=1                # Number of tasks
#SBATCH --nodes=1                 # Ensure that all cores are on one machine
#SBATCH --time=3-00:00            # Runtime in D-HH:MM (3 days)
#SBATCH --cpus-per-task=8         # Number of CPU cores per task
#SBATCH --mem=120G                # Memory to allocate
#SBATCH --output=slurm_logs/%j.out # File to which STDOUT will be written
#SBATCH --error=slurm_logs/%j.err  # File to which STDERR will be written
#SBATCH --gres=gpu:1              # Request one GPU
#SBATCH --partition=h100-ferranti # Partition name

# Diagnostic and Analysis Phase
scontrol show job $SLURM_JOB_ID
pwd
nvidia-smi
ls $WORK

# Setup Phase
echo "Setting up environment..."
cd /home/vernade/vns988/attention_svd/KPCA 
source /home/vernade/vns988/attention_svd/KPCA/env/bin/activate 
cd /home/vernade/vns988/attention_svd/KPCA/RPCA/Reconstruction

# Display Python and environment info for debugging
which python
python --version
pip list | grep torch

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "Started at: $(date)"

# Create logs directory if it doesn't exist
mkdir -p slurm_logs

# Run the training script
python main_train.py \
    --model deit_tiny_patch16_224 \
    --batch-size 256 \
    --data-path "/weka/datasets/ImageNet2012" \  
    --output_dir "checkpoints" \
    --lr 1e-4 \
    --warmup-epochs 0 \
    --wandb \
    --wandb_project "project" \
    --wandb_entity "entity" \
    --wandb_run_name "reconstruction" \

echo "Finished at: $(date)"