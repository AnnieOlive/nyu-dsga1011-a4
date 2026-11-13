#!/bin/bash
#SBATCH --job-name=a4-t5-scratch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/part2-ec-%j.out
#SBATCH --error=logs/part2-ec-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}/part2"

source "${HOME}/.bashrc"
conda activate dsga1011-hw4-part2

python train_t5.py \
  --experiment_name experiment_ec \
  --output_prefix t5_ft_experiment_ec \
  --learning_rate 1e-3 \
  --batch_size 16 \
  --test_batch_size 16 \
  --max_n_epochs 80 \
  --patience_epochs 8 \
  --num_beams 4

