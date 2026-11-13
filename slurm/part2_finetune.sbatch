#!/bin/bash
#SBATCH --job-name=a4-t5-ft
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/part2-ft-%j.out
#SBATCH --error=logs/part2-ft-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}/part2"

source "${HOME}/.bashrc"
conda activate dsga1011-hw4-part2

python train_t5.py \
  --finetune \
  --experiment_name experiment \
  --learning_rate 5e-4 \
  --batch_size 16 \
  --test_batch_size 16 \
  --max_n_epochs 20 \
  --patience_epochs 4 \
  --num_beams 4

