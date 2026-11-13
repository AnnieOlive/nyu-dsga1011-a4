#!/bin/bash
#SBATCH --job-name=a4-bert
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/part1-%j.out
#SBATCH --error=logs/part1-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}/part1"

source "${HOME}/.bashrc"
conda activate dsga1011-hw4-part1

python main.py --train --eval
python main.py --eval_transformed --model_dir out
python main.py --train_augmented --eval --eval_transformed

