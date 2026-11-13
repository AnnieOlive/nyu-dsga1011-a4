# GPU job submission

Create the two conda environments from the part-specific `requirements.txt` files before submitting.
The partition is deliberately supplied at submission time so the scripts can use whichever GPU
partition is available to the account:

```bash
mkdir -p logs
sbatch --partition=<PARTITION> slurm/part1.sbatch
sbatch --partition=<PARTITION> slurm/part2_finetune.sbatch
sbatch --partition=<PARTITION> slurm/part2_extra_credit.sbatch
```

The jobs write all assignment outputs into `part1/`, `part2/results/`, and `part2/records/`.
