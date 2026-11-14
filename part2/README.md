# DS-GA 1011: Homework 4 Part 2

## Environment

Use a separate Python environment for Part 2 because its package versions differ from Part 1.

```bash
conda create -n dsga1011-hw4-part2 python=3.11
conda activate dsga1011-hw4-part2
python -m pip install -r requirements.txt
```

## Evaluation

```bash
python evaluate.py \
  --predicted_sql results/t5_ft_experiment_dev.sql \
  --predicted_records records/t5_ft_experiment_dev.pkl \
  --development_sql data/dev.sql \
  --development_records records/ground_truth_dev.pkl
```

## Submission

Submit `t5_ft_experiment_test.sql` and `t5_ft_experiment_test.pkl`. For extra credit, submit `t5_ft_experiment_ec_test.sql` and `t5_ft_experiment_ec_test.pkl`. Predictions and record entries remain in the same order as `data/test.nl`.

## Extra credit: T5-small from scratch

The final run uses `train_t5_native_scratch.py`: randomly initialized T5-small,
the native T5 tokenizer, and reversible encoding of the less-than SQL operator.
Train with `python train_t5_native_scratch.py`; use `--inference-only` to generate
from the saved best checkpoint. See `README_NATIVE_SCRATCH.md` for the complete
configuration and acknowledgment of the classmate-shared reference design.

The selected checkpoint is epoch 80 of 100, selected by Dev Record F1 evaluated
every five epochs. Dev Query EM is 54.936%, Record EM 66.953%, and Record F1
69.350%. It uses four-beam generation from one independently trained checkpoint.
The older WordLevel scripts remain as historical experiments and do not generate
the final submitted EC predictions.
