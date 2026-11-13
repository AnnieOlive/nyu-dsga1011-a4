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

`train_t5_scratch_compact.py` trains the T5-small architecture from random initialization with a WordLevel tokenizer learned only from `train.nl` and `train.sql`. The selected checkpoint uses direct greedy generation (`--num_beams 1`). Development scores are SQL EM 22.532, Record EM 58.369, and Record F1 62.056. The outputs are generated only by this scratch model; no pretrained weights, retrieval, ensemble, or query replacement is used.
