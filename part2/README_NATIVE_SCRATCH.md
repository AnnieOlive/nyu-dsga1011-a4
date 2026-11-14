# Final extra-credit run

From `part2`, run `python train_t5_native_scratch.py` to train, or add
`--inference-only` to use `checkpoints/scr_experiments/native_lt_v1/best`.
The checkpoint is random-initialized T5-small, with native T5 tokenization,
reversible `<` -> `lessthan` SQL encoding, and no train/dev mixing.

Training: seed 42; batch 16; AdamW 1e-4, weight decay 0; BF16; gradient norm
1.0; two warmup epochs then cosine decay across 100 epochs. Every five
epochs, evaluate Dev Record F1 using beam 4 and maximum 512 new tokens.
Select epoch 80: Dev Query EM 54.936, Record EM 66.953, Record F1 69.350.
Test predictions use one checkpoint and no retrieval or ensemble.

Acknowledgment: tokenizer, comparison encoding, and scheduling choices were
informed by a classmate-shared `extra_credit.zip`. This repository's final
run uses independently initialized weights, our own training and outputs.
The older compact-tokenizer scripts are historical experiments.
