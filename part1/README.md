# DS-GA 1011: Homework 4 Part 1

> ### Please start early.

## Set up environment
We recommend you set up a conda environment for the packages used in this part of the homework.
```bash
conda create -n dsga1011-hw4-part1 python=3.11
conda activate dsga1011-hw4-part1
pip install -r requirements.txt
```

After this, you will need to install certain packages in nltk.
```bash
python3
>>> import nltk
>>> nltk.download('wordnet')
>>> nltk.download('punkt')
>>> exit()
```

## Submission outputs and seeds

The code sets Python and PyTorch seeds to 0. Augmentation samples 5,000
training reviews with seed 42. Typo transformations use a SHA-256-derived
per-review seed and word probability 0.40.

To regenerate predictions from the saved original and augmented checkpoints:

```bash
python main.py --eval --eval_transformed --model_dir out
python main.py --eval --eval_transformed --model_dir out_augmented
```

The four output files contain alternating prediction and gold-label lines:
50,000 lines represent 25,000 reviews. Accuracy compares each adjacent pair.
Training flags are configurable; defaults (3 epochs, 5e-5) should not be
mistaken for a record of the submitted checkpoint's training run. The original
checkpoint's recorded run used one epoch at 2e-5. A complete preserved training
configuration for every BERT checkpoint is not included in this repository.
