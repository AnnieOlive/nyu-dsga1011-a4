import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import hashlib
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


detokenizer = TreebankWordDetokenizer()
# Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.
keyboard_neighbors = {
    'q': 'wa', 'w': 'qes', 'e': 'wrd', 'r': 'etf', 't': 'ryg', 'y': 'tuh', 'u': 'yij', 'i': 'uok', 'o': 'ipl', 'p': 'ol',
    'a': 'qsz', 's': 'awdx', 'd': 'sefc', 'f': 'drgv', 'g': 'fthb', 'h': 'gyjn', 'j': 'hukm', 'k': 'jil', 'l': 'kop',
    'z': 'asx', 'x': 'zsdc', 'c': 'xdfv', 'v': 'cfgb', 'b': 'vghn', 'n': 'bhjm', 'm': 'njk'
}

# A moderately typo-heavy but still readable mobile/keyboard input setting.
# A pilot with 0.22 produced only a mild distribution shift; 0.40 creates a
# clearer robustness stress test while every edit remains one adjacent-key
# typo and therefore preserves the review's sentiment.
TYPO_WORD_PROBABILITY = 0.40

# Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation

    text = example["text"]

    # Use a stable per-example RNG so that evaluation is reproducible across
    # runs and independent of datasets.map's caching/worker order.
    seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
    rng = random.Random(seed)

    # 1. 使用 NLTK 进行分词
    tokens = word_tokenize(text)
    new_tokens = []

    for word in tokens:
        # 只对纯字母单词进行扰动，保留标点符号原样
        if (word.isalpha() and len(word) >= 3
                and rng.random() < TYPO_WORD_PROBABILITY):
            # Simulate a realistic adjacent-key typo. Unlike unrestricted
            # WordNet replacement, this leaves the review's semantics intact.
            chars = list(word)
            candidates = [i for i, char in enumerate(chars)
                          if char.lower() in keyboard_neighbors]
            if candidates:
                idx = rng.choice(candidates)
                replacement = rng.choice(keyboard_neighbors[chars[idx].lower()])
                chars[idx] = replacement.upper() if chars[idx].isupper() else replacement
                word = "".join(chars)

        new_tokens.append(word)

    # 2. 把修改后的词重新拼成句子
    example["text"] = detokenizer.detokenize(new_tokens)

    ##### YOUR CODE ENDS HERE ######

    return example
