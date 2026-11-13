"""Build the compact tokenizer used by the scratch T5 extra-credit model.

The pretrained T5 SentencePiece vocabulary contains many tokens irrelevant to
ATIS SQL.  A WordLevel tokenizer trained only on the assignment corpus makes
random-initialization training substantially more data efficient while keeping
the T5-small encoder/decoder architecture unchanged.
"""

from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast


SPECIAL_TOKENS = ["<pad>", "</s>", "<unk>"]


def build_tokenizer(data_dir="data", output_dir="scratch_tokenizer", vocab_size=8000):
    data_dir = Path(data_dir)
    tokenizer = Tokenizer(models.WordLevel(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.WordLevelTrainer(
        vocab_size=vocab_size,
        min_frequency=1,
        special_tokens=SPECIAL_TOKENS,
    )
    tokenizer.train(
        [str(data_dir / "train.nl"), str(data_dir / "train.sql")], trainer
    )
    wrapped = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="<pad>",
        eos_token="</s>",
        unk_token="<unk>",
    )
    wrapped.save_pretrained(output_dir)
    return wrapped


if __name__ == "__main__":
    tok = build_tokenizer()
    print(
        {
            "vocab_size": len(tok),
            "pad_token_id": tok.pad_token_id,
            "eos_token_id": tok.eos_token_id,
            "unk_token_id": tok.unk_token_id,
        }
    )
