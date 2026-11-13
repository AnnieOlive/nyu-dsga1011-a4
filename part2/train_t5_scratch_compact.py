"""Train the extra-credit T5-small model from random initialization.

This variant uses a corpus-specific tokenizer, native T5 label shifting,
gradient clipping, mixed precision, and dev-loss checkpoint selection.
"""

import argparse
import os
import re
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import T5Config, T5ForConditionalGeneration

from scratch_sql_tokenizer import build_tokenizer
from utils import compute_metrics, save_queries_and_records, set_random_seeds


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_lines(path):
    return Path(path).read_text(encoding="utf-8").splitlines()


class Text2SQLDataset(Dataset):
    def __init__(self, data_dir, split, tokenizer, max_source=128, max_target=512):
        self.tokenizer = tokenizer
        inputs = read_lines(Path(data_dir) / f"{split}.nl")
        targets = [] if split == "test" else read_lines(Path(data_dir) / f"{split}.sql")
        self.rows = []
        for i, text in enumerate(inputs):
            source = tokenizer.encode(
                "translate English to SQL: " + text,
                add_special_tokens=False,
                truncation=True,
                max_length=max_source - 1,
            ) + [tokenizer.eos_token_id]
            target = None
            if split != "test":
                target = tokenizer.encode(
                    targets[i], add_special_tokens=False, truncation=True,
                    max_length=max_target - 1,
                ) + [tokenizer.eos_token_id]
            self.rows.append((source, target))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def collate(rows, pad_id):
    x = pad_sequence([torch.tensor(r[0]) for r in rows], True, padding_value=pad_id)
    mask = x.ne(pad_id).long()
    if rows[0][1] is None:
        return x, mask, None
    y = pad_sequence([torch.tensor(r[1]) for r in rows], True, padding_value=pad_id)
    return x, mask, y


def loss_epoch(model, loader, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    total_loss = total_tokens = 0
    for x, mask, y in tqdm(loader):
        x, mask, y = x.to(DEVICE), mask.to(DEVICE), y.to(DEVICE)
        labels = y.clone()
        labels[labels == 0] = -100
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16,
                            enabled=training and DEVICE.type == "cuda"):
            loss = model(input_ids=x, attention_mask=mask, labels=labels).loss
        if training:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        tokens = labels.ne(-100).sum().item()
        total_loss += loss.item() * tokens
        total_tokens += tokens
    return total_loss / total_tokens


def detokenize_sql(text):
    """Undo spaces introduced by the WordLevel decoder around SQL atoms."""
    text = re.sub(r"\s*\.\s*", ".", text)
    text = re.sub(r"'\s*([^']*?)\s*'", lambda m: "'" + m.group(1).strip() + "'", text)
    text = re.sub(r"<\s*=", "<=", text)
    text = re.sub(r">\s*=", ">=", text)
    text = re.sub(r"!\s*=", "!=", text)
    return text.strip()


@torch.no_grad()
def generate(model, loader, tokenizer, max_new_tokens=512, num_beams=2):
    model.eval()
    outputs = []
    for x, mask, _ in tqdm(loader):
        ids = model.generate(
            input_ids=x.to(DEVICE), attention_mask=mask.to(DEVICE),
            max_new_tokens=max_new_tokens, num_beams=num_beams,
            early_stopping=True,
        )
        outputs.extend(
            detokenize_sql(text)
            for text in tokenizer.batch_decode(ids, skip_special_tokens=True)
        )
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--experiment", default="compact_scratch")
    parser.add_argument("--num_beams", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    set_random_seeds(42)

    root = Path(".")
    tok_dir = root / "checkpoints" / "scr_experiments" / args.experiment / "tokenizer"
    tokenizer = build_tokenizer("data", tok_dir)
    loaders = {}
    for split, batch, shuffle in [
        ("train", args.batch_size, True), ("dev", args.eval_batch_size, False),
        ("test", args.eval_batch_size, False),
    ]:
        ds = Text2SQLDataset("data", split, tokenizer)
        loaders[split] = DataLoader(
            ds, batch_size=batch, shuffle=shuffle,
            collate_fn=lambda rows, p=tokenizer.pad_token_id: collate(rows, p),
        )

    config = T5Config.from_pretrained("google-t5/t5-small")
    config.vocab_size = len(tokenizer)
    config.pad_token_id = tokenizer.pad_token_id
    config.eos_token_id = tokenizer.eos_token_id
    config.decoder_start_token_id = tokenizer.pad_token_id
    ckpt = root / "checkpoints" / "scr_experiments" / args.experiment
    ckpt.mkdir(parents=True, exist_ok=True)
    model = T5ForConditionalGeneration(config).to(DEVICE)
    if args.resume:
        model.load_state_dict(torch.load(ckpt / "best_model.pt", map_location=DEVICE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
    best_loss = loss_epoch(model, loaders["dev"]) if args.resume else float("inf")
    print(f"START_BEST_LOSS {best_loss:.6f}", flush=True)
    for epoch in range(args.epochs):
        train_loss = loss_epoch(model, loaders["train"], optimizer, scaler)
        dev_loss = loss_epoch(model, loaders["dev"])
        print(f"Epoch {epoch}: train_loss={train_loss:.6f} dev_loss={dev_loss:.6f}", flush=True)
        torch.save(model.state_dict(), ckpt / "last_model.pt")
        if dev_loss < best_loss:
            best_loss = dev_loss
            torch.save(model.state_dict(), ckpt / "best_model.pt")

    model.load_state_dict(torch.load(ckpt / "best_model.pt", map_location=DEVICE))
    dev_sql = generate(model, loaders["dev"], tokenizer, num_beams=args.num_beams)
    save_queries_and_records(dev_sql, "results/t5_ft_experiment_ec_compact_dev.sql",
                             "records/t5_ft_experiment_ec_compact_dev.pkl")
    metrics = compute_metrics(
        "data/dev.sql", "results/t5_ft_experiment_ec_compact_dev.sql",
        "records/ground_truth_dev.pkl", "records/t5_ft_experiment_ec_compact_dev.pkl",
    )
    print("COMPACT_DEV", metrics[:3], flush=True)
    test_sql = generate(model, loaders["test"], tokenizer, num_beams=args.num_beams)
    save_queries_and_records(test_sql, "results/t5_ft_experiment_ec_test.sql",
                             "records/t5_ft_experiment_ec_test.pkl")


if __name__ == "__main__":
    main()
