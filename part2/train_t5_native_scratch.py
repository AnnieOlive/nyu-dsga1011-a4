"""Random-init T5-small with native tokenizer and reversible '<' encoding.

Design choices informed by a classmate's shared extra_credit.zip implementation;
this run uses independently initialized weights and our own training outputs.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import (T5Config, T5ForConditionalGeneration,
                          T5TokenizerFast, get_cosine_schedule_with_warmup)
from utils import compute_metrics, save_queries_and_records


class NativeDataset(Dataset):
    def __init__(self, split, tokenizer):
        text = Path(f'data/{split}.nl').read_text().splitlines()
        self.x = tokenizer([s.strip() for s in text], add_special_tokens=True)['input_ids']
        self.y = None
        if split != 'test':
            sql = Path(f'data/{split}.sql').read_text().splitlines()
            assert len(sql) == len(text)
            self.y = tokenizer([s.strip().replace('<', 'lessthan') for s in sql],
                               add_special_tokens=True)['input_ids']

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], None if self.y is None else self.y[i]


def collate(rows):
    x = pad_sequence([torch.tensor(r[0]) for r in rows], batch_first=True, padding_value=0)
    y = None if rows[0][1] is None else pad_sequence(
        [torch.tensor(r[1]) for r in rows], batch_first=True, padding_value=-100)
    return x, x.ne(0).long(), y


@torch.no_grad()
def generate(model, loader, tokenizer, device):
    model.eval()
    queries = []
    for x, mask, _ in loader:
        ids = model.generate(input_ids=x.to(device), attention_mask=mask.to(device),
                             num_beams=4, max_new_tokens=512)
        queries.extend(s.replace('lessthan', '<').replace('\n', ' ').strip()
                       for s in tokenizer.batch_decode(ids, skip_special_tokens=True,
                                                       clean_up_tokenization_spaces=False))
    return queries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inference-only', action='store_true')
    parser.add_argument('--experiment', default='native_lt_v1')
    args = parser.parse_args()
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
    loaders = {s: DataLoader(NativeDataset(s, tokenizer), batch_size=16,
                             shuffle=s == 'train', collate_fn=collate)
               for s in ['train', 'dev', 'test']}
    directory = Path('checkpoints/scr_experiments') / args.experiment
    directory.mkdir(parents=True, exist_ok=True)
    for p in ['results', 'records']:
        Path(p).mkdir(exist_ok=True)
    tokenizer.save_pretrained(directory / 'tokenizer')
    if not args.inference_only:
        model = T5ForConditionalGeneration(T5Config.from_pretrained('google-t5/t5-small')).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, 2 * len(loaders['train']), 100 * len(loaders['train']))
        best, history = -1, []
        for epoch in range(100):
            model.train()
            total = count = 0
            for x, mask, y in loaders['train']:
                x, mask, y = x.to(device), mask.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                    loss = model(input_ids=x, attention_mask=mask, labels=y).loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                n = y.ne(-100).sum().item()
                total += loss.item() * n
                count += n
            print('EPOCH', epoch + 1, 'LOSS', total / count, flush=True)
            if (epoch + 1) % 5 == 0:
                queries = generate(model, loaders['dev'], tokenizer, device)
                sp, rp = 'results/native_lt_dev.sql', 'records/native_lt_dev.pkl'
                save_queries_and_records(queries, sp, rp)
                metrics = compute_metrics('data/dev.sql', sp, 'records/ground_truth_dev.pkl', rp)
                f1 = float(metrics[2])
                history.append(dict(epoch=epoch + 1, f1=f1, loss=total / count))
                (directory / 'history.json').write_text(json.dumps(history, indent=2))
                if f1 > best:
                    best = f1
                    model.save_pretrained(directory / 'best')
                print('DEV', metrics[:3], flush=True)
            model.save_pretrained(directory / 'last')
        del model, optimizer, scheduler
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    model = T5ForConditionalGeneration.from_pretrained(directory / 'best').to(device)
    for split in ['dev', 'test']:
        queries = generate(model, loaders[split], tokenizer, device)
        stem = 'native_lt_best_dev' if split == 'dev' else 't5_ft_experiment_ec_test'
        save_queries_and_records(queries, f'results/{stem}.sql', f'records/{stem}.pkl')


if __name__ == '__main__':
    main()
