import os
import argparse
import sqlite3
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np
import wandb

from t5_utils import initialize_model, initialize_optimizer_and_scheduler, save_model, load_model_from_checkpoint, setup_wandb
from transformers import GenerationConfig
from load_data import load_t5_data
from utils import compute_metrics, save_queries_and_records
from utils import set_random_seeds

DEVICE = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
PAD_IDX = 0


def get_args():
    '''
    Arguments for training. You may choose to change or extend these as you see fit.
    '''
    parser = argparse.ArgumentParser(description='T5 training loop')

    # Model hyperparameters
    parser.add_argument('--finetune', action='store_true',
                        help="Whether to finetune T5 or not")

    # Training hyperparameters
    parser.add_argument('--optimizer_type', type=str, default="AdamW", choices=["AdamW"],
                        help="What optimizer to use")
    parser.add_argument('--learning_rate', type=float, default=None,
                        help="Defaults to 5e-4 for fine-tuning and 1e-3 from scratch")
    parser.add_argument('--weight_decay', type=float, default=0)

    parser.add_argument('--scheduler_type', type=str, default="cosine", choices=["none", "cosine", "linear"],
                        help="Whether to use a LR scheduler and what type to use if so")
    parser.add_argument('--num_warmup_epochs', type=int, default=0,
                        help="How many epochs to warm up the learning rate for if using a scheduler")
    parser.add_argument('--max_n_epochs', type=int, default=20,
                        help="How many epochs to train the model for")
    parser.add_argument('--patience_epochs', type=int, default=4,
                        help="If validation performance stops improving, how many epochs should we wait before stopping?")

    parser.add_argument('--use_wandb', action='store_true',
                        help="If set, we will use wandb to keep track of experiments")
    parser.add_argument('--experiment_name', type=str, default='experiment',
                        help="How should we name this experiment?")

    # Data hyperparameters
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=16)
    parser.add_argument('--num_beams', type=int, default=4)
    parser.add_argument('--num_return_sequences', type=int, default=1,
                        help="Number of beam candidates to generate per example")
    parser.add_argument('--execution_guided', action='store_true',
                        help="Select the first beam candidate that executes successfully")
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--eval_only', action='store_true',
                        help="Load the best checkpoint and only generate dev/test outputs")
    parser.add_argument('--resume_training', action='store_true',
                        help="Resume training from last_model.pt (not best_model.pt)")
    parser.add_argument('--dev_only', action='store_true',
                        help="Evaluate the development set without running test inference")
    parser.add_argument('--output_prefix', type=str, default=None,
                        help="Submission filename prefix, e.g. t5_ft_experiment_ec")

    args = parser.parse_args()
    if args.learning_rate is None:
        args.learning_rate = 5e-4 if args.finetune else 1e-3
    return args


def select_executable_candidates(decoded, batch_size, num_candidates):
    """Select the highest-ranked executable SQL candidate for each example.

    Hugging Face returns candidates contiguously for each input.  We retain
    beam order and only fall back to the top beam when none executes.  Query
    execution is read-only against the assignment database.
    """
    selected = []
    conn = sqlite3.connect('data/flight_database.db')
    try:
        for i in range(batch_size):
            candidates = decoded[i * num_candidates:(i + 1) * num_candidates]
            choice = candidates[0]
            for candidate in candidates:
                try:
                    deadline = time.monotonic() + 0.25
                    conn.set_progress_handler(
                        lambda: 1 if time.monotonic() > deadline else 0, 10000)
                    conn.execute(candidate).fetchall()
                    choice = candidate
                    break
                except sqlite3.Error:
                    continue
            selected.append(choice)
    finally:
        conn.close()
    return selected


def generate_sql(args, model, tokenizer, encoder_input, encoder_mask):
    num_candidates = args.num_return_sequences if args.execution_guided else 1
    generated_ids = model.generate(
        input_ids=encoder_input,
        attention_mask=encoder_mask,
        max_new_tokens=args.max_new_tokens,
        num_beams=max(args.num_beams, num_candidates),
        num_return_sequences=num_candidates,
        early_stopping=True,
        length_penalty=1.0,
    )
    decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    if args.execution_guided:
        return select_executable_candidates(decoded, encoder_input.size(0), num_candidates)
    return decoded


def train(args, model, train_loader, dev_loader, optimizer, scheduler):
    best_f1 = -1
    epochs_since_improvement = 0

    model_type = 'ft' if args.finetune else 'scr'
    checkpoint_dir = os.path.join(
        'checkpoints', f'{model_type}_experiments', args.experiment_name)
    gt_sql_path = os.path.join(f'data/dev.sql')
    gt_record_path = os.path.join(f'records/ground_truth_dev.pkl')
    model_sql_path = os.path.join(
        f'results/t5_{model_type}_{args.experiment_name}_dev.sql')
    model_record_path = os.path.join(
        f'records/t5_{model_type}_{args.experiment_name}_dev.pkl')
    for epoch in range(args.max_n_epochs):
        tr_loss = train_epoch(args, model, train_loader, optimizer, scheduler)
        print(f"Epoch {epoch}: Average train loss was {tr_loss}")

        eval_loss, record_f1, record_em, sql_em, error_rate = eval_epoch(args, model, dev_loader,
                                                                         gt_sql_path, model_sql_path,
                                                                         gt_record_path, model_record_path)
        print(
            f"Epoch {epoch}: Dev loss: {eval_loss}, Record F1: {record_f1}, Record EM: {record_em}, SQL EM: {sql_em}")
        print(
            f"Epoch {epoch}: {error_rate*100:.2f}% of the generated outputs led to SQL errors")

        if args.use_wandb:
            result_dict = {
                'train/loss': tr_loss,
                'dev/loss': eval_loss,
                'dev/record_f1': record_f1,
                'dev/record_em': record_em,
                'dev/sql_em': sql_em,
                'dev/error_rate': error_rate,
            }
            wandb.log(result_dict, step=epoch)

        if record_f1 > best_f1:
            best_f1 = record_f1
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        save_model(checkpoint_dir, model, best=False)
        if epochs_since_improvement == 0:
            save_model(checkpoint_dir, model, best=True)

        if args.patience_epochs > 0 and epochs_since_improvement >= args.patience_epochs:
            break


def train_epoch(args, model, train_loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    total_tokens = 0
    for encoder_input, encoder_mask, decoder_input, decoder_targets, _ in tqdm(train_loader):
        optimizer.zero_grad()
        encoder_input = encoder_input.to(DEVICE)
        encoder_mask = encoder_mask.to(DEVICE)
        decoder_input = decoder_input.to(DEVICE)
        decoder_targets = decoder_targets.to(DEVICE)

        labels = decoder_targets.clone()
        labels[labels == PAD_IDX] = -100
        output = model(
            input_ids=encoder_input,
            attention_mask=encoder_mask,
            labels=labels,
        )
        loss = output.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        with torch.no_grad():
            num_tokens = torch.sum(labels != -100).item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    return total_loss / total_tokens


def eval_epoch(args, model, dev_loader, gt_sql_pth, model_sql_path, gt_record_path, model_record_path):
    '''
    You must implement the evaluation loop to be using during training. We recommend keeping track
    of the model loss on the SQL queries, the metrics compute_metrics returns (save_queries_and_records should be helpful)
    and the model's syntax error rate. 

    To compute non-loss metrics, you will need to perform generation with the model. Greedy decoding or beam search
    should both provide good results. If you find that this component of evaluation takes too long with your compute,
    we found the cross-entropy loss (in the evaluation set) to be well (albeit imperfectly) correlated with F1 performance.
    '''
    model.eval()
    total_loss = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss()
    all_generated_sqls = []

    # 获取 tokenizer 用于解码
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")

    with torch.no_grad():
        for encoder_input, encoder_mask, decoder_input, decoder_targets, _ in tqdm(dev_loader):
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)
            decoder_input = decoder_input.to(DEVICE)
            decoder_targets = decoder_targets.to(DEVICE)

            # 1. 计算 Dev Loss（与 train_epoch 逻辑完全一致）
            logits = model(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                decoder_input_ids=decoder_input,
            )['logits']

            non_pad = decoder_targets != PAD_IDX
            loss = criterion(logits[non_pad], decoder_targets[non_pad])

            num_tokens = torch.sum(non_pad).item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

            # 2. 生成预测的 SQL (Greedy 解码或 Beam Search)
            generated_sqls = generate_sql(
                args, model, tokenizer, encoder_input, encoder_mask)
            all_generated_sqls.extend(generated_sqls)

    eval_loss = total_loss / total_tokens if total_tokens > 0 else 0

    # 4. 保存生成的 SQL 并去数据库执行得到 records
    # 注意：如果你的 utils.py 里的 save_queries_and_records 需要传数据库路径，请记得加上
    save_queries_and_records(all_generated_sqls, model_sql_path, model_record_path)

    # 5. 计算指标 (Record F1, Record EM, SQL EM, 错误率)
    sql_em, record_em, record_f1, error_msgs = compute_metrics(
        gt_sql_pth, model_sql_path, gt_record_path, model_record_path)
    error_rate = (sum(bool(msg) for msg in error_msgs) / len(error_msgs)
                  if error_msgs else 0.0)

    return eval_loss, record_f1, record_em, sql_em, error_rate


def test_inference(args, model, test_loader, model_sql_path, model_record_path):
    '''
    You must implement inference to compute your model's generated SQL queries and its associated 
    database records. Implementation should be very similar to eval_epoch.
    '''
    model.eval()
    all_generated_sqls = []

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")

    with torch.no_grad():
        for encoder_input, encoder_mask, _, _, _ in tqdm(test_loader):
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)

            generated_sqls = generate_sql(
                args, model, tokenizer, encoder_input, encoder_mask)
            all_generated_sqls.extend(generated_sqls)

    # 保存测试集的 SQL 和 Records
    save_queries_and_records(all_generated_sqls, model_sql_path, model_record_path)
    print(
        f"Test inference completed. SQL saved to {model_sql_path}, Records saved to {model_record_path}")


def main():
    # Get key arguments
    args = get_args()
    set_random_seeds(42)
    if args.use_wandb:
        # Recommended: Using wandb (or tensorboard) for result logging can make experimentation easier
        setup_wandb(args)

    # Load the data and the model
    train_loader, dev_loader, test_loader = load_t5_data(
        args.batch_size, args.test_batch_size)
    if not args.eval_only:
        model = (load_model_from_checkpoint(args, best=False)
                 if args.resume_training else initialize_model(args))
        optimizer, scheduler = initialize_optimizer_and_scheduler(
            args, model, len(train_loader))
        train(args, model, train_loader, dev_loader, optimizer, scheduler)

    # Evaluate
    model = load_model_from_checkpoint(args, best=True)
    model.eval()

    # Dev set
    experiment_name = args.experiment_name
    model_type = 'ft' if args.finetune else 'scr'
    output_prefix = args.output_prefix or f't5_{model_type}_{experiment_name}'
    gt_sql_path = os.path.join(f'data/dev.sql')
    gt_record_path = os.path.join(f'records/ground_truth_dev.pkl')
    model_sql_path = os.path.join(
        f'results/{output_prefix}_dev.sql')
    model_record_path = os.path.join(
        f'records/{output_prefix}_dev.pkl')
    dev_loss, dev_record_f1, dev_record_em, dev_sql_em, dev_error_rate = eval_epoch(args, model, dev_loader,
                                                                                    gt_sql_path, model_sql_path,
                                                                                    gt_record_path, model_record_path)
    print(
        f"Dev set results: Loss: {dev_loss}, Record F1: {dev_record_f1}, Record EM: {dev_record_em}, SQL EM: {dev_sql_em}")
    print(
        f"Dev set results: {dev_error_rate*100:.2f}% of the generated outputs led to SQL errors")

    if args.dev_only:
        return

    # Test set
    model_sql_path = os.path.join(
        f'results/{output_prefix}_test.sql')
    model_record_path = os.path.join(
        f'records/{output_prefix}_test.pkl')
    test_inference(args, model, test_loader, model_sql_path, model_record_path)


if __name__ == "__main__":
    main()
