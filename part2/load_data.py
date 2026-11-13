import torch
from transformers import T5TokenizerFast
import os
import random
import re
import string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

PAD_IDX = 0


class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        self.tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
        self.data = self.process_data(data_folder, split, self.tokenizer)
        self.split = split

    def process_data(self, data_folder, split, tokenizer):
        nl_path = os.path.join(data_folder, f'{split}.nl')
        sql_path = os.path.join(data_folder, f'{split}.sql')

        nl_lines = load_lines(nl_path) if os.path.exists(nl_path) else []

        if split == 'test':
            sql_lines = []
        else:
            sql_lines = load_lines(sql_path)

        processed = []
        for i in range(len(nl_lines)):
            # T5 的输入长度限制一般为 512，这里安全起见都设 512
            # A task prefix matches T5's text-to-text pretraining interface.
            source = f"translate English to SQL: {nl_lines[i]}"
            x_enc = tokenizer(source, max_length=128, truncation=True, padding=False)[
                'input_ids']
            if split != 'test':
                y_enc = tokenizer(sql_lines[i], max_length=512, truncation=True, padding=False)[
                    'input_ids']
                processed.append((x_enc, y_enc))
            else:
                processed.append((x_enc, None))

        return processed

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    x_ids = [torch.tensor(item[0]) for item in batch]
    y_ids = [torch.tensor(item[1]) for item in batch]
    # 1. 动态 Padding 输入
    encoder_ids = pad_sequence(x_ids, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).long()

    # 2. 动态 Padding 输出 (targets)
    decoder_targets = pad_sequence(
        y_ids, batch_first=True, padding_value=PAD_IDX)

    # 3. 为 Decoder 构造输入 (Shift Right)
    # T5 使用 pad_token_id (也就是 0) 作为起始 token (start token)
    # decoder_inputs 就是把 targets 往右移一格，前面补 0
    decoder_inputs = torch.cat([
        torch.zeros((decoder_targets.size(0), 1), dtype=torch.long),
        decoder_targets[:, :-1]
    ], dim=1)

    initial_decoder_inputs = decoder_inputs[:, :1]

    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs


def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    x_ids = [torch.tensor(item[0]) for item in batch]

    encoder_ids = pad_sequence(x_ids, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).long()

    initial_decoder_inputs = torch.zeros(
        (encoder_ids.size(0), 1), dtype=torch.long)

    # 为了兼容 train_t5.py 中解包 5 个变量的写法，这里多返回两个 dummy tensor
    dummy_decoder_inputs = initial_decoder_inputs
    dummy_decoder_targets = initial_decoder_inputs

    return encoder_ids, encoder_mask, dummy_decoder_inputs, dummy_decoder_targets, initial_decoder_inputs


def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size,
                            shuffle=shuffle, collate_fn=collate_fn)
    return dataloader


def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")

    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines


def load_prompting_data(data_folder):
    train_x = load_lines(os.path.join(data_folder, 'train.nl'))
    train_y = load_lines(os.path.join(data_folder, 'train.sql'))
    dev_x = load_lines(os.path.join(data_folder, 'dev.nl'))
    dev_y = load_lines(os.path.join(data_folder, 'dev.sql'))
    test_x = load_lines(os.path.join(data_folder, 'test.nl'))
    return train_x, train_y, dev_x, dev_y, test_x
