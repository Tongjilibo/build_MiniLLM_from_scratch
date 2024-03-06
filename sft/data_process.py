import json
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
from bert4torch.snippets import sequence_padding


MAX_LENGTH = 1024
HUMAN = '<human>'
ROBOT = '<robot>'

def process_alpaca(data_path, tokenizer):
    '''alpaca_gpt4_data_zh.json'''
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    res = []
    for per in data:
        q = per['instruction']
        i = per['input']
        a = per['output']
        q = tokenizer.encode(HUMAN + q + i + ROBOT, add_special_tokens=False)
        a = tokenizer.encode(a, add_special_tokens=False)
        if len(q) + len(a) > MAX_LENGTH-1:
            continue
        res.append((q,a))
    return res


def process_belle(data_path, tokenizer):
    '''Belle_open_source_1M.json'''
    f = open(data_path, 'r', encoding='utf-8')
    
    res = []
    while True:
        line = f.readline()
        if not line:
            break
        per = json.loads(line)
        q = per['instruction']
        i = per['input']
        a = per['output']
        q = tokenizer.encode(HUMAN + q + i + ROBOT, add_special_tokens=False)
        a = tokenizer.encode(a, add_special_tokens=False)
        if len(q) + len(a) > MAX_LENGTH-1:
            continue
        res.append((q,a))
    return res

def process_deepctrl(data_path, tokenizer):
    '''deepctrl-sft-data'''
    f = open(data_path, 'r', encoding='utf-8')
    
    res = []
    while True:
        line = f.readline()
        if not line:
            break
        per = json.loads(line)
        q = per['instruction']
        i = per['input']
        a = per['output']
        h = ''
        for human, robot in per['history']:
            h += HUMAN + human + ROBOT + robot
        q = tokenizer.encode(h + HUMAN + q + i + ROBOT, add_special_tokens=False)
        a = tokenizer.encode(a, add_special_tokens=False)
        if len(q) + len(a) > MAX_LENGTH-1:
            continue
        res.append((q,a))
    return res


MAPPING = {
    'alpaca-zh/alpaca_gpt4_data_zh.json': process_alpaca,
    'train_1M_CN/Belle_open_source_1M.json': process_belle,
    'train_0.5M_CN/Belle_open_source_0.5M.json': process_belle,
    'school_math_0.25M/school_math_0.25M.json': process_belle,
    'deepctrl-sft-data/sft_data_zh.jsonl': process_deepctrl
}

class SFTDataset(Dataset):
    def __init__(self, filename, tokenizer):
        super().__init__()
        self.MAX_LENGTH = MAX_LENGTH
        self.tokenizer = tokenizer
        self.eos = self.tokenizer.special_tokens['<eos>']
        self.pad = 0 # self.tokenizer.special_tokens['<pad>']
        self.data = self.load_data(filename)
    
    def load_data(self, filename):
        postfix = filename.split('@')[-1]
        return MAPPING[postfix](filename, self.tokenizer)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index: int):
        prompt, answer = self.data[index]
        input_ids = prompt + answer + [self.eos]
        labels = [-100] * len(prompt) + input_ids[len(prompt)+1:]

        return input_ids, labels

def collate_train_fn(batch):
    batch_token_ids = [i[0] for i in batch]
    batch_labels = [i[1] for i in batch]
    batch_token_ids = torch.tensor(sequence_padding(batch_token_ids, value=0), dtype=torch.long)
    batch_labels = torch.tensor(sequence_padding(batch_labels, value=-100), dtype=torch.long)
    return [batch_token_ids[..., :-1]], batch_labels[..., 1:]