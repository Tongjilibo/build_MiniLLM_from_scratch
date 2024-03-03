import json
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
from bert4torch.snippets import sequence_padding


MAX_LENGTH = 1024
PROMPT_MAX_LEN = 512
ANSWER_MAX_LEN = 512


def process_alpaca(data_path):
    '''alpaca_gpt4_data_zh.json'''
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    res = []
    for per in data:
        q = per['instruction']
        i = per['input']
        a = per['output']
        q = q + i
        if len(q) > PROMPT_MAX_LEN or len(a) > ANSWER_MAX_LEN:
            continue
        res.append((q,a))
    return res


def process_belle_1m(data_path):
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
        q = q + i
        if len(q) > PROMPT_MAX_LEN or len(a) > ANSWER_MAX_LEN:
            continue
        res.append((q,a))
    return res

MAPPING = {
    'alpaca_gpt4_data_zh.json': process_alpaca,
    'Belle_open_source_1M.json': process_belle_1m

}

class SFTDataset(Dataset):
    def __init__(self, filename, tokenizer):
        super().__init__()
        self.data = self.load_data(filename)
        self.MAX_LENGTH = MAX_LENGTH
        self.prompt_max_len = PROMPT_MAX_LEN
        self.answer_max_len = ANSWER_MAX_LEN
        self.tokenizer = tokenizer
        self.bos = self.tokenizer.special_tokens['<bos>']
        self.eos = self.tokenizer.special_tokens['<eos>']
        self.pad = 0 # self.tokenizer.special_tokens['<pad>']
    
    @staticmethod
    def load_data(filename):
        postfix = filename.split('\\')[-1].split('/')[-1]
        return MAPPING[postfix](filename)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index: int):
        prompt, answer = self.data[index]
        prompt = self.tokenizer.encode(prompt, add_special_tokens=False)
        answer = self.tokenizer.encode(answer, add_special_tokens=False)
        if len(prompt) > self.prompt_max_len:
            prompt = prompt[:self.prompt_max_len-2]
        if len(answer) > self.answer_max_len:
            answer = answer[:self.answer_max_len-2]
        #
        input_ids = prompt + [self.bos] + answer + [self.eos]
        context_length = input_ids.index(self.bos)
        mask_position = context_length - 1
        labels = [-100] * context_length + input_ids[mask_position+1:]

        return input_ids, labels

def collate_train_fn(batch):
    batch_token_ids = [i[0] for i in batch]
    batch_labels = [i[1] for i in batch]
    batch_token_ids = torch.tensor(sequence_padding(batch_token_ids, value=0), dtype=torch.long)
    batch_labels = torch.tensor(sequence_padding(batch_labels, value=-100), dtype=torch.long)
    return [batch_token_ids[..., :-1]], batch_labels[..., 1:]