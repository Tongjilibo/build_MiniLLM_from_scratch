''' 数据处理模块
1. 增加HUMAN和ROBOT标记，可以用于多轮对话问答
2. 不限制prompt和answer的长度，仅限制总长度，可容纳更多的样本
3. 多轮对话中，同时计算多个answer的loss, 提升训练效率
'''
import json
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
from bert4torch.snippets import sequence_padding
import re
from torch4keras.snippets.log import log_info, log_warn
from tqdm import tqdm


MAX_LENGTH = 1024
HUMAN = '<human>'
ROBOT = '<robot>'
PAD_TOKEN_ID = 0
EOS_TOKEN_ID = 2
MAX_SAMPLES = None  # None表示不限制，不为None用于测试小样本快速验证
if MAX_SAMPLES is not None:
    log_warn(f'Only use {MAX_SAMPLES} samples for each sft file')


def get_probable_samples(filenames):
    '''获取可能的训练样本总量'''

    total_samples = 0
    bar = tqdm(total=len(filenames))
    for id, filename in enumerate(filenames):
        bar.n = id + 1
        bar.refresh()
        bar.set_description(filename.split('/')[-1])

        if any([n in filename for n in ['alpaca_gpt4_data_zh', 'fnlp@moss-002-sft-data']]):
            #　json格式文件
            data = json.load(open(filename, 'r', encoding='utf-8'))
            if MAX_SAMPLES is not None:
                total_samples += min(len(data), MAX_SAMPLES)
            else:
                total_samples += len(data)
        else:
            # 每一行一个json文件
            with open(filename, 'r', encoding='utf-8') as f:
                for id_, _ in enumerate(f):
                    if MAX_SAMPLES is not None and id_ >= MAX_SAMPLES:
                        break
                    total_samples += 1
    bar.close()
    log_info(f'probable_total_samples={total_samples}')
    return total_samples


def process_alpaca(data_path, tokenizer):
    '''alpaca_gpt4_data_zh.json'''
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    res = []
    for per in data:
        q = tokenizer.encode(HUMAN + per['instruction'] + per['input'] + ROBOT, add_special_tokens=False)
        a = tokenizer.encode(per['output'], add_special_tokens=False)
        if len(q) + len(a) >= MAX_LENGTH:
            continue
        input_ids = q + a
        labels = [PAD_TOKEN_ID] * (len(q)-1) + input_ids[len(q):] + [EOS_TOKEN_ID]

        assert len(input_ids) == len(labels)
        res.append((input_ids, labels))
        if (MAX_SAMPLES is not None) and (len(res) >= MAX_SAMPLES):
            break
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
        q = tokenizer.encode(HUMAN + per['instruction'] + per['input'] + ROBOT, add_special_tokens=False)
        a = tokenizer.encode(per['output'], add_special_tokens=False)
        if len(q) + len(a) >= MAX_LENGTH:
            continue
        input_ids = q + a
        labels = [PAD_TOKEN_ID] * (len(q)-1) + input_ids[len(q):] + [EOS_TOKEN_ID]

        assert len(input_ids) == len(labels)
        res.append((input_ids, labels))
        if (MAX_SAMPLES is not None) and (len(res) >= MAX_SAMPLES):
            break
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

        input_ids, labels = [], []
        for human, robot in per['history']:
            q = tokenizer.encode(HUMAN + human + ROBOT, add_special_tokens=False)
            a = tokenizer.encode(robot, add_special_tokens=False)
            # 轮次太多的话，则进行截断
            if len(input_ids + q + a) >= MAX_LENGTH:
                break
            input_ids.extend(q + a)
            labels.extend([PAD_TOKEN_ID]*(len(q)-1) + a + [EOS_TOKEN_ID])
            
        q = tokenizer.encode(HUMAN + per['instruction'] + per['input'] + ROBOT, add_special_tokens=False)
        a = tokenizer.encode(per['output'], add_special_tokens=False)
        input_ids.extend(q + a)
        labels.extend([PAD_TOKEN_ID]*(len(q)-1) + a + [EOS_TOKEN_ID])
        
        if len(input_ids) >= MAX_LENGTH:
            continue
        assert len(input_ids) == len(labels)
        res.append((input_ids, labels))
        if (MAX_SAMPLES is not None) and (len(res) >= MAX_SAMPLES):
            break
    return res


def process_moss002(data_path, tokenizer):
    '''fnlp@moss-002-sft-data'''
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    res = []
    for per in data:
        history = re.split(r'<eoh> \[MOSS\]: |<eoa> \[Human\]: |\[Human\]: |<eoa>', per['plain_text'])
        history = [i.strip() for i in history if i]
        input_ids, labels = [], []
        for human, robot in zip(history[0::2], history[1::2]):
            human = tokenizer.encode(HUMAN + human + ROBOT, add_special_tokens=False)
            robot = tokenizer.encode(robot, add_special_tokens=False)
            # 轮次太多的话，则进行截断
            if len(input_ids + human + robot) >= MAX_LENGTH:
                break
            input_ids.extend(human + robot)
            labels.extend([PAD_TOKEN_ID]*(len(human)-1) + robot + [EOS_TOKEN_ID])

        if len(input_ids) >= MAX_LENGTH:
            continue
        assert len(input_ids) == len(labels)
        res.append((input_ids, labels))
        if (MAX_SAMPLES is not None) and (len(res) >= MAX_SAMPLES):
            break
    return res


def process_moss003(data_path, tokenizer):
    '''fnlp@moss-003-sft-data'''
    f = open(data_path, 'r', encoding='utf-8')
    
    res = []
    while True:
        line = f.readline()
        if not line:
            break
        per = json.loads(line)
        input_ids, labels = [], []
        for turn in per['chat'].values():
            if not re.search(r'[\u4e00-\u9fff]', turn['MOSS']):
                continue

            human = turn['Human'].replace('<|Human|>: ', '').replace('<eoh>\n', '')
            robot = turn['MOSS'].replace('<|MOSS|>: ', '').replace('<eom>\n', '')
            robot = re.sub(r'<sup><\|[0-9]+\|></sup>', '', robot).strip()

            human = tokenizer.encode(HUMAN + human + ROBOT, add_special_tokens=False)
            robot = tokenizer.encode(robot, add_special_tokens=False)
            # 轮次太多的话，则进行截断
            if len(input_ids + human + robot) >= MAX_LENGTH:
                break
            input_ids.extend(human + robot)
            labels.extend([PAD_TOKEN_ID]*(len(human)-1) + robot + [EOS_TOKEN_ID])

        if len(input_ids) >= MAX_LENGTH:
            continue
        assert len(input_ids) == len(labels)
        res.append((input_ids, labels))
        if (MAX_SAMPLES is not None) and (len(res) >= MAX_SAMPLES):
            break
    return res


def process_shareai(data_path, tokenizer):
    '''shareAI'''
    f = open(data_path, 'r', encoding='utf-8')
    
    res = []
    while True:
        line = f.readline()
        if not line:
            break
        per = json.loads(line)
        input_ids, labels = [], []
        for turn in per['conversation']:
            human = turn['human']
            robot = turn['assistant']

            human = tokenizer.encode(HUMAN + human + ROBOT, add_special_tokens=False)
            robot = tokenizer.encode(robot, add_special_tokens=False)
            # 轮次太多的话，则进行截断
            if len(input_ids + human + robot) >= MAX_LENGTH:
                break
            input_ids.extend(human + robot)
            labels.extend([PAD_TOKEN_ID]*(len(human)-1) + robot + [EOS_TOKEN_ID])

        if len(input_ids) >= MAX_LENGTH:
            continue
        assert len(input_ids) == len(labels)
        res.append((input_ids, labels))
        if (MAX_SAMPLES is not None) and (len(res) >= MAX_SAMPLES):
            break
    return res


def process_firefly(data_path, tokenizer):
    '''YeungNLP@firefly-train-1.1M'''
    f = open(data_path, 'r', encoding='utf-8')
    
    res = []
    while True:
        line = f.readline()
        if not line:
            break
        per = json.loads(line)
        q = tokenizer.encode(HUMAN + per['input'] + ROBOT, add_special_tokens=False)
        a = tokenizer.encode(per['target'], add_special_tokens=False)
        if len(q) + len(a) >= MAX_LENGTH:
            continue
        input_ids = q + a
        labels = [PAD_TOKEN_ID] * (len(q)-1) + input_ids[len(q):] + [EOS_TOKEN_ID]
        assert len(input_ids) == len(labels)
        res.append((input_ids, labels))
        if (MAX_SAMPLES is not None) and (len(res) >= MAX_SAMPLES):
            break
    return res


MAPPING = {
    'alpaca-zh/alpaca_gpt4_data_zh.json': process_alpaca,
    'train_1M_CN/Belle_open_source_1M.json': process_belle,
    'train_0.5M_CN/Belle_open_source_0.5M.json': process_belle,
    'school_math_0.25M/school_math_0.25M.json': process_belle,
    'deepctrl-sft-data/sft_data_zh.jsonl': process_deepctrl,
    'moss-002-sft-data/zh_helpfulness.json': process_moss002,
    'moss-002-sft-data/zh_honesty.json': process_moss002,
    'moss-003-sft-data/conversations_with_tools_with_inner_instruction_no_text2image_train_all_random_meta0.5_0.1_0.01_moss_0709.jsonl': process_moss003,
    'moss-003-sft-data/moss-003-sft-no-tools.jsonl': process_moss003,
    'CodeChat/continue_zh.jsonl': process_shareai,
    'CodeChat/continue_zh_2.jsonl': process_shareai,
    'ShareGPT-Chinese-English-90k/common_zh_70k.jsonl': process_shareai,
    'ShareGPT-Chinese-English-90k/computer_cn_26k_continue.jsonl': process_shareai,
    'ShareGPT-Chinese-English-90k/computer_en_26k(fixed).jsonl': process_shareai,
    'ShareGPT-Chinese-English-90k/computer_zh_26k(fixed).jsonl': process_shareai,
    'ShareGPT-Chinese-English-90k/computer_zh_26k.jsonl': process_shareai,
    'ShareGPT-Chinese-English-90k/unknow_zh_38k.jsonl': process_shareai,
    'ShareGPT-Chinese-English-90k/unknow_zh_38k_continue.jsonl': process_shareai,
    'firefly-train-1.1M/firefly-train-1.1M.jsonl': process_firefly
}

class SFTDataset(Dataset):
    def __init__(self, filename, tokenizer):
        super().__init__()
        self.MAX_LENGTH = MAX_LENGTH
        self.tokenizer = tokenizer
        self.data = self.load_data(filename)
    
    def load_data(self, filename):
        postfix = filename.split('@')[-1]
        return MAPPING[postfix](filename, self.tokenizer)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index: int):
        return self.data[index]

def collate_train_fn(batch):
    batch_token_ids = [i[0] for i in batch]
    batch_labels = [i[1] for i in batch]
    batch_token_ids = torch.tensor(sequence_padding(batch_token_ids, value=PAD_TOKEN_ID), dtype=torch.long)
    batch_labels = torch.tensor(sequence_padding(batch_labels, value=PAD_TOKEN_ID), dtype=torch.long)
    return [batch_token_ids], batch_labels


if __name__ == '__main__':
    get_probable_samples(['F:/data/corpus/sft/common/deepctrl@deepctrl-sft-data/sft_data_zh.jsonl'])

    # from transformers import AutoTokenizer
    # tokenizer = AutoTokenizer.from_pretrained('../config', trust_remote_code=True)
    # process_deepctrl('F:/data/corpus/sft/common/deepctrl@deepctrl-sft-data/sft_data_zh.jsonl', tokenizer)
    # process_moss002('F:/data/corpus/sft/common/fnlp@moss-002-sft-data/zh_helpfulness.json', tokenizer)
    # process_moss003('F:/data/corpus/sft/common/fnlp@moss-003-sft-data/conversations_with_tools_with_inner_instruction_no_text2image_train_all_random_meta0.5_0.1_0.01_moss_0709.jsonl', tokenizer)
    # process_moss003('F:/data/corpus/sft/common/fnlp@moss-003-sft-data/moss-003-sft-no-tools.jsonl', tokenizer)
    # process_shareai('F:/data/corpus/sft/common/shareAI@CodeChat/continue_zh_2.jsonl', tokenizer)
    # process_firefly('F:/data/corpus/sft/common/YeungNLP@firefly-train-1.1M/firefly-train-1.1M.jsonl', tokenizer)