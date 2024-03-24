''' 数据处理模块
处理数据, tokenize并保存到硬盘

1. 增加HUMAN和ROBOT标记，可以用于多轮对话问答
2. 不限制prompt和answer的长度，仅限制总长度，可容纳更多的样本
3. 多轮对话中，同时计算多个answer的loss, 提升训练效率
4. linux下可调用多进程处理，加快数据处理速度，推荐！
'''

import json
from torch.utils.data import Dataset
import torch
from bert4torch.snippets import sequence_padding, Timeit, log_info, log_warn, parallel_apply, log_info_once
from typing import Literal
import re
from tqdm import tqdm
import os
import random
from transformers import AutoTokenizer
import numpy as np


# ================================数据处理的参数================================
MAX_LENGTH = 1024
HUMAN = '<human>'
ROBOT = '<robot>'
PAD_TOKEN_ID = 0
EOS_TOKEN_ID = 2
MAX_SAMPLES = 1000  # None表示不限制，不为None用于测试小样本快速验证
MAX_SAMPLES_PER_FILE = 100000  # 每个文件最多能容纳的样本量
DEBUG = False

# 多进程参数, linux下可用
USE_PARALLEL = False if os.name == 'nt' else True
WORKERS = 8 # os.cpu_count()
MAX_QUEUE_SIZE = 2000

DATASET_SRC_DIR = '/data/corpus/sft/common/'  # 源数据路径
DATASET_SAVE_DIR = '../sft_data'  # 处理完的数据保存路径
# 待处理的数据集，因为数据集很大，按照实际情况按需使用，比如只使用alpaca-zh
FILE_NAMES = [
    'alpaca-zh/alpaca_gpt4_data_zh.json',
    'BelleGroup/Belle_open_source_0.5M.json',
    'BelleGroup/Belle_open_source_1M.json',
    'BelleGroup/school_math_0.25M.json',
    'deepctrl-sft-data/sft_data_zh.jsonl',
    'moss-002-sft-data/zh_helpfulness.json',
    'moss-002-sft-data/zh_honesty.json',
    'moss-003-sft-data/moss-003-sft-no-tools.jsonl',
    'CodeChat/continue_zh.jsonl',
    'CodeChat/continue_zh_2.jsonl',
    'ShareGPT-Chinese-English-90k/common_zh_70k.jsonl',
    'ShareGPT-Chinese-English-90k/computer_cn_26k_continue.jsonl',
    'ShareGPT-Chinese-English-90k/computer_zh_26k.jsonl',
    'ShareGPT-Chinese-English-90k/unknow_zh_38k.jsonl',
    'ShareGPT-Chinese-English-90k/unknow_zh_38k_continue.jsonl',
    'firefly-train-1.1M/firefly-train-1.1M.jsonl'
]
tokenizer = AutoTokenizer.from_pretrained('../config', trust_remote_code=True)


# ================================数据处理过程================================
def get_samples_count(filenames):
    '''获取训练样本总量'''

    total_samples = 0
    bar = tqdm(total=len(filenames))
    for id, filename in enumerate(filenames):
        bar.n = id + 1
        bar.refresh()
        bar.set_description(filename.split('/')[-1])

        # 每一行一个json文件
        with open(filename, 'r', encoding='utf-8') as f:
            for _ in enumerate(f):
                total_samples += 1
    bar.close()
    log_info(f'total_samples={total_samples}')
    return total_samples


def collect_tokens(process_one, filename, data_format:Literal['jsonl', 'json']='jsonl'):
    '''各个函数通用的处理token的方式'''
    def process_data_slice(data_slice, fi):
        # 是否并行处理数据
        if not USE_PARALLEL:
            log_info_once('Use single process to process data, maybe slow')
            train_samples = [process_one(line) for line in data_slice]
        else:
            log_info_once('Use multiprocess to accelerate data process')
            train_samples = parallel_apply(func=process_one, iterable=data_slice, workers=WORKERS, max_queue_size=MAX_QUEUE_SIZE,
                                        dummy=False, callback=None, unordered=False)
        train_samples = [{'input_ids':i[0], 'labels':i[1]} for i in train_samples if i[0] is not None and len(i[0])>1]

        save_path = os.path.join(DATASET_SAVE_DIR, filename.replace('/', '--').replace('.jsonl', '').replace('.json', '') + f'_{fi}.jsonl')
        with open(save_path, 'w') as f:
            for item in train_samples:
                json.dump(item, f)  # 使用json.dump直接将字典转换为JSON格式写入文件
                f.write('\n')  # 每个JSON对象后面添加换行符
        return len(train_samples)


    # 读入数据，分json和jsonl两种格式
    data_path = os.path.join(DATASET_SRC_DIR, filename)
    os.makedirs(DATASET_SAVE_DIR, exist_ok=True)

    all_count = 0  # 总的样本量
    if data_format == 'json':
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if MAX_SAMPLES is not None:
            data = data[:MAX_SAMPLES]

        # 按照每个文件的最大样本数量来保存
        total_steps = int(np.ceil(len(data) / MAX_SAMPLES_PER_FILE) * MAX_SAMPLES_PER_FILE)
        for fi, start in enumerate(range(0, total_steps, MAX_SAMPLES_PER_FILE), start=1):
            data_slice = data[start: start+MAX_SAMPLES_PER_FILE]
            if len(data_slice) == 0:
                continue
            all_count += process_data_slice(data_slice, fi)
    else:
        data = []
        f = open(data_path, 'r', encoding='utf-8')
        fi = 1
        while True:
            line = f.readline()
            if not line:
                break
            data.append(line)
            data_len = len(data)

            if data_len >= MAX_SAMPLES_PER_FILE:
                all_count += process_data_slice(data, fi)
                data = []
                fi += 1

            if (MAX_SAMPLES is not None) and (data_len >= MAX_SAMPLES):
                break
        
        if len(data) > 0:
            all_count += process_data_slice(data, fi)
    
    # debug使用
    if DEBUG:
        print('='*60)
        print('data_path:', data_path)
        print('len(train_samples):', all_count)
        print('='*60)
        print()
    return all_count


def process_alpaca(filename, tokenizer):
    '''alpaca_gpt4_data_zh.json'''

    def process_one(per):
        q = tokenizer.encode(HUMAN + per['instruction'] + per['input'] + ROBOT, add_special_tokens=False)
        a = tokenizer.encode(per['output'], add_special_tokens=False)
        if len(q) + len(a) >= MAX_LENGTH:
            return None, None
        input_ids = q + a
        labels = [PAD_TOKEN_ID] * (len(q) - 1) + input_ids[len(q):] + [EOS_TOKEN_ID]

        assert len(input_ids) == len(labels)
        return input_ids, labels

    return collect_tokens(process_one, filename, data_format='json')


def process_belle(filename, tokenizer):
    '''Belle_open_source_1M.json'''

    def process_one(line):
        if not line:
            return None, None
        per = json.loads(line)
        q = tokenizer.encode(HUMAN + per['instruction'] + per['input'] + ROBOT, add_special_tokens=False)
        a = tokenizer.encode(per['output'], add_special_tokens=False)
        if len(q) + len(a) >= MAX_LENGTH:
            return None, None
        input_ids = q + a
        labels = [PAD_TOKEN_ID] * (len(q) - 1) + input_ids[len(q):] + [EOS_TOKEN_ID]

        assert len(input_ids) == len(labels)
        return input_ids, labels
    
    return collect_tokens(process_one, filename, data_format='jsonl')


def process_deepctrl(filename, tokenizer):
    '''deepctrl-sft-data'''

    def process_one(line):
        if not line:
            return None, None
        per = json.loads(line)

        input_ids, labels = [], []
        for human, robot in per['history']:
            q = tokenizer.encode(HUMAN + human + ROBOT, add_special_tokens=False)
            a = tokenizer.encode(robot, add_special_tokens=False)
            # 轮次太多的话，则进行截断
            if len(input_ids + q + a) >= MAX_LENGTH:
                return None, None
            input_ids.extend(q + a)
            labels.extend([PAD_TOKEN_ID] * (len(q) - 1) + a + [EOS_TOKEN_ID])

        q = tokenizer.encode(HUMAN + per['instruction'] + per['input'] + ROBOT, add_special_tokens=False)
        a = tokenizer.encode(per['output'], add_special_tokens=False)
        input_ids.extend(q + a)
        labels.extend([PAD_TOKEN_ID] * (len(q) - 1) + a + [EOS_TOKEN_ID])

        if len(input_ids) >= MAX_LENGTH:
            return None, None
        assert len(input_ids) == len(labels)
        return input_ids, labels

    return collect_tokens(process_one, filename, data_format='jsonl')


def process_moss002(filename, tokenizer):
    '''fnlp@moss-002-sft-data'''

    def process_one(per):
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
            labels.extend([PAD_TOKEN_ID] * (len(human) - 1) + robot + [EOS_TOKEN_ID])

        if len(input_ids) >= MAX_LENGTH:
            return None, None
        assert len(input_ids) == len(labels)
        return input_ids, labels

    return collect_tokens(process_one, filename, data_format='json')


def process_moss003(filename, tokenizer):
    '''fnlp@moss-003-sft-data'''

    def process_one(line):
        if not line:
            return None, None
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
            labels.extend([PAD_TOKEN_ID] * (len(human) - 1) + robot + [EOS_TOKEN_ID])

        if len(input_ids) >= MAX_LENGTH:
            return None, None
        assert len(input_ids) == len(labels)
        return input_ids, labels

    return collect_tokens(process_one, filename, data_format='jsonl')


def process_shareai(filename, tokenizer):
    '''shareAI'''

    def process_one(line):
        if not line:
            return None, None
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
            labels.extend([PAD_TOKEN_ID] * (len(human) - 1) + robot + [EOS_TOKEN_ID])

        if len(input_ids) >= MAX_LENGTH:
            return None, None
        assert len(input_ids) == len(labels)
        return input_ids, labels

    return collect_tokens(process_one, filename, data_format='jsonl')


def process_firefly(filename, tokenizer):
    '''YeungNLP@firefly-train-1.1M'''

    def process_one(line):
        if not line:
            return None, None
        per = json.loads(line)
        q = tokenizer.encode(HUMAN + per['input'] + ROBOT, add_special_tokens=False)
        a = tokenizer.encode(per['target'], add_special_tokens=False)
        if len(q) + len(a) >= MAX_LENGTH:
            return None, None
        input_ids = q + a
        labels = [PAD_TOKEN_ID] * (len(q) - 1) + input_ids[len(q):] + [EOS_TOKEN_ID]
        assert len(input_ids) == len(labels)
        return input_ids, labels

    return collect_tokens(process_one, filename, data_format='jsonl')


MAPPING = {
    'alpaca-zh/alpaca_gpt4_data_zh.json': process_alpaca,
    'BelleGroup/Belle_open_source_1M.json': process_belle,
    'BelleGroup/Belle_open_source_0.5M.json': process_belle,
    'BelleGroup/school_math_0.25M.json': process_belle,
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
    def __init__(self, datadir, verbose=1):
        super().__init__()
        self.verbose = verbose
        self.data = self.load_data(datadir)

    def load_data(self, filenames):
        all_res = []
        for filename in filenames:
            res = []
            f = open(filename, 'r', encoding='utf-8')
            while True:
                line = f.readline()
                if not line:
                    break
                res.append(json.loads(line))
            if self.verbose:
                log_info(f'Loading {filename}: len={len(res)}')
            all_res.extend(res)
        random.shuffle(all_res)
        if self.verbose:
            log_info(f'Training samples: {len(all_res)}')
        return all_res

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        return self.data[index]

def collate_train_fn(batch):
    batch_token_ids = [i['input_ids'] for i in batch]
    batch_labels = [i['labels'] for i in batch]
    batch_token_ids = torch.tensor(sequence_padding(batch_token_ids, value=PAD_TOKEN_ID), dtype=torch.long)
    batch_labels = torch.tensor(sequence_padding(batch_labels, value=PAD_TOKEN_ID), dtype=torch.long)
    return [batch_token_ids], batch_labels


def main():
    '''数据处理主函数
    :param FILE_NAMES: 处理的数据集名称
    :param DATASET_SRC_DIR: 数据源文件夹
    :param DATASET_SAVE_DIR: 数据保存的文件夹
    :param tokenizer: tokenizer
    '''
    if MAX_SAMPLES is not None:
        log_warn(f'Only use {MAX_SAMPLES} samples for each sft dataset.')
    else:
        log_warn(f'Use all samples for each sft dataset, may be slow.')
    
    with Timeit() as ti:
        for filename in FILE_NAMES:
            sample_count = MAPPING[filename](filename, tokenizer)
            ti.lap(name=f'{filename}: len={sample_count}'.ljust(70) + '-', reset=True)

if __name__ == '__main__':
    main()
