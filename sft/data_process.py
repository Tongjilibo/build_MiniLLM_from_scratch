''' 数据处理模块
本模块不用跑，跑sft的时候，会使用到本模块处理数据和保存到硬盘

1. 增加HUMAN和ROBOT标记，可以用于多轮对话问答
2. 不限制prompt和answer的长度，仅限制总长度，可容纳更多的样本
3. 多轮对话中，同时计算多个answer的loss, 提升训练效率
4. linux下可调用多进程处理，加快数据处理速度，推荐！
'''
import json

from torch.utils.data import Dataset
import torch
from bert4torch.snippets import sequence_padding, Timeit, log_info, log_warn, parallel_apply
from typing import Literal
import re
from tqdm import tqdm
import os
import pickle
import random


MAX_LENGTH = 1024
HUMAN = '<human>'
ROBOT = '<robot>'
PAD_TOKEN_ID = 0
EOS_TOKEN_ID = 2
MAX_SAMPLES = None  # None表示不限制，不为None用于测试小样本快速验证
if MAX_SAMPLES is not None:
    log_warn(f'Only use {MAX_SAMPLES} samples for each sft file')

# 多进程参数, linux下可用
USE_PARALLEL = True if os.name == 'nt' else True
WORKERS = 8 # os.cpu_count()
MAX_QUEUE_SIZE = 2000


def get_probable_samples(filenames):
    '''获取可能的训练样本总量'''

    total_samples = 0
    bar = tqdm(total=len(filenames))
    for id, filename in enumerate(filenames):
        bar.n = id + 1
        bar.refresh()
        bar.set_description(filename.split('/')[-1])

        if any([n in filename for n in ['alpaca_gpt4_data_zh', 'moss-002-sft-data']]):
            # 　json格式文件
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


def collect_tokens(process_one, data_path, data_format:Literal['jsonl', 'json']='jsonl'):
    '''各个函数通用的处理token的方式'''
    # 读入数据，分json和jsonl两种格式
    if data_format == 'json':
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if MAX_SAMPLES is not None:
            data = data[:MAX_SAMPLES]
    
    else:
        data = []
        f = open(data_path, 'r', encoding='utf-8')
        while True:
            line = f.readline()
            if not line:
                break
            data.append(line)
            if (MAX_SAMPLES is not None) and (len(data) >= MAX_SAMPLES):
                break

    # 是否并行处理数据
    if not USE_PARALLEL:        
        train_samples = [process_one(line) for line in data]
    else:
        train_samples = parallel_apply(func=process_one, iterable=data, workers=WORKERS, max_queue_size=MAX_QUEUE_SIZE,
                                    dummy=False, callback=None, unordered=False)
    train_samples = [i for i in train_samples if i[0] is not None and len(i[0])>1]

    # debug使用
    # print('='*60)
    # print('data_path:', data_path)
    # print('len(train_samples):', len(train_samples))
    # print('sample[0]:', train_samples[0])
    return train_samples


def process_alpaca(data_path, tokenizer):
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

    return collect_tokens(process_one, data_path, data_format='json')


def process_belle(data_path, tokenizer):
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
    
    return collect_tokens(process_one, data_path, data_format='jsonl')


def process_deepctrl(data_path, tokenizer):
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

    return collect_tokens(process_one, data_path, data_format='jsonl')


def process_moss002(data_path, tokenizer):
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

    return collect_tokens(process_one, data_path, data_format='json')


def process_moss003(data_path, tokenizer):
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

    return collect_tokens(process_one, data_path, data_format='jsonl')


def process_shareai(data_path, tokenizer):
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

    return collect_tokens(process_one, data_path, data_format='jsonl')


def process_firefly(data_path, tokenizer):
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

    return collect_tokens(process_one, data_path, data_format='jsonl')


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
    def __init__(self, filenames, tokenizer, dataset_dir="../data", save_dir='../sft_data/'):
        super().__init__()
        self.MAX_LENGTH = MAX_LENGTH
        self.tokenizer = tokenizer
        self.save_dir = save_dir
        self.dataset_dir = dataset_dir
        self.data = self.load_data(filenames)

    def load_data(self, filenames):
        all_res = []
        for filename in filenames:
            save_path = os.path.join(self.save_dir, filename.replace('/', '--').replace('.jsonl', '').replace('.json', '') + '.pkl')
            if os.path.exists(save_path):
                with open(save_path, 'rb') as f:
                    res = pickle.load(f)
            else:
                res = MAPPING[filename](self.dataset_dir + filename, self.tokenizer)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, 'wb') as f:
                    pickle.dump(all_res, f)
            log_info(f'Loading {filename}: len={len(res)}')
            all_res.extend(res)
        random.shuffle(all_res)

        log_info(f'Training samples: {len(all_res)}')
        return all_res

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
    # 获取可能
    # get_probable_samples(['/data/corpus/sft/common/fnlp@moss-002-sft-data/zh_helpfulness.json'])

    # 测试各个文件的处理
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('../config', trust_remote_code=True)
    with Timeit() as ti:
        # pass
        process_alpaca('/data/corpus/sft/common/shibing624@alpaca-zh/alpaca_gpt4_data_zh.json', tokenizer)
        process_belle('/data/corpus/sft/common/BelleGroup@train_0.5M_CN/Belle_open_source_0.5M.json', tokenizer)
        process_belle('/data/corpus/sft/common/BelleGroup@train_1M_CN/Belle_open_source_1M.json', tokenizer)
        process_belle('/data/corpus/sft/common/BelleGroup@school_math_0.25M/school_math_0.25M.json', tokenizer)
        process_deepctrl('/data/corpus/sft/common/deepctrl@deepctrl-sft-data/sft_data_zh.jsonl', tokenizer)
        process_moss002('/data/corpus/sft/common/fnlp@moss-002-sft-data/zh_helpfulness.json', tokenizer)
        process_moss002('/data/corpus/sft/common/fnlp@moss-002-sft-data/zh_honesty.json', tokenizer)
        process_moss003('/data/corpus/sft/common/fnlp@moss-003-sft-data/conversations_with_tools_with_inner_instruction_no_text2image_train_all_random_meta0.5_0.1_0.01_moss_0709.jsonl', tokenizer)
        process_moss003('/data/corpus/sft/common/fnlp@moss-003-sft-data/moss-003-sft-no-tools.jsonl', tokenizer)
        process_shareai('/data/corpus/sft/common/shareAI@CodeChat/continue_zh.jsonl', tokenizer)
        process_shareai('/data/corpus/sft/common/shareAI@CodeChat/continue_zh_2.jsonl', tokenizer)
        process_shareai('/data/corpus/sft/common/shareAI@ShareGPT-Chinese-English-90k/common_zh_70k.jsonl', tokenizer)
        process_shareai('/data/corpus/sft/common/shareAI@ShareGPT-Chinese-English-90k/computer_en_26k_continue.jsonl', tokenizer)
        process_shareai('/data/corpus/sft/common/shareAI@ShareGPT-Chinese-English-90k/computer_zh_26k.jsonl', tokenizer)
        process_shareai('/data/corpus/sft/common/shareAI@ShareGPT-Chinese-English-90k/unknow_zh_38k_continue.jsonl', tokenizer)
        process_shareai('/data/corpus/sft/common/shareAI@ShareGPT-Chinese-English-90k/unknow_zh_38k.jsonl', tokenizer)
        process_firefly('/data/corpus/sft/common/YeungNLP@firefly-train-1.1M/firefly-train-1.1M.jsonl', tokenizer)