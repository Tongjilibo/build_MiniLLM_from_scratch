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
from bert4torch.snippets import sequence_padding, Timeit, log_info, log_warn
from bert4torch.snippets import parallel_apply, log_info_once, YamlConfig
from typing import Literal
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm
import os
import random
from transformers import AutoTokenizer
import numpy as np


# ================================数据处理的参数================================
args = YamlConfig('../config/dpo/MiniLLM-0.2B-DPO/dpo_args.yaml')['data_process']
HUMAN = '<human>'  # human标记符
ROBOT = '<robot>'  # answer标记符

# 多进程参数, linux下可用
USE_PARALLEL = False # if os.name == 'nt' else True
WORKERS = 8 # os.cpu_count(), 根据硬件条件配置
MAX_QUEUE_SIZE = 2000
tokenizer = AutoTokenizer.from_pretrained('../tokenizer', trust_remote_code=True)


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


def collect_tokens(process_one, filename, data_format:Literal['jsonl', 'json', 'table']='jsonl'):
    '''各个函数通用的处理token的方式'''
    def process_data_slice(data_slice, fi):
        # 是否并行处理数据
        if not USE_PARALLEL:
            log_info_once('Use single process to process data, maybe slow')
            train_samples = [process_one(line) for line in data_slice]
        else:
            log_info_once('Use multiprocess to accelerate data process')
            train_samples = parallel_apply(func=process_one, iterable=data_slice, workers=WORKERS, 
                                           max_queue_size=MAX_QUEUE_SIZE,
                                           dummy=False, callback=None, unordered=False)
        train_samples = [{'prompt_ids':i[0], 'chosen_ids':i[1], 'rejected_ids':i[2]} for i in train_samples if i[0] is not None and len(i[0])>1]

        save_path = os.path.join(args.dataset_save_dir, filename.replace('/', '--').replace('.jsonl', '').replace('.json', '') + f'_{fi}.jsonl')
        with open(save_path, 'w') as f:
            for item in train_samples:
                json.dump(item, f)  # 使用json.dump直接将字典转换为JSON格式写入文件
                f.write('\n')  # 每个JSON对象后面添加换行符
        return len(train_samples)


    # 读入数据，分json和jsonl两种格式
    data_path = os.path.join(args.dataset_src_dir, filename)
    os.makedirs(args.dataset_save_dir, exist_ok=True)

    all_count = 0  # 总的样本量
    data = None
    if data_format == 'parquet':
        df = pq.read_table(data_path).to_pandas()
        data = [df.loc[i].to_dict() for i in df.index]
        data_format = 'json'
    elif data_format == 'table':
        df = pd.read_table(data_path)
        data = [df.loc[i].to_dict() for i in df.index]
        data_format = 'json'

    if data_format == 'json':
        if data is None:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        if args.max_samples is not None:
            data = data[:args.max_samples]

        # 按照每个文件的最大样本数量来保存
        total_steps = int(np.ceil(len(data) / args.max_samples_per_file) * args.max_samples_per_file)
        for fi, start in enumerate(range(0, total_steps, args.max_samples_per_file), start=1):
            data_slice = data[start: start+args.max_samples_per_file]
            if len(data_slice) == 0:
                continue
            all_count += process_data_slice(data_slice, fi)
    elif data_format == 'jsonl':
        data = []
        f = open(data_path, 'r', encoding='utf-8')
        fi = 1
        while True:
            line = f.readline()
            if not line:
                break
            data.append(line)
            data_len = len(data)

            if data_len >= args.max_samples_per_file:
                all_count += process_data_slice(data, fi)
                data = []
                fi += 1

            if (args.max_samples is not None) and (data_len >= args.max_samples):
                break
        
        if len(data) > 0:
            all_count += process_data_slice(data, fi)
    
    # debug使用
    # print('='*60)
    # print('data_path:', data_path)
    # print('len(train_samples):', all_count)
    # print('='*60)
    # print()
    return all_count


def process_DPO_En_Zh_20k(filename, tokenizer):
    '''hiyouga/DPO-En-Zh-20k'''

    def process_one(per):
        prompt_ids = tokenizer.encode(HUMAN + per['system'] + per['prompt'] + ROBOT, add_special_tokens=False)
        chosen_ids = tokenizer.encode(per['answer'][0], add_special_tokens=False)
        rejected_ids = tokenizer.encode(per['answer'][1], add_special_tokens=False)

        if len(prompt_ids) + len(chosen_ids) >= args.MAX_LENGTH or len(prompt_ids) + len(rejected_ids) >= args.MAX_LENGTH:
            return None, None
        # 这里labels要比input_ids少1位
        return prompt_ids, chosen_ids, rejected_ids

    return collect_tokens(process_one, filename, data_format='json')


def hh_rlhf_cn(filename, tokenizer):
    '''dikw/hh_rlhf_cn'''
    def process_one(line):
        per = json.loads(line)
        prompt_ids = []
        for context in per['context']:  # 最后一句是human
            if context['role'] == 'human':
                q = tokenizer.encode(HUMAN + context['text'] + ROBOT, add_special_tokens=False)
                prompt_ids.extend(q)
            elif context['role'] == 'assistant':
                a = tokenizer.encode(context['text'], add_special_tokens=False)
                prompt_ids.extend(a)
            
            # 轮次太多的话，则进行截断
            if len(prompt_ids) >= args.MAX_LENGTH:
                break
            
        chosen_ids = tokenizer.encode(per['chosen']['text'], add_special_tokens=False)
        rejected_ids = tokenizer.encode(per['rejected']['text'], add_special_tokens=False)
        
        if len(prompt_ids) + len(chosen_ids) >= args.MAX_LENGTH or len(prompt_ids) + len(rejected_ids) >= args.MAX_LENGTH:
            return None, None
        # 这里labels要比input_ids少1位
        return prompt_ids, chosen_ids, rejected_ids

    return collect_tokens(process_one, filename)


def CValues_Comparison(filename, tokenizer):
    '''diic/CValues-Comparison'''
    def process_one(per):
        prompt_ids = tokenizer.encode(HUMAN + per['prompt'] + ROBOT, add_special_tokens=False)
        chosen_ids = tokenizer.encode(per['pos_resp'], add_special_tokens=False)
        rejected_ids = tokenizer.encode(per['neg_resp'], add_special_tokens=False)

        if len(prompt_ids) + len(chosen_ids) >= args.MAX_LENGTH or len(prompt_ids) + len(rejected_ids) >= args.MAX_LENGTH:
            return None, None
        # 这里labels要比input_ids少1位
        return prompt_ids, chosen_ids, rejected_ids

    return collect_tokens(process_one, filename, data_format='jsonl')


def zhihu_rlhf_3k(filename, tokenizer):
    '''liyucheng/zhihu_rlhf_3k'''
    def process_one(per):
        prompt_ids = tokenizer.encode(HUMAN + per['prompt'] + ROBOT, add_special_tokens=False)
        chosen_ids = tokenizer.encode(per['chosen'], add_special_tokens=False)
        rejected_ids = tokenizer.encode(per['rejected'], add_special_tokens=False)

        if len(prompt_ids) + len(chosen_ids) >= args.MAX_LENGTH or len(prompt_ids) + len(rejected_ids) >= args.MAX_LENGTH:
            return None, None
        # 这里labels要比input_ids少1位
        return prompt_ids, chosen_ids, rejected_ids

    return collect_tokens(process_one, filename, data_format='table')


def rlhf_reward_single_round_trans_chinese(filename, tokenizer):
    def process_one(per):
        prompt_ids = tokenizer.encode(HUMAN + per['prompt'] + ROBOT, add_special_tokens=False)
        chosen_ids = tokenizer.encode(per['chosen'], add_special_tokens=False)
        rejected_ids = tokenizer.encode(per['rejected'], add_special_tokens=False)

        if len(prompt_ids) + len(chosen_ids) >= args.MAX_LENGTH or len(prompt_ids) + len(rejected_ids) >= args.MAX_LENGTH:
            return None, None
        # 这里labels要比input_ids少1位
        return prompt_ids, chosen_ids, rejected_ids

    return collect_tokens(process_one, filename, data_format='parquet')


MAPPING = {
    'hiyouga@DPO-En-Zh-20k/dpo_zh.json': process_DPO_En_Zh_20k,
    'AI-ModelScope@hh_rlhf_cn/hh_rlhf_train.jsonl': hh_rlhf_cn,
    'AI-ModelScope@hh_rlhf_cn/hh_rlhf_test.jsonl': hh_rlhf_cn,
    "iic@CValues-Comparison/train.jsonl": CValues_Comparison,
    "iic@CValues-Comparison/test.jsonl": CValues_Comparison,
    "liyucheng@zhihu_rlhf_3k/zhihu_3k_rlfh.tsv": zhihu_rlhf_3k,
    "beyond@rlhf-reward-single-round-trans_chinese/train-00000-of-00001-789dc5dece0f1fc1.parquet": rlhf_reward_single_round_trans_chinese,
    "beyond@rlhf-reward-single-round-trans_chinese/test-00000-of-00001-8ecd46436fadcf7f.parquet": rlhf_reward_single_round_trans_chinese,
}


class DPODataset(Dataset):
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
    chosen_ids, chosen_labels, rejected_ids, rejected_labels = [], [], [], []
    for smp in batch:
        prompt_id, chosen_id, rejected_id = smp['prompt_ids'], smp['chosen_ids'], smp['rejected_ids']
        chosen_ids.append(prompt_id + chosen_id)
        chosen_labels.append([args.pad_token_id] * (len(prompt_id)-1) + chosen_id + [args.eos_token_id])  # prompt部分用padding位
        rejected_ids.append(prompt_id + rejected_id)
        rejected_labels.append([args.pad_token_id] * (len(prompt_id)-1) + rejected_id + [args.eos_token_id])

    # 这里是把chosen和rejected放到同一个batch中，前半部分是chosen，后半部分是rejected
    input_ids = torch.tensor(sequence_padding(chosen_ids+rejected_ids, value=args.pad_token_id), dtype=torch.long)
    input_labels = torch.tensor(sequence_padding(chosen_labels+rejected_labels, value=args.pad_token_id), dtype=torch.long)
    return input_ids, input_labels


def main():
    '''数据处理主函数
    :param FILE_NAMES: 处理的数据集名称
    :param DATASET_SRC_DIR: 数据源文件夹
    :param DATASET_SAVE_DIR: 数据保存的文件夹
    :param tokenizer: tokenizer
    '''
    if args.max_samples is not None:
        log_warn(f'Only use {args.max_samples} samples for each dpo dataset.')
    else:
        log_warn(f'Use all samples for each dpo dataset, may be slow.')
    
    with Timeit() as ti:
        for filename in args.file_names:
            sample_count = MAPPING[filename](filename, tokenizer)
            ti.lap(name=f'{filename}: len={sample_count}'.ljust(70) + '-', reset=True)


if __name__ == '__main__':
    main()

    # 测试使用
    # filename = 'Tongjilibo/self_cognition.json'
    # USE_PARALLEL = False
    # sample_count = MAPPING[filename](filename, tokenizer)
