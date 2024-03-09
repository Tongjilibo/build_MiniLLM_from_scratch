#! -*- coding: utf-8 -*-
'''
指令微调
启动命令: nohup torchrun --standalone --nproc_per_node=4 pretrain.py --name baby > nohup.log&
'''
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from data_process import SFTDataset, collate_train_fn
from torch.utils.data.distributed import DistributedSampler
from bert4torch.models import build_transformer_model, BaseModel, BaseModelDDP
from bert4torch.snippets import ListDataset, DottableDict, log_info, get_weight_decay_optim_groups
from bert4torch.callbacks import Callback, Checkpoint, Logger, EarlyStopping, Tensorboard, Evaluator
from bert4torch.optimizers import get_linear_schedule_with_warmup
from collections import deque
import os
import numpy as np
import inspect
from transformers import AutoTokenizer


# 基本参数
args = DottableDict()
args.ddp_config = BaseModelDDP.init_process_group() if int(os.environ.get("RANK", -1)) != -1 else None
args.lr = 2e-5
args.batch_size = 8
args.grad_accumulation_steps = 1
args.pad_token_id = 0
args.max_length = 1024
args.epochs = 1
args.weight_decay = 0.1
args.interval = 2000
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
args.config_path = '../config'
args.model_path = '../ckpt/L12_H1024_A8-Wudao/final/model.pt'
args.save_dir = '../ckpt/L12_H1024_A8-Wudao-SFT'
args.filenames = [
    'shibing624@alpaca-zh/alpaca_gpt4_data_zh.json',
    'BelleGroup@train_0.5M_CN/Belle_open_source_0.5M'
    'BelleGroup@train_1M_CN/Belle_open_source_1M.json',
    'BelleGroup@school_math_0.25M/school_math_0.25M.json',
    'deepctrl@deepctrl-sft-data/sft_data_zh.jsonl',
    'fnlp@moss-002-sft-data/zh_helpfulness.json',
    'fnlp@moss-002-sft-data/zh_honesty.json',
    'fnlp@moss-003-sft-data/conversations_with_tools_with_inner_instruction_no_text2image_train_all_random_meta0.5_0.1_0.01_moss_0709.jsonl',
    'fnlp@moss-003-sft-data/moss-003-sft-no-tools.jsonl',
    'shareAI@CodeChat/continue_zh.jsonl',
    'shareAI@CodeChat/continue_zh_2.jsonl',
    'shareAI@ShareGPT-Chinese-English-90k/common_zh_70k.jsonl',
    'shareAI@ShareGPT-Chinese-English-90k/computer_cn_26k_continue.jsonl',
    'shareAI@ShareGPT-Chinese-English-90k/computer_en_26k(fixed).jsonl',
    'shareAI@ShareGPT-Chinese-English-90k/computer_zh_26k(fixed).jsonl',
    'shareAI@ShareGPT-Chinese-English-90k/computer_zh_26k.jsonl',
    'shareAI@ShareGPT-Chinese-English-90k/unknow_zh_38k.jsonl',
    'shareAI@ShareGPT-Chinese-English-90k/unknow_zh_38k_continue.jsonl',
    'YeungNLP@firefly-train-1.1M/firefly-train-1.1M.jsonl'
    ]
args.filenames = deque(['F:/data/corpus/sft/common/' + i for i in args.filenames])

# ========================加载数据集========================
tokenizer = AutoTokenizer.from_pretrained(args.config_path, trust_remote_code=True)
def get_trainloader(args):
    filename = args.filenames.popleft()
    dataset = SFTDataset(filename, tokenizer)
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, pin_memory=False, 
                                drop_last=False, shuffle=False, num_workers=0 if os.name == 'nt' else 2,
                                sampler=DistributedSampler(dataset) if args.ddp_config is not None else None,
                                collate_fn=collate_train_fn)
    return train_dataloader

train_dataloader = get_trainloader(args)
model = build_transformer_model(config_path=args.config_path, checkpoint_path=None, add_trainer=True)
model.to(args.device)
model.load_weights(args.model_path, mapping=lambda x: x.replace('module.', ''))  # 加载预训练权重

if args.ddp_config is not None:
    model = BaseModelDDP(model, master_rank=0, device_ids=[args.ddp_config.local_rank], output_device=args.ddp_config.local_rank, find_unused_parameters=False)
model.print_trainable_parameters()

class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def forward(self, logits, labels):
        '''
        logits: [btz, seq_len, vocab_size]
        labels: token_ids: [btz, seq_len]
        '''
        raw_dtyps = logits.dtype
        logits = logits.to(torch.float32)        
        logits = logits.reshape(-1, logits.shape[-1])
        labels = labels.flatten()
        loss = super().forward(logits, labels)

        return loss.to(raw_dtyps)

# 创建optimizer
optim_groups = get_weight_decay_optim_groups(model, weight_decay=args.weight_decay)
use_fused = 'fused' in inspect.signature(torch.optim.AdamW).parameters
extra_args = dict(fused=True) if use_fused else dict()
optimizer = optim.AdamW(optim_groups, lr=args.lr, betas=(0.9, 0.95), **extra_args)

model.compile(loss=CrossEntropyLoss(ignore_index=-100), optimizer=optimizer, 
              grad_accumulation_steps=args.grad_accumulation_steps, clip_grad_norm=1.0, mixed_precision=True)

class GenTrainLoader(Callback):
    """自动保存最新模型
    """
    def on_dataloader_end(self, logs=None):
        model.train_dataloader = get_trainloader(args)
    
if __name__ == '__main__':
    logger = Logger(args.save_dir+'/log_sft.log')
    checkpoint = Checkpoint(monitor='loss', epoch_or_step='step', min_max='min', verbose=0, interval=args.interval, 
                            save_dir=args.save_dir+'/{step}_{loss:.4f}', max_save_count=5, save_on_train_end=True)
    early_stop = EarlyStopping(monitor='loss', verbose=1, patience=3*args.interval)
    ts_board = Tensorboard(args.save_dir+'/tensorboard')  # tensorboard
    callbacks=[checkpoint, logger, ts_board]
    if args.ddp_config is not None:
        model.disable_run_callbacks(callbacks)

    model.fit(train_dataloader, steps_per_epoch=None, epochs=args.epochs, callbacks=[GenTrainLoader()]+callbacks)
