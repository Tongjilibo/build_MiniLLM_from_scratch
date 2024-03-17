#! -*- coding: utf-8 -*-
"""
指令微调
单机多卡训练
torchrun --standalone --nproc_per_node=4 sft.py
多机多卡训练
NCCL_DEBUG=INFO TORCH_NCCL_BLOCKING_WAIT=1 NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=你的网卡类型  torchrun --nnodes=你的主机数量 --node_rank=编号 --master_addr=你的master节点IP --master_port=12346 --nproc_per_node=8 sft.py
"""
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from data_process import SFTDataset, collate_train_fn
from torch.utils.data.distributed import DistributedSampler
from bert4torch.models import build_transformer_model, BaseModelDDP
from bert4torch.snippets import DottableDict, get_weight_decay_optim_groups
from bert4torch.callbacks import Checkpoint, Logger, EarlyStopping, Tensorboard
from bert4torch.optimizers import get_linear_schedule_with_warmup
import os
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
args.epochs = 5
args.weight_decay = 0.1
args.interval = 2000
args.torch_dtype = None  # 默认使用混合精度训练，可以制定为torch.float32，torch.float16或者torch.bfloat16
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
args.config_path = '../config'
args.model_path = '../ckpt/MiniLLM-L12_H1024_A8-NoWudao/final/model.pt'
args.save_dir = '../ckpt/MiniLLM-L12_H1024_A8-NoWudao-SFT'
args.dataset_path = '/share/home/zyx/Dataset/sft_dataset/'
args.dataset_save_path = '../sft_data/'

filenames = [
    'alpaca-zh/alpaca_gpt4_data_zh.json',
    'BelleGroup/Belle_open_source_0.5M.jsonl',
    'BelleGroup/Belle_open_source_1M.jsonl',
    'BelleGroup/school_math_0.25M.jsonl',
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

# ========================加载数据集========================

### 这可能需要很久，因为数据集很大

tokenizer = AutoTokenizer.from_pretrained(args.config_path, trust_remote_code=True)

dataset = SFTDataset(
    filenames=filenames,
    tokenizer=tokenizer,
    dataset_dir=args.dataset_path,
    save_dir=args.dataset_save_path
)

train_dataloader = DataLoader(
    dataset=dataset,
    batch_size=args.batch_size,
    pin_memory=False,
    drop_last=False,
    shuffle=False,
    num_workers=0 if os.name == 'nt' else 2,
    sampler=DistributedSampler(dataset) if args.ddp_config is not None else None,
    collate_fn=collate_train_fn
)

model = build_transformer_model(
    config_path=args.config_path,
    checkpoint_path=None,
    add_trainer=True
)

model.to(args.device)
model.load_weights(args.model_path, mapping=lambda x: x.replace('module.', ''))

if args.ddp_config is not None:
    model = BaseModelDDP(
        model,
        master_rank=0,
        device_ids=[args.ddp_config.local_rank],
        output_device=args.ddp_config.local_rank,
        find_unused_parameters=False
    )
model.print_trainable_parameters()


class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, logits, labels):
        """
        logits: [btz, seq_len, vocab_size]
        labels: token_ids: [btz, seq_len]
        """
        raw_dtyps = logits.dtype
        logits = logits.to(torch.float32)
        logits = logits.reshape(-1, logits.shape[-1])
        labels = labels.flatten()
        loss = super().forward(logits, labels)

        return loss.to(raw_dtyps)


optim_groups = get_weight_decay_optim_groups(model, weight_decay=args.weight_decay)
use_fused = 'fused' in inspect.signature(torch.optim.AdamW).parameters
extra_args = dict(fused=True) if use_fused else dict()
optimizer = optim.AdamW(
    optim_groups,
    lr=args.lr,
    betas=(0.9, 0.95),
    **extra_args
)

total_steps = len(train_dataloader) * args.epochs // args.grad_accumulation_steps
scheduler = get_linear_schedule_with_warmup(optimizer, min(5000, int(0.1 * total_steps)), total_steps)
model.compile(
    loss=CrossEntropyLoss(ignore_index=args.pad_token_id),
    optimizer=optimizer,
    scheduler=scheduler,
    grad_accumulation_steps=args.grad_accumulation_steps,
    clip_grad_norm=1.0,
    mixed_precision=True if args.torch_dtype is None else False
)

if __name__ == '__main__':
    logger = Logger(args.save_dir + '/log_sft.log')
    checkpoint = Checkpoint(
        monitor='loss',
        epoch_or_step='step',
        min_max='min',
        verbose=0,
        nterval=args.interval,
        save_dir=args.save_dir + '/{step}_{loss:.4f}',
        max_save_count=5,
        save_on_train_end=True
    )
    early_stop = EarlyStopping(
        monitor='loss',
        verbose=1,
        patience=3 * args.interval
    )

    ts_board = Tensorboard(args.save_dir + '/tensorboard')  # tensorboard
    callbacks = [checkpoint, logger, ts_board]
    if args.ddp_config is not None:
        model.disable_run_callbacks(callbacks)

    model.fit(train_dataloader, steps_per_epoch=None, epochs=args.epochs, callbacks=callbacks)
