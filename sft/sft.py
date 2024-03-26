#! -*- coding: utf-8 -*-
"""
========指令微调========
单机多卡训练
torchrun --standalone --nproc_per_node=4 sft.py
多机多卡训练
NCCL_DEBUG=INFO TORCH_NCCL_BLOCKING_WAIT=1 NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=你的网卡类型  torchrun --nnodes=你的主机数量 --node_rank=编号 --master_addr=你的master节点IP --master_port=12346 --nproc_per_node=8 sft.py

注意事项：
1. data_process下有个MAX_SAMPLES参数, 可设置比如1000先在小数据集上验证跑通
2. 目前支持数据加载方式：
    1）一次性加载所有数据，适合数据集不大的情况
    2）每次从训练datasets随机挑选一个dataset进行训练，训练完成后再重新选择一个dataset，适合数据集较大时候
       优点是节省内存空间，缺点是仅在该dataset内部shuffle，loss曲线可能会抖动
"""
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from data_process import SFTDataset, collate_train_fn, get_samples_count
from torch.utils.data.distributed import DistributedSampler
from bert4torch.models import build_transformer_model, BaseModelDDP
from bert4torch.snippets import JsonConfig, get_weight_decay_optim_groups, log_info
from bert4torch.callbacks import Checkpoint, Logger, EarlyStopping, Tensorboard, Callback
from bert4torch.optimizers import get_linear_schedule_with_warmup
import os
import inspect
from glob import glob
from collections import deque
import random


# 训练使用到的参数，可加载不同的文件
args = JsonConfig('../config/MiniLLM-0.2B-WithWudao-SFT_Alpaca/sft_args.json')
args.ddp_config = BaseModelDDP.init_process_group() if int(os.environ.get("RANK", -1)) != -1 else None
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
filenames = glob(args.dataset_save_dir + '/*.jsonl')
random.shuffle(filenames)
args.filenames = deque(filenames)


# ========================加载数据集========================
def get_trainloader(args):
    if not args.one_dataset_every_time:
        # 一次吃进去所有训练数据集，对内存要求较高
        dataset = SFTDataset(datadir=filenames)
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
    else:
        # 一次使用一个数据文件
        if len(args.filenames) == 0:
            args.filenames = deque(filenames)
            # log_info('all datasets consumed, start a new epoch')

        filename = args.filenames.popleft()
        dataset = SFTDataset([filename], verbose=0)
        train_dataloader = DataLoader(dataset, batch_size=args.batch_size, pin_memory=False, 
                                    drop_last=False, shuffle=False, num_workers=0 if os.name == 'nt' else 2,
                                    sampler=DistributedSampler(dataset) if args.ddp_config is not None else None,
                                    collate_fn=collate_train_fn)
    return train_dataloader
train_dataloader = get_trainloader(args)

if args.one_dataset_every_time:
    sample_count = get_samples_count(filenames)
    total_steps = sample_count * args.epochs // (args.batch_size * args.grad_accumulation_steps)
    if args.ddp_config is not None:
        total_steps = total_steps // args.ddp_config.world_size
else:
    total_steps = len(train_dataloader) * args.epochs // args.grad_accumulation_steps


# ========================加载预训练模型========================
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


# ========================配置loss, optimizer等参数========================
optim_groups = get_weight_decay_optim_groups(model, weight_decay=args.weight_decay)
use_fused = 'fused' in inspect.signature(torch.optim.AdamW).parameters
extra_args = dict(fused=True) if use_fused else dict()
optimizer = optim.AdamW(
    optim_groups,
    lr=args.lr,
    betas=(0.9, 0.95),
    **extra_args
)

scheduler = get_linear_schedule_with_warmup(optimizer, min(5000, int(0.1 * total_steps)), total_steps)
model.compile(
    loss=CrossEntropyLoss(ignore_index=args.pad_token_id),
    optimizer=optimizer,
    scheduler=scheduler,
    grad_accumulation_steps=args.grad_accumulation_steps,
    clip_grad_norm=1.0,
    mixed_precision=True if args.torch_dtype is None else False
)

class GenTrainLoader(Callback):
    """当前dataloader消耗完，自动用下一个文件生成dataloder
    """
    def on_dataloader_end(self, logs=None):
        model.train_dataloader = get_trainloader(args)


if __name__ == '__main__':
    logger = Logger(args.save_dir + '/log_sft.log')
    checkpoint = Checkpoint(
        monitor='loss',
        epoch_or_step='step',
        min_max='min',
        verbose=0,
        interval=args.interval,
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
    if args.one_dataset_every_time:
        callbacks = [GenTrainLoader()] + callbacks
    
    model.fit(train_dataloader, 
              steps_per_epoch=None if not args.one_dataset_every_time else total_steps // args.epochs, 
              epochs=args.epochs, 
              callbacks=callbacks)
