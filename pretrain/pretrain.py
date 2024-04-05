#! -*- coding: utf-8 -*-
""" 预训练

1. 单机多卡训练
torchrun --standalone --nproc_per_node=4 pretrain.py

2. 多机多卡训练
NCCL_DEBUG=INFO TORCH_NCCL_BLOCKING_WAIT=1 NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=你的网卡类型  torchrun --nnodes=你的主机数量 --node_rank=编号 --master_addr=你的master节点IP --master_port=12346 --nproc_per_node=8 pretrain.py

3. deepspeed方式训练
deepspeed --num_gpus=1 --master_port $(shuf -n 1 -i 10000-65535) pretrain.py  --deepspeed ../config/MiniLLM-0.2B-WithWudao/ds_config.json
"""

import os
import numpy as np
import inspect
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from bert4torch.models import build_transformer_model, BaseModelDDP, DeepSpeedTrainer
from bert4torch.snippets import YamlConfig, log_info, get_weight_decay_optim_groups, argument_parse
from bert4torch.callbacks import Checkpoint, Logger, Tensorboard
from bert4torch.optimizers import get_linear_schedule_with_warmup
from bert4torch.losses import CausalLMLoss
from glob import glob


# 训练使用到的参数，可加载不同的文件
args = YamlConfig('../config/MiniLLM-0.2B-WithWudao/pretrain_args.yaml')
if os.environ.get("RANK") is None:  # 单卡
    args.train_mode = 'single'
elif argument_parse('deepspeed').deepspeed is not None:  # deepspeed
    args.train_mode = 'deepspeed'
else:  # DDP
    args.train_mode = 'ddp'
    args.ddp_config = BaseModelDDP.init_process_group()

args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
args.resume_path = None
args.filenames = [i for i in glob(args.data_path, recursive=True)]
if not args.include_wudao_corpus:
    args.filenames = [i for i in args.filenames if 'wudaocorpus' not in i]


class MyDataset(Dataset):
    def __init__(self, filenames):
        """加载数据"""
        self.data = []
        self.index_map = {}
        self.token_size, self.smp_size = 0, 0
        for fi, filename in enumerate(filenames):
            with open(filename, 'r') as f:
                nbytes = f.seek(0, 2)
                flen = f.tell() // np.dtype('uint16').itemsize
            self.token_size += flen
            self.index_map.update({self.smp_size + i: (fi, i) for i in range(flen // args.max_length)})
            self.smp_size += flen // args.max_length
            self.data.append(
                np.memmap(filename, dtype=np.dtype('uint16'), shape=(flen // args.max_length, args.max_length)))
        log_info(f'token_size: {self.token_size}, smp_size: {self.smp_size}')

    def __len__(self):
        return self.smp_size

    def __getitem__(self, index: int):
        fi, i = self.index_map[index]
        sample = self.data[fi][i]
        X = np.array(sample[:-1]).astype(np.int64)
        Y = np.array(sample[1:]).astype(np.int64)
        return torch.from_numpy(X), torch.from_numpy(Y)


dataset = MyDataset(args.filenames)
train_dataloader = DataLoader(dataset,
                              batch_size=args.batch_size,
                              pin_memory=False,
                              drop_last=False, shuffle=False,
                              num_workers=0 if os.name == 'nt' else 4,
                              sampler=DistributedSampler(dataset) if args.train_mode == 'ddp' else None)

model = build_transformer_model(config_path=args.config_path, checkpoint_path=None, add_trainer=True,
                                torch_dtype=args.torch_dtype)
model.to(args.device)
model.print_trainable_parameters()

if args.train_mode in {'single', 'ddp'}:
    model = BaseModelDDP(model) if args.train_mode == 'ddp' else model

    optim_groups = get_weight_decay_optim_groups(model, weight_decay=args.weight_decay)
    use_fused = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = optim.AdamW(
        optim_groups,
        lr=args.lr,
        betas=(0.9, 0.95),
        **extra_args
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        5000,
        len(train_dataloader) * args.epochs // args.grad_accumulation_steps
    )

    model.compile(
        loss=CausalLMLoss(ignore_index=args.pad_token_id),
        optimizer=optimizer,
        scheduler=scheduler,
        grad_accumulation_steps=args.grad_accumulation_steps,
        clip_grad_norm=1.0,
        mixed_precision=True if args.torch_dtype is None else False
    )

elif args.train_mode == 'deepspeed':
    model = DeepSpeedTrainer(model)
    model.compile(loss=CausalLMLoss(ignore_index=args.pad_token_id))


if args.resume_path:
    mapping = None if args.train_mode == 'ddp' else lambda x: x.replace('module.', '')
    model.resume_from_checkpoint(args.resume_path, mapping=None)

if __name__ == '__main__':
    logger = Logger(args.save_dir + '/log_pretrain.log')
    checkpoint = Checkpoint(monitor='loss',
                            epoch_or_step='step',
                            min_max='min',
                            verbose=0,
                            interval=args.interval,
                            save_dir=args.save_dir + '/{step}_{loss:.4f}',
                            max_save_count=5,
                            save_on_train_end=True)
    ts_board = Tensorboard(args.save_dir + '/tensorboard')  # tensorboard
    callbacks = [checkpoint, logger, ts_board]
    if args.train_mode == 'ddp':
        model.disable_workers_callback(callbacks)
    model.fit(train_dataloader, steps_per_epoch=None, epochs=args.epochs, callbacks=callbacks)
