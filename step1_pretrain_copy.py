#! -*- coding: utf-8 -*-

import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch
from bert4torch.models import build_transformer_model, BaseModel, BaseModelDDP
from bert4torch.snippets import ListDataset, DottableDict, log_info, get_weight_decay_optim_groups
from bert4torch.callbacks import Callback, Checkpoint, Logger, EarlyStopping, Tensorboard, Evaluator
from bert4torch.optimizers import get_linear_schedule_with_warmup
from collections import deque
from glob import glob
import os
import numpy as np
import inspect


# 基本参数
args = DottableDict()
args.compile = False
args.ddp_config = BaseModelDDP.init_process_group() if int(os.environ.get("RANK", -1)) != -1 else None
args.lr = 3e-4
args.batch_size = 8
args.eval_batch_size = 4
args.grad_accumulation_steps = 1
args.pad_token_id = 0
args.max_length = 1024
args.epochs = 1
args.weight_decay = 0.1
args.interval = 2000
args.data_path = 'F:/data/pretrain_data_bin/**/*.bin'
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
args.dir_path = './config'
args.config_path = os.path.join(args.dir_path, 'bert4torch_config.json')

    
# ========================加载数据集========================
args.filenames = glob(args.data_path, recursive=True)
args.filenames = deque([i for i in args.filenames if 'wudaocorpus' not in i])  # 除去wudao外的140亿tokens
token_size, smp_size = 0, 0
for filename in args.filenames:
    with open(filename,'r') as f:
        nbytes = f.seek(0,2)
        flen = f.tell() // np.dtype('uint16').itemsize
    token_size += flen
    smp_size += flen//args.max_length
args.steps_per_epoch = smp_size // (args.batch_size * args.grad_accumulation_steps)
if args.ddp_config is not None:
    torch.manual_seed(1337 + args.ddp_config.rank)
    args.steps_per_epoch = int(args.steps_per_epoch/args.ddp_config.world_size)
log_info(f'token_size: {token_size}, smp_size: {smp_size}, steps_per_epoch: {args.steps_per_epoch}')

def get_trainloader(args):
    class MyDataset(ListDataset):
        def load_data(self, filename):
            """加载数据，并尽量分为不超过maxlen的句子
            """
            with open(filename,'r') as f:
                data=np.fromfile(f,dtype=np.uint16)
            data = data[:args.max_length*int(len(data)/args.max_length)]
            data = data.reshape(-1, args.max_length)
            data = torch.from_numpy(np.array(data).astype(np.int64))
            return data
        def __getitem__(self, index):
            return self.data[index][..., :-1], self.data[index][..., 1:]

    filename = args.filenames.popleft()
    print(f'=============={len(args.filenames)}===================')
    dataset = MyDataset(filename)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset) if args.ddp_config is not None else None
    return DataLoader(dataset, batch_size=args.batch_size, pin_memory=False, 
                                drop_last=False, shuffle=False, num_workers=0 if os.name == 'nt' else 4,
                                sampler=train_sampler) 

train_dataloader = get_trainloader(args)
model = build_transformer_model(config_path=args.config_path, checkpoint_path=None, add_trainer=True).to(args.device)

if args.compile:
    print("compiling the model... (takes a ~minute)")
    model = torch.compile(model)

if args.ddp_config is not None:
    prefix = "_orig_mod." if args.compile else ""
    model._ddp_params_and_buffers_to_ignore = {prefix + "freqs_cis"}
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

scheduler = get_linear_schedule_with_warmup(optimizer, 5000, args.steps_per_epoch*args.epochs)
model.compile(loss=CrossEntropyLoss(ignore_index=args.pad_token_id), optimizer=optimizer, scheduler=scheduler, 
              grad_accumulation_steps=args.grad_accumulation_steps, clip_grad_norm=1.0, mixed_precision=True)


class GenTrainLoader(Callback):
    """自动保存最新模型
    """
    def on_dataloader_end(self, logs=None):
        model.train_dataloader = get_trainloader(args)

if __name__ == '__main__':
    logger = Logger('./ckpt/log_pretrain.log')
    checkpoint = Checkpoint(monitor='loss', epoch_or_step='step', min_max='min', verbose=0, interval=args.interval, save_dir='./ckpt/{step}_{loss:.4f}', max_save_count=3)
    early_stop = EarlyStopping(monitor='loss', verbose=1, patience=3*args.interval)
    ts_board = Tensorboard('./ckpt/tensorboard')  # tensorboard
    callbacks=[checkpoint, logger, ts_board, early_stop]
    if args.ddp_config is not None:
        model.disable_run_callbacks(callbacks)

    model.fit(train_dataloader, steps_per_epoch=args.steps_per_epoch, epochs=args.epochs, callbacks=callbacks)
else:
    model.load_weights('./best_model_pretain.pt', strict=False)
