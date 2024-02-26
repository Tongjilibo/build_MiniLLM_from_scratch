#! -*- coding: utf-8 -*-

import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch
from bert4torch.models import build_transformer_model, BaseModel
from bert4torch.snippets import IterDataset, DottableDict, log_info, get_weight_decay_optim_groups
from bert4torch.callbacks import Checkpoint, Logger, EarlyStopping, Tensorboard, Evaluator
from bert4torch.optimizers import get_linear_schedule_with_warmup
from tqdm import tqdm
from glob import glob
import os
import numpy as np
import inspect


# 基本参数
args = DottableDict()
args.lr = 3e-4
args.batch_size = 32
args.eval_batch_size = 4
args.grad_accumulation_steps = 1
args.pad_token_id = 0
args.max_length = 1024
args.epochs = 1
args.weight_decay = 0.1
args.data_path = '/home/hfai/data/pretrain/pretrain_data_bin/**/*.bin'
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
args.dir_path = './config'
args.config_path = os.path.join(args.dir_path, 'bert4torch_config.json')


# ========================加载数据集========================
filenames = glob(args.data_path, recursive=True)
filenames = [i for i in filenames if 'wudaocorpus' not in i]  # 除去wudao外的140亿tokens
token_size, smp_size = 0, 0
for filename in filenames:
    with open(filename,'r') as f:
        nbytes = f.seek(0,2)
        flen = f.tell() // np.dtype('uint16').itemsize
    token_size += flen
    smp_size += flen//args.max_length
args.steps_per_epoch = smp_size // (args.batch_size * args.grad_accumulation_steps)
log_info(f'token_size: {token_size}, smp_size: {smp_size}, steps_per_epoch: {args.steps_per_epoch}')

class MyDataset(IterDataset):
    def __init__(self, file_path=None, memmap=False, **kwargs):
        self.memmap = memmap
        super().__init__(file_path, **kwargs)

    def load_data(self, filenames):
        """加载数据，并尽量分为不超过maxlen的句子
        """
        for filename in filenames:
            if self.memmap:
                with open(filename,'r') as f:
                    nbytes = f.seek(0,2)
                    flen = f.tell() // np.dtype('uint16').itemsize
                data = np.memmap(filename, dtype=np.dtype('uint16'),shape=(flen//args.max_length, args.max_length))
            else:
                with open(filename,'r') as f:
                    data=np.fromfile(f,dtype=np.uint16)
                data = data[:args.max_length*int(len(data)/args.max_length)]
                data = data.reshape(-1, args.max_length)
                data = torch.from_numpy(np.array(data).astype(np.int64))
            for item in data:
                yield item[..., :-1], item[..., 1:]

train_dataloader = DataLoader(MyDataset(filenames), batch_size=args.batch_size, pin_memory=False, 
                              drop_last=False, shuffle=False, num_workers=0 if os.name == 'nt' else 4) 

# ========================建立模型========================
def init_model():
    # model init
    # model init
    model_args = dict(
        dim=1024,
        n_layers=12,
        n_heads=8,
        n_kv_heads=8,
        vocab_size=64793,
        multiple_of=32,
        max_seq_len=1024,
        dropout=0,
    )  # start with model_args from command line
        # init a new model from scratch
    print("Initializing a new model from scratch")
    from model import Transformer, ModelArgs
    gptconf = ModelArgs(**model_args)
    model = Transformer(gptconf)
    return model

# class Model(BaseModel):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.module = init_model()
#     def forward(self, *inputs, **input_kwargs):
#         return self.module.forward(*inputs, **input_kwargs)
# model = Model().to(args.device)
model = build_transformer_model(config_path=args.config_path, checkpoint_path=None, add_trainer=True).to(args.device)
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


if __name__ == '__main__':
    logger = Logger('./ckpt/log_pretrain.log')
    evaluator = Checkpoint(monitor='loss', method='step', mode='min', verbose=0, interval=2000, save_dir='./ckpt/{step}', max_save_count=3)
    early_stop = EarlyStopping(monitor='loss', verbose=1, patience=3, interval=2000)
    ts_board = Tensorboard('./ckpt/tensorboard')  # tensorboard

    model.fit(train_dataloader, steps_per_epoch=args.steps_per_epoch, epochs=args.epochs, 
              callbacks=[evaluator, logger, ts_board, early_stop])
else:
    model.load_weights('./best_model_pretain.pt', strict=False)
