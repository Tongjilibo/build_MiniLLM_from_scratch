#! -*- coding: utf-8 -*-

from bert4torch.models import build_transformer_model
from bert4torch.snippets import sequence_padding
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch
from bert4torch.models import build_transformer_model
from bert4torch.snippets import IterDataset, DottableDict
from bert4torch.callbacks import Callback, Logger, EarlyStopping, Tensorboard, Evaluator
from bert4torch.optimizers import get_linear_schedule_with_warmup
from tqdm import tqdm
from glob import glob
import os
import numpy as np


# 基本参数
args = DottableDict()
args.lr = 5e-5
args.batch_size = 8
args.eval_batch_size = 4
args.grad_accumulation_steps = 1
args.pad_token_id = 0
args.max_length = 512
args.epochs = 1
args.data_path = 'F:/data/pretrain_data_bin/**/*.bin'
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
args.dir_path = './config'
args.config_path = os.path.join(args.dir_path, 'bert4torch_config.json')


# ========================加载数据集========================
filenames = glob(args.data_path, recursive=True)
smp_size = 0
for filename in filenames:
    with open(filename,'r') as f:
        nbytes = f.seek(0,2)
        flen = f.tell() // np.dtype('uint16').itemsize
    smp_size += flen//args.max_length
args.steps_per_epoch = smp_size // (args.batch_size * args.grad_accumulation_steps)

class MyDataset(IterDataset):
    @staticmethod
    def load_data(filenames):
        """加载数据，并尽量分为不超过maxlen的句子
        """
        for filename in filenames:
            with open(filename,'r') as f:
                nbytes = f.seek(0,2)
                flen = f.tell() // np.dtype('uint16').itemsize
            data = np.memmap(filename, dtype=np.dtype('uint16'),shape=(flen//args.max_length, args.max_length))
            for item in data:
                yield item

def collate_fn(batch_token_ids):
    batch_token_ids = torch.from_numpy(sequence_padding(batch_token_ids, value=args.pad_token_id).astype(np.int64))
    return [batch_token_ids[..., :-1]], batch_token_ids[..., 1:]

train_dataloader = DataLoader(MyDataset(filenames), batch_size=args.batch_size, collate_fn=collate_fn) 


# ========================建立模型========================
model = build_transformer_model(config_path=args.config_path, checkpoint_path=None, add_trainer=True).to(args.device)

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

optimizer = optim.AdamW(model.parameters(), args.lr)
scheduler = get_linear_schedule_with_warmup(optimizer, 0, args.steps_per_epoch*args.epochs)
model.compile(loss=CrossEntropyLoss(ignore_index=args.pad_token_id), optimizer=optimizer, scheduler=scheduler, 
              grad_accumulation_steps=args.grad_accumulation_steps, clip_grad_norm=1.0)



if __name__ == '__main__':
    logger = Logger('./ckpt/log_pretrain.log')
    evaluator = Evaluator(monitor='loss', method='step', mode='min', verbose=0, save_dir='./ckpt/best')
    early_stop = EarlyStopping(monitor='loss', verbose=1)
    ts_board = Tensorboard('./ckpt/tensorboard')  # tensorboard

    model.fit(train_dataloader, steps_per_epoch=args.steps_per_epoch, epochs=args.epochs, 
              callbacks=[evaluator, logger, ts_board, early_stop])
else:
    model.load_weights('./best_model_pretain.pt', strict=False)
