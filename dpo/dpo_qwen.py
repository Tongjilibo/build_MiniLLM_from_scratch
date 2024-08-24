#! -*- coding: utf-8 -*-
"""
========DPO========
单机多卡训练
torchrun --standalone --nproc_per_node=4 dpo.py
多机多卡训练
NCCL_DEBUG=INFO TORCH_NCCL_BLOCKING_WAIT=1 NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=你的网卡类型  torchrun --nnodes=你的主机数量 --node_rank=编号 --master_addr=你的master节点IP --master_port=12346 --nproc_per_node=8 dpo.py

注意事项：
1. data_process下有个MAX_SAMPLES参数, 可设置比如1000先在小数据集上验证跑通
2. 目前支持数据加载方式：
    1）一次性加载所有数据，适合数据集不大的情况
    2）每次从训练datasets随机挑选一个dataset进行训练，训练完成后再重新选择一个dataset，适合数据集较大时候
       优点是节省内存空间，缺点是仅在该dataset内部shuffle，loss曲线可能会抖动
"""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from bert4torch.models import build_transformer_model, BaseModelDDP
from bert4torch.trainer import DPOTrainer
from bert4torch.snippets import YamlConfig, get_weight_decay_optim_groups, find_all_linear_names
from bert4torch.callbacks import Checkpoint, Logger, EarlyStopping, Tensorboard, Callback
from bert4torch.optimizers import get_cosine_schedule_with_warmup
from bert4torch.losses import DPOLoss
import os
import inspect
from glob import glob
from collections import deque, Counter
import random
from copy import deepcopy


# 训练使用到的参数，可加载不同的文件
args = YamlConfig('./config/MiniLLM-0.2B-WithWudao-DPO/dpo_args.yaml')['dpo']
filenames = glob(args.dataset_save_dir + '/*.jsonl')
random.seed(100)
random.shuffle(filenames)
args.filenames = deque(filenames)
args.ddp_config = BaseModelDDP.init_process_group() if int(os.environ.get("RANK", -1)) != -1 else None
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ========================加载数据集========================
from torch.utils.data import Dataset
class DPODataset(Dataset):
    def __init__(self, datadir, verbose=1):
        super().__init__()
        self.verbose = verbose
        self.data = self.load_data(datadir)

    def load_data(self, filenames):
        all_res = []
        for i in range(100):
            all_res.append(torch.load(filenames, map_location='cpu'))
        # random.shuffle(all_res)
        return all_res

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        return self.data[index]

def collate_train_fn(batch):
    input_ids, input_mask, input_labels = [], [], []
    for smp in batch:
        input_ids.append(smp['concatenated_input_ids'])
        input_mask.append(smp['concatenated_attention_mask'])
        input_labels.append(smp['concatenated_labels'])

    # 这里是把chosen和rejected放到同一个batch中，前半部分是chosen，后半部分是rejected
    input_ids = torch.cat(input_ids)
    input_labels = torch.cat(input_labels)
    return input_ids, input_labels


dataset = DPODataset(datadir='/home/lb/projects/MedicalGPT/data/tmp.data')
train_dataloader = DataLoader(
    dataset=dataset,
    batch_size=1,
    pin_memory=False,
    drop_last=False,
    shuffle=False,
    num_workers=0 if os.name == 'nt' else 2,
    sampler=DistributedSampler(dataset) if args.ddp_config is not None else None,
    collate_fn=collate_train_fn
)
steps_per_epoch = len(train_dataloader)
total_steps = steps_per_epoch * args.grad_accumulation_steps


# ========================加载预训练模型========================
net = build_transformer_model(
    config_path='/data/pretrain_ckpt/Qwen/Qwen1.5-0.5B-Chat',
    checkpoint_path='/data/pretrain_ckpt/Qwen/Qwen1.5-0.5B-Chat',
    pad_token_id = 151643,
    torch_dtype = 'float16',
    is_causal=False,
    tie_word_embeddings=True
)
for param in filter(lambda p: p.requires_grad, net.parameters()):
    param.data = param.data.to(torch.float32)

net.to(args.device)

if args.use_peft:
    from peft import LoraConfig
    peft_config = LoraConfig(
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=find_all_linear_names(
            net, 
            int4=getattr(args, 'load_in_4bit', False), 
            int8=getattr(args, 'load_in_8bit', False))
    )

trainer = DPOTrainer(
    net, 
    ref_model=None if args.use_peft else deepcopy(net),
    args=args,
    peft_config=peft_config if args.use_peft else None)
optimizer = optim.AdamW(net.parameters(), args.lr)


if args.ddp_config is not None:
    trainer = BaseModelDDP(
        trainer,
        master_rank=0,
        device_ids=[args.ddp_config.local_rank],
        output_device=args.ddp_config.local_rank,
        find_unused_parameters=False
    )


# ========================配置loss, optimizer等参数========================
optim_groups = get_weight_decay_optim_groups(net, weight_decay=0) #args.weight_decay)
use_fused = 'fused' in inspect.signature(torch.optim.AdamW).parameters
extra_args = dict(fused=True) if use_fused else dict()
# optimizer = optim.AdamW(
#     optim_groups,
#     lr=args.lr,
#     betas=(0.9, 0.99),
#     **extra_args
# )

from transformers.optimization import AdamW
optimizer = AdamW(
    optim_groups,
    lr=args.lr,
    betas=(0.9, 0.999),
    eps = 1e-8
)

scheduler = get_cosine_schedule_with_warmup(optimizer, 100, total_steps)
trainer.compile(
    loss=DPOLoss(pad_token_id=-100, offset=True),
    optimizer=optimizer,
    scheduler=scheduler,
    grad_accumulation_steps=args.grad_accumulation_steps,
    clip_grad_norm=1.0,
    # mixed_precision=True if args.torch_dtype is None else False
    stateful_metrics='loss'
)


if __name__ == '__main__':
    logger = Logger(args.save_dir + '/log_dpo.log')
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
        trainer.disable_workers_callback(callbacks)
    
    trainer.fit(train_dataloader, 
              steps_per_epoch=steps_per_epoch, 
              epochs=args.epochs, 
              callbacks=callbacks)
