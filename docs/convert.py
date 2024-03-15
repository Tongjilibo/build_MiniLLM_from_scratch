#! -*- coding: utf-8 -*-
"""
权重转化为transformers可以使用的格式
"""
import os
import torch
from bert4torch.models import build_transformer_model
from bert4torch.snippets import DottableDict
from transformers import AutoTokenizer

config_path = '../config/bert4torch_config.json'
model_path = '../ckpt/L12_H1024_A8-WithWudao/final/model.pt'

model = build_transformer_model(config_path=config_path, checkpoint_path=None, add_trainer=True)
model.load_weights(model_path, mapping=lambda x: x.replace('module.', ''))

save_path = os.path.join(os.path.dirname(model_path), 'pytorch_model.bin')
model.save_pretrained(save_path)