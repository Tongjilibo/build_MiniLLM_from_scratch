#! -*- coding: utf-8 -*-
"""
权重转化为transformers可以使用的格式
"""
import os
from bert4torch.models import build_transformer_model
from bert4torch.snippets import copytree

config_path = '../config'
# model_path = '../ckpt/MiniLLM-L12_H1024_A8-WithWudao/final/model.pt'
model_path = '../ckpt/L12_H1024_A8-Wudao-SFT_Alpaca/final/model.pt'
save_dir = os.path.dirname(model_path) + '_transformers'

model = build_transformer_model(config_path=config_path, checkpoint_path=None, add_trainer=True)
model.load_weights(model_path, mapping=lambda x: x.replace('module.', ''))

model.save_pretrained(save_dir)
copytree(config_path, save_dir, dirs_exist_ok=True)