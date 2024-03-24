#! -*- coding: utf-8 -*-
"""
权重转化为transformers可以使用的格式
"""
import os
from bert4torch.models import build_transformer_model
from bert4torch.snippets import copytree

config_path = '../config'
model_path = 'your absolute path to the model checkpoint include model.pt'
save_dir = '../output_transformer'

if not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)

model = build_transformer_model(config_path=config_path, checkpoint_path=None, add_trainer=True)
model.load_weights(model_path, mapping=lambda x: x.replace('module.', ''))

model.save_pretrained(save_dir)
copytree(config_path, save_dir, dirs_exist_ok=True)