#! -*- coding: utf-8 -*-
"""
权重转化为transformers可以使用的格式
"""
import os
from bert4torch.models import build_transformer_model
from bert4torch.snippets import copytree
import shutil

config_dir = '../config'
config_path = config_dir + '/MiniLLM-1.1B-WithWudao-SFT/bert4torch_config.json'
model_path = '../ckpt/MiniLLM-1.1B-WithWudao-SFT/final/model.pt'
save_dir = '../ckpt/MiniLLM-1.1B-WithWudao-SFT/final_transformers'

if not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)

model = build_transformer_model(config_path=config_path, checkpoint_path=None, add_trainer=True)
model.load_weights(model_path, mapping=lambda x: x.replace('module.', ''))

# 保存pytorch_model.bin
model.save_pretrained(save_dir)

# copy一些必须文件，config.json, tokenizer所需文件
if os.path.isfile(config_path):
    copy_dir = os.path.dirname(config_path)
elif os.path.isdir(config_path):
    copy_dir = config_path
else:
    raise TypeError('config_path only support path/dir')
copytree(copy_dir, save_dir, dirs_exist_ok=True)

for file_name in ['tokenization_chatglm.py', 'tokenizer_config.json', 'tokenizer.model']:
    shutil.copy(f'{config_dir}/{file_name}', save_dir)
