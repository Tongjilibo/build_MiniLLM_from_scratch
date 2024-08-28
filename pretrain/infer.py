#! -*- coding: utf-8 -*-
"""
预训练-推理：使用的是模型保存下来的pt文件
"""
import os
import torch
from bert4torch.models import build_transformer_model
from bert4torch.snippets import DottableDict
from transformers import AutoTokenizer

args = DottableDict()
args.max_length = 128
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
args.dir_path = '../config/pretrain'
args.config_path = os.path.join(args.dir_path, 'MiniLLM-0.2B-WithWudao/bert4torch_config.json')
args.model_path = '../ckpt/pretrain/MiniLLM-0.2B-WithWudao/final_3.1822/model.pt'

tokenizer = AutoTokenizer.from_pretrained('../tokenizer', trust_remote_code=True)
model = build_transformer_model(config_path=args.config_path, checkpoint_path=None, add_trainer=True)
model.to(args.device)

model.load_weights(args.model_path, mapping=lambda x: x.replace('module.', ''))  # 加载权重-单卡/ddp
# model.load_state_dict(torch.load(args.model_path)['module'], strict=True)  # deepspeed

generation_config = {
    'tokenizer': tokenizer,
    'tokenizer_config': {'skip_special_tokens': True, 'add_special_tokens': False},
    'bos_token_id': None,
    'eos_token_id': tokenizer.eos_token_id,
    'top_k': 40,
    'top_p': 0.8,
    'repetition_penalty': 1.1,
    'mode': 'random_sample',
    'max_length': args.max_length,
    'default_rtype': 'logits',
    'use_states': True,
    'include_input': True
}


def build_prompt(history):
    prompt = ''
    for query, response in history:
        prompt += f"\n\n用户：{query}"
        prompt += f"\n\n续写：{response}"
    return prompt

if __name__ == '__main__':
    history = []
    clear_command = 'cls' if os.name == 'nt' else 'clear'
    while True:
        query = input('\n输入:')
        if query.strip() == "stop":
            break
        if query.strip() == "clear":
            history = []
            os.system(clear_command)
            continue
        for response in model.stream_generate(query, **generation_config):
            os.system(clear_command)
            print(build_prompt(history + [(query, response)]), flush=True)
        os.system(clear_command)
        print(build_prompt(history + [(query, response)]), flush=True)
