#! -*- coding: utf-8 -*-
"""
预训练-推理
"""
import os
import torch
from transformers import AutoTokenizer, LlamaForCausalLM

max_length = 1024
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dir_path = '../config'

tokenizer = AutoTokenizer.from_pretrained(dir_path, trust_remote_code=True)
model = LlamaForCausalLM.from_pretrained(dir_path).to(device)

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
        inputs = tokenizer(query, return_tensors='pt').to(device)
        response = model.generate(**inputs)
        os.system(clear_command)
        print(build_prompt(history + [(query, response)]), flush=True)
        os.system(clear_command)
        print(build_prompt(history + [(query, response)]), flush=True)
