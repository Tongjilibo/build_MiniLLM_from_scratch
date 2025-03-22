#! -*- coding: utf-8 -*-
"""
预训练-推理：transformers格式
- 使用的是docs/convert.py对pt文件转换后的pytorch_model.bin文件
- 也可参考readme直接从huggingface下载并运行
"""
import os
import torch
from transformers import AutoTokenizer, LlamaForCausalLM
from transformers import TextIteratorStreamer
from threading import Thread

max_length = 1024
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# pretrained_model_name_or_path：预训练模型的本地路径或model_name(连接huggingface下载)
# Tongjilibo/MiniLLM-0.2B-NoWudao-Base
# Tongjilibo/MiniLLM-0.2B-Base
# Tongjilibo/MiniLLM-1.1B-Base
pretrained_model_name_or_path = '../ckpt/pretrain/MiniLLM-0.2B-Base/final_transformers'

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)
model = LlamaForCausalLM.from_pretrained(pretrained_model_name_or_path).to(device)


def build_cli_history(history):
    prompt = ''
    for query, response in history:
        prompt += f"\n\n用户：{query}"
        prompt += f"\n\n续写：{response}"
    return prompt


def chat():
    """
    非流式输出
    """
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
        inputs = tokenizer.encode(query, return_tensors='pt', add_special_tokens=False).to(device)
        response = model.generate(inputs)
        response = tokenizer.decode(response[0].cpu(), skip_special_tokens=True)

        os.system(clear_command)
        print(build_cli_history(history + [(query, response)]), flush=True)


def stream_chat():
    """
    流式输出
    """
    streamer = TextIteratorStreamer(tokenizer)

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
        inputs = tokenizer.encode(query, return_tensors='pt', add_special_tokens=False).to(device)
        generation_kwargs = dict({'input_ids': inputs}, streamer=streamer, max_new_tokens=512)
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        # 流式输出
        response = ''
        for new_text in streamer:
            os.system(clear_command)
            response += new_text
            print(build_cli_history(history + [(query, response)]), flush=True)

        os.system(clear_command)
        print(build_cli_history(history + [(query, response)]), flush=True)


if __name__ == '__main__':
    # chat()
    stream_chat()
