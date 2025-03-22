#! -*- coding: utf-8 -*-
"""
sft-推理：transformers格式
- 使用的是docs/convert.py对pt文件转换后的pytorch_model.bin文件
- 也可参考readme直接从huggingface下载并运行
"""
import os
import torch
from transformers import AutoTokenizer, LlamaForCausalLM
from transformers import TextIteratorStreamer
from threading import Thread
from data_process import HUMAN, ROBOT

device = 'cuda' if torch.cuda.is_available() else 'cpu'
pretrained_model_name_or_path = '../ckpt/sft/MiniLLM-1.1B-Base-SFT/final_transformers'

# pretrained_model_name_or_path：预训练模型的本地路径或model_name(连接huggingface下载)
# Tongjilibo/MiniLLM-0.2B-SFT-Alpaca
# zRzRzRzRzRzRzR/zR-Llama-1b-ChatGLM2-6b-tokenizer
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)
model = LlamaForCausalLM.from_pretrained(pretrained_model_name_or_path).to(device)


def build_cli_history(history):
    prompt = ''
    for query, response in history:
        prompt += f"\n\nUser：{query.strip()}"
        prompt += f"\n\nRobot：{response.strip()}"
    return prompt


def build_prompt(query, history) -> str:
    texts = ''
    for user_input, response in history:
        texts += f'{HUMAN}{user_input}{ROBOT}{response}'

    texts += f'{HUMAN}{query}{ROBOT}'
    return texts


def chat():
    '''非流式'''
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

        inputs = tokenizer.encode(build_prompt(query, history), return_tensors='pt', add_special_tokens=False).to(
            device)
        response = model.generate(inputs)
        response = tokenizer.decode(response[0].cpu(), skip_special_tokens=True)

        os.system(clear_command)
        print(build_cli_history(history + [(query, response)]), flush=True)


def stream_chat():
    '''流式'''
    streamer = TextIteratorStreamer(tokenizer)

    history = []
    clear_command = 'cls' if os.name == 'nt' else 'clear'
    while True:
        query = input('\nUser:')
        if query.strip() == "stop":
            break
        if query.strip() == "clear":
            history = []
            os.system(clear_command)
            continue

        query_new = build_prompt(query, history)
        inputs = tokenizer.encode(query_new, return_tensors='pt', add_special_tokens=False).to(device)
        generation_kwargs = dict({'input_ids': inputs}, streamer=streamer, max_new_tokens=512)
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        # 流式输出
        response = ''
        for new_text in streamer:
            os.system(clear_command)
            response += new_text
            print(build_cli_history(history + [(query, response[len(query_new):])]), flush=True)

        os.system(clear_command)
        print(build_cli_history(history + [(query, response[len(query_new):])]), flush=True)


if __name__ == '__main__':
    # chat()
    stream_chat()
