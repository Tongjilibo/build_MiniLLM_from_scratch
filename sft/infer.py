#! -*- coding: utf-8 -*-
"""
指令微调推理,建议单机单卡推理

python infer.py
"""
import os
import torch
from bert4torch.models import build_transformer_model
from bert4torch.snippets import DottableDict
from bert4torch.pipelines.chat import ChatCli, LLaMA2
from transformers import AutoTokenizer
from data_process import HUMAN, ROBOT

args = DottableDict()
args.max_length = 1024
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
args.dir_path = '../config'
args.config_path = os.path.join(args.dir_path, 'MiniLLM-0.2B-WithWudao-SFT/bert4torch_config.json')
args.model_path = '../ckpt/MiniLLM-0.2B-WithWudao-SFT/final_2.1458/model.pt'

tokenizer = AutoTokenizer.from_pretrained(args.dir_path, trust_remote_code=True)

generation_config = {
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
    'include_input': False
}


class Chat(ChatCli, LLaMA2):
    def build_prompt(self, query, history, functions) -> str:
        texts = ''
        for query_or_response in history:
            role, content = query_or_response['role'], query_or_response['content'].strip()
            if role == 'user':
                texts += f'{HUMAN}{content}'
            elif role == 'assistant':
                texts += f"{ROBOT}{content}"

        texts += f'{HUMAN}{query}{ROBOT}'
        history.append({"role": "user", "content": query})
        return texts

    def build_model(self):
        model = build_transformer_model(config_path=args.config_path, checkpoint_path=None, add_trainer=True)
        model.to(args.device)
        model.load_weights(args.model_path, mapping=lambda x: x.replace('module.', ''))
        return model


if __name__ == '__main__':
    chat = Chat(args.dir_path, generation_config=generation_config, history_maxlen=1)
    chat.run()
