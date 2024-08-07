#! -*- coding: utf-8 -*-
"""
指令微调推理,建议单机单卡推理

python infer.py
"""
import os
import torch
from bert4torch.models import build_transformer_model
from bert4torch.snippets import DottableDict
from bert4torch.pipelines import ChatLLaMA2Cli
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
    'start_id': None,
    'end_id': tokenizer.eos_token_id,
    'topk': 40,
    'topp': 0.8,
    'repetition_penalty': 1.1,
    'mode': 'random_sample',
    'max_length': args.max_length,
    'default_rtype': 'logits',
    'use_states': True,
    'include_input': False
}


class Chat(ChatLLaMA2Cli):
    def build_prompt(self, query, history) -> str:
        texts = ''
        for user_input, response in history:
            texts += f'{HUMAN}{user_input}{ROBOT}{response}'

        texts += f'{HUMAN}{query}{ROBOT}'
        return texts

    def build_model(self):
        model = build_transformer_model(config_path=args.config_path, checkpoint_path=None, add_trainer=True)
        model.to(args.device)
        model.load_weights(args.model_path, mapping=lambda x: x.replace('module.', ''))
        return model


if __name__ == '__main__':
    # history_maxlen设置保留的历史对话的轮次，需要sft的chat数据集中有多轮对话数据
    chat = Chat(args.dir_path, generation_config=generation_config, history_maxlen=1)
    chat.run()
