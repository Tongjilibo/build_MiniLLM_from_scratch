#! -*- coding: utf-8 -*-
"""
预训练-推理
"""
import os
import torch
from bert4torch.models import build_transformer_model
from bert4torch.snippets import DottableDict
from bert4torch.pipelines import ChatLLaMA2Cli
from transformers import AutoTokenizer
from data_process import HUMAN, ROBOT

args = DottableDict()
args.compile = False
args.eval_batch_size = 4
args.pad_token_id = 0
args.max_length = 1024
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
args.dir_path = '../config'
args.config_path = os.path.join(args.dir_path, 'bert4torch_config.json')
args.model_path = '../ckpt/L12_H1024_A8-Wudao-SFT/final_1.1675/model.pt'

tokenizer = AutoTokenizer.from_pretrained(args.dir_path, trust_remote_code=True)

generation_config = {
    'tokenizer_config':  {'skip_special_tokens': True, 'add_special_tokens': False},
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
        if (not hasattr(self, 'model')) or (self.model is None):
            model = build_transformer_model(config_path=args.config_path, checkpoint_path=None, add_trainer=True)
            model.to(args.device)
            model.load_weights(args.model_path, mapping=lambda x: x.replace('module.', ''))
            return model
        return self.model

if __name__ == '__main__':
    chat = Chat(args.dir_path, generation_config=generation_config)
    chat.run()