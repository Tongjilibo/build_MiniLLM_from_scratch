# build_MiniLLM_from_scratch

[![licence](https://img.shields.io/github/license/Tongjilibo/build_MiniLLM_from_scratch.svg?maxAge=3600)](https://github.com/Tongjilibo/build_MiniLLM_from_scratch/blob/master/LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/Tongjilibo/build_MiniLLM_from_scratch?style=social)](https://github.com/Tongjilibo/build_MiniLLM_from_scratch)
[![GitHub Issues](https://img.shields.io/github/issues/Tongjilibo/build_MiniLLM_from_scratch.svg)](https://github.com/Tongjilibo/build_MiniLLM_from_scratch/issues)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/Tongjilibo/build_MiniLLM_from_scratch/issues)
[![Generic badge](https://img.shields.io/badge/wechat-join-green.svg?logo=wechat)](https://github.com/Tongjilibo/build_MiniLLM_from_scratch/blob/master/docs/pics/wechat_group.jpg)

[Bert4torch](https://github.com/Tongjilibo/bert4torch) |
[Torch4keras](https://github.com/Tongjilibo/torch4keras)

## 1. 介绍
- **初衷**：本项目旨在构建一个小参数量的llm，完整走完`预训练` -> `指令微调`  -> `奖励模型`  -> `强化学习` 四个阶段，以可控的成本完成一个可以完成简单聊天任务的chat模型
- **特色**: 
  - 使用[bert4torch](https://github.com/Tongjilibo/bert4torch)训练框架，代码简洁高效；
  - 训练的checkpoint可以直接使用`transformers`包进行推理
  - 优化了训练时候内存占用；
  - 提供了完整训练log供复现比对
- **声明**: 本实验训练出来的模型，目前只具备简单的聊天功能（受限于语料大小、模型规模、sft语料大小和质量），不具备回答复杂问题的能力。

## 2. 快速开始
- 环境安装
```shell
pip install git+https://github.com/Tongjilibo/torch4keras.git
pip install git+https://github.com/Tongjilibo/bert4torch.git@dev
```
- 脚本说明
```shell
# 为防止terminal关闭，可以使用nohup, tmux, screen方式来启动
# eg. nohup torchrun --standalone --nproc_per_node=4 pretrain.py --name baby > nohup.log&
# config/bert4torch_config.py: 配置文件默认为0.2B模型训练文件，如果你希望更换为1B，你需要自行将config文件中的`bert4torch_config_1.json`的内容黏贴到`bert4torch_config.json`
# 预训练
cd pretrain
torchrun --standalone --nproc_per_node=4 pretrain.py  # 部分反映ddp训到一般会崩，需设置`export NCCL_IB_DISABLE=1`

# 预训练推理（命令行聊天）
cd pretrain
python infer.py  # python infer_transformers.py

# 指令微调训练
cd sft
python sft.py

# 指令微调推理（命令行聊天）
cd sft
python infer.py  # python infer_transformers.py

# 把ckpt转化成transformers可以运行的格式
cd docs
python convert.py
```

## 3. 更新历史
- **20240325**: 增加1.1B模型（源于[zRzRzRzRzRzRzR](https://github.com/zRzRzRzRzRzRzR)）
- **20240316**: 初始提交，预训练模型`MiniLLM-MiniLLM-0.2B-NoWudao`和`MiniLLM-MiniLLM-0.2B-WithWudao`; SFT模型`MiniLLM-0.2B-WithWudao-SFT_Alpaca`

## 4. 预训练
### 4.1 预训练语料
| 中文预训练语料               | 描述                                      |
|-------------------------|----------------------------------------|
| [Wiki中文百科](https://huggingface.co/datasets/pleisto/wikipedia-cn-20230720-filtered)| 中文Wikipedia的数据 |
| [BaiduBaiKe](https://pan.baidu.com/s/1jIpCHnWLTNYabftavo3DVw?pwd=bwvb) 提取码: bwvb| 中文BaiduBaiKe的数据|
| [C4_zh：part1](https://pan.baidu.com/s/18O2Tj_PPB718K8gnaWrWUQ) 提取码：zv4r；[C4_zh：part2](https://pan.baidu.com/s/11PTgtUfFXvpNkOige9Iw4w) 提取码：sb83；[C4_zh：part3](https://pan.baidu.com/s/1248QfTS8QHPojYW-0fd5jQ) 提取码：l89d | C4是可用的最大语言数据集之一，收集了来自互联网上超过3.65亿个域的超过1560亿个token。C4_zh是其中的一部分 |
| [WuDaoCorpora](https://data.baai.ac.cn/details/WuDaoCorporaText) | 中文悟道开源的200G数据|
| [shibing624/medical](https://huggingface.co/datasets/shibing624/medical/tree/main)| 源自shibing624的一部分医学领域的预训练数据 |

项目开源了经过ChatGLM2-6B的分词器处理后的预训练语料，共计**634亿Tokens**的数据量，链接如下：[Corpus](https://pan.baidu.com/s/18o4gF-G68qfgOGWQXgAg3g) 提取码：6unr。

### 4.2 预训练权重和过程
- 预训练细节

|预训练权重 | 模型设置                    | 硬件占用和训练时长                       | 下载地址                       |
|----------------------------|--------------------------|---------------------|---------------------|
| MiniLLM-0.2B-NoWudao       | ✅140亿 Tokens: Wiki中文百科、BaiduBaiKe、hibing624/medical、C4_zh<br/>✅btz=32*4gpu; lr=3e-4; warmup_steps=5000; maxlen=1024 | 4×A800(80G), 单卡占用约60G，耗时20h|[百度网盘](https://pan.baidu.com/s/1ixjSR3IW9YXRhQ08RX-lMQ?pwd=lrj5), [HuggingFace](https://huggingface.co/Tongjilibo/MiniLLM-0.2B-NoWudao)|
| MiniLLM-0.2B-WithWudao       | ✅640亿 Tokens: Wiki中文百科、BaiduBaiKe、shibing624/medical、C4_zh、WuDaoCorpora<br/>✅btz=32*4gpu; lr=1.5e-4; warmup_steps=5000; maxlen=1024 |✅ 4×A800(80G), 单卡占用约60G，耗时3.79d<br/>✅ baby-llama2项目2×4090，耗时26d<br/>✅ 个人测试单卡btz=8下, gpu占用约17G，时长未知（可配合梯度累计进一步降低占用） | [百度网盘](https://pan.baidu.com/s/1ixjSR3IW9YXRhQ08RX-lMQ?pwd=lrj5), [HuggingFace](https://huggingface.co/Tongjilibo/MiniLLM-0.2B-WithWudao)|
| MiniLLM-1.1B-WithWudao| ✅640亿 Tokens: Wiki中文百科、BaiduBaiKe、shibing624/medical、C4_zh、WuDaoCorpora<br/>✅btz=32*8gpu; lr=1.5e-4; warmup_steps=5000; maxlen=896 |8×A800(80G), 耗时1天| [HuggingFace](https://huggingface.co/Tongjilibo/MiniLLM-1.1B-WithWudao)|

- loss记录

![tensorboard](./docs/pics/tensorboard.png)

### 4.3 预训练模型调用
```python
# 以下两句视网络情况添加
import os
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"

from transformers import AutoTokenizer, LlamaForCausalLM
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = 'Tongjilibo/MiniLLM-0.2B-WithWudao'  # 'Tongjilibo/MiniLLM-0.2B-NoWudao'

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = LlamaForCausalLM.from_pretrained(model_name).to(device)

query = '王鹏是一名'
inputs = tokenizer.encode(query, return_tensors='pt', add_special_tokens=False).to(device)
output_ids = model.generate(inputs)
response = tokenizer.decode(output_ids[0].cpu(), skip_special_tokens=True)
print(response)
```

### 4.4 预训练续写效果
- MiniLLM-0.2B-NoWudao
```shell
用户：小明学习优异、身体健康、是一名

续写：小明学习优异、身体健康、是一名优秀的大连理工大学学生，从大学一年级起，他就对体育十分痴迷，成绩名列前茅，还参加了多项比赛，在全校体育活动中，他成绩名列前茅，被大连理工大学授予“优秀运动员”称号。
2006年，高中二年级，小明的同学杨帆来到大连理工大学，他决心要改变自己的命运。他每天放学后，都会跑到操场上练习篮球，练到下午三点多钟时，小明的篮球技术开始显现出来，他的篮球动作敏捷，力量大，而且，他经常利用空档时间练习跳投


用户：床前明月光，疑是地上霜。举头望明月，

续写：床前明月光，疑是地上霜。举头望明月，低头思故乡……”
我读着这些诗句时也倍感亲切和亲切了！
```
- MiniLLM-0.2B-WithWudao
```shell
用户：小明学习优异、身体健康、是一名

续写：小明学习优异、身体健康、是一名优秀的教师。他为人诚实、有责任心。在课堂上，他能认真听讲，积极思考，对每节课都积极举手发言。在课后，他主动要求帮助老师，认真做好作业，按时完成老师布置的作业，受到同学们的好评。


用户：床前明月光，疑是地上霜。举头望明月，

续写：床前明月光，疑是地上霜。举头望明月，低头思故乡。”这诗不仅描述了中秋夜月夜的温馨与宁静，还写出了作者对故土深情的眷恋和思乡之情。“月上柳梢头”一语，是写月下所见。“欲将心事付瑶琴”，指欲诉别情； “举头望明月”，写中秋之夜，月上高挂、皓月当空、群星闪耀的景象；“低头思故乡”，写思念故土的深情厚意。
这首诗在写作手法上，主要运用象征
```

## 5、指令微调
### 5.1 指令微调语料（筛选的可用数据集）
| 数据集名称     | 介绍               |
| ---------------- | -------------------- |
|[Tongjilibo/self_cognition](https://huggingface.co/datasets/Tongjilibo/self_cognition)|整理的自我认知数据集，目前有100多条|
|[shibing624/alpaca-zh](https://huggingface.co/datasets/shibing624/alpaca-zh)|参考Alpaca方法基于GPT4得到的self-instruct数据，约5万条|
|[BelleGroup/Belle-0.5M-cn](https://huggingface.co/datasets/BelleGroup/train_0.5M_CN)|包含约50万条由BELLE项目生成的中文指令数据||
|[BelleGroup/Belle-1M-cn](https://huggingface.co/datasets/BelleGroup/train_1M_CN)| 包含约100万条由BELLE项目生成的中文指令数据|
|[BelleGroup/Belle-school_math_0.25M](https://huggingface.co/datasets/BelleGroup/school_math_0.25M)| Belle开放的0.25M数学指令数据集|
|[BelleGroup/Belle-multiturn_chat_0.8M](https://huggingface.co/datasets/BelleGroup/multiturn_chat_0.8M)| Belle开放的0.8M多轮任务对话数据集|
|[YeungNLP/firefly-train-1.1M](https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M)|流萤23种常见的中文NLP任务的数据，并且构造了许多与中华文化相关的数据，如对联、作诗、文言文翻译、散文、金庸小说等。对于每个任务，由人工书写若干种指令模板，保证数据的高质量与丰富度，数据量为115万|
|[fnlp/moss-002-sft-data](https://huggingface.co/datasets/fnlp/moss-002-sft-data)|MOSS-002所使用的多轮对话数据，覆盖有用性、忠实性、无害性三个层面，包含由text-davinci-003生成的约57万条英文对话和59万条中文对话|
|[fnlp/moss-003-sft-data](https://huggingface.co/datasets/fnlp/moss-003-sft-data)|moss-moon-003-sft所使用的多轮对话数据，基于MOSS-002内测阶段采集的约10万用户输入数据和gpt-3.5-turbo构造而成，相比moss-002-sft-data，moss-003-sft-data更加符合真实用户意图分布，包含更细粒度的有用性类别标记、更广泛的无害性数据和更长对话轮数，约含110万条对话数据|
|[shareAI/CodeChat](https://huggingface.co/datasets/shareAI/CodeChat)      | 主要包含逻辑推理、代码问答、代码生成相关语料样本。 |
|[shareAI/ShareGPT-Chinese-English-90k](https://huggingface.co/datasets/shareAI/ShareGPT-Chinese-English-90k)     | 中英文平行双语优质人机问答数据集，覆盖真实复杂场景下的用户提问。|
|[deepctrl/deepctrl-sft-data](https://www.modelscope.cn/datasets/deepctrl/deepctrl-sft-data/summary)|匠数大模型SFT数据集是一个由匠数科技精心搜集整理的高质量数据集,包含10M条数据的中文数据集和包含2M条数据的英文数据集|

### 5.2 指令微调权重和过程
- 指令微调细节

|         权重                  |   模型设置                    | 硬件占用和训练时长                       | 下载地址 |
|-------------------------------|--------------------------|---------------------|---------------------|
| MiniLLM-0.2B-WithWudao-SFT_Alpaca  |✅[shibing624/alpaca-zh](https://huggingface.co/datasets/shibing624/alpaca-zh)<br/>✅btz=8; lr=2e-5; 5epoch |  单卡4090，显存17G, 耗时45min| [百度网盘](https://pan.baidu.com/s/1ixjSR3IW9YXRhQ08RX-lMQ?pwd=lrj5), [HuggingFace](https://huggingface.co/Tongjilibo/MiniLLM-0.2B-WithWudao-SFT_Alpaca) |
| zR-Llama-1b-ChatGLM2-6b-tokenizer  |✅全部语料<br/>✅btz=8; lr=2e-5; 5epoch|单卡A800, 耗时 3d 12h|[HuggingFace](https://huggingface.co/zRzRzRzRzRzRzR/zR-Llama-1b-ChatGLM2-6b-tokenizer)

- loss

![tensorboard](./docs/pics/tensorboard_sft.png)

### 5.3 指令微调模型调用
```python
# 以下两句视网络情况添加
import os
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"

from transformers import AutoTokenizer, LlamaForCausalLM
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = 'Tongjilibo/MiniLLM-0.2B-WithWudao-SFT_Alpaca'

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = LlamaForCausalLM.from_pretrained(model_name).to(device)

query = '你好'
query = f'<human>{query}<robot>'
inputs = tokenizer.encode(query, return_tensors='pt', add_special_tokens=False).to(device)
output_ids = model.generate(inputs)
response = tokenizer.decode(output_ids[0].cpu(), skip_special_tokens=True)[len(query):]
print(response)
```

### 5.4 指令微调Chat效果
- MiniLLM-0.2B-WithWudao-SFT_Alpaca
```shell
User：你好

Assistant：你好！很高兴见到你。

User：你能做什么？

Assistant：你好！我是一个人工智能助手，有什么我可以为你做的吗？

User：番茄炒蛋怎么制作？

Assistant：如果你想要制作一个番茄炒蛋，那么下面这些步骤可能会帮助你：

1. 准备食材：准备好鸡蛋、盐、香菜、胡萝卜丝、黄瓜丝等食材。
2. 清洗鸡蛋：在搅拌碗中打散鸡蛋，使其表面可以清洁。
3. 准备材料：准备一个锅，倒入适量的水，将鸡蛋液倒入锅中。
4. 煮鸡蛋：用汤锅或锅煎至鸡蛋表面金黄色，熟透即可。
5. 炒蔬菜：在锅中加入适量的油，将胡萝卜丝和黄瓜丝个人喜欢的，翻炒几下，直到胡萝卜熟透。
6. 加入鸡蛋：从锅中取出鸡蛋，倒入锅中。
7. 调味：将炒好的鸡蛋倒入锅中，与蔬菜一起翻炒几下即可。

注意：想要炒出更丰富的蔬菜味道，可以适当加入适量的盐和香菜，增加风味。
```

## 6. Todo
- ⬜deepspeed方式训练
- ⬜加入更多的sft数据
- ⬜更大的模型规模来

## 7. 鸣谢

- 感谢[baby-llama2-chinese](https://github.com/DLLXW/baby-llama2-chinese)，本实现有不少地方参考该项目

## 8. 引用

```
@misc{build_MiniLLM_from_scratch,
  title={build_MiniLLM_from_scratch},
  author={Bo Li},
  year={2024},
  howpublished={\url{https://github.com/Tongjilibo/build_MiniLLM_from_scratch}},
}
```

## 9. 其他

- Wechat & Star History Chart

<table border="0">
  <tbody>
    <tr align="center" >
      <td>
         <a href="https://github.com/Tongjilibo"><img width="200" height="250" src="./docs/pics/wechat.jpg" alt="pic"></a><br>
         <a href="https://github.com/Tongjilibo">微信号</a> 
      </td>
      <td>
         <a href="https://github.com/Tongjilibo"><img width="190" height="250" src="./docs/pics/wechat_group.jpg" alt="pic"></a><br>
         <a href="https://github.com/Tongjilibo">微信群</a> 
      </td>
      <td>
         <a href="https://star-history.com/#Tongjilibo/build_MiniLLM_from_scratch&Date"><img width="400" height="250" src="https://api.star-history.com/svg?repos=Tongjilibo/build_MiniLLM_from_scratch&type=Date" alt="pic"></a><br>
         <a href="https://star-history.com/#Tongjilibo/build_MiniLLM_from_scratch&Date">Star History Chart</a> 
      </td>    
      </tr>
  </tbody>
</table>
