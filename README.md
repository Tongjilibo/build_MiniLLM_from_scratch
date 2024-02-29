# build_llm_from_scratch

[![licence](https://img.shields.io/github/license/Tongjilibo/build_llm_from_scratch.svg?maxAge=3600)](https://github.com/Tongjilibo/build_llm_from_scratch/blob/master/LICENSE)
[![GitHub release](https://img.shields.io/github/release/Tongjilibo/build_llm_from_scratch.svg?maxAge=3600)](https://github.com/Tongjilibo/build_llm_from_scratch/releases)
[![GitHub stars](https://img.shields.io/github/stars/Tongjilibo/build_llm_from_scratch?style=social)](https://github.com/Tongjilibo/build_llm_from_scratch)
[![GitHub Issues](https://img.shields.io/github/issues/Tongjilibo/build_llm_from_scratch.svg)](https://github.com/Tongjilibo/build_llm_from_scratch/issues)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/Tongjilibo/build_llm_from_scratch/issues)
[![Generic badge](https://img.shields.io/badge/wechat-join-green.svg?logo=wechat)](https://github.com/Tongjilibo/build_llm_from_scratch/blob/master/docs/pics/wechat_group.jpg)

[Bert4torch](https://github.com/Tongjilibo/bert4torch) |
[Torch4keras](https://github.com/Tongjilibo/torch4keras)

## 1、介绍
- **初衷**：本项目旨在构建一个小参数量的llm，完整走完`预训练` -> `指令微调`  -> `奖励模型`  -> `强化学习` 四个阶段，以可控的成本完成一个可以完成简单聊天任务的chat模型
- **特色**: 使用[bert4torch](https://github.com/Tongjilibo/bert4torch)训练框架，代码简介高效；优化了训练时候内存占用；提供了完整训练log供复现比对

## 2、快速开始
- 环境安装
```shell
pip install git+https://github.com/Tongjilibo/torch4keras.git
pip install git+https://github.com/Tongjilibo/bert4torch
```
- 训练脚本
```shell
# 预训练
cd pretrain
nohup torchrun --standalone --nproc_per_node=4 pretrain.py --name baby > nohup.log&

# 预训练推理（命令行聊天）
cd pretrain
python infer.py

# 指令微调训练

# 指令微调推理
```

## 3、更新历史

## 4、预训练
1. 预训练语料（源于[baby-llama2-chinese](https://github.com/DLLXW/baby-llama2-chinese)）
   | 中文预训练语料               | 描述                                      |
   |-------------------------|----------------------------------------|
   | [Wiki中文百科](https://huggingface.co/datasets/pleisto/wikipedia-cn-20230720-filtered)| 中文Wikipedia的数据 |
   | [BaiduBaiKe](https://pan.baidu.com/s/1jIpCHnWLTNYabftavo3DVw?pwd=bwvb) 提取码: bwvb| 中文BaiduBaiKe的数据|
   | [C4_zh：part1](https://pan.baidu.com/s/18O2Tj_PPB718K8gnaWrWUQ) 提取码：zv4r；[C4_zh：part2](https://pan.baidu.com/s/11PTgtUfFXvpNkOige9Iw4w) 提取码：sb83；[C4_zh：part3](https://pan.baidu.com/s/1248QfTS8QHPojYW-0fd5jQ) 提取码：l89d | C4是可用的最大语言数据集之一，收集了来自互联网上超过3.65亿个域的超过1560亿个token。C4_zh是其中的一部分 |
   | [WuDaoCorpora](https://data.baai.ac.cn/details/WuDaoCorporaText) | 中文悟道开源的200G数据|
   | [shibing624/medical](https://huggingface.co/datasets/shibing624/medical/tree/main)| 源自shibing624的一部分医学领域的预训练数据 |

项目开源了经过ChatGLM2-6B的分词器处理后的预训练语料，共计**634亿Tokens**的数据量，链接如下：[Corpus](https://pan.baidu.com/s/18o4gF-G68qfgOGWQXgAg3g) 提取码：6unr。将下载好的数据放到./data目录下即可。

1. 预训练权重
   |预训练权重 | 预训练语料         | 模型参数              | 下载地址                       |
   |----------------------------|-------------------------|--------------------------|---------------------|
   | L12_H1024_A8-NoWudao       | （140亿 Tokens）<br/>Wiki中文百科<br/>+BaiduBaiKe<br/>+shibing624/medical<br/>+C4_zh | max_seq_len=1024<br/>dim=1024<br/>n_layers=12<br/>n_heads=8  | [百度网盘](https://pan.baidu.com/s/1Zvk2Nuf7fsOPKJOk-Tjjcg?pwd=1oel) 提取码:1oel |

2. 预训练过程

   ![tensorboard](./docs/pics/tensorboard.png)

## 5、指令微调（Todo）

## 6、效果演示
- 预训练续写
```shell
用户：小明学习优异、身体健康、是一名

续写：小明学习优异、身体健康、是一名优秀的大连理工大学学生，从大学一年级起，他就对体育十分痴迷，成绩名列前茅，还参加了多项比赛，在全校体育活动中，他成绩名列前茅，被大连理工大学授予“优秀运动员”称号。
2006年，高中二年级，小明的同学杨帆来到大连理工大学，他决心要改变自己的命运。他每天放学后，都会跑到操场上练习篮球，练到下午三点多钟时，小明的篮球技术开始显现出来，他的篮球动作敏捷，力量大，而且，他经常利用空档时间练习跳投


用户：床前明月光，疑是地上霜。举头望明月，

续写：床前明月光，疑是地上霜。举头望明月，低头思故乡……”
我读着这些诗句时也倍感亲切和亲切了！
```

- 指令微调聊天
```shell
# Todo
```

## 7. 鸣谢

- 感谢[baby-llama2-chinese](https://github.com/DLLXW/baby-llama2-chinese)，本实现有不少地方参考该项目

## 8. 引用

```
@misc{build_llm_from_scratch,
  title={build_llm_from_scratch},
  author={Bo Li},
  year={2024},
  howpublished={\url{https://github.com/Tongjilibo/build_llm_from_scratch}},
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
         <a href="https://star-history.com/#Tongjilibo/bert4torch&Date"><img width="400" height="250" src="https://api.star-history.com/svg?repos=Tongjilibo/bert4torch&type=Date" alt="pic"></a><br>
         <a href="https://star-history.com/#Tongjilibo/bert4torch&Date">Star History Chart</a> 
      </td>    
      </tr>
  </tbody>
</table>
