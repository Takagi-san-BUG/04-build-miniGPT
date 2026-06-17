# <div align="center">01-build-miniGPT</div>

<div align="center">
  <p><strong>从零实现一个可训练、可推理的 miniGPT，并逐步加入更现代的结构设计。</strong></p>
  <p>Base GPT -> RoPE + GQA -> RMSNorm + SwiGLU -> Structured training pipeline</p>
</div>

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Tokenizer](https://img.shields.io/badge/Tokenizer-tiktoken-2F855A?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Learning%20Project-6B46C1?style=for-the-badge)

</div>

---

## Overview

这个仓库聚焦于一个小型 GPT 风格语言模型的完整最小闭环：

- 用 `tiktoken` 做 GPT-2 词表分词
- 从 JSONL 语料构建训练样本
- 使用 PyTorch 实现模型、训练和推理
- 在基础实现上加入 `RoPE`、`GQA`、`RMSNorm`、`SwiGLU`
- 保存 checkpoint、loss 曲线和 TensorBoard 日志

项目更偏学习型实验代码，而不是通用训练框架。它适合用来理解小型语言模型从数据到训练、再到生成的整条路径。

---

## Features

- 从零实现 GPT 风格自回归语言模型
- 支持 `RoPE` 旋转位置编码
- 支持 `GQA` 分组查询注意力
- 使用 `RMSNorm` 和 `SwiGLU`
- 使用 `AdamW + CosineAnnealingLR`
- 支持 `mixed precision`
- 支持 `torch.compile`
- 支持 early stopping
- 保存模型权重、训练曲线和 TensorBoard 日志

---

## Repository Layout

```text
01-build-miniGPT/
├─ README.md
├─ requirements.txt
├─ logs/
├─ models/
└─ src/
   ├─ config.py
   ├─ dataset.py
   ├─ model.py
   ├─ predict.py
   └─ train.py
```

核心文件说明：

- `src/config.py`：路径、模型超参数和训练超参数
- `src/dataset.py`：JSONL 数据读取、tokenize、切块、DataLoader 构建
- `src/model.py`：GPT 主体结构与注意力/前馈模块
- `src/train.py`：训练、验证、日志记录、checkpoint 保存
- `src/predict.py`：加载最新 checkpoint 并进行文本生成

---

## Environment

推荐先激活该环境，再安装项目依赖：

```powershell
conda activate your_environment
pip install -r requirements.txt
```

`requirements.txt` 按当前项目真实用到的最小依赖整理，其他常用包版本不限。

---

## Dependencies

当前 `requirements.txt` 包含：

```text
torch==2.10.0
tiktoken==0.12.0
tqdm==4.67.3
keyboard==0.13.5
matplotlib==3.10.8
tensorboard==2.20.0
```

说明：

- 如果你本机需要特定 CUDA 版本，可以按自己的显卡环境安装匹配的 PyTorch 构建
- 本项目代码没有直接依赖 `torchvision`

---

## Dataset

本项目使用 **出门问问 序列猴子（Seq-Monkey）开源数据集 1.0** 中的 **中文通用文本语料** 作为预训练语料，文件名为 `mobvoi_seq_monkey_general_open_corpus.jsonl`。

### 数据集介绍

「序列猴子」是出门问问（Mobvoi）发布的超大规模语言模型，序列猴子开源数据集即用于训练序列猴子模型的部分公开数据。其中的「中文通用文本语料」是大语言模型预训练语料，由网页、百科、博客、问答、开源代码、书籍、报刊、专利、教材、考题等多种公开可获取的数据汇总清洗而成。

构建流程统一处理了 HTML / TEXT / PDF / EPUB 等多种格式的原始数据，并进行：

- 语言判别、正文抽取、格式标准化
- 多尺度去重
- 基于规则与模型的多重数据过滤与清洗
- 多维度数据质量检验
- 与中文主流价值观的内容对齐

最终面向公众开放的版本，是从其中文通用文本数据集中抽取的 **13,000,000 条** 数据，统一整理为 `JSONL` 格式。

特点：

- **处理细致**：经过完整的清洗与质检流水线
- **价值对齐**：内容更符合中文主流价值观
- **简洁易用**：统一的 `JSONL` 格式，可直接喂给 LM 训练流程

数据采用 [Apache 2.0](https://github.com/mobvoi/seq-monkey-data/blob/main/LICENSE) 协议开源，原始仓库见 [`mobvoi/seq-monkey-data`](https://github.com/mobvoi/seq-monkey-data)。

### 数据格式

每行是一个独立的 JSON 对象，至少包含 `text` 字段：

```json
{"text": "你好，今天我们来训练一个 miniGPT。"}
{"text": "语言模型的核心任务是预测下一个 token。"}
```

`<text>` 字段中存放的就是清洗后的中文文档正文。

### 默认存放位置

训练数据默认读取：

```text
../dataset/mobvoi_seq_monkey_general_open_corpus.jsonl
```

这里的 `../dataset` 是相对仓库根目录的上一级目录，不是在当前仓库内部。也就是说，当前代码默认期望目录结构类似：

```text
99-上传github/
├─ 01-build-miniGPT/
└─ dataset/
   └─ mobvoi_seq_monkey_general_open_corpus.jsonl
```

### 获取数据

本仓库**不**附带数据集本体，需要自行从原始仓库下载并解压：

- 原始仓库：https://github.com/mobvoi/seq-monkey-data
- 中文通用文本语料说明：[`docs/pretrain_open_corpus.md`](https://github.com/mobvoi/seq-monkey-data/blob/main/docs/pretrain_open_corpus.md)
- 下载链接：`http://share.mobvoi.com:5000/sharing/O91blwPkY`
- 文件 MD5：`ffacae345d22ab4f1464221d8ecf92c6`
- 压缩格式：`*.tar.bz2`，解压后即可得到 `mobvoi_seq_monkey_general_open_corpus.jsonl`

下载完成后建议先做完整性校验

> 数据规模较大，请确保磁盘空间充足；首次跑通训练可以只用其中一小部分样本做 sanity check。

---

## Default Config

当前默认超参数来自 `src/config.py`：

- `max_seq_len = 512`
- `batch_size = 12`
- `n_layer = 4`
- `n_head = 8`
- `n_kv_head = 4`
- `n_embd = 768`
- `dropout = 0.05`
- `epoch = 10`
- `patience = 3`

---

## Training

在仓库根目录直接运行：

```powershell
python src/train.py
```

训练过程中会：

- 从 `../dataset/mobvoi_seq_monkey_general_open_corpus.jsonl` 读取语料
- 在 `models/` 下保存 checkpoint
- 在 `logs/` 下保存 loss 图和 TensorBoard 日志
- 在验证集长期不提升时提前停止

如果你想看 TensorBoard：

```powershell
tensorboard --logdir logs
```

---

## Inference

推理脚本会自动加载 `models/` 下最新的 `model_epoch*.pt`：

```powershell
python src/predict.py
```

注意：

- 首次推理前需要先训练出至少一个 checkpoint
- `predict.py` 会兼容 `torch.compile` 训练后保存的 `_orig_mod.` 权重前缀

---

## Notes

- 数据集文件默认放在仓库外层目录，使用前需要自己准备
- 本仓库中没有暂无预训练好的模型，像尝试效果可以自行租赁GPU算力进行测试

---

## Acknowledgement

- [PyTorch](https://pytorch.org/)
- [tiktoken](https://github.com/openai/tiktoken)
- [出门问问 序列猴子开源数据集](https://github.com/mobvoi/seq-monkey-data) — 提供了本项目所用的中文预训练语料

同时也参考了 GPT 类模型常见实现思路，以及 `RoPE / GQA / RMSNorm / SwiGLU` 等现代结构设计。

---

