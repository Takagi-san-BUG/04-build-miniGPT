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

你当前指定的环境是：

```powershell
E:\Anaconda\envs\Pytorch_All
```

推荐先激活该环境，再安装项目依赖：

```powershell
conda activate Pytorch_All
pip install -r requirements.txt
```

`requirements.txt` 按当前项目真实用到的最小依赖整理，不包含整个 Conda 环境里的其它实验包。

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

## Dataset Format

训练数据默认读取：

```text
../dataset/mobvoi_seq_monkey_general_open_corpus.jsonl
```

这里的 `../dataset` 是相对仓库根目录的上一级目录，不是在当前仓库内部。

也就是说，当前代码默认期望目录类似这样：

```text
99-上传github/
├─ 01-build-miniGPT/
└─ dataset/
   └─ mobvoi_seq_monkey_general_open_corpus.jsonl
```

JSONL 每行需要是一个 JSON 对象，并至少包含 `text` 字段，例如：

```json
{"text": "你好，今天我们来训练一个 miniGPT。"}
{"text": "语言模型的核心任务是预测下一个 token。"}
```

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

## Current Limitations

- 数据集文件默认放在仓库外层目录，使用前需要自己准备
- 仓库默认不上传训练日志和模型权重
- 推理依赖已有 checkpoint，纯 clone 后不能立即生成文本
- 当前更适合学习和实验，不是面向生产的训练框架

---

## Review Notes

这次整理仓库时，我额外确认了几件和运行相关的事实：

- `README` 里原先关于多阶段目录的描述，与当前仓库结构不再一致，现已改为和实际目录对齐
- `requirements.txt` 之前缺失，现已补充为最小依赖集合
- 代码层面目前仍有几处需要手动修正的 `.py` 问题；如果你不想让我改代码，可以按我前面给你的行号自行修改

---

## Acknowledgement

- [PyTorch](https://pytorch.org/)
- [tiktoken](https://github.com/openai/tiktoken)

同时也参考了 GPT 类模型常见实现思路，以及 `RoPE / GQA / RMSNorm / SwiGLU` 等现代结构设计。

---

## License

当前仓库未显式提供 License。  
如果后续计划公开长期维护，建议补一个 `MIT` 或 `Apache-2.0`。
