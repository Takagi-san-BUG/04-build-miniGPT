# <div align="center">04-build-miniGPT</div>

<div align="center">
  <p><strong>从零手搓一个 miniGPT，并沿着真实实验路径持续迭代优化。</strong></p>
  <p>Base GPT -> Training polish -> RoPE + GQA -> Structured training pipeline</p>
</div>

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Tokenizer](https://img.shields.io/badge/Tokenizer-tiktoken-2F855A?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Learning%20Project-6B46C1?style=for-the-badge)

</div>

---

## Project Overview

这个仓库不是单一版本的 demo，而是一条完整的 miniGPT 学习与迭代路线。

我把项目拆成了 4 个阶段：

1. `01-基础版本`
   最原始的 GPT 风格实现，先把训练和生成跑通。
2. `02-基础版本细节优化`
   对训练流程和实验结果做了一轮更工程化的整理。
3. `03-增加RoPE和GQA及优化`
   引入更现代的结构设计，包括 `RoPE`、`GQA`、`RMSNorm`、`SwiGLU`。
4. `04-03基础上优化结构`
   在第 3 阶段基础上做目录重构，把训练、数据、预测、配置拆分成独立模块。

如果你想看“一个小型语言模型项目是怎么从能跑，逐渐长成更像样的实验代码”的过程，这个仓库就是按这个思路组织的。

---

## Highlights

- 从零实现一个可训练、可推理的 GPT 风格语言模型
- 使用 `tiktoken` 的 GPT-2 词表做分词
- 支持基于 JSONL 文本语料的自定义训练
- 引入 `RoPE` 旋转位置编码，替代绝对位置编码
- 引入 `GQA`，减少 KV 头数量，提升注意力结构效率
- 使用 `RMSNorm` 与 `SwiGLU`，贴近更现代的大模型结构
- 训练流程包含 `mixed precision`、`gradient clipping`、`CosineAnnealingLR`
- 支持 `torch.compile`
- 支持训练过程手动中断并保存 checkpoint
- 自动保存 loss 曲线、batch loss 曲线和模型权重

---

## Final Stage Architecture

当前最新版本位于 [`04-03基础上优化结构`](./04-03基础上优化结构)，核心模块拆分如下：

- `src/config.py`
  统一管理路径、模型超参数、训练超参数
- `src/dataset.py`
  负责读取 JSONL 数据、tokenize、切块、构建 dataloader
- `src/model.py`
  定义 GPT 主体结构，以及 `RoPE / GQA / RMSNorm / SwiGLU`
- `src/train.py`
  负责训练、验证、日志记录、checkpoint 保存
- `src/predict.py`
  加载最近 checkpoint 并进行文本生成

当前代码默认配置大致为：

- `max_seq_len = 512`
- `batch_size = 12`
- `n_layer = 4`
- `n_head = 8`
- `n_kv_head = 4`
- `n_embd = 768`
- `epoch = 10`

---

## Model Improvements

相较于最初版本，后续阶段逐步引入了这些优化：

| Module | Purpose |
|---|---|
| RoPE | 让位置编码更自然地融入注意力计算 |
| GQA | 减少 Key/Value 头数量，降低注意力计算和存储开销 |
| RMSNorm | 相比 LayerNorm 更轻量 |
| SwiGLU | 更现代的前馈网络激活结构 |
| Weight Tying | 共享 token embedding 与 lm head 权重 |
| Mixed Precision | 在 CUDA 上减少显存占用并加速训练 |
| Gradient Clipping | 让训练更稳定 |
| Early Stopping | 验证集长期不提升时提前停止 |

---

## Repository Layout

```text
04-搭建miniGPT
├─ 01-基础版本
├─ 02-基础版本细节优化
├─ 03-增加RoPE和GQA及优化
├─ 04-03基础上优化结构
│  ├─ src
│  │  ├─ config.py
│  │  ├─ dataset.py
│  │  ├─ model.py
│  │  ├─ predict.py
│  │  └─ train.py
│  ├─ logs
│  └─ models
└─ dataset
```

---

## Quick Start

### 1. Clone

```bash
git clone https://github.com/Takagi-san-BUG/04-build-miniGPT.git
cd 04-build-miniGPT
```

### 2. Install Dependencies

当前仓库里没有稳定保留可直接复用的 `requirements.txt`，所以建议先手动安装下面这些依赖：

```bash
pip install torch tiktoken tqdm keyboard matplotlib tensorboard
```

如果你打算运行 notebook，也建议额外安装：

```bash
pip install jupyter notebook
```

### 3. Prepare Dataset

最新结构化版本默认读取：

```text
dataset/mobvoi_seq_monkey_general_open_corpus.jsonl
```

数据格式需要是逐行 JSON，例如：

```json
{"text": "你好，今天我们来训练一个 miniGPT。"}
{"text": "语言模型的核心任务是预测下一个 token。"}
```

### 4. Train

进入最新版本目录后运行：

```bash
cd "04-03基础上优化结构/src"
python train.py
```

训练过程中会：

- 在 `../models/` 下保存 checkpoint
- 在 `../logs/` 下保存曲线图和 TensorBoard 日志
- 在验证集效果不再提升时触发 early stopping

### 5. Inference

```bash
cd "04-03基础上优化结构/src"
python predict.py
```

程序会自动加载最新的 `model_epoch*.pt`。

---

## Training Notes

这个项目目前更偏“教学实验项目”而不是通用训练框架，所以有几个现实约束需要提前说明：

- 数据集、模型权重、TensorBoard 日志默认不随仓库上传
- 最新训练脚本依赖本地 `dataset/` 目录中的语料文件
- `predict.py` 依赖已有 checkpoint，纯 clone 仓库后不能直接生成文本
- 项目里保留了不同阶段的 notebook 和脚本，方便对照演进过程

---

## Visual Preview

<div align="center">
  <img src="./02-基础版本细节优化/结果1/Snipaste_2026-04-26_20-15-05.png" alt="training-preview-1" width="80%">
</div>

<div align="center">
  <img src="./03-增加RoPE和GQA及优化/GQA与RoPE原理/01旋转矩阵.png" alt="rope-preview" width="70%">
</div>

这些图分别对应：

- 训练过程中的实验结果截图
- RoPE 原理理解过程中的示意图

如果后面继续迭代，我建议再补一张：

- 生成样例截图
- loss 曲线截图
- 最新模型结构图

这样项目首页会更完整。

---

## Why This Repo Exists

做这个仓库的目标，不只是“实现一个能跑的 GPT”，而是把下面几件事串起来：

- 理解语言模型从 token 到 next-token prediction 的最小闭环
- 理解训练代码从脚本堆叠到模块化结构的重构过程
- 理解现代大模型里常见结构为什么会逐步替换掉早期设计
- 用一个小项目把“模型结构、训练流程、实验记录、代码组织”连成一条线

---

## Future Work

- 增加更完整的推理示例与生成效果展示
- 补充统一的 `requirements.txt`
- 支持命令行参数或配置文件覆盖默认超参数
- 增加 checkpoint resume 功能
- 增加采样策略，例如 temperature / top-k / top-p
- 增加更规范的实验记录表
- 提供更清晰的模型结构图

---

## Acknowledgement

这个仓库使用了：

- [PyTorch](https://pytorch.org/)
- [tiktoken](https://github.com/openai/tiktoken)

同时也参考了 GPT 类模型的经典实现思路，以及 RoPE / GQA / RMSNorm / SwiGLU 等现代结构设计。

---

## License

当前仓库未显式提供 License。  
如果后续计划公开长期维护，建议补一个 `MIT` 或 `Apache-2.0`。
