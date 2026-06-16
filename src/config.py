from pathlib import Path
from dataclasses import dataclass

ROOT_DIR = Path(__file__).parent.parent

RAW_DATA_DIR = ROOT_DIR.parent / "dataset" 
LOGS_DIR = ROOT_DIR / "logs"
MODELS_DIR = ROOT_DIR / "models"


# 模型参数
max_seq_len: int = 512  #最大文本长度512
batch_size: int = 12
n_layer: int = 4 #6
n_head: int = 8 #12
n_kv_head: int = 4 #4
n_embd: int =768  #768   
hidden_dim = n_embd
head_size: int = n_embd // n_head
dropout: float = 0.05
# 使用 GPT-2 的词表，大约有 50257 个token
vocab_size: int = 50257

# 数据与训练参数
max_lines = 100 # 读取多少行训练技术及
block_size = 512 # 训练数据每条序列的最多token
epoch = 10 # 训练轮数
key = "F12" # 中断训练键盘监听
patience = 3 # 早停参数，多少轮每下降就暂停训练


