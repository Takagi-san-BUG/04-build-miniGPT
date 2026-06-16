from torch.utils.data import Dataset,DataLoader,random_split
from dataclasses import dataclass
import math
import tiktoken
import torch
import config
import json

class MyDataset(Dataset):

    def __init__(self, path,block_size, max_lines):
        
        self.enc = tiktoken.get_encoding("gpt2")
        self.block_size = block_size
        self.max_lines = max_lines
        
        self.eos_token = self.enc.encode(
            "<|endoftext|>",
            allowed_special={"<|endoftext|>"}
        )[0]

        self.encoded_data = []
        raw_data = []
        
        with open(path, 'r',encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= self.max_lines:
                    break
                try:
                    text = json.loads(line.strip())['text']
                    raw_data.append(text)
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    continue
                
        full_encoded = []
        for text in raw_data:
            encoded_text = self.enc.encode(text)
            full_encoded.extend(encoded_text + [self.eos_token])
        
        # 将长文本分割成训练样本
        for i in range(0, len(full_encoded), self.block_size):
            # 多取一个 Token 作为目标
            chunk = full_encoded[i:i+self.block_size+1]
            # 如果长度不够，用 eos_token 填充
            if len(chunk) < self.block_size + 1:
                chunk = chunk + [self.eos_token] * (self.block_size + 1 - len(chunk))
            self.encoded_data.append(chunk)
    
    def __len__(self):
        return len(self.encoded_data)
    
    def __getitem__(self, idx):
        chunk = self.encoded_data[idx]
        input_tensor = torch.tensor(chunk[:-1], dtype=torch.long)
        target_tensor = torch.tensor(chunk[1:], dtype=torch.long)
        return input_tensor, target_tensor

    def encode(self, text):
        """将文本编码为token IDs"""
        return self.enc.encode(text)

    def decode(self, ids):
        """将token IDs解码为文本"""
        return self.enc.decode(ids)
    
def get_dataloader(train=True,split_ratio=0.9, shuffle=True):
    path = config.RAW_DATA_DIR / "mobvoi_seq_monkey_general_open_corpus.jsonl"
    dataset = MyDataset(
        path=path,
        block_size=config.max_seq_len,
        max_lines=config.max_lines,
    )
    train_size = int(len(dataset) * split_ratio)
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator,
    )
    
    current_dataset = train_dataset if train else val_dataset
    
    return DataLoader(
        current_dataset,
        batch_size=config.batch_size,
        shuffle=shuffle if train else False,
    )

if __name__ == "__main__":
    train_dataloader = get_dataloader(train=True)
    val_dataloader = get_dataloader(train=False)
    print(f"训练批次数: {len(train_dataloader)}")
    print(f"验证批次数: {len(val_dataloader)}")
    print("dataloader 准备就绪")
    for input_tensor, target_tensor in train_dataloader:
        print(input_tensor.shape, target_tensor.shape)
        break