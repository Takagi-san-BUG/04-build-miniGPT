import torch.nn as nn 
import torch
import math
import torch.nn.functional as F

#RoPE部分
def get_rotary_frequencies(hidden_dim: int, seq_len: int, theta: float = 10000.0):
    i = torch.arange(0, hidden_dim // 2, dtype=torch.float32)
    freqs = theta ** (-2 * i / hidden_dim)
    positions = torch.arange(seq_len, dtype=torch.float32)
    angles = torch.outer(positions, freqs)
    return angles
def get_rotary_embedding(dim: int, seq_len: int, theta: float = 10000.0):
    """
    计算 RoPE 的 sin 和 cos 值
    """
    angles = get_rotary_frequencies(dim, seq_len, theta)
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    # 拼接相同值，因为两两分组，一组用一个角度即可
    cos = torch.cat([cos, cos], dim=-1) #(seq_len, dim)
    sin = torch.cat([sin, sin], dim=-1)
    return cos, sin
def rotate_half(x):
    """
    将向量的前半部分和后半部分交换，并对后半部分取负
    [x1, x2, x3, x4] -> [-x3, -x4, x1, x2]
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)
def apply_rotary_pos_emb(q, k, cos, sin):
    """
    旋转公式：
        q' = q * cos + rotate_half(q) * sin
        k' = k * cos + rotate_half(k) * sin
    """
    # 调整 cos/sin 形状以便广播: (seq_len, head_dim) -> (batch, seq_len, heads, head_dim)
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)
    # 应用旋转
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
class RotaryPositionEmbedding(nn.Module):
    """
    完整的 RoPE 实现
    """
    def __init__(self, head_size: int, max_seq_len: int = 512, theta: float = 10000.0):
        super().__init__()
        self.head_size = head_size
        self.max_seq_len = max_seq_len
        self.theta = theta
        # 预计算并缓存 sin/cos 值
        self.cos_cached: torch.Tensor
        self.sin_cached: torch.Tensor
        cos, sin = get_rotary_embedding(head_size, max_seq_len, theta)
        self.register_buffer('cos_cached', cos)
        self.register_buffer('sin_cached', sin)

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        """
            q: Query，shape (batch, seq_len, num_heads, head_dim)
            k: Key，shape (batch, seq_len, num_heads, head_dim)
        """
        seq_len = q.shape[1]
        # 在多头注意力中, q, k 的 shape 已经是 (batch, num_heads, seq_len, head_dim)
        # 我们需要调整 cos/sin 以匹配
        # q 的 shape: (batch, n_head, seq, head_dim) -> 调整 cos/sin
        seq_len_q = q.shape[2]
        seq_len_k = k.shape[2]

        cos_q = self.cos_cached[:seq_len_q].unsqueeze(0).unsqueeze(0) # (1, 1, seq_len, head_dim)
        sin_q = self.sin_cached[:seq_len_q].unsqueeze(0).unsqueeze(0)
        
        cos_k = self.cos_cached[:seq_len_k].unsqueeze(0).unsqueeze(0)
        sin_k = self.sin_cached[:seq_len_k].unsqueeze(0).unsqueeze(0)
        
        q_rot = (q * cos_q) + (rotate_half(q) * sin_q)
        k_rot = (k * cos_k) + (rotate_half(k) * sin_k)
        # # 获取当前序列长度的 cos/sin
        # cos = self.cos_cached[:seq_len]
        # sin = self.sin_cached[:seq_len]
        # # 应用旋转
        # q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
        return q_rot, k_rot
    
#RMSNorm
class RMSNorm(nn.Module):
    """
    实现 RMSNorm (Root Mean Square Layer Normalization)。
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # gamma 参数，初始化为 1
        self.weight = nn.Parameter(torch.ones(dim))
    def _norm(self, x):
        # 计算均方根并进行归一化
        # rsqrt(x) is 1/sqrt(x) for numerical stability
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    def forward(self, x):
        # x shape: (batch, seq_len, n_embd)
        # weight shape: (n_embd)
        # PyTorch 会自动广播 weight
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class GroupQueryAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 从 config 对象中提取参数
        self.hidden_dim = config.n_embd
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.n_embd // config.n_head
        self.max_seq_len = config.max_seq_len
        
        # 确保头数可以被正确整除
        assert self.hidden_dim % self.n_head == 0
        assert self.n_head % self.n_kv_head == 0
        # 初始化 Q, K, V 的线性投影层
        self.q_proj = nn.Linear(self.hidden_dim, self.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_dim, self.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_dim, self.n_kv_head * self.head_dim, bias=False)
        # 初始化输出投影层
        self.o_proj = nn.Linear(self.n_head * self.head_dim, self.hidden_dim, bias=False)
        # 增加 Dropout 层
        self.dropout = nn.Dropout(config.dropout)
        
        # --- 从旧的 MultiHeadAttention 迁移过来的功能 ---
        # 1. RoPE (旋转位置编码)
        self.rope = RotaryPositionEmbedding(
            head_size=self.head_dim,
            max_seq_len=config.max_seq_len
        )
        
        # 2. 因果掩码 (Causal Mask)
        self.causal_mask: torch.Tensor
        self.register_buffer(
            'causal_mask',
            torch.tril(torch.ones(self.max_seq_len, self.max_seq_len))
        )
        # --- 迁移结束 ---
    def forward(self, X, attention_mask=None):
        # X shape (batch, seq, hidden_dim)
        batch_size, seq_len, _ = X.size()
        # 1. Q, K, V 投影
        q = self.q_proj(X)
        k = self.k_proj(X)
        v = self.v_proj(X)
        # 2. Reshape Q, K, V 以便进行多头/分组注意力计算
        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_kv_head, self.head_dim).transpose(1, 2)
        # 3. 应用 RoPE
        # 注意：RoPE 是在 K, V 被重复之前应用的
        q, k = self.rope(q, k)
        # 4. 重复 K 和 V 以匹配 Q 的头数
        # repeat_interleave 用于沿指定维度重复张量
        repeat_factor = self.n_head // self.n_kv_head
        k = k.repeat_interleave(repeat_factor, dim=1)
        v = v.repeat_interleave(repeat_factor, dim=1)
        # 5. 计算注意力分数
        attention_score = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # 6. 应用因果掩码
        mask = self.causal_mask[:seq_len, :seq_len] == 0
        attention_score = attention_score.masked_fill(mask, float('-inf'))
        
        # 7. 计算注意力权重 (Softmax) 和应用 Dropout
        attention_weight = F.softmax(attention_score, dim=-1)
        attention_weight = self.dropout(attention_weight)
        # 8. 加权求和得到输出
        output = attention_weight @ v
        # 9. Reshape 并进行输出投影
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        final_output = self.o_proj(output)
        return final_output

#SwiGLU FFN
class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        n_embd = config.n_embd
        hidden_dim = 4 * n_embd * 2 // 3   # LLaMA 官方最优配置
        
        self.up = nn.Linear(n_embd, hidden_dim, bias=False)
        self.gate = nn.Linear(n_embd, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout( self.down( F.silu(self.up(x)) * self.gate(x) ) )

#使用RMSNorm
class Block(nn.Module): 
    def __init__(self, config):
        super().__init__()
        head_size = config.n_embd // config.n_head
        self.att = GroupQueryAttention(config)
        self.ffn = FeedForward(config)
        self.ln1 = RMSNorm(config.n_embd)
        self.ln2 = RMSNorm(config.n_embd)


    def forward(self, x):
        x = x + self.att(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x
 

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.max_seq_len = config.max_seq_len
        #self.position_embedding_table = nn.Embedding(config.max_seq_len, config.n_embd)  有了RoPE就不要了
        self.blocks = nn.Sequential(
            *[Block(config) for _ in range(config.n_layer)]
        )
        self.ln_final = RMSNorm(config.n_embd)

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # 参数共享
        self.lm_head.weight = self.token_embedding_table.weight 

        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # 正态分布初始化
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, idx, targets=None):
        
        batch, seq_len = idx.size()
        token_emb = self.token_embedding_table(idx)

        # # seq_len 输入的最大长度
        # pos_emb = self.position_embedding_table(
        #     # 要确保 位置编码和输入的 idx 在同一个设备上
        #     torch.arange(seq_len, device=idx.device)
        # )

        x = token_emb #不要脸+ pos_emb   # shape is (batch, seq_len, n_embd)
        x = self.blocks(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)   # shape is (batch, seq_len, vocab_size)
        
        if targets is None:
            loss = None
        else:
            batch, seq_len, vocab_size = logits.size()
            logits = logits.view(batch * seq_len, vocab_size)
            targets = targets.view(batch * seq_len)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # 如果序列太长，只取最后 max_seq_len个token
            idx_cond = idx if idx.size(1) <= self.max_seq_len else idx[:, -self.max_seq_len:]
            # 获取预测
            logits, _ = self(idx_cond)
            # 只关注最后一个时间步的预测
            logits = logits[:, -1, :]  # becomes (B, vocab_size)
            # 应用softmax获取概率
            probs = F.softmax(logits, dim=-1)
            # 采样下一个token
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # 附加到序列上
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx