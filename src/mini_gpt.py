import math
 
import torch
import torch.nn as nn
 
 
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, block_size, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model phải chia hết cho n_heads"
        self.h = n_heads
        self.dk = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)     # tạo Q, K, V một lượt
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        mat_na = torch.tril(torch.ones(block_size, block_size)).bool()
        self.register_buffer("mat_na", mat_na.view(1, 1, block_size,
                                                   block_size),
                             persistent=False)
 
    def forward(self, x):                              # x: [B, T, C]
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # [B, T, C] -> [B, h, T, dk]
        q = q.view(B, T, self.h, self.dk).transpose(1, 2)
        k = k.view(B, T, self.h, self.dk).transpose(1, 2)
        v = v.view(B, T, self.h, self.dk).transpose(1, 2)
 
        diem = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)  # [B,h,T,T]
        diem = diem.masked_fill(~self.mat_na[:, :, :T, :T], float("-inf"))
        w = self.drop(torch.softmax(diem, dim=-1))
        y = (w @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)

class MLP(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
 
    def forward(self, x):
        return self.net(x)
 
 
class Block(nn.Module):
    def __init__(self, d_model, n_heads, block_size, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.att = CausalSelfAttention(d_model, n_heads, block_size, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, dropout)
 
    def forward(self, x):
        x = x + self.att(self.ln1(x))      # residual quanh Attention
        x = x + self.mlp(self.ln2(x))      # residual quanh FFN
        return x
 
class MiniGPT(nn.Module):
    def __init__(self, vocab_size, block_size=128, d_model=256, n_heads=4,
                 n_layers=4, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(block_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, block_size, dropout)
             for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok.weight          # weight tying
        self.apply(self._init)
 
    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)
 
    def forward(self, idx):                         # idx: [B, T] các ID
        B, T = idx.shape
        assert T <= self.block_size, "chuỗi dài hơn block_size"
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok(idx) + self.pos(pos))
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln_f(x))              # logits [B, T, V]
