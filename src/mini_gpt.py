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
