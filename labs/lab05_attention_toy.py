import math
 
import torch
 
torch.manual_seed(0)
T, d = 4, 8                         # 4 token, mỗi token là vector 8 chiều
x = torch.randn(T, d)
 
Wq, Wk, Wv = (torch.randn(d, d) for _ in range(3))
q, k, v = x @ Wq, x @ Wk, x @ Wv
 
diem = q @ k.T / math.sqrt(d)       # T x T: token i "thích" token j cỡ nào
mat_na = torch.tril(torch.ones(T, T)).bool()        # tam giác dưới
diem = diem.masked_fill(~mat_na, float("-inf"))     # cấm nhìn tương lai
w = torch.softmax(diem, dim=-1)
 
print("Ma trận Attention (mỗi hàng là 100%):")
print(w.round(decimals=2))
print("Tổng mỗi hàng:", w.sum(dim=-1))
out = w @ v
print("Đầu ra:", out.shape)
