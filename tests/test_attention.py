import torch
 
from src.mini_gpt import CausalSelfAttention
 
 
def test_hinh_dang():
    att = CausalSelfAttention(32, 4, block_size=16)
    x = torch.randn(2, 10, 32)
    assert att(x).shape == (2, 10, 32)
 
 
def test_khong_nhin_trom_tuong_lai():
    torch.manual_seed(0)
    att = CausalSelfAttention(32, 4, block_size=16, dropout=0.0).eval()
    x = torch.randn(1, 10, 32)
    y1 = att(x)
    x2 = x.clone()
    x2[:, 7:, :] = torch.randn(1, 3, 32)       # đổi 3 token CUỐI
    y2 = att(x2)
    # 7 vị trí đầu không được thay đổi dù tương lai đã đổi
    assert torch.allclose(y1[:, :7], y2[:, :7], atol=1e-6)
    # còn các vị trí cuối thì phải đổi
    assert not torch.allclose(y1[:, 7:], y2[:, 7:])
