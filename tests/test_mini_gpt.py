import math
 
import torch
import torch.nn.functional as F
 
from src.mini_gpt import MiniGPT
 
 
def test_hinh_dang_logits():
    m = MiniGPT(100, block_size=16, d_model=32, n_heads=4, n_layers=2)
    assert m(torch.randint(0, 100, (3, 10))).shape == (3, 10, 100)
 
 
def test_loss_khoi_dau_gan_ln_V():
    torch.manual_seed(0)
    V = 1000
    m = MiniGPT(V, block_size=32, d_model=64, n_heads=4, n_layers=2).eval()
    x = torch.randint(0, V, (8, 32))
    y = torch.randint(0, V, (8, 32))
    loss = F.cross_entropy(m(x).reshape(-1, V), y.reshape(-1))
    assert abs(loss.item() - math.log(V)) < 0.5
 
 
def test_hoc_thuoc_mot_batch_nho():
    torch.manual_seed(0)
    V = 50
    m = MiniGPT(V, block_size=16, d_model=64, n_heads=4, n_layers=2,
                dropout=0.0)
    x = torch.randint(0, V, (4, 16))
    y = torch.roll(x, shifts=-1, dims=1)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    for _ in range(300):
        loss = F.cross_entropy(m(x).reshape(-1, V), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.2