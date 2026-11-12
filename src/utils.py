import random
 
import numpy as np
import torch
 
 
def get_device():
    """Chọn thiết bị tốt nhất: CUDA (NVIDIA) > MPS (Apple M1) > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
 
 
def set_seed(seed=42):
    """Cố định số ngẫu nhiên để chạy lại ra cùng kết quả."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
