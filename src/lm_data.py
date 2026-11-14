from array import array
from pathlib import Path
 
import numpy as np
import torch
from torch.utils.data import Dataset
 
 
def encode_file(path, encode_fn, eos_id, batch_lines=2000):
    """Đọc file, mã hoá thành một dãy ID dài (mỗi dòng kết thúc bằng EOS)."""
    ids = array("i")          # tiết kiệm RAM hơn list Python
 
    def flush(buf):
        for e in encode_fn(buf):
            ids.extend(e)
            ids.append(eos_id)
 
    buf = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            buf.append(line)
            if len(buf) == batch_lines:
                flush(buf)
                buf = []
    if buf:
        flush(buf)
    return np.frombuffer(ids, dtype=np.int32).copy()
 
 
def load_ids(split, tag, encode_fn, eos_id, proc_dir="data/processed"):
    """Mã hoá một lần rồi lưu cache .npy; tag phân biệt từng tokenizer."""
    cache = Path(proc_dir) / f"{split}.{tag}.npy"
    if cache.exists():
        return np.load(cache)
    ids = encode_file(Path(proc_dir) / f"{split}.txt", encode_fn, eos_id)
    np.save(cache, ids)
    return ids
 
 
class BlockDataset(Dataset):
    """Cắt dãy ID thành các khối không chồng lấn. y = x dịch trái 1 vị trí."""
 
    def __init__(self, ids, block_size):
        self.ids = ids
        self.bs = block_size
        self.n = (len(ids) - 1) // block_size
 
    def __len__(self):
        return self.n
 
    def __getitem__(self, i):
        s = i * self.bs
        chunk = torch.from_numpy(self.ids[s: s + self.bs + 1].astype(np.int64))
        return chunk[:-1], chunk[1:]
