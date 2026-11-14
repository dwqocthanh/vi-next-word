import numpy as np
import torch
 
from src.lm_data import BlockDataset
from src.lstm_baseline import LSTMLM
from src.train_lm import train
 
 
def test_block_dataset_dich_trai_mot_vi_tri():
    ds = BlockDataset(np.arange(11, dtype=np.int32), 5)
    assert len(ds) == 2
    x, y = ds[0]
    assert x.tolist() == [0, 1, 2, 3, 4]
    assert y.tolist() == [1, 2, 3, 4, 5]
 
 
def test_lstm_shape():
    m = LSTMLM(100, emb=16, hid=32)
    assert m(torch.randint(0, 100, (2, 7))).shape == (2, 7, 100)
 
 
def test_train_chay_va_luu_checkpoint(tmp_path):
    ids = np.random.randint(0, 50, size=3000).astype(np.int32)
    tr, va = BlockDataset(ids, 16), BlockDataset(ids[:1000], 16)
    m = LSTMLM(50, emb=16, hid=32)
    train(m, tr, va, torch.device("cpu"), tmp_path, max_steps=20,
          batch_size=8, eval_every=10, warmup=2, log=lambda s: None)
    assert (tmp_path / "best.pt").exists()
