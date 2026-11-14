import math
 
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
 
 
def count_syllables(path):
    """Đếm âm tiết (tách khoảng trắng) để chuẩn hoá perplexity."""
    with open(path, encoding="utf-8") as f:
        return sum(len(line.split()) for line in f)
 
 
@torch.no_grad()
def evaluate_full(model, ds, device, batch_size=16, ks=(1, 3),
                  n_syllables=None):
    model.to(device).eval()
    nll, n = 0.0, 0
    hits = {k: 0 for k in ks}
    for x, y in DataLoader(ds, batch_size=batch_size):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        if hasattr(logits, "logits"):
            logits = logits.logits
        logits = logits.float()
        nll += F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               y.reshape(-1), reduction="sum").item()
        n += y.numel()
        top = logits.topk(max(ks), dim=-1).indices          # B, T, K
        for k in ks:
            hits[k] += (top[..., :k] == y.unsqueeze(-1)).any(-1).sum().item()
    res = {"tokens": n, "nll_per_token": nll / n,
           "ppl_token": math.exp(nll / n)}
    for k in ks:
        res[f"top{k}_acc"] = hits[k] / n
    if n_syllables:
        token_per_syl = len(ds.ids) / n_syllables
        res["ppl_syllable"] = math.exp(nll / n * token_per_syl)
    return res
 
 
def frequency_baseline(train_ids, test_ids, ks=(1, 3)):
    """Mốc 'ngây thơ': luôn đoán các token phổ biến nhất."""
    order = np.argsort(-np.bincount(train_ids))
    return {f"top{k}_acc": float(np.isin(test_ids, order[:k]).mean())
            for k in ks}
