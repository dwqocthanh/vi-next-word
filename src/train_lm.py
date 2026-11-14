import json
import math
import time
from contextlib import nullcontext
from pathlib import Path
 
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
 
 
def lm_loss(model, x, y):
    logits = model(x)
    if hasattr(logits, "logits"):            # mô hình của Hugging Face
        logits = logits.logits
    logits = logits.float()
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                           y.reshape(-1))
 
 
@torch.no_grad()
def quick_eval(model, loader, device, max_batches=50):
    model.eval()
    tong, n = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        tong += lm_loss(model, x, y).item() * y.numel()
        n += y.numel()
    model.train()
    return tong / n
 
 
def get_lr(step, max_steps, base_lr, warmup):
    if step < warmup:
        return base_lr * (step + 1) / warmup          # tăng dần
    p = (step - warmup) / max(1, max_steps - warmup)
    return base_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))
 
 
def train(model, train_ds, val_ds, device, out_dir, max_steps=2000,
          batch_size=32, lr=3e-4, warmup=100, eval_every=200,
          grad_clip=1.0, weight_decay=0.01, amp=True, save_fn=None,
          log=print):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.to(device).train()
    tl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                    drop_last=True)
    vl = DataLoader(val_ds, batch_size=batch_size)
    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=weight_decay)
    use_amp = amp and device.type == "cuda"      # fp16 chỉ dùng trên NVIDIA
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
 
    tok_per_step = batch_size * train_ds.bs
    epochs = max_steps * tok_per_step / len(train_ds.ids)
    log(f"Bắt đầu: {max_steps} step x {tok_per_step} token "
        f"= khoảng {epochs:.2f} epoch | AMP: {use_amp} | {device}")
 
    step, best, history, t0 = 0, float("inf"), [], time.time()
    while step < max_steps:
        for x, y in tl:
            for g in opt.param_groups:
                g["lr"] = get_lr(step, max_steps, lr, warmup)
            x, y = x.to(device), y.to(device)
            ctx = (torch.autocast("cuda", dtype=torch.float16)
                   if use_amp else nullcontext())
            with ctx:
                loss = lm_loss(model, x, y)
            opt.zero_grad(set_to_none=True)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
            step += 1
 
            if step % 50 == 0:
                log(f"step {step}/{max_steps} | loss {loss.item():.3f} "
                    f"| {time.time() - t0:.0f}s")
            if step % eval_every == 0 or step == max_steps:
                v = quick_eval(model, vl, device)
                history.append({"step": step, "train_loss": loss.item(),
                                "val_loss": v})
                log(f"[eval] step {step} | val_loss {v:.3f} "
                    f"| ppl(token) {math.exp(v):.1f}")
                if v < best:                       # chỉ lưu khi val tốt hơn
                    best = v
                    if save_fn is None:
                        torch.save({"model": model.state_dict(),
                                    "step": step, "val_loss": v},
                                   out_dir / "best.pt")
                    else:
                        save_fn(out_dir / "best")
            if step >= max_steps:
                break
    (out_dir / "history.json").write_text(json.dumps(history, indent=1))
    return history
