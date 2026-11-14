import argparse
import json
from pathlib import Path
 
import sentencepiece as spm
import torch
 
from src.config import PROC_DIR, SPM_PREFIX
from src.evaluate import count_syllables, evaluate_full, frequency_baseline
from src.lm_data import BlockDataset, load_ids
from src.train_lm import train
from src.utils import get_device, set_seed
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["lstm", "gpt"], required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-every", type=int, default=200)
    args = ap.parse_args()
 
    set_seed(42)
    dev = get_device()
    sp = spm.SentencePieceProcessor(model_file=SPM_PREFIX + ".model")
    enc = lambda lines: sp.encode(lines)
    ids = {s: load_ids(s, "spm8k", enc, sp.eos_id())
           for s in ("train", "val", "test")}
    ds = {s: BlockDataset(ids[s], args.block) for s in ids}
    V = sp.get_piece_size()
 
    if args.model == "lstm":
        from src.lstm_baseline import LSTMLM
        model = LSTMLM(V)
    else:
        from src.mini_gpt import MiniGPT
        model = MiniGPT(V, block_size=args.block)
    n_par = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"{args.model}: {n_par:.2f}M tham số | thiết bị: {dev}")
 
    out = Path("models") / args.model
    train(model, ds["train"], ds["val"], dev, out, max_steps=args.steps,
          batch_size=args.batch, lr=args.lr, eval_every=args.eval_every)
 
    best = torch.load(out / "best.pt", map_location=dev)
    model.load_state_dict(best["model"])
    n_syl = count_syllables(f"{PROC_DIR}/test.txt")
    res = evaluate_full(model, ds["test"], dev, n_syllables=n_syl)
    res["params_M"] = round(n_par, 2)
    res["freq_baseline"] = frequency_baseline(ids["train"], ids["test"])
    Path("results").mkdir(exist_ok=True)
    Path(f"results/{args.model}.json").write_text(
        json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))
 
 
if __name__ == "__main__":
    main()
