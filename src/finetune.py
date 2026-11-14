import argparse
import json
import re
from pathlib import Path
 
from transformers import AutoModelForCausalLM, AutoTokenizer
 
from src.config import PRETRAINED, PROC_DIR
from src.evaluate import count_syllables, evaluate_full, frequency_baseline
from src.lm_data import BlockDataset, load_ids
from src.train_lm import train
from src.utils import get_device, set_seed
 
 
def save_json(name, obj):
    Path("results").mkdir(exist_ok=True)
    Path(f"results/{name}.json").write_text(
        json.dumps(obj, indent=2), encoding="utf-8")
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=PRETRAINED)
    ap.add_argument("--name", default="finetune")
    ap.add_argument("--out", default="models/final")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--zero-shot-only", action="store_true",
                    help="chỉ đo mô hình gốc, chưa huấn luyện")
    args = ap.parse_args()
 
    set_seed(42)
    dev = get_device()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    n_emb = model.get_input_embeddings().num_embeddings
    if len(tok) > n_emb:
        print(f"Tokenizer ({len(tok)}) lớn hơn embedding ({n_emb}), mở rộng.")
        model.resize_token_embeddings(len(tok))
    cfg = model.config
    max_pos = (getattr(cfg, "n_positions", None)
               or getattr(cfg, "max_position_embeddings", 1024))
    block = min(args.block, max_pos)
    n_par = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"{args.model}: {n_par:.0f}M tham số | context tối đa {max_pos} "
          f"| từ điển {len(tok)} | block dùng {block} | thiết bị {dev}")
 
    tag = re.sub(r"[^A-Za-z0-9]+", "_", args.model)
    eos = tok.eos_token_id
    enc = lambda lines: tok(lines, add_special_tokens=False)["input_ids"]
    ids = {s: load_ids(s, tag, enc, eos) for s in ("train", "val", "test")}
    print({k: f"{len(v):,} token" for k, v in ids.items()})
    ds = {s: BlockDataset(ids[s], block) for s in ids}
    n_syl = count_syllables(f"{PROC_DIR}/test.txt")
 
    zs = evaluate_full(model, ds["test"], dev, n_syllables=n_syl)
    zs["params_M"] = round(n_par, 1)
    print("Trước khi fine-tune:", json.dumps(zs, indent=1))
    save_json(f"{args.name}_zeroshot", zs)
    if args.zero_shot_only:
        return
 
    def save_fn(d):
        model.save_pretrained(d)
        tok.save_pretrained(d)
 
    train(model, ds["train"], ds["val"], dev, args.out,
          max_steps=args.steps, batch_size=args.batch, lr=args.lr,
          eval_every=args.eval_every, save_fn=save_fn)
 
    best = AutoModelForCausalLM.from_pretrained(Path(args.out) / "best")
    res = evaluate_full(best, ds["test"], dev, n_syllables=n_syl)
    res["params_M"] = round(n_par, 1)
    res["freq_baseline"] = frequency_baseline(ids["train"], ids["test"])
    print("Sau khi fine-tune:", json.dumps(res, indent=1))
    save_json(args.name, res)
 
 
if __name__ == "__main__":
    main()
