import argparse
import glob
import html
import json
import os
import random
import re
import unicodedata
from pathlib import Path
 
DINH_CHU = re.compile(r"([.!?…])([A-Za-zÀ-ỹ])")   # "đạm.gió" -> "đạm. gió"
KHOANG_TRANG = re.compile(r"[ \t ]+")
 
 
def clean_line(line):
    line = html.unescape(line)                    # &amp; -> &
    line = unicodedata.normalize("NFC", line)     # chuẩn hoá dấu tiếng Việt
    line = DINH_CHU.sub(r"\1 \2", line)
    line = KHOANG_TRANG.sub(" ", line).strip()
    return line
 
 
def read_lines(path, min_words=3):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = clean_line(raw)
            if len(line.split()) >= min_words:    # bỏ dòng quá ngắn
                yield line
 
 
def split_files(files, seed=42, val_frac=0.05, test_frac=0.05):
    files = sorted(files)
    random.Random(seed).shuffle(files)
    n = len(files)
    n_test = max(1, int(n * test_frac))
    n_val = max(1, int(n * val_frac))
    return {
        "test": files[:n_test],
        "val": files[n_test:n_test + n_val],
        "train": files[n_test + n_val:],
    }
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--max-train-mb", type=float, default=0,
                    help="0 = không giới hạn")
    args = ap.parse_args()
 
    pattern = os.path.join(args.raw, "**", "*.txt")
    files = glob.glob(pattern, recursive=True)
    if not files:
        raise SystemExit(f"Không thấy file .txt nào trong {args.raw}")
 
    parts = split_files(files)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
 
    seen_eval = set()      # các dòng của val/test: train không được chứa
    stats = {}
    for name in ["test", "val", "train"]:
        lines, seen, nbytes = [], set(), 0
        for fp in parts[name]:
            for line in read_lines(fp):
                if line in seen or line in seen_eval:
                    continue                       # bỏ dòng trùng
                seen.add(line)
                lines.append(line)
                nbytes += len(line.encode("utf-8")) + 1
            limit = args.max_train_mb * 1024 * 1024
            if name == "train" and limit and nbytes >= limit:
                break
        if name != "train":
            seen_eval |= seen
        (out / f"{name}.txt").write_text("\n".join(lines) + "\n",
                                         encoding="utf-8")
        stats[name] = {"files": len(parts[name]), "lines": len(lines),
                       "mb": round(nbytes / 1024 / 1024, 2)}
 
    names = {k: [os.path.basename(p) for p in v] for k, v in parts.items()}
    (out / "split_files.json").write_text(
        json.dumps(names, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
 
 
if __name__ == "__main__":
    main()
