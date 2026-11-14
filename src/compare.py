import json
from pathlib import Path
 
rows = []
for p in sorted(Path("results").glob("*.json")):
    r = json.loads(p.read_text(encoding="utf-8"))
    rows.append((p.stem, r.get("params_M", "?"),
                 r.get("ppl_syllable", float("nan")),
                 r.get("top1_acc", float("nan")),
                 r.get("top3_acc", float("nan"))))
 
print("| mô hình | tham số (M) | ppl/âm tiết | Top-1 | Top-3 |")
print("|---|---|---|---|---|")
for name, par, ppl, t1, t3 in rows:
    print(f"| {name} | {par} | {ppl:.1f} | {t1:.3f} | {t3:.3f} |")
