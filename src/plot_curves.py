import json
import sys
 
import matplotlib.pyplot as plt
 
for name in sys.argv[1:]:                 # ví dụ: lstm gpt
    h = json.load(open(f"models/{name}/history.json", encoding="utf-8"))
    steps = [r["step"] for r in h]
    plt.plot(steps, [r["val_loss"] for r in h], label=f"{name} (val)")
plt.xlabel("step")
plt.ylabel("val loss")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("results/curves.png", dpi=150, bbox_inches="tight")
print("Đã lưu results/curves.png")
