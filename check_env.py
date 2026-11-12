import platform
import sys
 
import torch
 
from src.utils import get_device
 
print("Python:", sys.version.split()[0], "|", platform.platform())
print("Kiến trúc CPU:", platform.machine())
print("PyTorch:", torch.__version__)
print("CUDA:", torch.cuda.is_available(),
      "| MPS:", torch.backends.mps.is_available())
 
dev = get_device()
print("Thiết bị sẽ dùng:", dev)
x = torch.randn(1000, 1000, device=dev)
y = x @ x
print("Nhân ma trận OK, tổng =", float(y.sum()))
 
import transformers, gradio, sentencepiece  # noqa: E402,F401
print("transformers", transformers.__version__,
      "| gradio", gradio.__version__)