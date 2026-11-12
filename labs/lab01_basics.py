import torch
import torch.nn.functional as F
 
# 1. Tensor
x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print("shape:", x.shape, "| dtype:", x.dtype)
 
# 2. Softmax: logits -> xác suất
logits = torch.tensor([2.0, 1.0, 0.1])
probs = torch.softmax(logits, dim=0)
print("xác suất:", probs, "| tổng =", probs.sum().item())
 
# 3. Cross-entropy: tính tay và so với PyTorch (từ đúng là vị trí 0)
loss_tay = -torch.log(probs[0])
loss_torch = F.cross_entropy(logits.unsqueeze(0), torch.tensor([0]))
print("loss tính tay:", loss_tay.item(), "| PyTorch:", loss_torch.item())
 
# 4. Autograd: loss = (2w - 10)^2, đạo hàm = 4(2w - 10), tại w=3 là -16
w = torch.tensor(3.0, requires_grad=True)
loss = (2 * w - 10) ** 2
loss.backward()
print("gradient của w:", w.grad.item(), "(kỳ vọng -16)")
 
# 5. Vòng huấn luyện đầu tiên: học y = 2x + 1 từ dữ liệu có nhiễu
torch.manual_seed(0)
X = torch.rand(200, 1)
Y = 2 * X + 1 + 0.05 * torch.randn(200, 1)
model = torch.nn.Linear(1, 1)
opt = torch.optim.Adam(model.parameters(), lr=0.05)
for step in range(1000):
    pred = model(X)
    loss = F.mse_loss(pred, Y)
    opt.zero_grad()
    loss.backward()
    opt.step()
print("w =", round(model.weight.item(), 2),
      "| b =", round(model.bias.item(), 2), "(kỳ vọng ≈ 2 và 1)")