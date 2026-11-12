import sys, numpy as np, torch, torch.nn as nn
sys.path.insert(0, ".")
from torch.utils.data import DataLoader, TensorDataset
from eicore import IrisSigmoidNet, load_iris_tensors
torch.manual_seed(42); np.random.seed(42)
x_tr, y_tr, *_ = load_iris_tensors(42)
model = IrisSigmoidNet().to(dtype=torch.float64)
loss_fn = nn.MSELoss(); opt = torch.optim.SGD(model.parameters(), lr=0.01)
g = torch.Generator().manual_seed(42)
loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=10, shuffle=True, generator=g)
print("权重矩阵 W2 (5x5) 自身的奇异值：")
for ep in range(1, 621):
    model.train()
    for xb, yb in loader:
        opt.zero_grad(); loss_fn(model(xb), yb).backward(); opt.step()
    if ep in (1, 150, 170, 185, 200, 595):
        s = np.linalg.svd(model.layers[1].weight.detach().numpy(), compute_uv=False)
        print(f"  ep={ep:3d}  " + "  ".join(f"{v:.2e}" for v in s) + f"   cond={s[0]/s[-1]:.1f}")
d = np.load(str(__import__("pathlib").Path(__file__).resolve().parents[2] / "transport_map_EI" / "data" / "iris_weights_seed42.npz"))
s = np.linalg.svd(d["W2"], compute_uv=False)
print(f"  ep=4000 (存档)  " + "  ".join(f"{v:.2e}" for v in s) + f"   cond={s[0]/s[-1]:.1f}")
