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
def stats(ep):
    W = model.layers[1].weight.detach().numpy()
    Wn = W / np.linalg.norm(W, axis=1, keepdims=True)
    C = np.abs(Wn @ Wn.T); np.fill_diagonal(C, 0)
    i, j = np.unravel_index(np.argmax(C), C.shape)
    gen = torch.Generator().manual_seed(42 + ep)
    S = torch.rand((2000, 5), generator=gen, dtype=torch.float64)
    z = S @ model.layers[1].weight.t(); sp = torch.sigmoid(z)*(1-torch.sigmoid(z))
    sv = np.linalg.svd((sp[:,:,None]*model.layers[1].weight.detach()[None,:,:]).detach().numpy(), compute_uv=False)
    return W[i], W[j], C[i,j], sv[:,4].mean()
for ep in range(1, 621):
    model.train()
    for xb, yb in loader:
        opt.zero_grad(); loss_fn(model(xb), yb).backward(); opt.step()
    if ep in (150, 170, 180, 185, 190, 200, 580, 595, 610):
        wi, wj, cos, sv5 = stats(ep)
        print(f"ep={ep:3d}  最相似权重行 cos={cos:.6f}  sv5={sv5:.2e}  ||Wi||={np.linalg.norm(wi):.3f} ||Wj||={np.linalg.norm(wj):.3f}")
