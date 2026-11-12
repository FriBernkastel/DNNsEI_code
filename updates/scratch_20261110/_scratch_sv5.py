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
def probe(ep):
    L = model.layers[1]
    gen = torch.Generator().manual_seed(42 + ep)
    S = torch.rand((2000, L.in_features), generator=gen, dtype=torch.float64)
    z = S @ L.weight.t(); sp = torch.sigmoid(z)*(1-torch.sigmoid(z))
    J = sp[:,:,None] * L.weight[None,:,:]
    return torch.linalg.svdvals(J).detach().numpy()
print("epoch   sv1     sv2     sv3     sv4      sv5        mean_logpdet  frac(sv5<1e-10)")
for ep in range(1, 621):
    model.train()
    for xb, yb in loader:
        opt.zero_grad(); loss_fn(model(xb), yb).backward(); opt.step()
    if ep % 5 == 0 and 170 <= ep <= 215 or ep in (580,585,590,595,600,605,610):
        sv = probe(ep)
        lp = np.sum(np.log(np.clip(sv, 1e-12, None)), axis=1)
        m = sv.mean(0)
        print(f"{ep:5d} {m[0]:.4f} {m[1]:.4f} {m[2]:.4f} {m[3]:.4f}  {m[4]:.2e}   {lp.mean():8.3f}     {np.mean(sv[:,4]<1e-10)*100:6.2f}%")
