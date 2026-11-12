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
    sv = torch.linalg.svdvals(J).detach().numpy()
    lp = np.sum(np.log(np.clip(sv, 1e-12, None)), axis=1)
    return sv, lp
for ep in range(1, 621):
    model.train()
    for xb, yb in loader:
        opt.zero_grad(); loss_fn(model(xb), yb).backward(); opt.step()
    if ep in (180, 185, 190, 195, 200, 205, 210, 580, 590, 600, 610):
        sv, lp = probe(ep)
        frac = np.mean(sv[:,3] < 1e-6)
        print(f"ep={ep:3d}  mean_logpdet={lp.mean():8.3f}  p1={np.percentile(lp,1):8.2f} "
              f"min_logpdet={lp.min():9.2f}  frac(sv4<1e-6)={frac*100:6.2f}%  "
              f"sv4: mean={sv[:,3].mean():.2e} min={sv[:,3].min():.2e}")
