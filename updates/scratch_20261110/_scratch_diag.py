import sys, numpy as np, torch, torch.nn as nn, math
sys.path.insert(0, ".")
from torch.utils.data import DataLoader, TensorDataset
from eicore import IrisSigmoidNet, load_iris_tensors
torch.manual_seed(42); np.random.seed(42)
x_tr, y_tr, *_ = load_iris_tensors(42)
model = IrisSigmoidNet().to(dtype=torch.float64)
loss_fn = nn.MSELoss(); opt = torch.optim.SGD(model.parameters(), lr=0.01)
g = torch.Generator().manual_seed(42)
loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=10, shuffle=True, generator=g)
print(f"{'ep':>4} {'ei':>7} {'logpdet':>8} {'min_sv':>9} {'sv2':>9} {'sv3':>9} {'sv4':>9} {'max|z|':>7} {'min_sigp':>9}")
for ep in range(1, 621):
    model.train()
    for xb, yb in loader:
        opt.zero_grad(); yp = model(xb); loss_fn(yp, yb).backward(); opt.step()
    if ep % 10 == 0 and 90 <= ep <= 620:
        L = model.layers[1]
        gen = torch.Generator().manual_seed(42 + ep)
        S = torch.rand((2000, L.in_features), generator=gen, dtype=torch.float64)
        z = S @ L.weight.t()
        sp = torch.sigmoid(z) * (1 - torch.sigmoid(z))
        J = sp[:, :, None] * L.weight[None, :, :]
        sv = torch.linalg.svdvals(J)
        logp = torch.sum(torch.log(torch.clamp(sv, min=1e-12)), dim=1)
        zabs = z.abs().max().item(); minsigp = sp.min().item()
        print(f"{ep:4d} {float(torch.mean(logp))/math.log(2):7.3f} {float(torch.mean(logp)):8.3f} "
              f"{sv[:,0].mean():9.3e} {sv[:,1].mean():9.3e} {sv[:,2].mean():9.3e} {sv[:,3].mean():9.3e} {zabs:7.2f} {minsigp:9.2e}")
