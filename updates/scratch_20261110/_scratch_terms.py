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
for ep in range(1, 186):
    model.train()
    for xb, yb in loader:
        opt.zero_grad(); loss_fn(model(xb), yb).backward(); opt.step()
L = model.layers[1]
gen = torch.Generator().manual_seed(42 + 185)
S = torch.rand((2000, L.in_features), generator=gen, dtype=torch.float64)
z = S @ L.weight.t(); sp = torch.sigmoid(z)*(1-torch.sigmoid(z))
sv = torch.linalg.svdvals(sp[:,:,None]*L.weight.detach()[None,:,:]).detach().numpy()
logs = np.log(sv)
ln2 = np.log(2)
print("L2 (5x5) @ epoch 185 —— 每个奇异值对 term2 的贡献")
print(f"{'k':>2} {'mean s_k':>10} {'mean ln s_k':>12} {'bit':>9}")
tot = 0
for k in range(5):
    m = logs[:,k].mean(); tot += m
    print(f"{k+1:2d} {sv[:,k].mean():10.3e} {m:12.3f} {m/ln2:9.2f}")
print(f"   {'sum':>10} {'':>12} {tot:12.3f} {tot/ln2:9.2f}")
sigma = 0.0026188
t1 = -0.5*5*math.log(2*math.pi*math.e*sigma**2)
print(f"\nterm1 = {t1:+.3f} nats = {t1/ln2:+.2f} bit")
print(f"term2 = 0 + {tot:+.3f} nats = {tot/ln2:+.2f} bit")
print(f"EI = (term1+term2)/5 = {(t1+tot)/5/ln2:+.4f} bit")
print(f"\n若 s5 正常(0.05): ln= {math.log(0.05):+.3f} nats -> EI={((t1+tot-logs[:,4].mean()+math.log(0.05))/5/ln2):+.3f} bit")
