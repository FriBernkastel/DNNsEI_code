import numpy as np, math
d = np.load(str(TM_DIR / "data" / "iris_weights_seed42.npz"))
sig = lambda z: 1/(1+np.exp(-z))
rng = np.random.default_rng(0)
res = float(d["residual_sigma"])
h = 0.5*math.log(2*math.pi*math.e*res*res)
print(f"residual sigma = {res:.4f}")
print(f"h_noise = 0.5*ln(2*pi*e*sigma^2) = {h:+.4f} nats   -> 方向要贡献, 需 ln s > {h:+.4f}, 即 s > {math.exp(h):.4f}")
print(f"噪声等效宽度 sigma*sqrt(2*pi*e) = {res*math.sqrt(2*math.pi*math.e):.3f}\n")
for i in (1,2,3):
    W = d[f"W{i}"]; n_in = W.shape[1]; n_out = W.shape[0]
    U = rng.random((20000, n_in))
    z = U @ W.T; sp = sig(z)*(1-sig(z))
    sv = np.linalg.svd(sp[:,:,None]*W[None,:,:], compute_uv=False)
    m = sv.mean(0)
    print(f"L{i} ({n_in}->{n_out}) sigma_layer={float(d[f'sigma{i}']):.5f}  奇异值均值: {np.round(m,4)}")
    print(f"     最大 = {m.max():.4f}  vs 残差阈值 {math.exp(h):.4f} -> {'全部低于底噪' if m.max() < math.exp(h) else 'ok'}")
    hl = 0.5*math.log(2*math.pi*math.e*float(d[f"sigma{i}"])**2)
    print(f"     若用层 sigma: h_noise={hl:+.3f} 阈值={math.exp(hl):.4f}")
