import sys, time, numpy as np
from scipy.stats import norm
TM_DIR = __import__("pathlib").Path(__file__).resolve().parents[1]
PACKAGE = TM_DIR.parent
sys.path.insert(0, str(TM_DIR))
sys.path.insert(0, str(PACKAGE))
from transport_map_ei import estimate_transport_map_mutual_information as tm
from eicore.discrete import discrete_layer_ei
LN2 = np.log(2.0)
sig = lambda z: 1/(1+np.exp(-z))

def truth_1d(w, s):
    c = np.linspace(0.0, 1.0, 4001)
    m = sig(w*c)
    lo, hi = m.min()-8*s, m.max()+8*s
    e = np.linspace(lo, hi, 8001)
    dens = np.exp(-0.5*((e[:,None]-m[None,:])/s)**2).mean(axis=1)/(s*np.sqrt(2*np.pi))
    h_e = -np.trapezoid(np.where(dens>0, dens*np.log(np.where(dens>0,dens,1)), 0.0), e)
    h_cond = 0.5*np.log(2*np.pi*np.e*s*s)
    return (h_e - h_cond)/LN2

def cont_1d(w, s):
    c = np.linspace(0.0, 1.0, 4001)
    jac = np.abs(w*sig(w*c)*(1-sig(w*c)))
    e_log_j = np.trapezoid(np.log(jac), c)
    return (e_log_j - 0.5*np.log(2*np.pi*np.e*s*s))/LN2

print("== 1D sigmoid 层，真值用数值积分，bit ==")
print(f"{'w':>4} {'sigma':>6} {'truth':>8} {'continuous':>11} {'err%':>7} {'TM-EI':>8} {'err%':>7} {'discrete32':>10} {'err%':>7}")
rng = np.random.default_rng(0)
n = 20000
for w in (0.5, 2.0, 4.0, 8.0):
    for s in (0.01, 0.05, 0.2):
        t = truth_1d(w, s); c_est = cont_1d(w, s)
        c = rng.random((n,1)); e = sig(w*c) + s*rng.normal(size=(n,1))
        tmv = tm(c, e)["mi"]/LN2
        dv = discrete_layer_ei(np.array([[w]]), n, 32, np.random.default_rng(0))["ei_bits"]
        pc = abs(c_est-t)/abs(t)*100; pt = abs(tmv-t)/abs(t)*100; pd = abs(dv-t)/abs(t)*100
        print(f"{w:4.1f} {s:6.2f} {t:8.4f} {c_est:11.4f} {pc:7.1f} {tmv:8.4f} {pt:7.1f} {dv:10.4f} {pd:7.1f}")
