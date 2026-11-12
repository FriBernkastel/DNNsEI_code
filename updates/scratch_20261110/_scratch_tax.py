import numpy as np, math
sigma = 0.0026188
s = np.array([2.175e-01, 1.826e-01, 1.110e-01, 8.615e-03, 1.331e-05])
h_noise = 0.5*math.log(2*math.pi*math.e*sigma**2)
ln2 = math.log(2)
w = sigma*math.sqrt(2*math.pi*math.e)
print(f"sigma={sigma:.6f}   h_noise=0.5*ln(2*pi*e*sigma^2)={h_noise:+.4f} nats   equiv_width={w:.4f}")
print(f"{'k':>2} {'s_k':>10} {'ln s_k':>9} {'tax':>8} {'net nats':>9} {'bit':>8}")
net=[]
for k,v in enumerate(s,1):
    a=math.log(v); n=a+h_noise; net.append(n)
    print(f"{k:2d} {v:10.3e} {a:9.3f} {h_noise:8.3f} {n:+9.3f} {n/ln2:8.2f}")
print(f"  sum{'':>9} {sum(math.log(v) for v in s):9.3f} {5*h_noise:8.3f} {sum(net):+9.3f} {sum(net)/ln2:8.2f}")
print(f"EI(repo)      = {sum(net)/5/ln2:+.4f} bit/dim")
print(f"EI(floor 0)   = {sum(max(0,n) for n in net)/5/ln2:+.4f} bit/dim")
s2=s.copy(); s2[4]=0.05
print(f"EI(s5=0.05)   = {sum(math.log(v)+h_noise for v in s2)/5/ln2:+.4f} bit/dim")
