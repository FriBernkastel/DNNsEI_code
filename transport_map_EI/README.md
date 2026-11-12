# transport_map_EI —— 第三种 EI 估计方法（TM-EI）

把"用传输映射估计互信息"这条路线的**库 + 驱动脚本 + 数据**收拢在一起，方便日后单独调用，
不依赖主实验流程（个别脚本需要父目录的 `eicore`，见下）。

```
transport_map_EI/
├── transport_map_ei.py     核心库（与 experiments/transport_map_ei.py 逐字节相同）
├── __init__.py             把核心函数 re-export，方便 import
├── data/
│   └── iris_weights_seed42.npz   复现出来的 IRIS 网络权重（可由 demos 重新生成）
├── demos/                  10 个驱动脚本
└── README.md
```

> 产物（图 / CSV）仍按项目约定写在 `../results/` 下，不在这里重复存一份。

---

## 1. 这个方法在做什么

前两种口径分别是"分箱数格子"（离散）和"高斯噪声 + 雅可比体积"（连续）。TM-EI 是第三条路：

1. 用样本估计**联合分布**的均值与协方差；
2. 用 Cholesky 分解构造一个**仿射三角传输映射**，把数据空间映射到标准正态参考空间；
3. 在参考空间里，对数密度有解析式，于是
   `h(X,Y) = h(ref) + log|det L|`，从而
   `MI = h(X) + h(Y) − h(X,Y)`；
4. 用 **Wishart 模型的 log-det 偏差修正**（`gaussian_logdet_bias_correction`）抵消有限样本导致的协方差偏小。

本质是"**在提升（lift）后的特征空间里做高斯互信息**"。因为高斯互信息是**非负**的（也能被偏差修正拉回 0 附近），
所以它天生不会出现连续法的负值问题——代价是：**它假设（提升后的）联合分布近似高斯**。

三种口径的关键差别：

| | 离散法 | 连续法 | TM-EI |
|---|---|---|---|
| 需要的额外假设 | 分箱粒度 bins | 噪声尺度 σ | 联合高斯（或提升后近似高斯） |
| 自由参数 | bins | σ | 提升特征（x, x², x³…）与 jitter |
| 符号 | 恒非负 | 可为负 | 通常非负（实现里有 clip） |
| 确定性映射 | 直接可算 | 需要 σ | MI 发散，靠 jitter / 噪声兜底 |
| 逐边分解 | 可以（sensitivity） | 可以（雅可比） | 不做逐边，做"多源分解"（joint/left/right/syn） |

---

## 2. 快速使用

```python
import numpy as np
import transport_map_EI as tm          # 需把本目录的父目录加入 sys.path

x = np.random.default_rng(0).normal(size=(20000, 1))
y = 0.8 * x + 0.6 * np.random.default_rng(1).normal(size=(20000, 1))

res = tm.estimate_transport_map_mutual_information(x, y)
print(res["mi"] / np.log(2), "bits")   # 返回的是 nats，除以 ln2 得 bits
```

常用函数：

| 函数 | 作用 |
|---|---|
| `estimate_transport_map_mutual_information(source, target)` | 单次 MI 估计（nats），附 `bias_correction` 与逐点值 |
| `compute_two_source_transport_map_ei(left, right, target)` | 双源分解：`joint_ei / left_ei / right_ei / syn / syn_ratio` |
| `lift_single_source_features(x)` | 单源提升 `[x, x², x³]` |
| `lift_joint_source_features(x, y)` | 双源提升 `[x, y, xy, x², y²]` |
| `fit_affine_transport_map_density(samples)` | 拟合密度模型（后续可 `.log_prob()` / `.marginal()`） |
| `gaussian_logdet_bias_correction(d, n)` | Wishart log-det 偏差修正项 |
| `simulate_alpha_nonlinear_case(...)` | 论文式双源仿真 `q1_next = α·sin(q2·q3) + (1−α)·q2 + noise` |

---

## 3. `demos/` 里有什么

| 脚本 | 作用 | 产物 |
|---|---|---|
| `run_transport_map_ei.py` | 冒烟测试：二元高斯解析校验、lift、α 扫描、nats/偏差拆解 | 终端 |
| `probe_tm_ei_domain.py` | 探它能否处理离散对象、jitter 敏感性（确定性映射下 MI 发散） | 终端 |
| `reproduce_iris_weights.py` | 复现 seed=42 的 IRIS 网络，与 `results/latest` 对账，导出权重 | `data/iris_weights_seed42.npz` |
| `tm_ei_on_iris.py` | 用复现权重跑 TM-EI，与连续/离散 EI 对比、噪声敏感性 | 终端 |
| `fig_three_routes.py` | 三条路线柱状对比 + 噪声扫描 | `../results/iris_ei_three_routes.png` |
| `fig_benchmark_1d.py` | 1D 真值 vs 连续 vs TM-EI vs 离散；另测收敛与耗时 | `../results/benchmark_1d_sigmoid.png` |
| `bench_1d.py` | 1D 基准误差表 | 终端 |
| `sample_size_check.py` | 样本量对 TM-EI 的影响（n=20000→150） | 终端 |
| `sv_tax.py` | 奇异值"噪声税"拆解（解释 EI 符号来源） | 终端 |
| `why_noise_floor.py` | 解释残差 σ 下 Overall EI 为何低于噪声下限 | 终端 |

运行（任选其一，脚本会自动把本目录与父包加入 `sys.path`）：

```
python demos/run_transport_map_ei.py
python demos/fig_three_routes.py
python -m transport_map_EI.demos...   # 本目录不是 package 的子包时可忽略
```

> `reproduce_iris_weights.py`、`tm_ei_on_iris.py`、`bench_1d.py`、`fig_benchmark_1d.py`
> 会用到父目录的 `eicore`（加载 IRIS、离散 EI 对照）。其余脚本只依赖 numpy/scipy。

---

## 4. 历史产物在哪

| 内容 | 位置 |
|---|---|
| 三条路线对比图 | `../results/iris_ei_three_routes.png` |
| 1D 基准图 | `../results/benchmark_1d_sigmoid.png` |
| IRIS 上的 TM-EI 历史 | `../results/iris_transport_map_ei_20261110_150843/`、`..._155630/` |
| 噪声下限分析 | `../results/iris_ei_noise_floor_20261110_224905/` |

---

## 5. 两个使用陷阱

1. **确定性映射下 MI 发散**：`y = f(x)` 且无噪声时，联合协方差奇异，MI 只由 `jitter`（默认 1e-6）决定，
   数值随 jitter 变化剧烈。要得到有意义的数，必须加真实噪声（如 `+ σ·N(0,1)`）。
2. **提升（lift）决定能捕捉多少非线性**：`[x, x², x³]` 只能表达多项式型依赖；
   如果真实关系是 `sin(xy)` 这类，单靠 lift 拟合不出来，需要自行扩展特征。

---

## 6. 与主项目的关系

- 核心库原文件：`../experiments/transport_map_ei.py`（本目录是逐字节副本，改动请两边同步）；
- 这些脚本最初是 2026-11-10 探索会话的临时文件，归档在 `../updates/scratch_20261110/`，
  本目录是从那里整理出来的可调用版本。
