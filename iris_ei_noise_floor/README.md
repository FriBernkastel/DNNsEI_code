# iris_ei_noise_floor —— 连续 EI 的"噪声下限"修正实验

连续版 EI 有一个结构性问题：某些奇异方向的信号（$\ln s_k$）比噪声底 $h_{\text{noise}}$ 还弱，
原公式仍然给它们记负分，导致 EI 变得很负、甚至失去区分度。这个实验把"每个方向"单独结算：

```
净贡献_k = ln s_k − h_noise        （每个奇异方向）
         = max(0, ln s_k − h_noise)    （低于噪声底的方向截断为 0）
```

即：**噪声只对真正有信号的方向计成本**，低于噪声底的方向不计负分。

```
iris_ei_noise_floor/
├── iris_ei_noise_floor.py      实验入口（与 experiments/ 那份相同 + sys.path 引导）
├── demos/
│   ├── nf.py                   打印 old/new 对比表（floor_gain、tax_fix）
│   ├── nf2.py                  聚焦 L2 尖谷，统计 floor_gain>0 的 epoch 数
│   └── nfplot.py               用 plot_comparison 重画 + 镜像 latest
└── README.md
```

---

## 1. 两个版本的区别

| 版本 | 公式 | 特点 |
|---|---|---|
| **original** | $\mathrm{EI} = \frac{1}{n_{in}}\left[\text{term1} + H(X) + \sum_k \ln s_k\right]$ | 可负；饱和方向上会被"噪声税"拖到很负 |
| **noise-floor corrected** | 每个方向先 `max(0, ln s_k − h_noise)`，再乘噪声 | 非负；信号弱于噪声时该方向贡献 0 |

核心库 `eicore/continuous.py` 同时给出两套字段（`ei_bits` 与 `ei_bits_floored`），
以及 `floor_gain_bits`（被截断补回的量）和 `tax_fix_bits`（方向数修正）。

---

## 2. 运行

从**项目根目录**执行：

```
python iris_ei_noise_floor/iris_ei_noise_floor.py
```

参数与主实验一致（`--epochs / --record-every / --ei-samples / --outdir`）。小规模试跑：

```
python iris_ei_noise_floor/iris_ei_noise_floor.py --epochs 40
```

产物写到 `../results/iris_ei_noise_floor_<时间戳>/`：

```
noise_floor_history.csv        每 epoch × 每层：ei_bits / ei_bits_floored / floor_gain_bits / tax_fix_bits / overall_*
noise_floor_comparison.png     2×2：三层 old vs corrected + Overall（残差 σ 与 Σ layers 两种口径）
results/latest/iris_ei_noise_floor/   ← 自动镜像
```

---

## 3. 已知结果（seed 42）

- 训练初期（epoch 1）几乎所有方向都低于噪声底，修正后 Overall **从 −12.55 变成 0.000**；
- 随训练推进，部分方向浮出噪声底，corrected EI 从 0 开始上升；
- `floor_gain_bits` 正好量化了"原公式多扣了多少"，`tax_fix_bits` 量化"方向数是否算多了"。

> 换句话说：原公式的负值并不全是"网络没有信息"，有相当一部分是**噪声成本记在了没有信号的方向上**。
> 修正版把这部分截断掉，得到一个非负、且与 σ 相关的 EI。

---

## 4. 依赖与注意

- 依赖与主项目相同；
- 原始文件在 `../experiments/iris_ei_noise_floor.py`；
- 该修正依赖 `eicore/continuous.py` 里新增的字段（`ei_bits_floored` 等），
  若换回旧版 eicore，本脚本会报 `KeyError`。
