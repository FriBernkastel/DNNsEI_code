# iris_transport_map_ei —— 用 TM-EI 跑完 IRIS 的整条训练轨迹

第三种 EI 口径（传输映射估计器）的**实验入口**：训练 4→5→5→3 的同时，每 N 轮用 TM-EI
重算一次各层 EI，产出与 `iris_continuous_ei` / `iris_discrete_ei` 同构的曲线。

```
iris_transport_map_ei/
├── iris_transport_map_ei.py     实验入口（与 experiments/ 那份相同 + sys.path 引导）
├── demos/                       5 个后处理脚本
│   ├── replot.py                读最新 CSV 重画曲线 + 趋势相关性诊断
│   ├── split.py                 重画到最新 run 目录
│   ├── split2.py                重画 + publish_latest 镜像
│   ├── sp.py                    Spearman 秩相关（TM vs 连续）
│   └── cmp.py                   对比新旧两次 TM-EI 运行
└── README.md
```

---

## 1. 它测什么

同一条训练轨迹上，每 `--record-every` 轮测一次三路结果：

| 口径 | cause（输入） | 说明 |
|---|---|---|
| **continuous** | $U(0,1)^{n_{in}}$ | 原连续公式（雅可比 + σ），作为参照 |
| **TM-EI uniform** | $U(0,1)^{n_{in}}$，effect 加 $\mathcal N(0,\sigma^2)$ | 与连续版**同一个 estimand**，换估计器 |
| **TM-EI real** | 本 epoch 真实流经权重的 (输入, 输出) 样本对 | 只有采样型估计器能算，连续解析式给不出 |

另外记录一个"不加人为噪声、只靠 jitter"的版本，用来诊断噪声模型的必要性。

**目的**：验证"换估计器、同一估计对象，结论是否一致"。已知结论是**趋势相关、量级不同**：
TM-EI 依赖提升特征与高斯假设，绝对值不能和连续版直接比。

---

## 2. 运行

从**项目根目录**执行（脚本会自动把根目录加入 `sys.path`）：

```
python iris_transport_map_ei/iris_transport_map_ei.py
```

常用参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--epochs` | 4000 | 训练轮数 |
| `--record-every` | 10 | 每多少轮测一次 |
| `--tm-samples` | 20000 | TM-EI 每层样本数 |
| `--ei-samples` | 2000 | 连续版样本数 |
| `--outdir` | `results` | 输出根目录 |

小规模试跑：

```
python iris_transport_map_ei/iris_transport_map_ei.py --epochs 20 --tm-samples 2000
```

> 4000 轮 + 每 10 轮测一次 ≈ 401 个测点 × 3 层，单次 TM 估计 2 万样本，整体约 20–40 分钟（CPU）。

---

## 3. 产物（写到 `../results/`）

```
results/iris_transport_map_ei_<时间戳>/
├── tm_ei_history.csv             每 epoch × 每层：sigma / cont_ei_bits / tm_ei_bits / tm_ei_real_bits / ...
├── tm_ei_routes_comparison.png   图 A：三个估计器、同一个 estimand
└── tm_ei_sample_source.png       图 B：同一估计器，只换样本来源/噪声处理
results/latest/iris_transport_map_ei/   ← 自动镜像（publish_latest）
```

---

## 4. demos 说明

均需从项目根目录运行（它们用相对 `results/...` 路径）：

| 命令 | 作用 |
|---|---|
| `python iris_transport_map_ei/demos/replot.py` | 重画图 + 打印 `corr(cont, −log₂σ)`、`corr(TM, cont)` |
| `python iris_transport_map_ei/demos/split.py` | 重画到最新 run 目录 |
| `python iris_transport_map_ei/demos/split2.py` | 重画 + 镜像到 `results/latest` |
| `python iris_transport_map_ei/demos/sp.py` | Spearman 秩相关（全段 / epoch≥200） |
| `python iris_transport_map_ei/demos/cmp.py` | 新旧两次运行的 `tm_ei_real_bits` 差异 |

---

## 5. 依赖与注意

- 依赖：numpy / scipy / torch / matplotlib / scikit-learn（与主项目一致）；
- 估计器本体在 [`../transport_map_EI/`](../transport_map_EI/README.md)，本目录只负责"跑训练 + 对照"；
- 原始文件在 `../experiments/iris_transport_map_ei.py`（本目录副本加了 `sys.path` 引导，其余相同）。
