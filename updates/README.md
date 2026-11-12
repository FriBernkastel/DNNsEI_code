# updates/ —— 本轮分析新增的脚本

这些脚本都是**只读**的：不修改 `eicore/`、`experiments/`、`results/latest/`，也不改原始实验流程。
它们读取已有的运行结果（CSV/权重），做诊断、对照或画图，输出统一写到 `results/updates/`。

运行方式（在项目根目录 `DNNsEI_code_20261029/` 下）：

```
python -m updates.<脚本名>
```

---

## 1. 画图的脚本（会产生 PNG/CSV）

| 脚本 | 作用 | 输出 |
|---|---|---|
| `make_relative_ei_figure.py` | 图 2 左图的"相对初始化"版本：每条曲线减去自己在 epoch 1 的值，Overall 由 0 涨到 +3.63 bits | `results/updates/ei_decomposition_relative.png`<br>`results/updates/ei_relative.csv` |
| `run_continuous_w3.py` | 把注入区间改成 w=3（采样 U(0,3)）重跑连续版，对照 w=1 | `results/updates/ei_decomposition_w3.png`<br>`results/updates/ei_w3.csv` |
| `run_positive_overall.py` | 尝试用"w=3+Adam"和"tanh+Adam"把 Overall EI 拉正（结论：失败，越训越负） | `results/updates/overall_positive.png`<br>`results/updates/overall_positive.csv` |

## 2. 打印表格的脚本（输出已存档到 `results/updates/logs/`）

| 脚本 | 作用 | 存档 |
|---|---|---|
| `diagnose_mse2.py` | 用原模型类诊断 MSE：4000 轮 / 20000 轮 / Adam 三组对比与逐样本误差分布 | `logs/diagnose_mse2.txt` |
| `diagnose_mse.py` | 早期版本（手工搭 Sequential，初始化与 `IrisSigmoidNet` 不一致，**结论以 diagnose_mse2 为准**） | — |
| `scan_volume_term.py` | 体积项的三种杠杆：权重缩放 / 激活函数 / 注入宽度，各自能换多少 bits、对应多低的 MSE 门槛 | `logs/scan_volume_term.txt` |
| `scan_combo.py` | 上面三者的组合扫描，找"体积项最不负"的配置 | `logs/scan_combo.txt` |
| `volume_term_4000_vs_20000.py` | 固定 sigmoid+SGD+w=1，拆解 4000 轮与 20000 轮的体积项（权重幅度 / 饱和 / 奇异值） | `logs/volume_term_4000_vs_20000.txt` |
| `run_long_training_ei.py` | 训练 20000 轮后测 EI（Overall 由 −12.55 涨到 −4.18，仍为负） | `logs/run_long_training_ei.txt` |
| `knn_entropy_preview.py` | 方法②预览：用 Kozachenko–Leonenko kNN 熵替代局部体积近似，EI 变为正（逐层 4.31/2.67/1.16 bits） | `logs/knn_entropy_preview.txt` |

> `logs/` 里是当时运行的输出记录（含运行日期与命令），不是每次自动生成；重跑脚本会得到同样的结果（seed 固定）。

---

## 3. 与本轮其它改动的对应关系

| 改动 | 位置 | 说明 |
|---|---|---|
| 报告固定路径镜像 | `eicore/report.py` + `tools/publish_latest.py` | 实验结束时自动把结果镜像到 `results/latest/<实验名>/`，HTML 只引用固定路径 |
| `results/latest/`、`results/updates/` | `results/` | 前者是原实验的固定镜像，后者是本轮分析产物 |
| CUDA 设备修复 | `eicore/sigma.py` | 累加器按梯度所在设备创建，GPU 上不再报 device mismatch |
| 原理文档 | `keypoints/EI计算原理（连续与离散）.md` | 从 60 个权重讲起的连续/离散方法说明 + 12 条误区 |

---

## 4. 复现全部输出

```
python -m updates.make_relative_ei_figure
python -m updates.run_continuous_w3 --w 3
python -m updates.run_positive_overall
python -m updates.diagnose_mse2
python -m updates.scan_volume_term
python -m updates.scan_combo
python -m updates.volume_term_4000_vs_20000
python -m updates.run_long_training_ei
python -m updates.knn_entropy_preview
```

依赖与主项目相同（numpy / matplotlib / torch / scikit-learn；`knn_entropy_preview` 另需 scipy）。
