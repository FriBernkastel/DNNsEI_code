# scratch_20261110/ —— 2026-11-10 探索会话的临时脚本（归档）

这些是当天做方法论探索时的一次性脚本，共 **26 个**，不属于主项目流程。
2026-11-12 从包根目录收拢到这里；其中 **8 个已迁出**到两个正式包：

| 迁到 | 脚本 |
|---|---|
| `../../iris_transport_map_ei/demos/` | `replot / split / split2 / sp / cmp` |
| `../../iris_ei_noise_floor/demos/` | `nf / nf2 / nfplot` |

另外 **10 个 TM-EI 驱动脚本**在 [`../../transport_map_EI/demos/`](../../transport_map_EI/README.md)
有整理版（路径已改为相对），本目录保留同名原始副本：

`run_transport_map_ei / probe_tm_ei_domain / reproduce_iris_weights / tm_ei_on_iris /
fig / fig2 / bench1d / ncheck / tax / why0`

---

## 本目录剩余的两类脚本

### A. 连续 EI 的探针口径诊断（5 个）

真实 IRIS 输入（范围 0.1–7.9）和仓库用的 $U(0,1)$ 探针不是同一个分布，这组脚本量化两者差异：

| 脚本 | 作用 |
|---|---|
| `_scratch_z.py` | 两种输入下各层加权和 z 的范围对比 |
| `_scratch_h.py` | 真实输入的高斯熵 h(x) vs $U(0,1)$ 的 0，预测 L1 的 EI gap |
| `_scratch_j.py` | `E log|det J|` 在两种输入下的差，解释 EI_real − EI_uniform |
| `_scratch_range.py` | 三种注入（U(0,1) / 特征范围箱 / 真实样本对）下 L1 的 EI |
| `_scratch_overall.py` | 用存档权重按残差 σ 重算 Overall EI 的 old/new |

### B. 奇异值塌缩 / 噪声下限机理（6 个）

| 脚本 | 作用 |
|---|---|
| `_scratch_diag.py` | 训 1–620 轮，逐点打印 L2 的 logpdet、各奇异值、max\|z\|、min σ′ |
| `_scratch_tail.py` | 聚焦 180–610 轮：1% 分位、最小 logpdet、sv₄<1e−6 的比例 |
| `_scratch_sv5.py` | 5×5 的五个奇异值随训练变化、sv₅→0 的比例 |
| `_scratch_rows.py` | 最相似两行权重的 cos 与最小奇异值的关系 |
| `_scratch_w2.py` | W2 自身奇异值与条件数随时间 |
| `_scratch_terms.py` | epoch 185 逐项拆 term1/term2，验证 EI 符号来源 |

### C. 桌面报告排版修复（5 个）

目标文件在项目外：`D:\我的文档\OneDrive\Desktop\连续EI奇异值凹陷与噪声下限修正分析记录.md`，
作用是把 `$$…$$` 规范成块级公式、清理对中文不友好的宏、检查 `$` 配对与空行。
`_scratch_fix_md.py / _scratch_verify_md.py / _scratch_verify2.py / _scratch_gfm_check.py / _scratch_final_check.py`

---

## 注意

- 这些脚本里仍有 `sys.path.insert(0, ".")` 和相对 `results/...` 路径，**必须在包根目录运行**；
- 依赖 `iris_weights_seed42.npz` 的脚本（z / j / range / w2 / overall）已改为相对路径，
  指向 `../../transport_map_EI/data/iris_weights_seed42.npz`；
- 它们的结果不需要重跑——相关图与 CSV 都在 `results/` 下。
