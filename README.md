# DNNsEI：深度神经网络有效信息（EI）的连续/离散双口径复现

本项目复现并扩展论文 *Quantifying the Causal Structure of Deep Neural Networks*（Entropy 2020, 22, 1429，`entropy-22-01429-v2.pdf`）的有效信息（Effective Information, EI）分析框架，跟踪 DNN 训练过程中 EI、sensitivity、degeneracy 的演化，并实现论文原版的离散（分箱互信息）口径与连续（闭式 Jacobian + 噪声近似）口径的双路线对照。

## 目录结构

```text
eicore/        核心库
               common(EPS) / models(IrisSigmoidNet, evaluate_mse) / data(IRIS 加载) /
               sigma(梯度&残差 sigma 累加器) / continuous(连续 EI+sensitivity) /
               discrete(分箱+互信息+离散层 EI) / plotting(统一绘图风格)
experiments/   实验入口（用 python -m 从根目录运行）
               iris_continuous_ei / iris_discrete_ei / mnist_continuous_ei /
               plot_iris_comparison / figure1_ei_decomposition
tests/         36 个 unittest：python -m unittest discover -s tests -t .
tools/         报告路径镜像脚本（python -m tools.publish_latest）
updates/       本轮分析新增的只读脚本（体积项拆解、MSE 诊断、kNN 熵预览等），见 updates/README.md
transport_map_EI/  第三种 EI 估计方法（传输映射）的独立包：库 + 10 个 demo + 数据，见其 README
iris_transport_map_ei/  用 TM-EI 跑 IRIS 训练全程的实验入口 + 5 个后处理 demo
iris_ei_noise_floor/    连续 EI 的"噪声下限"修正实验 + 3 个分析 demo
keypoints/     原理说明文档（连续/离散 EI 的推导、常见误区）
legacy/        旧脚本原样归档（figure1_reproduction.py、figure1_continuous_ei_L1/L3.py、
               ei.py、sensitivity_degeneracy_comparison.py、notebooks、旧日志）
data/          MNIST 原始数据（不随代码包分发）
results/       实验输出（每次运行生成时间戳目录；latest/ 为说明文档用的固定镜像，
               updates/ 为 updates 脚本的产物，均不随代码包分发）
```

参考材料：`entropy-22-01429-v2.pdf`（主论文）、`NIS+.pdf`、`ei_continuous_discrete_summary.html`（连续/离散公式口径说明）、`experiment_figures_20261022.html`（实验图与推导汇总）。

## 环境

Python 3.12。核心依赖：NumPy、Matplotlib、PyTorch、scikit-learn；`legacy/ei.py` 另需 torchmetrics。

```bash
python -m pip install numpy matplotlib torch scikit-learn torchmetrics
```

PyTorch 按本机 CUDA 环境选择版本，CPU 版即可运行全部实验。

## 复现命令

在项目根目录执行：

```bash
python -m experiments.iris_continuous_ei --outdir results                                   # ~2.5 min
python -m experiments.iris_discrete_ei --record-every 40 --ei-samples 30000 --outdir results # ~6 min
python -m experiments.figure1_ei_decomposition                                              # ~30 min
python -m experiments.plot_iris_comparison                                                  # 生成连续/离散对比图
python -m unittest discover -s tests -t .                                                   # 36 个单元测试
```

各实验脚本支持 `--help` 查看训练轮数、批量大小、随机种子、样本量、分箱数等参数。MNIST 脚本默认从 `data/MNIST/raw/` 读取 IDX 文件（需自行下载，支持 `.gz`）。

实验结束时会把本次运行的 png/csv 自动镜像到 `results/latest/<实验名>/`，说明文档（`experiment_figures_*.html`）只引用这个固定路径，所以重跑后不需要再改 HTML 里的时间戳。需要单独补刷（或换了 `--outdir`）时执行：

```bash
python -m tools.publish_latest
```

## 核心口径

### 连续版三层公式

```text
EI_full = (1/n)[ −½Σ_j log(2πe σ_j²) + Σ_i log w_i + E_x Σ_k log s_k(J) ]   J = diag(a′(Wx))·W
Sens    = (1/n)[ −½Σ_j log(2πe σ_j²) + H(X) + Σ_{i,j} E log|J_ji| ]          （约定 B，当前实现）
Deg     = Sens − EI_full
```

训练实验中 σ_{l,j} 取每层每个输出神经元的反传梯度 std；Overall EI 用输出残差 std 代入逐层公式求和；log pdet 用 SVD 奇异值。

### 离散版（论文原方法，依据论文 §3 与 Appendix A.1 自行实现）

```text
EI   = I(C;E) = H(E) − H(E|C)        （分箱 + 联合直方图，bins=b，cause 状态数 b^n）
Sens = Σ_{i,j} I(T_i;T_j)            （每次只扰动输入 i、其余固定 0，逐对 MI 求和）
Deg  = Sens − EI ≥ 0（通常）
```

论文使用 10⁷ 注入样本/测量点（或倍增收敛/曲线外推协议）；本项目默认 3×10⁴ 样本的轻量口径，可用 `--ei-samples` 加大。

### 两口径的可比性

- 离散版基于香农熵，EI/Deg 恒非负、受 log₂(bⁿ) 上限约束；连续版基于微分熵，**可以为负，零点任意**。
- 当前连续版 sensitivity 采用约定 B（term1、H(X) 只计一份再除 n），degeneracy 恒为负；离散版 degeneracy 恒为正。
- **结论：两口径只有趋势可比，符号和绝对值不可直接比。**

## 主要实验结论

- IRIS 训练（4→5→5→3，sigmoid，no bias，MSE，lr=0.01，batch=10，4000 epochs）连续/离散两版均与论文 Figure 2/4 趋势一致：H(E|C)≈0（前馈近似确定性映射）；先 sensitivity 上升、后 degeneracy 加速并抵消大部分 EI 增长；L2（中间隐藏层）路径重叠增加最多；每层 EI 增量与论文一致（约 +1.5~2 bits）。
- Figure 1 简单系统：单边 A→B 下 sens≡EI、deg≡0；sigmoid 连续闭式解 ≈ term1+log w−w/2（w*≈2 峰值）。双入单出网格上离散 deg≥0、连续 deg≤0，空间结构一致、符号相反（口径差异）。

## 被测试锁定的解析性质

- 单边：sens = EI，deg = 0；w=0 时 EI = 0。
- 2→1 某权重为 0 时离散 deg ≈ 0；恒等层 deg ≈ 0（分箱跨界噪声允许约 ±0.2 bits 偏差）。
- 连续 2→1 deg ≤ 0（约定 B）。
- deg = sens − EI 恒等式（两口径各自成立）。

## 打包说明

代码包包含 `eicore/`、`experiments/`、`tests/`、`legacy/`、参考论文 PDF、HTML 报告与本说明文件。`data/`、`results/`、`__pycache__` 不纳入归档。
