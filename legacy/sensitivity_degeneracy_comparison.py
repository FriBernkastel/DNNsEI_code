"""Compare discrete and continuous sensitivity/degeneracy on a 2-to-1 network.

The experiment follows the project's Figure 1 conventions. It scans the
symmetric path w_A = w_B = w, estimates single-edge EI contributions and the
joint layer EI, then defines degeneracy as sensitivity minus joint EI.
"""

import csv
import html
import io
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from figure1_continuous_ei_L1 import (
    EPS,
    apply_output_clip_to_jacobian,
    decompose_ei,
    get_sampling_bounds,
    relu_prime,
    sigmoid_prime,
    tanh_prime,
    to_bits,
)
from figure1_reproduction import (
    causal_metrics_from_encoded_states,
    discretize_uniform,
    encode_states,
)


ACTIVATIONS = {
    "sigmoid": sigmoid_prime,
    "tanh": tanh_prime,
    "relu": relu_prime,
}
BINS_LIST = (8, 16, 32, 64)
SIGMA_LIST = (0.01, 0.05, 0.1)
WEIGHTS = np.linspace(0.2, 9.0, 45)
N_SAMPLES = 100_000
SEED = 20261015
CENTRAL_BINS = 32
CENTRAL_SIGMA = 0.05


def activation_value(name, z):
    if name == "sigmoid":
        return 1.0 / (1.0 + np.exp(-z))
    if name == "tanh":
        return np.tanh(z)
    return np.maximum(0.0, z)


def empirical_mi_bits(cause, effect, n_cause_states):
    return causal_metrics_from_encoded_states(
        cause.astype(np.int64), effect.astype(np.int64), n_cause_states
    )["ei_bits"]


def save_rows(rows, path):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_experiment():
    rows = []
    rng = np.random.default_rng(SEED)

    for activation, derivative in ACTIVATIONS.items():
        x_low, x_high, y_low, y_high = get_sampling_bounds(activation)
        width = x_high - x_low
        a = rng.uniform(x_low, x_high, size=N_SAMPLES).astype(np.float64)
        b = rng.uniform(x_low, x_high, size=N_SAMPLES).astype(np.float64)
        edge_z = WEIGHTS[:, None] * a[None, :]
        full_z = WEIGHTS[:, None] * (a + b)[None, :]

        for bins in BINS_LIST:
            a_codes = discretize_uniform(a, x_low, x_high, bins)
            b_codes = discretize_uniform(b, x_low, x_high, bins)
            full_cause = encode_states(np.stack([a_codes, b_codes], axis=1), bins)
            n_cause_states = bins**2

            for idx, weight in enumerate(WEIGHTS):
                # Paper-style sensitivity: assess each edge independently while
                # holding the other input at the baseline value zero.
                edge_y = activation_value(activation, weight * a)
                edge_codes = discretize_uniform(
                    edge_y, y_low, y_high, bins, clip=(activation != "relu")
                )
                edge_ei = empirical_mi_bits(a_codes, edge_codes, bins)
                discrete_sensitivity = 2.0 * edge_ei

                full_y = activation_value(activation, weight * (a + b))
                full_codes = discretize_uniform(
                    full_y, y_low, y_high, bins, clip=(activation != "relu")
                )
                discrete_full_ei = empirical_mi_bits(
                    full_cause, full_codes, n_cause_states
                )
                rows.append(
                    {
                        "activation": activation,
                        "method": "discrete",
                        "setting": f"bins={bins}",
                        "bins": bins,
                        "sigma": "",
                        "weight": float(weight),
                        "edge_ei_bits": float(edge_ei),
                        "sensitivity_bits": float(discrete_sensitivity),
                        "full_ei_bits": float(discrete_full_ei),
                        "degeneracy_bits": float(
                            discrete_sensitivity - discrete_full_ei
                        ),
                    }
                )

        for sigma in SIGMA_LIST:
            for idx, weight in enumerate(WEIGHTS):
                # With the other input fixed at zero, each edge is a scalar map.
                edge_jac = weight * derivative(edge_z[idx])
                edge_jac = apply_output_clip_to_jacobian(
                    activation, edge_z[idx], edge_jac
                )
                edge_log_jac = float(
                    np.mean(np.log(np.maximum(np.abs(edge_jac), EPS)))
                )
                _, _, edge_ei_nats = decompose_ei(
                    edge_log_jac,
                    n_in=1,
                    n_out=1,
                    input_width=width,
                    sigma=sigma,
                )
                edge_ei_bits = to_bits(edge_ei_nats)
                continuous_sensitivity = 2.0 * edge_ei_bits

                z = full_z[idx]
                sp = derivative(z)
                ja = apply_output_clip_to_jacobian(activation, z, weight * sp)
                jb = apply_output_clip_to_jacobian(activation, z, weight * sp)
                pseudo_det = np.sqrt(ja * ja + jb * jb)
                full_log_pdet = float(
                    np.mean(np.log(np.maximum(np.abs(pseudo_det), EPS)))
                )
                _, _, full_ei_nats = decompose_ei(
                    full_log_pdet,
                    n_in=2,
                    n_out=1,
                    input_width=width,
                    sigma=sigma,
                )
                continuous_full_ei = to_bits(full_ei_nats)
                rows.append(
                    {
                        "activation": activation,
                        "method": "continuous",
                        "setting": f"sigma={sigma:g}",
                        "bins": "",
                        "sigma": sigma,
                        "weight": float(weight),
                        "edge_ei_bits": float(edge_ei_bits),
                        "sensitivity_bits": float(continuous_sensitivity),
                        "full_ei_bits": float(continuous_full_ei),
                        "degeneracy_bits": float(
                            continuous_sensitivity - continuous_full_ei
                        ),
                    }
                )

    return rows


def rows_for(rows, activation, method, setting):
    return sorted(
        [
            row
            for row in rows
            if row["activation"] == activation
            and row["method"] == method
            and row["setting"] == setting
        ],
        key=lambda row: row["weight"],
    )


def rankdata(values):
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.float64)
    return ranks


def corr(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if np.std(a) == 0.0 or np.std(b) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def shape_metrics(rows, activation, metric):
    continuous = rows_for(
        rows, activation, "continuous", f"sigma={CENTRAL_SIGMA:g}"
    )
    discrete = rows_for(rows, activation, "discrete", f"bins={CENTRAL_BINS}")
    c = np.asarray([row[metric] for row in continuous], dtype=np.float64)
    d = np.asarray([row[metric] for row in discrete], dtype=np.float64)
    dc = np.diff(c)
    dd = np.diff(d)
    peak_c = float(continuous[int(np.argmax(c))]["weight"])
    peak_d = float(discrete[int(np.argmax(d))]["weight"])
    return {
        "activation": activation,
        "metric": metric.replace("_bits", ""),
        "pearson_level": corr(c, d),
        "spearman_level": corr(rankdata(c), rankdata(d)),
        "pearson_slope": corr(dc, dd),
        "continuous_peak_weight": peak_c,
        "discrete_peak_weight": peak_d,
    }


def normalize_curve(values):
    values = np.asarray(values, dtype=np.float64)
    lo = float(np.min(values))
    hi = float(np.max(values))
    if math.isclose(lo, hi):
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


def render_figure(rows, normalized=False):
    fig, axes = plt.subplots(3, 3, figsize=(17, 12), constrained_layout=True)
    metrics = ("full_ei_bits", "sensitivity_bits", "degeneracy_bits")
    methods = (
        ("continuous", "#2563eb", f"Continuous (σ={CENTRAL_SIGMA:g})"),
        ("discrete", "#d97706", f"Discrete (B={CENTRAL_BINS})"),
    )
    setting_lists = {
        "continuous": [f"sigma={sigma:g}" for sigma in SIGMA_LIST],
        "discrete": [f"bins={bins}" for bins in BINS_LIST],
    }

    for row_idx, activation in enumerate(ACTIVATIONS):
        for col_idx, metric in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            for method, color, label in methods:
                setting_curves = []
                x_values = None
                for setting in setting_lists[method]:
                    selected = rows_for(rows, activation, method, setting)
                    x_values = np.asarray([item["weight"] for item in selected])
                    values = np.asarray(
                        [item[metric] for item in selected], dtype=np.float64
                    )
                    if normalized:
                        values = normalize_curve(values)
                    setting_curves.append(values)
                matrix = np.stack(setting_curves, axis=0)
                ax.fill_between(
                    x_values,
                    np.min(matrix, axis=0),
                    np.max(matrix, axis=0),
                    color=color,
                    alpha=0.14,
                    linewidth=0,
                    label=f"{method} parameter range",
                )
                central_setting = (
                    f"sigma={CENTRAL_SIGMA:g}"
                    if method == "continuous"
                    else f"bins={CENTRAL_BINS}"
                )
                central = rows_for(rows, activation, method, central_setting)
                central_values = np.asarray(
                    [item[metric] for item in central], dtype=np.float64
                )
                if normalized:
                    central_values = normalize_curve(central_values)
                ax.plot(x_values, central_values, color=color, lw=2.2, label=label)

            title = metric.replace("_bits", "").replace("_", " ").title()
            ax.set_title(f"{activation.title()} · {title}", loc="left", weight="bold")
            ax.set_xlabel("Symmetric edge weight w (wA = wB)")
            ax.set_ylabel("Normalized shape [0, 1]" if normalized else "bits")
            ax.grid(alpha=0.22)
            if row_idx == 0 and col_idx == 0:
                ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.suptitle(
        "Sensitivity and degeneracy: continuous vs discrete"
        + (" (shape normalized)" if normalized else " (raw values)"),
        fontsize=15,
        weight="bold",
    )
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    plt.close(fig)
    svg = buf.getvalue()
    start = svg.find("<svg")
    return svg[start:]


def format_number(value, digits=3):
    if not np.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def summary_table(rows):
    body = []
    for activation in ACTIVATIONS:
        for metric in ("full_ei_bits", "sensitivity_bits", "degeneracy_bits"):
            stats = shape_metrics(rows, activation, metric)
            body.append(
                "<tr>"
                f"<td>{html.escape(activation)}</td>"
                f"<td>{html.escape(stats['metric'])}</td>"
                f"<td>{format_number(stats['pearson_level'])}</td>"
                f"<td>{format_number(stats['spearman_level'])}</td>"
                f"<td>{format_number(stats['pearson_slope'])}</td>"
                f"<td>{stats['continuous_peak_weight']:.2f}</td>"
                f"<td>{stats['discrete_peak_weight']:.2f}</td>"
                "</tr>"
            )
    return "\n".join(body)


def findings_html(rows):
    items = []
    for activation in ACTIVATIONS:
        full = shape_metrics(rows, activation, "full_ei_bits")
        sensitivity = shape_metrics(rows, activation, "sensitivity_bits")
        degeneracy = shape_metrics(rows, activation, "degeneracy_bits")
        note = ""
        if activation == "relu":
            note = "本项目 ReLU 的输出/Jacobian 口径不同，相关性只描述当前两段代码的结果。"
        elif activation == "sigmoid" and degeneracy["spearman_level"] < 0.6:
            note = "整层 EI 与敏感度对得较好，但简并度曲线排序差异明显；连续端最大值落在扫描上界。"
        elif activation == "tanh":
            note = "整层 EI、敏感度和简并度的曲线排序均高度一致。"
        items.append(
            "<li><b>"
            + html.escape(activation.title())
            + ":</b> 整层 EI ρ="
            + format_number(full["spearman_level"])
            + "，敏感度 ρ="
            + format_number(sensitivity["spearman_level"])
            + "，简并度 ρ="
            + format_number(degeneracy["spearman_level"])
            + "。"
            + html.escape(note)
            + "</li>"
        )
    return "\n".join(items)


def build_html(rows, raw_svg, normalized_svg):
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>连续与离散 EI 的敏感度、简并度曲线对比</title>
  <style>
    :root {{ color-scheme: light; --ink:#172033; --muted:#526078; --line:#dbe2ec; --panel:#fff; --bg:#f4f7fb; --blue:#2563eb; --orange:#d97706; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--bg); color:var(--ink); font:15px/1.65 "Segoe UI","Microsoft YaHei",sans-serif; }}
    main {{ width:min(1180px, calc(100% - 32px)); margin:32px auto 60px; }}
    header, section {{ background:var(--panel); border:1px solid var(--line); border-radius:16px; padding:24px 28px; margin:18px 0; box-shadow:0 6px 24px #182b4a0a; }}
    h1 {{ margin:0 0 8px; font-size:30px; line-height:1.25; }} h2 {{ margin:0 0 14px; font-size:21px; }} h3 {{ margin:18px 0 8px; font-size:16px; }}
    p {{ margin:8px 0; }} .muted {{ color:var(--muted); }} .formula {{ background:#f1f5fb; padding:14px 16px; border-radius:10px; overflow:auto; font-family:Consolas,monospace; }}
    table {{ width:100%; border-collapse:collapse; font-size:13px; }} th,td {{ padding:9px 10px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; }} th {{ background:#f5f8fc; }}
    .chart svg {{ display:block; width:100%; height:auto; }} .tag {{ display:inline-block; padding:3px 9px; border-radius:99px; background:#eaf0fb; color:#29466f; font-size:12px; margin-right:6px; }}
    .callout {{ border-left:4px solid var(--orange); background:#fff8ed; padding:12px 16px; border-radius:8px; }} .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
    code {{ background:#eef2f7; padding:1px 5px; border-radius:4px; }} footer {{ text-align:center; color:var(--muted); padding:18px; font-size:12px; }}
    @media(max-width:760px) {{ main {{ width:min(100% - 16px, 1180px); margin-top:8px; }} header,section {{ padding:18px; }} h1 {{ font-size:24px; }} .grid {{ grid-template-columns:1fr; }} table {{ display:block; overflow:auto; white-space:nowrap; }} }}
  </style>
</head>
<body><main>
  <header>
    <span class="tag">2 输入 → 1 输出</span><span class="tag">对称权重路径</span><span class="tag">固定随机种子 {SEED}</span>
    <h1>连续与离散 EI：敏感度和简并度曲线</h1>
    <p class="muted">沿 wA = wB = w 扫描权重。每个权重点都计算边单独作用的 EI 总和、整层 EI，以及二者之差。</p>
  </header>

  <section>
    <h2>定义与计算口径</h2>
    <div class="formula">Sensitivity S(w) = EI(edge A → C | B fixed at 0) + EI(edge B → C | A fixed at 0)<br>
Degeneracy D(w) = S(w) − EI_full(A, B → C)<br>
Therefore: EI_full(w) = S(w) − D(w)</div>
    <p>对称设置下两个 edgewise EI 相同，所以 S(w) = 2 × EI_single(w)。离散端每个 EI 是分箱后的经验互信息；连续端沿用项目的高斯噪声项与 Jacobian 对数体积近似。</p>
    <p>离散端报告 bits，使用 B = 8、16、32、64；连续端报告 bits，使用 σ = 0.01、0.05、0.1。输入样本数为 {N_SAMPLES:,}，权重范围为 0.2–9.0，共 {len(WEIGHTS)} 点。绘图中心设置为 B = {CENTRAL_BINS} 与 σ = {CENTRAL_SIGMA:g}。</p>
  </section>

    <section><h2>原始数值曲线</h2><p class="muted">每行对应一个激活函数，三列依次为整层 EI、敏感度、简并度。实线为中心设置；阴影表示改变 bins 或 σ 后的参数范围，不是置信区间。连续与离散的绝对值可能存在方法偏移，重点看曲线形状及趋势统计。</p><div class="chart">{raw_svg}</div></section>
  <section><h2>各自归一化后的趋势</h2><p class="muted">每条设置曲线分别线性缩放到 [0, 1]，用于查看峰值位置和整体升降形状，不代表 EI 大小可直接比较。</p><div class="chart">{normalized_svg}</div></section>

  <section>
    <h2>中心设置下的趋势统计</h2>
    <p>中心设置（B = {CENTRAL_BINS}, σ = {CENTRAL_SIGMA:g}）的主要观察：</p>
    <ul>{findings_html(rows)}</ul>
    <p>Spearman 相关比较曲线取值的排序趋势；一阶差分 Pearson 相关比较相邻权重点的升降方向与幅度。表中列的是扫描网格内的最大值位置；若落在边界，代表峰值可能在扫描范围之外。</p>
    <table><thead><tr><th>激活</th><th>指标</th><th>曲线 Pearson</th><th>曲线 Spearman</th><th>斜率 Pearson</th><th>连续最大值 w</th><th>离散最大值 w</th></tr></thead><tbody>{summary_table(rows)}</tbody></table>
  </section>

  <section>
    <h2>如何解读</h2>
    <ul>
      <li><b>敏感度</b>表示两条边分别单独作用时 EI 的加和；它不等于现有离散脚本输出的 determinism。</li>
      <li><b>简并度</b>按论文关系定义为敏感度减整层 EI，反映边独立贡献与联合输入作用之间的差。</li>
      <li>连续曲线的敏感度与简并度继承 Jacobian/高斯噪声近似及其数值下限 EPS；简并度是由两个估计量相减得到。连续近似可能给出负值，而经验互信息非负，因此绝对数值不能直接视为同一种估计量。</li>
      <li>离散端使用有限样本的 plug-in 互信息估计。B = 64 时 cause 联合状态空间有 4096 个状态，100,000 个样本仍可能有稀疏偏差；阴影显示 bins 参数敏感性，不是置信区间。</li>
      <li>整层 EI 面板用作对照：若它的趋势相近而敏感度/简并度不同，差异来自“逐边相加再相减”的分解，而不是原 EI 的对照本身。</li>
    </ul>
    <div class="callout"><b>ReLU 口径提示：</b>当前离散脚本对 ReLU 输出不做 [0,1] clip；连续 Figure 1 脚本则在 z ≥ 1 时把 Jacobian 人为设为 1。ReLU 面板按项目现有实现计算，两个方法不是严格同一输出处理口径。Sigmoid 与 tanh 的输出分箱范围与连续 Jacobian 计算口径更直接对应。</div>
  </section>

  <section>
    <h2>数据文件</h2>
    <p>本报告由实验脚本生成。完整逐点结果见同目录的 <code>sensitivity_degeneracy_curves.csv</code>；列包含激活函数、方法、参数设置、权重、edgewise EI、敏感度、整层 EI 和简并度。</p>
  </section>
  <footer>DNNsEI · Sensitivity / Degeneracy experiment · {SEED}</footer>
</main></body></html>"""


def main():
    out_dir = Path("results") / "sensitivity_degeneracy_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = run_experiment()
    csv_path = out_dir / "sensitivity_degeneracy_curves.csv"
    save_rows(rows, csv_path)

    raw_svg = render_figure(rows, normalized=False)
    normalized_svg = render_figure(rows, normalized=True)
    (out_dir / "curves_raw.svg").write_text(raw_svg, encoding="utf-8")
    (out_dir / "curves_normalized.svg").write_text(normalized_svg, encoding="utf-8")
    html_path = out_dir / "sensitivity_degeneracy_comparison.html"
    html_path.write_text(build_html(rows, raw_svg, normalized_svg), encoding="utf-8")

    print(f"rows={len(rows)}")
    print(f"csv={csv_path.resolve()}")
    print(f"html={html_path.resolve()}")
    for activation in ACTIVATIONS:
        for metric in ("sensitivity_bits", "degeneracy_bits"):
            stats = shape_metrics(rows, activation, metric)
            print(
                activation,
                stats["metric"],
                "spearman=", format_number(stats["spearman_level"]),
                "slope_r=", format_number(stats["pearson_slope"]),
                "peaks=", f'{stats["continuous_peak_weight"]:.2f}/{stats["discrete_peak_weight"]:.2f}',
            )


if __name__ == "__main__":
    main()
