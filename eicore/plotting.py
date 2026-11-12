import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LAYER_COLORS = {
    "L1": "#1f77b4",
    "L2": "#ff7f0e",
    "L3": "#2ca02c",
    "L4": "#d62728",
    "L5": "#9467bd",
}
TOTAL_COLOR = "#222222"


def apply_style():
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 160,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.8,
            "legend.fontsize": 9.5,
            "legend.framealpha": 0.9,
            "lines.linewidth": 2.0,
        }
    )


def layer_color(layer):
    return LAYER_COLORS.get(str(layer), None)


def layer_rows(history, layer):
    rows = [row for row in history if str(row["layer"]) == str(layer)]
    return sorted(rows, key=lambda item: item["epoch"])


def plot_layer_metric(ax, history, metric_key, total_label=None, markevery=None):
    layers = sorted({str(row["layer"]) for row in history})
    total_by_epoch = {}
    for layer in layers:
        rows = layer_rows(history, layer)
        epochs = [row["epoch"] for row in rows]
        values = [row[metric_key] for row in rows]
        every = markevery or max(1, len(rows) // 10)
        ax.plot(
            epochs,
            values,
            color=layer_color(layer),
            marker="o",
            markersize=3.5,
            markevery=every,
            label=f"{layer}",
        )
        for row in rows:
            total_by_epoch[row["epoch"]] = total_by_epoch.get(row["epoch"], 0.0) + row[metric_key]
    if total_label is not None:
        total_epochs = sorted(total_by_epoch)
        ax.plot(
            total_epochs,
            [total_by_epoch[e] for e in total_epochs],
            color=TOTAL_COLOR,
            linewidth=2.8,
            label=total_label,
        )
    ax.set_xlabel("Epoch")
    ax.legend(loc="best")


def plot_extra_series(ax, epochs, values, label):
    ax.plot(epochs, values, color=TOTAL_COLOR, linewidth=2.8, linestyle="--", label=label)
    ax.legend(loc="best")


def finalize_figure(fig, path, suptitle=None):
    if suptitle:
        fig.suptitle(suptitle, fontsize=15, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
    else:
        fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
