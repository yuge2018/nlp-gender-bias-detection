"""Report figures for P5: challenge composition and grouped metrics.

Matplotlib is imported lazily so the mock entry and the contract tests still run
with the standard library only. Figure labels stay in English to avoid missing
CJK fonts; the surrounding report text is Chinese.
"""
from pathlib import Path


def _matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError("drawing P5 figures needs matplotlib; install requirements/p5.txt") from exc
    return plt


def draw_composition(coverage, out_path):
    """Two panels: per-target polarity counts and per-phenomenon counts."""
    plt = _matplotlib()
    targets = list(coverage["by_target"])
    series = (("non_biased", "0"), ("biased", "1"))
    figure, axes = plt.subplots(1, 2, figsize=(13, 0.45 * max(len(targets), len(coverage["by_phenomenon"])) + 2.5))
    x = range(len(targets))
    width = 0.38
    for offset, (label, key) in enumerate(series):
        values = [coverage["by_target"][target][key] for target in targets]
        axes[0].bar([i + offset * width for i in x], values, width, label=label)
        for position, value in zip(x, values):
            axes[0].text(position + offset * width, value + 0.4, str(value), ha="center", va="bottom", fontsize=8)
    axes[0].set_xticks([i + width / 2 for i in x])
    axes[0].set_xticklabels(targets, fontsize=9)
    axes[0].set_title("Challenge rows per target (multi-label allowed)")
    axes[0].legend(fontsize=8)

    phenomena = list(coverage["by_phenomenon"])
    y = range(len(phenomena))
    for offset, (label, key) in enumerate(series):
        values = [coverage["by_phenomenon"][item][key] for item in phenomena]
        axes[1].barh([i + offset * width for i in y], values, width, label=label)
    axes[1].set_yticks([i + width / 2 for i in y])
    axes[1].set_yticklabels(phenomena, fontsize=8)
    axes[1].invert_yaxis()
    axes[1].set_title("Challenge rows per phenomenon")
    figure.tight_layout()
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=200)
    plt.close(figure)
    return str(target)


def draw_group_metrics(table, out_path, model_name):
    """Grouped macro-F1 / FPR / FNR per annotated target for one model."""
    plt = _matplotlib()
    rows = [entry for entry in table if entry["dimension"] == "target" and entry["n"] > 0]
    groups = [entry["group"] for entry in rows]
    figure, axes = plt.subplots(1, 3, figsize=(13, 0.4 * len(groups) + 3), sharey=False)
    specs = (("macro_f1", "Macro-F1 (fixed 0/1 average)"), ("fpr", "FPR (non-biased flagged as biased)"), ("fnr", "FNR (biased missed)"))
    positions = range(len(groups))
    for axis, (key, title) in zip(axes, specs):
        values = [entry["results"][model_name][key] for entry in rows]
        plotted = [0.0 if value is None else value for value in values]
        axis.barh(list(positions), plotted, color="#3f7fbf")
        axis.set_yticks(list(positions))
        axis.set_yticklabels(groups if axis is axes[0] else [""] * len(groups), fontsize=9)
        axis.set_xlim(0, 1)
        axis.set_title(title, fontsize=10)
        for position, entry, value in zip(positions, rows, values):
            text = "null" if value is None else f"{value:.2f} (n={entry['n']})"
            axis.text((value or 0) + 0.02, position, text, va="center", fontsize=8)
    figure.suptitle(f"{model_name}: challenge subgroup metrics (small n is descriptive only)", fontsize=11)
    figure.tight_layout()
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=200)
    plt.close(figure)
    return str(target)
