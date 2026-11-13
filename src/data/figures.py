"""Static data-audit charts; no model metrics or demographic inference."""
from pathlib import Path


def make_figures(summary, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stats = summary["packages"]["corgi-official-v1"]["audit"]["split_statistics"]
    names = ["train", "dev", "test"]
    positive = [stats[name]["positive_n"] for name in names]
    negative = [stats[name]["negative_n"] for name in names]
    total = [a + b for a, b in zip(positive, negative)]
    colors = ("#3176A8", "#D98D39")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "svg.fonttype": "none", "svg.hashsalt": "p1-data-v1"})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
    axes[0].barh(names, negative, label="No bias (0)", color=colors[0])
    axes[0].barh(names, positive, left=negative, label="Bias (1)", color=colors[1])
    for index, n in enumerate(total):
        axes[0].text(n + 350, index, f"{n:,}", va="center", fontsize=10)
    axes[0].set(xlabel="Samples", title="Official detection samples", xlim=(0, max(total) * 1.17))
    rates = [p / n for p, n in zip(positive, total)]
    axes[1].barh(names, [1 - p for p in rates], color=colors[0])
    axes[1].barh(names, rates, left=[1 - p for p in rates], color=colors[1])
    for index, p in enumerate(rates):
        axes[1].text(0.5, index, f"Bias: {p:.2%}", ha="center", va="center", color="white", fontweight="bold")
    axes[1].xaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1].set(xlabel="Class share", title="Class balance differs by split", xlim=(0, 1))
    for axis in axes:
        axis.invert_yaxis()
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(loc="lower right")
    for extension in ("png", "svg"):
        fig.savefig(directory / f"class_balance.{extension}", dpi=170, metadata={"Date": None} if extension == "svg" else {})
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 4.4), layout="constrained")
    xs = [0, 1, 2]
    for offset, threshold in enumerate((128, 256, 512)):
        percentages = [stats[name]["characters"]["over_threshold"][str(threshold)] / stats[name]["n"] for name in names]
        bars = axis.bar([x + (offset - 1) * 0.25 for x in xs], percentages, width=0.24, label=f"> {threshold} characters")
        axis.bar_label(bars, labels=[f"{p:.2%}" for p in percentages], padding=3, fontsize=9)
    axis.set_xticks(xs, names)
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.set(ylabel="Share of split", title="Long-text diagnostics (characters, not tokenizer tokens)", ylim=(0, max(0.1, axis.get_ylim()[1] * 1.2)))
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend()
    for extension in ("png", "svg"):
        fig.savefig(directory / f"length_thresholds.{extension}", dpi=170, metadata={"Date": None} if extension == "svg" else {})
    plt.close(fig)
    for name in ("class_balance", "length_thresholds"):
        path = directory / f"{name}.svg"
        normalized = "\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()) + "\n"
        path.write_text(normalized, encoding="utf-8", newline="\n")
