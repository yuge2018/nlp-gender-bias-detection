"""Report figures for P2, generated from the acceptance evidence.

Charts use English labels only so they render on any machine without Chinese
fonts. ``make_figures`` reads data structures already present in
``configs/baseline/acceptance.json`` and never recomputes metrics.
"""
from pathlib import Path


def _pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("drawing P2 figures needs matplotlib; install requirements/p2.txt") from exc
    # Chinese character n-grams appear in the feature chart: prefer a CJK font and
    # fall back to the default family (labels then degrade, but charts still render).
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Noto Sans CJK JP", "Arial Unicode MS", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    return plt


def _save(figure, output_dir, name):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / f"{name}.png", dpi=160, bbox_inches="tight")
    figure.savefig(output_dir / f"{name}.svg", bbox_inches="tight")
    return [str(output_dir / f"{name}.png"), str(output_dir / f"{name}.svg")]


def _variant_label(config):
    ngram = "1" if list(config["ngram_range"]) == [1, 1] else "1-3"
    weight = "none" if config["class_weight"] is None else "balanced"
    return f"C={config['C']:g} {weight} {ngram}gram"


def _sweep_figure(plt, acceptance, output_dir):
    sweep = acceptance["sweeps"]["corgi-official-v1"]
    labels = [_variant_label(variant["config"]) for variant in sweep["variants"]]
    default = [variant["dev_at_default_threshold"]["metrics"]["macro_f1"] for variant in sweep["variants"]]
    tuned = [variant["threshold_selection"]["metrics"]["macro_f1"] for variant in sweep["variants"]]
    best_index = sweep["best"]["variant_index"]
    figure, axis = plt.subplots(figsize=(11, 5.2))
    positions = range(len(labels))
    width = 0.42
    axis.bar([item - width / 2 for item in positions], default, width, label="dev Macro-F1 @0.50", color="#4c72b0")
    axis.bar([item + width / 2 for item in positions], tuned, width, label="dev Macro-F1 @dev-tuned threshold", color="#dd8452")
    axis.axhline(0.5, color="#888888", linewidth=0.8, linestyle="--", label="majority-class Macro-F1 reference")
    axis.set_xticks(list(positions))
    axis.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    axis.set_ylabel("dev Macro-F1")
    axis.set_ylim(0.0, 1.06)
    axis.set_title("P2 dev-only configuration sweep (test split not used)")
    axis.annotate(f"selected: {_variant_label(sweep['best']['config'])}", xy=(best_index, max(default[best_index], tuned[best_index])), xytext=(best_index, 0.99), ha="center", va="bottom", fontsize=8, arrowprops={"arrowstyle": "->", "color": "#333333"})
    axis.legend(fontsize=8, loc="lower right")
    figure.tight_layout()
    return _save(figure, output_dir, "dev_sweep_macro_f1")


def _threshold_figure(plt, acceptance, output_dir):
    sweep = acceptance["sweeps"]["corgi-official-v1"]
    variant = sweep["variants"][sweep["best"]["variant_index"]]
    grid = variant["threshold_selection"]["grid"]
    thresholds = [row["threshold"] for row in grid]
    figure, axis = plt.subplots(figsize=(7.5, 4.6))
    axis.plot(thresholds, [row["macro_f1"] for row in grid], marker="o", markersize=3, label="Macro-F1")
    axis.plot(thresholds, [row["positive_recall"] for row in grid], marker="s", markersize=3, label="biased-class recall")
    axis.plot(thresholds, [row["positive_precision"] for row in grid], marker="^", markersize=3, label="biased-class precision")
    selected = variant["threshold_selection"]["threshold"]
    axis.axvline(selected, color="#c44e52", linestyle="--", linewidth=1, label=f"selected threshold {selected:.2f}")
    axis.axvline(0.5, color="#888888", linestyle=":", linewidth=1, label="default threshold 0.50")
    axis.set_xlabel("decision threshold")
    axis.set_ylabel("dev metric")
    axis.set_ylim(0.0, 1.02)
    axis.set_title("Dev threshold grid for the selected P2 baseline")
    axis.legend(fontsize=8, loc="best")
    figure.tight_layout()
    return _save(figure, output_dir, "dev_threshold_grid")


def _ablation_figure(plt, acceptance, output_dir):
    ablations = acceptance["ablations"]["corgi-official-v1"]
    tags = [tag for tag in ("b0-majority", "default", "b2-balanced", "b3-char1gram", "b3-balanced-char1gram") if tag in ablations]
    labels = {"b0-majority": "B0 majority", "default": "B1 1-3gram", "b2-balanced": "B2 balanced", "b3-char1gram": "B3a 1gram", "b3-balanced-char1gram": "B3b 1gram+bal"}
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    splits = {"dev": axes[0], "test": axes[1]}
    width = 0.38
    for split, axis in splits.items():
        default = [ablations[tag].get(split, {}).get("default", {}).get("macro_f1", 0.0) for tag in tags]
        tuned = [ablations[tag].get(split, {}).get("tuned", {}).get("macro_f1", 0.0) for tag in tags]
        positions = range(len(tags))
        axis.bar([item - width / 2 for item in positions], default, width, label="threshold 0.50", color="#4c72b0")
        axis.bar([item + width / 2 for item in positions], tuned, width, label="dev-tuned threshold", color="#dd8452")
        axis.axhline(0.5, color="#888888", linestyle="--", linewidth=0.8)
        axis.set_xticks(list(positions))
        axis.set_xticklabels([labels[tag] for tag in tags], rotation=25, ha="right", fontsize=8)
        axis.set_ylim(0.0, 1.0)
        axis.set_ylabel(f"{split} Macro-F1 (reference implementation)")
        axis.set_title(f"{split} ablation")
        axis.legend(fontsize=8, loc="lower right")
    figure.suptitle("P2 feature/class-weight ablations (B0-B3)", fontsize=10)
    figure.tight_layout()
    return _save(figure, output_dir, "ablation_macro_f1")


def _clean_sensitivity_figure(plt, acceptance, output_dir):
    sensitivity = acceptance.get("clean_train_sensitivity", {})
    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    labels, series_default, series_tuned = [], [], []
    for tag, per_dataset in sensitivity.items():
        for dataset in acceptance["datasets"]:
            if dataset not in per_dataset:
                continue
            labels.append(f"{tag}\n{dataset.replace('corgi-', '').replace('-v1', '')}")
            series_default.append(per_dataset[dataset]["default"]["macro_f1"]["mean"])
            series_tuned.append(per_dataset[dataset]["tuned"]["macro_f1"]["mean"])
    positions = range(len(labels))
    width = 0.38
    axis.bar([item - width / 2 for item in positions], series_default, width, label="threshold 0.50", color="#4c72b0")
    axis.bar([item + width / 2 for item in positions], series_tuned, width, label="dev-tuned threshold", color="#dd8452")
    axis.set_xticks(list(positions))
    axis.set_xticklabels(labels, fontsize=8)
    axis.set_ylabel("test Macro-F1 mean (reference implementation)")
    axis.set_ylim(0.0, 1.0)
    axis.set_title("Official split vs clean-train sensitivity analysis")
    axis.legend(fontsize=8, loc="lower right")
    figure.tight_layout()
    return _save(figure, output_dir, "clean_train_sensitivity")


def _top_features_figure(plt, acceptance, output_dir):
    features = acceptance.get("top_features", {})
    positive = features.get("positive", [])[:12]
    negative = features.get("negative", [])[:12]
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 5.0))
    for axis, rows, title, color in ((axes[0], positive, "Top biased-class character n-grams", "#c44e52"), (axes[1], negative, "Top non-biased-class character n-grams", "#4c72b0")):
        labels = [row["feature"] for row in rows][::-1]
        weights = [row["weight"] for row in rows][::-1]
        axis.barh(labels, weights, color=color)
        axis.set_title(title, fontsize=9)
        axis.tick_params(labelsize=8)
    figure.suptitle("P2 logistic-regression coefficients (lexical evidence, not causal cues)", fontsize=10)
    figure.tight_layout()
    return _save(figure, output_dir, "top_features")


def make_figures(acceptance, output_dir):
    plt = _pyplot()
    output = []
    output += _sweep_figure(plt, acceptance, output_dir)
    output += _threshold_figure(plt, acceptance, output_dir)
    output += _ablation_figure(plt, acceptance, output_dir)
    output += _clean_sensitivity_figure(plt, acceptance, output_dir)
    output += _top_features_figure(plt, acceptance, output_dir)
    plt.close("all")
    return output
