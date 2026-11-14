"""Report figures for P3, built from saved run artifacts (CPU-only, no retraining)."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .logic import THRESHOLD_GRID, labels_from_scores, macro_f1

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "reports" / "figures" / "P3"


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_predictions(run_dir):
    return _read_jsonl(Path(run_dir) / "predictions.jsonl")


def _read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _gold(split):
    return [row["label"] for row in _read_jsonl(ROOT / "data" / "processed" / "corgi-official-v1" / "labels" / f"{split}.jsonl")]


def figure_training_curves(output_dir=FIGURES):
    """Dev Macro-F1 at 0.5 per epoch, I1 vs I2 (seed 42)."""
    epochs = [1, 2, 3]
    series = {}
    for tag, label in (("improved-i1-s42", "I1 无类别加权"), ("improved-i2-s42", "I2 类别加权")):
        history = _load_json(ROOT / "artifacts" / tag / "model_metadata.json")["selection"]["history"]
        series[label] = [item["dev_macro_f1_at_0.5"] for item in history]

    fig, ax = plt.subplots(figsize=(6, 4))
    for label, values in series.items():
        ax.plot(epochs, values, marker="o", label=label)
        for epoch, value in zip(epochs, values):
            ax.annotate(f"{value:.4f}", (epoch, value), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
    ax.set_xticks(epochs)
    ax.set_xlabel("epoch")
    ax.set_ylabel("dev Macro-F1 @ 阈值0.5")
    ax.set_title("开发集逐epoch表现（seed 42）")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(output_dir / f"p3_dev_curves.{suffix}", dpi=200)
    plt.close(fig)
    return output_dir / "p3_dev_curves.png"


def figure_threshold_sweep(output_dir=FIGURES):
    """Dev Macro-F1 across the frozen threshold grid, I1 vs I2 (seed 42)."""
    gold = _gold("dev")
    fig, ax = plt.subplots(figsize=(7, 4))
    for tag, label, color in (("improved-i1-s42", "I1 无类别加权", "tab:blue"), ("improved-i2-s42", "I2 类别加权", "tab:orange")):
        scores = [row["score"] for row in _read_predictions(ROOT / "runs" / f"{tag}-dev")]
        f1s = [macro_f1(gold, labels_from_scores(scores, threshold)) for threshold in THRESHOLD_GRID]
        best = THRESHOLD_GRID[max(range(len(f1s)), key=lambda i: f1s[i])]
        ax.plot(THRESHOLD_GRID, f1s, marker=".", label=label, color=color)
        ax.axvline(best, color=color, linestyle="--", alpha=0.5)
        ax.annotate(f"best {best:.2f}", (best, max(f1s)), textcoords="offset points", xytext=(5, -10), color=color, fontsize=8)
    ax.set_xlabel("正类判定阈值")
    ax.set_ylabel("dev Macro-F1")
    ax.set_title("dev 阈值扫描（网格 0.10–0.90，步长 0.05）")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(output_dir / f"p3_threshold_sweep.{suffix}", dpi=200)
    plt.close(fig)
    return output_dir / "p3_threshold_sweep.png"


def figure_threshold_comparison(output_dir=FIGURES):
    """Test Macro-F1 at 0.5 vs the dev-selected threshold, I1 vs I2 (seed 42)."""
    bars = {}
    for tag, label in (("improved-i1-s42", "I1 无类别加权"), ("improved-i2-s42", "I2 类别加权")):
        tuned_threshold = _load_json(ROOT / "artifacts" / tag / "model_metadata.json")["selection"]["threshold"]
        tuned_matches = sorted((ROOT / "runs").glob(f"{tag}-test-thr*"))
        if len(tuned_matches) != 1:
            raise FileNotFoundError(f"{tag}: expected one tuned-threshold run dir, found {len(tuned_matches)}")
        gold = _gold("test")
        tuned_scores = [row["score"] for row in _read_predictions(tuned_matches[0])]
        bars[label] = [
            macro_f1(gold, [row["label"] for row in _read_predictions(ROOT / "runs" / f"{tag}-test")]),
            macro_f1(gold, labels_from_scores(tuned_scores, tuned_threshold)),
        ]

    fig, ax = plt.subplots(figsize=(6, 4))
    positions = range(len(bars))
    width = 0.35
    for offset, (tick, color) in enumerate(((-width / 2, "tab:green"), (width / 2, "tab:red"))):
        values = [pair[offset] for pair in bars.values()]
        names = ["阈值0.5", "dev调优阈值"] if offset == 0 else None
        ax.bar([p + tick for p in positions], values, width, label=names, color=color)
        for p, value in zip(positions, values):
            ax.annotate(f"{value:.4f}", (p + tick, value), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)
    ax.set_xticks(list(positions))
    ax.set_xticklabels(bars.keys())
    ax.set_ylim(0.75, 0.90)
    ax.set_ylabel("test Macro-F1")
    ax.set_title("T1：默认阈值 vs dev调优阈值（seed 42）")
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(output_dir / f"p3_threshold_comparison.{suffix}", dpi=200)
    plt.close(fig)
    return output_dir / "p3_threshold_comparison.png"


def figure_ablation_overview(output_dir=FIGURES):
    """Test Macro-F1 for every variant at 0.5 and the dev-selected threshold."""
    labels = {
        "improved-i1-s42": "I1 s42",
        "improved-i1-s43": "I1 s43",
        "improved-i1-s44": "I1 s44",
        "improved-i2-s42": "I2 类别加权",
        "improved-i3-len256-s42": "I3 长度256",
        "improved-l1-clean-s42": "L1 清理train",
    }
    base, tuned_labels, tuned_values = [], [], []
    base_values, tuned_thresholds = [], []
    for tag, label in labels.items():
        meta_path = ROOT / "artifacts" / tag / "model_metadata.json"
        if not meta_path.is_file():
            continue
        threshold = _load_json(meta_path)["selection"]["threshold"]
        tuned_dir = next((ROOT / "runs").glob(f"{tag}-test-thr*"), None)
        if tuned_dir is None:
            continue
        gold = _gold_test(tag)
        base.append((label, macro_f1(gold, [row["label"] for row in _read_predictions(ROOT / "runs" / f"{tag}-test")])))
        tuned_labels.append(label)
        tuned_values.append(macro_f1(gold, [row["label"] for row in _read_predictions(tuned_dir)]))
        tuned_thresholds.append(threshold)

    fig, ax = plt.subplots(figsize=(9, 4.2))
    positions = range(len(base))
    width = 0.38
    ax.bar([p - width / 2 for p in positions], [v for _, v in base], width, label="阈值0.5", color="tab:green")
    ax.bar([p + width / 2 for p in positions], tuned_values, width, label="dev调优阈值", color="tab:red")
    for p, (_, value) in zip(positions, base):
        ax.annotate(f"{value:.4f}", (p - width / 2, value), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=7)
    for p, value, thr in zip(positions, tuned_values, tuned_thresholds):
        ax.annotate(f"{value:.4f}\n@{thr:.2f}", (p + width / 2, value), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=7)
    ax.set_xticks(list(positions))
    ax.set_xticklabels([label for label, _ in base])
    ax.set_ylim(0.75, 0.92)
    ax.set_ylabel("test Macro-F1")
    ax.set_title("P3 全部变体的 test 表现（同数据版本，L1 用清理train训练）")
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(output_dir / f"p3_ablation_overview.{suffix}", dpi=200)
    plt.close(fig)
    return output_dir / "p3_ablation_overview.png"


def _gold_test(tag):
    # L1 用 clean-train 包训练，但 test 金标两个包逐字节一致，统一按 official 读。
    return [row["label"] for row in _read_jsonl(ROOT / "data" / "processed" / "corgi-official-v1" / "labels" / "test.jsonl")]


if __name__ == "__main__":
    print(figure_training_curves())
    print(figure_threshold_sweep())
    print(figure_threshold_comparison())
    print(figure_ablation_overview())
