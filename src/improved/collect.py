"""Collect all P3 run metrics into one table (CPU-only; uses the contract reference metrics)."""
import json
from pathlib import Path

from src.common.contracts import binary_metrics, read_jsonl

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"

# (tag, data_dir, dev-threshold run suffix) — suffix read from model metadata at collection time.
VARIANTS = [
    ("improved-i1-s42", "corgi-official-v1"),
    ("improved-i1-s43", "corgi-official-v1"),
    ("improved-i1-s44", "corgi-official-v1"),
    ("improved-i2-s42", "corgi-official-v1"),
    ("improved-i3-len256-s42", "corgi-official-v1"),
    ("improved-l1-clean-s42", "corgi-clean-train-v1"),
]


def metrics_for(data_dir, split, run_dir):
    gold = read_jsonl(DATA / data_dir / "labels" / f"{split}.jsonl")
    preds = read_jsonl(ROOT / run_dir / "predictions.jsonl")
    return binary_metrics(gold, preds)


def main():
    rows = []
    for tag, data_dir in VARIANTS:
        model_dir = ROOT / "artifacts" / tag
        if not (model_dir / "model_metadata.json").is_file():
            print(f"-- {tag}: not trained yet, skipped")
            continue
        meta = json.loads((model_dir / "model_metadata.json").read_text(encoding="utf-8"))
        threshold = meta["selection"]["threshold"]
        best_epoch = meta["selection"]["best_epoch"]
        tuned_dir_name = next((ROOT / "runs").glob(f"{tag}-test-thr*"), None)
        dev = metrics_for(data_dir, "dev", ROOT / "runs" / f"{tag}-dev")
        test = metrics_for(data_dir, "test", ROOT / "runs" / f"{tag}-test")
        tuned = metrics_for(data_dir, "test", tuned_dir_name) if tuned_dir_name else None
        rows.append((tag, best_epoch, threshold, dev, test, tuned))
        tuned_str = f"{tuned['macro_f1']:.4f}@{threshold:.2f}" if tuned else "-"
        print(
            f"{tag}: best_ep={best_epoch} dev={dev['macro_f1']:.4f} | "
            f"test {test['macro_f1']:.4f}@0.5 / {tuned_str} | "
            f"posR {test['positive_recall']:.4f}->{tuned['positive_recall']:.4f}" if tuned else ""
        )

    # S1 aggregate over the three I1 seeds at the default threshold.
    seeds = [row for row in rows if row[0].startswith("improved-i1-s4")]
    if len(seeds) == 3:
        values = [row[4]["macro_f1"] for row in seeds]
        mean = sum(values) / 3
        std = (sum((v - mean) ** 2 for v in values) / 2) ** 0.5  # 样本标准差 n-1
        print(f"\nI1 test Macro-F1@0.5 over seeds 42/43/44: {values[0]:.4f} {values[1]:.4f} {values[2]:.4f} -> mean {mean:.4f} ± {std:.4f}")


if __name__ == "__main__":
    main()
