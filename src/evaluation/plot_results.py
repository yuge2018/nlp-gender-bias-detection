# -*- coding: utf-8 -*-
"""把 aggregate.csv 画成 Macro-F1 均值±标准差柱状图。

用法：python -m src.evaluation.plot_results --aggregate reports/results/aggregate.csv \
        --output reports/figures/P4/macro_f1_by_method.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", default="reports/results/aggregate.csv")
    parser.add_argument("--output", default="reports/figures/P4/macro_f1_by_method.png")
    args = parser.parse_args()

    with open(args.aggregate, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit("aggregate.csv 为空")

    labels = [f"{r['method']}\n(seed {r['seeds']})" for r in rows]
    means = [float(r["macro_f1_mean"]) for r in rows]
    stds = [float(r["macro_f1_std"]) if r["macro_f1_std"] not in ("", "None") else 0.0
            for r in rows]
    is_mock = any(r["is_mock"] == "True" for r in rows)

    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    bars = ax.bar(range(len(rows)), means, yerr=stds, capsize=6,
                  color=["#7f9cc0", "#c08552", "#8aa37b"][:len(rows)])
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        ax.text(bar.get_x() + bar.get_width() / 2, min(mean + std + 0.02, 0.97),
                f"{mean:.3f}±{std:.3f}", ha="center", fontsize=9)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    title = "Macro-F1（均值 ± 样本标准差）"
    if is_mock:
        title += "  [MOCK]"
    ax.set_title(title)
    fig.tight_layout()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
