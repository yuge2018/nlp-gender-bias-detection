"""多种子聚合：保留每次原始指标，另算均值与样本标准差。"""

from __future__ import annotations

import statistics

METRIC_KEYS = ("accuracy", "macro_f1", "positive_precision",
               "positive_recall", "positive_f1", "negative_f1")


def aggregate(records: list[dict]) -> list[dict]:
    """按 (method, split, is_mock, data_hash) 分组，对每个指标算 mean/std。

    单个 run 时 std 为 None（JSON null），不做任何填充。
    """
    groups: dict[tuple, list[dict]] = {}
    for rec in records:
        key = (rec["method"], rec["split"], rec["is_mock"], rec["dataset_manifest_sha256"])
        groups.setdefault(key, []).append(rec)

    rows = []
    for (method, split, is_mock, data_hash), items in sorted(groups.items()):
        items = sorted(items, key=lambda r: r["seed"])
        row = {
            "method": method,
            "split": split,
            "is_mock": is_mock,
            "data_hash": data_hash,
            "n_runs": len(items),
            "seeds": [r["seed"] for r in items],
        }
        for metric in METRIC_KEYS:
            values = [float(r[metric]) for r in items]
            row[f"{metric}_mean"] = statistics.fmean(values)
            row[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else None
        rows.append(row)
    return rows
