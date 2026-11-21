"""多种子聚合：保留每次原始指标，另算均值与样本标准差。"""
from __future__ import annotations

import statistics

METRIC_KEYS = ("accuracy", "macro_f1", "positive_precision",
               "positive_recall", "positive_f1", "negative_f1")


def aggregate(records: list[dict]) -> list[dict]:
    """按 (method, split, is_mock, data_hash, threshold) 分组汇总。

    阈值必须进分组键：同一模型在 dev 上调出的不同阈值属于不同决策口径，
    合并会把阈值差异误报成跨 seed 方差。单次 run 的标准差为 None（写表时记作 null）。
    """
    groups: dict[tuple, list[dict]] = {}
    for record in records:
        key = (record["method"], record["split"], bool(record["is_mock"]),
               record["dataset_manifest_sha256"], round(float(record["threshold"]), 6))
        groups.setdefault(key, []).append(record)

    rows = []
    for (method, split, is_mock, data_hash, threshold), items in sorted(groups.items()):
        items = sorted(items, key=lambda item: item["seed"])
        row = {
            "method": method,
            "split": split,
            "is_mock": is_mock,
            "data_hash": data_hash,
            "threshold": threshold,
            "n_runs": len(items),
            "seeds": " ".join(str(item["seed"]) for item in items),
        }
        for metric in METRIC_KEYS:
            values = [float(item[metric]) for item in items]
            row[f"{metric}_mean"] = statistics.fmean(values)
            row[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else None
        rows.append(row)
    return rows
