"""P4 评测执行：单 run 打分、跨 run 比较与复现记录。

指标定义直接复用 src/common/contracts.py（P4 维护的冻结契约），
本模块只负责编排、对比表、聚合与复现记录，不导入任何模型实现。
"""
from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.common.contracts import (
    VERSION,
    binary_metrics,
    load_dataset,
    read_jsonl,
    sha256,
    validate_predictions,
    write_json,
    write_jsonl,
)
from src.common.model_api import MockPredictor

from .aggregate import METRIC_KEYS, aggregate

ROOT = Path(__file__).resolve().parents[2]
EVALUATOR_VERSION = "1.0.0"

COMPARISON_COLUMNS = ("method", "data_hash", "split", "seed", "is_mock", "n",
                      "accuracy", "macro_f1", "positive_precision", "positive_recall",
                      "positive_f1", "negative_f1", "tn", "fp", "fn", "tp",
                      "threshold", "run_dir", "predictions_sha256")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_commit() -> str | None:
    try:
        done = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=5, cwd=ROOT)
        return done.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def evaluate_run(data_dir: str | Path, split: str, run_dir: str | Path) -> dict:
    """校验一个 run 并把 metrics.json 写回 run 目录，返回记录。"""
    if split not in ("dev", "test"):
        raise ValueError(f"split={split!r} 不支持，应为 dev 或 test")

    data_path = Path(data_dir)
    run_path = Path(run_dir)
    _, dataset = load_dataset(data_path)
    inputs = dataset[f"{split}.jsonl"]
    gold = dataset[f"labels/{split}.jsonl"]

    predictions_path = run_path / "predictions.jsonl"
    metadata_path = run_path / "run_metadata.json"
    if not predictions_path.is_file():
        raise ValueError(f"{predictions_path}: 文件不存在")
    if not metadata_path.is_file():
        raise ValueError(f"{metadata_path}: 文件不存在")

    predictions = read_jsonl(predictions_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    validate_predictions(inputs, predictions, metadata, data_path / "manifest.json")

    metrics = binary_metrics(gold, predictions)
    record = {
        "schema_version": VERSION,
        "is_mock": metadata["is_mock"],
        "method": metadata["method"],
        "split": split,
        "seed": metadata["seed"],
        "threshold": float(metadata["threshold"]),
        "n": metrics["n"],
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "positive_precision": metrics["positive_precision"],
        "positive_recall": metrics["positive_recall"],
        "positive_f1": metrics["positive_f1"],
        "negative_f1": metrics["negative_f1"],
        "confusion_matrix": metrics["confusion_matrix"],
        "dataset_manifest_sha256": metadata["dataset_manifest_sha256"],
        "predictions_sha256": sha256(predictions_path),
        "run_dir": str(run_path),
        "created_at": _now(),
        "evaluator": {"name": "p4-evaluation", "version": EVALUATOR_VERSION},
    }
    write_json(run_path / "metrics.json", record)
    return record


def load_metrics(run_dir: str | Path) -> dict:
    path = Path(run_dir) / "metrics.json"
    if not path.is_file():
        raise ValueError(f"{path}: 缺少 metrics.json，请先执行 evaluate")
    return json.loads(path.read_text(encoding="utf-8"))


def _check_comparable(records: list[dict]) -> None:
    hashes = {record["dataset_manifest_sha256"] for record in records}
    if len(hashes) > 1:
        raise ValueError(f"待比较的 run 使用了不同数据版本：{sorted(hashes)}")
    splits = {record["split"] for record in records}
    if len(splits) > 1:
        raise ValueError(f"待比较的 run 使用了不同 split：{sorted(splits)}")
    mocks = {bool(record["is_mock"]) for record in records}
    if len(mocks) > 1:
        raise ValueError("待比较的 run 混合了真实结果与 mock 结果，禁止比较")


def _write_csv(path: Path, columns: tuple[str, ...], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: ("null" if row.get(key) is None else row.get(key))
                             for key in columns})


def _format(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _markdown(records: list[dict], groups: list[dict]) -> str:
    lines = ["# P4 评测结果", ""]
    if any(record["is_mock"] for record in records):
        lines += ["> 全部为 mock 结果，仅用于接口验证，不能作为模型性能结论。", ""]

    lines += ["## 单次运行", "",
              "| method | threshold | seed | n | accuracy | macro-F1 | P(1) | R(1) | F1(1) | F1(0) |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for record in sorted(records, key=lambda item: (item["method"], item["threshold"],
                                                    item["seed"])):
        lines.append(
            f"| {record['method']} | {_format(record['threshold'])} | {record['seed']} | "
            f"{record['n']} | "
            f"{_format(record['accuracy'])} | {_format(record['macro_f1'])} | "
            f"{_format(record['positive_precision'])} | {_format(record['positive_recall'])} | "
            f"{_format(record['positive_f1'])} | {_format(record['negative_f1'])} |")

    lines += ["", "## 多种子汇总（均值 ± 样本标准差，按阈值分组）", "",
              "| method | threshold | runs | macro-F1 | accuracy | P(1) | R(1) |",
              "|---|---:|---:|---|---|---|---|"]

    def mean_std(group: dict, metric: str) -> str:
        mean, std = group[f"{metric}_mean"], group[f"{metric}_std"]
        return _format(mean) if std is None else f"{mean:.4f} ± {std:.4f}"

    for group in groups:
        lines.append(
            f"| {group['method']} | {_format(group['threshold'])} | {group['n_runs']} | "
            f"{mean_std(group, 'macro_f1')} | "
            f"{mean_std(group, 'accuracy')} | {mean_std(group, 'positive_precision')} | "
            f"{mean_std(group, 'positive_recall')} |")

    lines += ["", "混淆矩阵行是真值 0/1，列是预测 0/1，即 [[TN, FP], [FN, TP]]。", ""]
    return "\n".join(lines)


def compare(run_dirs: list[str], output_dir: str | Path, allow_mock: bool = False) -> list[dict]:
    """读取各 run 的 metrics.json，校验可比性并写出对比与复现产物。"""
    if not run_dirs:
        raise ValueError("没有可比较的 run")
    records = [load_metrics(run_dir) for run_dir in run_dirs]
    _check_comparable(records)
    if records[0]["is_mock"] and not allow_mock:
        raise ValueError("正式比较拒绝 mock 结果；确需联调请加 --allow-mock")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for record in records:
        (tn, fp), (fn, tp) = record["confusion_matrix"]
        rows.append({
            "method": record["method"],
            "data_hash": record["dataset_manifest_sha256"],
            "split": record["split"],
            "seed": record["seed"],
            "is_mock": record["is_mock"],
            "n": record["n"],
            "threshold": record["threshold"],
            "accuracy": record["accuracy"],
            "macro_f1": record["macro_f1"],
            "positive_precision": record["positive_precision"],
            "positive_recall": record["positive_recall"],
            "positive_f1": record["positive_f1"],
            "negative_f1": record["negative_f1"],
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "run_dir": record["run_dir"],
            "predictions_sha256": record["predictions_sha256"],
        })
    _write_csv(out / "comparison.csv", COMPARISON_COLUMNS, rows)

    groups = aggregate(records)
    group_columns = ("method", "split", "is_mock", "data_hash", "threshold",
                     "n_runs", "seeds")
    group_columns += tuple(f"{metric}_{suffix}" for metric in METRIC_KEYS
                           for suffix in ("mean", "std"))
    _write_csv(out / "aggregate.csv", group_columns, groups)

    (out / "comparison.md").write_text(_markdown(records, groups), encoding="utf-8")

    write_json(out / "environment.json", {
        "schema_version": VERSION,
        "created_at": _now(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cwd": os.getcwd(),
        "argv": sys.argv,
        "git_commit": _git_commit(),
        "evaluator": {"name": "p4-evaluation", "version": EVALUATOR_VERSION},
        "dataset_manifest_sha256": records[0]["dataset_manifest_sha256"],
        "is_mock": records[0]["is_mock"],
    })
    write_json(out / "reproduction.json", {
        "schema_version": VERSION,
        "created_at": _now(),
        "command": " ".join([Path(sys.executable).name, "-m", "src.evaluation"] + sys.argv[1:]),
        "dataset_manifest_sha256": records[0]["dataset_manifest_sha256"],
        "split": records[0]["split"],
        "runs": [{
            "run_dir": record["run_dir"],
            "method": record["method"],
            "seed": record["seed"],
            "predictions_sha256": record["predictions_sha256"],
            "metrics_sha256": sha256(Path(record["run_dir"]) / "metrics.json"),
        } for record in records],
        "outputs": ["comparison.csv", "aggregate.csv", "comparison.md",
                    "environment.json", "reproduction.json"],
    })
    return rows


def _prepare_fresh_dir(output_dir: Path) -> Path:
    out = output_dir.resolve()
    fixture = (ROOT / "fixtures").resolve()
    if out == fixture or fixture in out.parents:
        raise ValueError("输出目录不能位于 fixtures/ 内")
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"输出目录非空：{out}")
    out.mkdir(parents=True, exist_ok=True)
    return out


def run_mock(output_dir: str | Path) -> list[Path]:
    """用共享 mock 替身生成两个 mock run，跑通 evaluate 与 compare。"""
    out = _prepare_fresh_dir(Path(output_dir))
    fixture = ROOT / "fixtures" / "dataset"
    _, dataset = load_dataset(fixture)
    inputs = dataset["test.jsonl"]
    scores = MockPredictor().predict_proba([row["text"] for row in inputs])

    run_dirs = []
    for seed in (42, 43):
        run_path = out / f"mock-s{seed}-test"
        run_path.mkdir()
        predictions = [{"id": row["id"], "label": int(score >= 0.5), "score": score}
                       for row, score in zip(inputs, scores)]
        metadata = {
            "schema_version": VERSION,
            "method": "mock-evaluation",
            "split": "test",
            "dataset_manifest_sha256": sha256(fixture / "manifest.json"),
            "is_mock": True,
            "seed": seed,
            "threshold": 0.5,
            "code_commit": "MOCK",
            "model_name": "constant-0.5-development-substitute",
            "model_revision": "mock-v1",
            "config": {"mock": True},
            "environment": {"python": sys.version.split()[0]},
        }
        validate_predictions(inputs, predictions, metadata, fixture / "manifest.json")
        write_jsonl(run_path / "predictions.jsonl", predictions)
        write_json(run_path / "run_metadata.json", metadata)
        write_json(run_path / "config.json", {"mock": True})
        (run_path / "log.txt").write_text(
            "MOCK ONLY: development substitute, no detection capability.\n", encoding="utf-8")
        evaluate_run(fixture, "test", run_path)
        run_dirs.append(run_path)

    compare([str(path) for path in run_dirs], out, allow_mock=True)
    return run_dirs
