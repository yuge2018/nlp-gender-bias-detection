"""P4 评测 CLI。

用法：
    python -m src.evaluation --mock --output-dir runs/p4-smoke
    python -m src.evaluation evaluate --data-dir DATA --split test --run-dir RUN [--output-dir DIR]
    python -m src.evaluation compare --run-dirs RUN1 RUN2 ... --output-dir reports/results [--allow-mock]

退出码：0 成功；2 参数/契约/数据错误；1 其他运行失败。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from src.common.contracts import (
    SCHEMA_VERSION,
    ContractError,
    read_json,
    sha256_file,
    write_json,
)
from src.evaluation.aggregate import METRIC_KEYS, aggregate
from src.evaluation.checks import (
    ensure_comparable,
    validate_data_package,
    validate_run,
)
from src.evaluation.fixtures import build_fixtures
from src.evaluation.metrics import compute_metrics

EVALUATOR_VERSION = "1.0.0"
COMPARE_COLUMNS = ("method", "data_hash", "split", "seed", "is_mock", "n", "threshold",
                   "accuracy", "macro_f1", "positive_precision", "positive_recall",
                   "positive_f1", "negative_f1", "tn", "fp", "fn", "tp",
                   "run_dir", "predictions_sha256")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


def evaluate_run(package: dict, run_dir: str | Path, output_dir: Path | None = None) -> dict:
    """评测单个 run，写出 metrics.json，返回结果记录。"""
    checked = validate_run(run_dir, package)
    meta, preds = checked["meta"], checked["preds"]

    y_true = package["labels"]
    y_pred = [row["label"] for row in preds]
    metrics = compute_metrics(y_true, y_pred)
    tn, fp = metrics["confusion_matrix"][0]
    fn, tp = metrics["confusion_matrix"][1]

    pred_path = Path(run_dir) / "predictions.jsonl"
    record = {
        "schema_version": SCHEMA_VERSION,
        "is_mock": meta["is_mock"],
        "method": meta["method"],
        "split": meta["split"],
        "seed": meta["seed"],
        "threshold": float(meta["threshold"]),
        "n": metrics["n"],
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "positive_precision": metrics["positive_precision"],
        "positive_recall": metrics["positive_recall"],
        "positive_f1": metrics["positive_f1"],
        "negative_f1": metrics["negative_f1"],
        "confusion_matrix": metrics["confusion_matrix"],
        "dataset_manifest_sha256": meta["dataset_manifest_sha256"],
        "predictions_sha256": sha256_file(pred_path),
        "run_dir": str(Path(run_dir)),
        "created_at": _now(),
        "evaluator": {"name": "p4-evaluation", "version": EVALUATOR_VERSION},
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }
    write_json(Path(run_dir) / "metrics.json", record)
    if output_dir is not None:
        name = f"{meta['method']}-s{meta['seed']}-{meta['split']}.json"
        write_json(Path(output_dir) / "metrics" / name, record)
    return record


def _write_csv(path: Path, columns: tuple[str, ...], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in columns})


def _fmt(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _markdown(records: list[dict], agg: list[dict]) -> str:
    lines = ["# P4 评测结果", ""]
    mock = any(r["is_mock"] for r in records)
    if mock:
        lines += ["> **注意：以下全部为 mock 结果，仅用于接口验证，不能作为模型结论。**", ""]

    lines += ["## 单次运行", "",
              "| method | seed | n | accuracy | macro-F1 | P(1) | R(1) | F1(1) | F1(0) |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in sorted(records, key=lambda x: (x["method"], x["seed"])):
        lines.append(
            f"| {r['method']} | {r['seed']} | {r['n']} | {_fmt(r['accuracy'])} | "
            f"{_fmt(r['macro_f1'])} | {_fmt(r['positive_precision'])} | "
            f"{_fmt(r['positive_recall'])} | {_fmt(r['positive_f1'])} | "
            f"{_fmt(r['negative_f1'])} |")

    lines += ["", "## 多种子汇总（均值 ± 样本标准差）", "",
              "| method | runs | macro-F1 | accuracy | P(1) | R(1) |",
              "|---|---:|---|---|---|---|"]

    def mean_std(row, key):
        mean, std = row[f"{key}_mean"], row[f"{key}_std"]
        return _fmt(mean) if std is None else f"{mean:.4f} ± {std:.4f}"

    for row in agg:
        lines.append(
            f"| {row['method']} | {row['n_runs']} | {mean_std(row, 'macro_f1')} | "
            f"{mean_std(row, 'accuracy')} | {mean_std(row, 'positive_precision')} | "
            f"{mean_std(row, 'positive_recall')} |")
    lines += ["", "混淆矩阵行是真值 0/1，列是预测 0/1。", ""]
    return "\n".join(lines)


def compare(run_dirs: list[str], output_dir: str | Path, data_dir: str,
            split: str, allow_mock: bool) -> list[dict]:
    package = validate_data_package(data_dir, split)
    records = [evaluate_run(package, rd, Path(output_dir)) for rd in run_dirs]
    ensure_comparable(records)

    if any(r["is_mock"] for r in records) and not allow_mock:
        raise ContractError("待比较的结果包含 mock，正式比较必须拒绝；确需联调请加 --allow-mock")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "comparison.csv", COMPARE_COLUMNS, records)

    agg = aggregate(records)
    agg_cols = ("method", "split", "is_mock", "data_hash", "n_runs", "seeds")
    agg_cols += tuple(f"{k}_{s}" for k in METRIC_KEYS for s in ("mean", "std"))
    _write_csv(out / "aggregate.csv", agg_cols, agg)

    (out / "comparison.md").write_text(_markdown(records, agg), encoding="utf-8")

    write_json(out / "environment.json", {
        "schema_version": SCHEMA_VERSION,
        "created_at": _now(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cwd": os.getcwd(),
        "argv": sys.argv,
        "git_commit": _git_commit(),
        "evaluator": {"name": "p4-evaluation", "version": EVALUATOR_VERSION},
        "data_dir": str(Path(data_dir)),
        "dataset_manifest_sha256": package["manifest_sha256"],
        "is_mock": package["manifest"]["is_mock"],
    })
    write_json(out / "reproduction.json", {
        "schema_version": SCHEMA_VERSION,
        "created_at": _now(),
        "command": " ".join([sys.executable, "-m", "src.evaluation"] + sys.argv[1:]),
        "data_dir": str(Path(data_dir)),
        "dataset_manifest_sha256": package["manifest_sha256"],
        "split": split,
        "runs": [{"run_dir": r["run_dir"], "method": r["method"], "seed": r["seed"],
                  "predictions_sha256": r["predictions_sha256"]} for r in records],
        "outputs": ["comparison.csv", "aggregate.csv", "comparison.md",
                    "environment.json", "metrics/"],
    })
    return records


def run_smoke(output_dir: str | Path) -> int:
    fixtures = build_fixtures(Path("fixtures"))
    data_dir = str(fixtures["data_dir"])
    run_dirs = [str(p) for p in fixtures["run_dirs"]]
    records = compare(run_dirs, output_dir, data_dir, "test", allow_mock=True)

    best = max(records, key=lambda r: r["macro_f1"])
    print(f"[mock] 数据包：{data_dir}")
    print(f"[mock] 评测 {len(records)} 个 run，结果写入 {output_dir}")
    print(f"[mock] 最高 Macro-F1：{best['method']} seed={best['seed']} "
          f"{best['macro_f1']:.4f}")
    print("提示：这些结果全部来自合成 fixtures，不能写进报告主表。")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.evaluation",
        description="P4 统一评测：契约检查、指标计算、多种子汇总")
    parser.add_argument("--mock", action="store_true",
                        help="生成内置 fixtures 并跑通评测/对比全流程")
    parser.add_argument("--output-dir", default="runs/p4-smoke",
                        help="mock 模式输出目录（默认 runs/p4-smoke）")
    sub = parser.add_subparsers(dest="command")

    p_eval = sub.add_parser("evaluate", help="评测单个 run")
    p_eval.add_argument("--data-dir", required=True)
    p_eval.add_argument("--split", default="test", choices=("dev", "test"))
    p_eval.add_argument("--run-dir", required=True)
    p_eval.add_argument("--output-dir", default=None,
                        help="可选：把 metrics.json 另存到该目录的 metrics/ 下")

    p_cmp = sub.add_parser("compare", help="比较多个 run 并生成结果表")
    p_cmp.add_argument("--run-dirs", nargs="+", required=True)
    p_cmp.add_argument("--data-dir", required=True)
    p_cmp.add_argument("--split", default="test", choices=("dev", "test"))
    p_cmp.add_argument("--output-dir", default="reports/results")
    p_cmp.add_argument("--allow-mock", action="store_true",
                       help="允许 mock 结果参与比较（仅联调用）")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "evaluate":
            package = validate_data_package(args.data_dir, args.split)
            record = evaluate_run(package, args.run_dir,
                                  Path(args.output_dir) if args.output_dir else None)
            print(f"{record['method']} seed={record['seed']} split={record['split']} "
                  f"n={record['n']} macro_f1={record['macro_f1']:.4f} "
                  f"accuracy={record['accuracy']:.4f}")
            return 0

        if args.command == "compare":
            records = compare(args.run_dirs, args.output_dir, args.data_dir,
                              args.split, args.allow_mock)
            for rec in sorted(records, key=lambda r: (r["method"], r["seed"])):
                print(f"{rec['method']:<10} seed={rec['seed']:<3} "
                      f"macro_f1={rec['macro_f1']:.4f} accuracy={rec['accuracy']:.4f}")
            print(f"结果已写入 {args.output_dir}")
            return 0

        if args.mock:
            return run_smoke(args.output_dir)

        parser.print_help()
        print("\n错误：未指定子命令；开发联调用 --mock。", file=sys.stderr)
        return 2
    except ContractError as exc:
        print(f"契约错误：{exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001
        print(f"运行失败：{exc}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
