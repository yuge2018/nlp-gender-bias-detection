"""Read-only browser of the run artifacts already committed to `runs/`.

Purpose: the demo must be able to show REAL evidence even when the model
weights are not in the repository (`.gitignore` keeps large files out).

Strict rules:
* metrics are read from P4's `metrics.json`; this module never recomputes them;
* `is_mock=true` runs are excluded by default, so a development run can never be
  mistaken for a real result;
* a run with `predictions.jsonl` but no `metrics.json` is reported as
  "predictions only" instead of inventing numbers;
* a corrupt metrics file is reported, not silently dropped.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.demo import figures

ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = ROOT / "runs"

META_FILE = "run_metadata.json"
PREDICTIONS_FILE = "predictions.jsonl"
METRICS_FILE = "metrics.json"


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _read_metadata(run_dir: Path) -> dict:
    meta_path = run_dir / META_FILE
    if not meta_path.is_file():
        return {}
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _run_row(run_dir: Path, metrics: dict, root: Path) -> dict:
    meta = _read_metadata(run_dir)
    return {
        "run": _relative(run_dir, root),
        "method": meta.get("method") or "未记录",
        "split": meta.get("split") or metrics.get("split") or "test",
        "seed": meta.get("seed"),
        "threshold": meta.get("threshold"),
        "model_name": meta.get("model_name"),
        "n": metrics.get("n"),
        "accuracy": metrics.get("accuracy"),
        "macro_f1": metrics.get("macro_f1"),
        "positive_f1": metrics.get("positive_f1"),
        "negative_f1": metrics.get("negative_f1"),
        "confusion_matrix": metrics.get("confusion_matrix"),
        "is_mock": metrics.get("is_mock"),
        "data_hash": meta.get("dataset_manifest_sha256"),
    }


def _sort_key(row: dict) -> tuple:
    seed = row.get("seed")
    threshold = row.get("threshold")
    return (
        str(row.get("method")),
        str(row.get("run")),
        seed if isinstance(seed, int) else -1,
        threshold if isinstance(threshold, (int, float)) else -1.0,
    )


def discover_runs(root: str | Path | None = None, *, include_mock: bool = False) -> dict:
    """Scan for committed runs and classify them without recomputing anything."""
    base = Path(root) if root else RUNS_DIR
    rows: list[dict] = []
    predictions_only: list[str] = []
    invalid: list[dict] = []
    skipped_mock = 0

    if base.is_dir():
        for metrics_path in sorted(base.rglob(METRICS_FILE)):
            run_dir = metrics_path.parent
            try:
                metrics = figures.load_metrics(metrics_path)
            except ValueError as exc:
                invalid.append({"run": _relative(run_dir, base), "error": str(exc)})
                continue
            if bool(metrics.get("is_mock")) and not include_mock:
                skipped_mock += 1
                continue
            rows.append(_run_row(run_dir, metrics, base))

        for pred_path in sorted(base.rglob(PREDICTIONS_FILE)):
            run_dir = pred_path.parent
            if (run_dir / METRICS_FILE).is_file():
                continue
            predictions_only.append(_relative(run_dir, base))

    rows.sort(key=_sort_key)
    return {
        "root": str(base),
        "runs": rows,
        "predictions_only": predictions_only,
        "invalid": invalid,
        "skipped_mock": skipped_mock,
    }


def selected_run(result: dict, run_path: str) -> dict | None:
    """Look up one run row by its path (used by the page's detail panel)."""
    for row in result["runs"]:
        if row["run"] == run_path:
            return row
    return None


def as_display_rows(rows: list[dict]) -> list[dict]:
    """Round for display only; the stored values stay untouched."""
    display: list[dict[str, Any]] = []
    for row in rows:
        display.append(
            {
                "run": row["run"],
                "method": row["method"],
                "seed": row["seed"],
                "threshold": row["threshold"],
                "n": row["n"],
                "Macro-F1": round(float(row["macro_f1"]), 4) if isinstance(row["macro_f1"], (int, float)) else None,
                "正类F1": round(float(row["positive_f1"]), 4) if isinstance(row["positive_f1"], (int, float)) else None,
                "负类F1": round(float(row["negative_f1"]), 4) if isinstance(row["negative_f1"], (int, float)) else None,
                "mock": row["is_mock"],
            }
        )
    return display
