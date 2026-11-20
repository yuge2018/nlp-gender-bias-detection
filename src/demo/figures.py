"""Report figures built ONLY from P4's `metrics.json`.

Rules taken from docs/INTERFACES.md and docs/roles/P6-demo.md:

* Never recompute a metric here: read the frozen fields P4 wrote.
* Every figure derived from `is_mock=true` gets a `mock_` filename prefix so a
  development figure can never slip into the final report unnoticed.
* matplotlib is imported inside the plotting functions, so importing this module
  (and the contract tests) needs no plotting stack.
* Chinese text is avoided inside images; labels come from run directory names to
  dodge missing CJK fonts in CI.

CLI (works with mock metrics today, real metrics after D7):

    python -m src.demo.figures --run-dir runs/p4-smoke --output-dir reports/figures/P6
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "reports" / "figures" / "P6"

REQUIRED_FIELDS = (
    "schema_version",
    "is_mock",
    "n",
    "accuracy",
    "macro_f1",
    "positive_precision",
    "positive_recall",
    "positive_f1",
    "negative_f1",
    "confusion_matrix",
)
FLOAT_FIELDS = ("accuracy", "macro_f1", "positive_precision", "positive_recall", "positive_f1", "negative_f1")


def load_metrics(path: str | Path) -> dict:
    """Read and validate a P4 metrics file without trusting its content."""
    target = Path(path)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"metrics file not found: {target}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"metrics file is not valid JSON: {target} ({exc})") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"metrics file must contain an object: {target}")
    missing = [field for field in REQUIRED_FIELDS if field not in payload]
    if missing:
        raise ValueError(f"{target}: missing fields {missing}")
    if payload["schema_version"] != "1.0":
        raise ValueError(f"{target}: unexpected schema_version {payload['schema_version']!r}")
    if type(payload["is_mock"]) is not bool:
        raise ValueError(f"{target}: is_mock must be a boolean")
    if type(payload["n"]) is not int or payload["n"] <= 0:
        raise ValueError(f"{target}: n must be a positive integer")
    for field in FLOAT_FIELDS:
        value = payload[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"{target}: {field} must be a finite number")
    matrix = payload["confusion_matrix"]
    if not (
        isinstance(matrix, list)
        and len(matrix) == 2
        and all(isinstance(row, list) and len(row) == 2 for row in matrix)
        and all(type(cell) is int and cell >= 0 for row in matrix for cell in row)
    ):
        raise ValueError(f"{target}: confusion_matrix must be 2x2 non-negative integers")
    if sum(cell for row in matrix for cell in row) != payload["n"]:
        raise ValueError(f"{target}: confusion_matrix does not sum to n")
    return payload


def figure_name(metrics: dict, stem: str) -> str:
    """`mock_` prefix keeps development figures out of the final report."""
    return f"{'mock_' if metrics.get('is_mock') else ''}{stem}.png"


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_macro_f1(series: list[tuple[str, dict]], output_dir: str | Path = DEFAULT_OUTPUT) -> Path:
    """Horizontal Macro-F1 bars. `series` is [(label, metrics)]."""
    if not series:
        raise ValueError("plot_macro_f1 needs at least one (label, metrics) pair")
    for label, metrics in series:
        if not isinstance(label, str) or not label.strip():
            raise ValueError("plot_macro_f1 labels must be non-empty strings")
        if "macro_f1" not in metrics:
            raise ValueError(f"plot_macro_f1: {label!r} has no macro_f1")
    plt = _pyplot()
    labels = [label for label, _ in series]
    values = [float(metrics["macro_f1"]) for _, metrics in series]
    any_mock = any(bool(metrics["is_mock"]) for _, metrics in series)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / figure_name({"is_mock": any_mock}, "macro_f1_comparison")
    height = max(2.2, 0.55 * len(series) + 1.1)
    fig, ax = plt.subplots(figsize=(7.2, height))
    bars = ax.barh(range(len(series)), values, color="#4C72B0")
    ax.set_yticks(range(len(series)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Macro-F1 (from metrics.json, not recomputed)")
    ax.set_title("Macro-F1 comparison" + (" [MOCK]" if any_mock else ""))
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    for bar, value in zip(bars, values):
        ax.text(min(value + 0.01, 0.98), bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(target, dpi=160)
    plt.close(fig)
    return target


def plot_confusion(metrics: dict, output_dir: str | Path = DEFAULT_OUTPUT, label: str = "run") -> Path:
    """2x2 confusion matrix heatmap with counts (rows/cols are 0/1)."""
    matrix = metrics.get("confusion_matrix")
    if matrix is None:
        raise ValueError("plot_confusion needs confusion_matrix")
    plt = _pyplot()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / figure_name(metrics, f"confusion_{label}")
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    image = ax.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            ax.text(column, row, str(matrix[row][column]), ha="center", va="center", fontsize=12, color="black")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["pred 0", "pred 1"])
    ax.set_yticklabels(["true 0", "true 1"])
    ax.set_title(f"Confusion matrix ({label})" + (" [MOCK]" if metrics.get("is_mock") else ""))
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(target, dpi=160)
    plt.close(fig)
    return target


def build_figures(run_dirs: list[str | Path], output_dir: str | Path = DEFAULT_OUTPUT) -> dict:
    """Read `<run>/metrics.json` for each run and emit comparison + per-run matrices."""
    runs = []
    for run_dir in run_dirs:
        run_path = Path(run_dir)
        metrics_path = run_path / "metrics.json"
        metrics = load_metrics(metrics_path)
        runs.append((run_path.name, metrics))
    if not runs:
        raise ValueError("build_figures needs at least one run directory")
    produced = {"macro_f1": str(plot_macro_f1(runs, output_dir))}
    for label, metrics in runs:
        produced[f"confusion_{label}"] = str(plot_confusion(metrics, output_dir, label=label))
    return produced


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build P6 report figures from P4 metrics.json")
    parser.add_argument("--run-dir", action="append", required=True, help="run directory containing metrics.json; repeatable")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)
    try:
        produced = build_figures(args.run_dir, args.output_dir)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(produced, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
