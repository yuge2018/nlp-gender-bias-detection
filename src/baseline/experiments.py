"""Reproduce the frozen P2 experiment matrix and assemble shareable evidence.

One command trains the dev sweep, the seed runs and the feature ablations on both
frozen data versions, writes run directories through the public CLI helpers, then
builds ``configs/baseline/acceptance.json`` and the report figures.

Reference metrics written here use the frozen ``src.common.contracts.binary_metrics``
implementation and are explicitly *not* P4's official evaluation output; P4 must
recompute metrics from the run directories.
"""
import json
import statistics
from pathlib import Path

from src.common.contracts import binary_metrics, read_jsonl, write_json

from . import cli
from .config import load_config, load_sweep_config
from .data import load_training_data
from .model import load_model, repository_state
from .sweep import run_sweep

DATASETS = ("corgi-official-v1", "corgi-clean-train-v1")
ABLATIONS = ("b0-majority", "default", "b2-balanced", "b3-char1gram", "b3-balanced-char1gram")
FINAL_SEEDS = (42, 43, 44)
SWEEP_SEED = 42
SUMMARY_METRICS = ("macro_f1", "positive_precision", "positive_recall", "positive_f1", "negative_f1", "accuracy")
REFERENCE_METRIC_NOTE = "Metrics in this file come from the frozen reference implementation src/common/contracts.binary_metrics for P2's own evidence. P4 must recompute all official metrics from the run directories; these numbers are not P4 output."


def _resolve(root, path):
    target = Path(path)
    return target if target.is_absolute() else Path(root) / target


def _rel(root, path):
    return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()


def _mean_std(values):
    values = [float(value) for value in values]
    if not values:
        return {"n": 0, "mean": None, "std": None, "values": []}
    return {"n": len(values), "mean": sum(values) / len(values), "std": statistics.stdev(values) if len(values) > 1 else None, "values": values}


def _read_gold(data_dir, split):
    return read_jsonl(Path(data_dir) / "labels" / f"{split}.jsonl")


def _run_reference_metrics(run_dir, gold):
    predictions = read_jsonl(Path(run_dir) / "predictions.jsonl")
    return binary_metrics(gold, predictions)


def _model_evidence(model_dir):
    metadata = json.loads((Path(model_dir) / "model_metadata.json").read_text(encoding="utf-8"))
    return {"train_seconds": metadata["training"]["train_seconds"], "vocabulary_size": metadata["feature"]["vocabulary_size"], "threshold": metadata["threshold"], "dev_reference_metrics": metadata["dev"]["reference_metrics"], "threshold_selection": metadata["threshold_selection"], "environment": metadata["environment"], "code_commit": metadata["code_commit"], "code_dirty": metadata["code_dirty"]}


def _train_and_score(*, root, data_root, out_root, artifacts_root, configs_dir, dataset, tag, config_name, seed, select_threshold, gold):
    data_dir = Path(data_root) / dataset
    config_path = Path(configs_dir) / config_name
    config = load_config(config_path)
    model_dir = Path(artifacts_root) / dataset / f"{tag}-s{seed}"
    train_summary = cli.train_command(data_dir, config_path, model_dir, seed=seed, select_threshold=select_threshold)
    evidence = _model_evidence(model_dir)
    runs = []
    for split in ("dev", "test"):
        input_path = data_dir / f"{split}.jsonl"
        for mode, threshold in (("default", config["threshold"]), ("tuned", None)):
            run_name = f"{tag}-s{seed}-{split}" + ("" if mode == "default" else "-tuned")
            run_dir = Path(out_root) / dataset / run_name
            predict_summary = cli.predict_command(input_path, model_dir, data_dir / "manifest.json", run_dir, threshold=threshold)
            metrics = _run_reference_metrics(run_dir, gold[(dataset, split)])
            runs.append({"run_dir": _rel(root, run_dir), "model_dir": _rel(root, model_dir), "dataset_id": dataset, "tag": tag, "split": split, "seed": seed, "mode": mode, "threshold": predict_summary["threshold"], "is_mock": predict_summary["is_mock"], "reference_metrics": metrics, "model": {"train_seconds": evidence["train_seconds"], "vocabulary_size": evidence["vocabulary_size"], "threshold": evidence["threshold"]}})
    return train_summary, evidence, runs


def _ablation_table(runs, dataset):
    table = {}
    for tag in ABLATIONS:
        entry = {}
        for split in ("dev", "test"):
            for mode in ("default", "tuned"):
                matches = [run for run in runs if run["dataset_id"] == dataset and run["tag"] == tag and run["split"] == split and run["mode"] == mode]
                if matches:
                    entry.setdefault(split, {})[mode] = matches[0]["reference_metrics"]
        model = [run for run in runs if run["dataset_id"] == dataset and run["tag"] == tag]
        if model:
            entry["train_seconds"] = model[0]["model"]["train_seconds"]
            entry["vocabulary_size"] = model[0]["model"]["vocabulary_size"]
        table[tag] = entry
    return table


def _seed_summary(runs, dataset, tag):
    summary = {}
    for mode in ("default", "tuned"):
        selected = [run for run in runs if run["dataset_id"] == dataset and run["tag"] == tag and run["split"] == "test" and run["mode"] == mode]
        summary[mode] = {metric: _mean_std([run["reference_metrics"][metric] for run in selected]) for metric in SUMMARY_METRICS}
        summary[mode]["n_seeds"] = len(selected)
    return summary


def run_experiments(root, data_dir, output_dir, artifacts_dir, configs_dir, acceptance_path, figures_dir):
    root = Path(root)
    data_root = _resolve(root, data_dir)
    out_root = _resolve(root, output_dir)
    artifacts_root = _resolve(root, artifacts_dir)
    configs_root = _resolve(root, configs_dir)
    acceptance_file = _resolve(root, acceptance_path) if acceptance_path else configs_root / "acceptance.json"
    figures_root = _resolve(root, figures_dir)
    if out_root.exists() and (not out_root.is_dir() or any(out_root.iterdir())):
        raise ValueError(f"{out_root}: experiment output directory is not empty; choose a fresh path")
    out_root.mkdir(parents=True, exist_ok=True)
    selected = {key: value for key, value in load_config(configs_root / "selected.json").items() if key != "config_path"}
    _, grid, variants, sweep_config_path = load_sweep_config(configs_root / "sweep.json")

    loaded, datasets_info, gold = {}, {}, {}
    for dataset in DATASETS:
        data = load_training_data(data_root / dataset)
        loaded[dataset] = data
        train_positive = sum(row["label"] for row in data["train"])
        datasets_info[dataset] = {"dataset_id": data["manifest"]["dataset_id"], "manifest_sha256": data["manifest_sha256"], "source_commit": data["manifest"]["source_commit"], "rows": {"train": len(data["train"]), "dev": len(data["dev"]), "test": data["manifest"]["files"]["test.jsonl"]["rows"]}, "train_positive": train_positive, "train_positive_rate": train_positive / len(data["train"])}
        for split in ("dev", "test"):
            gold[(dataset, split)] = _read_gold(data_root / dataset, split)

    sweeps = {}
    for dataset in DATASETS:
        sweep_dir = out_root / dataset / "sweep"
        result = run_sweep(loaded[dataset], variants, grid, SWEEP_SEED)
        write_json(sweep_dir / "sweep.json", result)
        (sweep_dir / "log.txt").write_text("P2 dev-only sweep\n", encoding="utf-8")
        sweeps[dataset] = result

    runs, train_summaries = [], {}
    for dataset in DATASETS:
        for seed in FINAL_SEEDS:
            summary, evidence, produced = _train_and_score(root=root, data_root=data_root, out_root=out_root, artifacts_root=artifacts_root, configs_dir=configs_root, dataset=dataset, tag="selected", config_name="selected.json", seed=seed, select_threshold=True, gold=gold)
            runs.extend(produced)
            train_summaries[f"selected-{dataset}-s{seed}"] = {key: summary[key] for key in ("model_dir", "seed", "threshold", "dataset_id", "dataset_manifest_sha256", "is_mock", "vocabulary_size", "dev_reference_metrics", "model_name", "model_revision", "code_commit", "code_dirty")}
            train_summaries[f"selected-{dataset}-s{seed}"]["threshold_selection"] = evidence["threshold_selection"]
        for tag in ABLATIONS:
            summary, evidence, produced = _train_and_score(root=root, data_root=data_root, out_root=out_root, artifacts_root=artifacts_root, configs_dir=configs_root, dataset=dataset, tag=tag, config_name=f"{tag}.json", seed=SWEEP_SEED, select_threshold=True, gold=gold)
            runs.extend(produced)
            train_summaries[f"{tag}-{dataset}-s{SWEEP_SEED}"] = {key: summary[key] for key in ("model_dir", "seed", "threshold", "dataset_id", "dataset_manifest_sha256", "is_mock", "vocabulary_size", "dev_reference_metrics", "model_name", "model_revision", "code_commit", "code_dirty")}
            train_summaries[f"{tag}-{dataset}-s{SWEEP_SEED}"]["threshold_selection"] = evidence["threshold_selection"]

    model, metadata = load_model(artifacts_root / "corgi-official-v1" / f"selected-s{FINAL_SEEDS[0]}")
    top_features = model.top_features(20)
    top_features["caveat"] = "Character n-gram weights are lexical associations on the frozen training split, not a complete or causal list of bias cues; review context and negation before reading them as evidence."
    top_features["model_name"] = metadata["model_name"]
    top_features["seed"] = FINAL_SEEDS[0]

    acceptance = {
        "schema_version": "1.0",
        "is_mock": False,
        "owner": "cengjingdejiyi",
        "description": "P2 CPU character TF-IDF + logistic-regression baseline acceptance evidence.",
        "reference_metric_note": REFERENCE_METRIC_NOTE,
        "commands": [
            "python -m src.baseline experiments --data-dir data/processed --output-dir runs/p2 --artifacts-dir artifacts/p2 --configs-dir configs/baseline --figures-dir reports/figures/P2",
            "python -m unittest discover -s tests -v",
            "python -m src.baseline --mock --output-dir runs/p2-smoke",
        ],
        "sweep_command_example": f"python -m src.baseline sweep --data-dir data/processed/corgi-official-v1 --config {_rel(root, sweep_config_path)} --output-dir runs/p2/corgi-official-v1/sweep",
        "datasets": datasets_info,
        "selection_rule": sweeps["corgi-official-v1"]["selection"]["rule"],
        "selected_config": selected,
        "sweeps": sweeps,
        "train_summaries": train_summaries,
        "runs": runs,
        "seed_summary": {dataset: _seed_summary(runs, dataset, "selected") for dataset in DATASETS},
        "ablations": {dataset: _ablation_table(runs, dataset) for dataset in DATASETS},
        "clean_train_sensitivity": {tag: {dataset: _seed_summary(runs, dataset, tag) for dataset in DATASETS} for tag in ("selected", "default")},
        "top_features": top_features,
        "limitations": [
            "Lexical character n-gram features cannot represent context, negation or quoted rebuttals reliably.",
            "The frozen official split contains exact cross-split duplicate texts; the clean-train version is a separate sensitivity analysis and must not be mixed into one table.",
            "class_weight=balanced changes the positive/negative trade-off; it is not guaranteed to raise Macro-F1.",
            "Logistic regression on a convex objective is deterministic here, so seeds 42/43/44 can produce identical scores; the seed spread then reports zero variance honestly.",
            "Scores are model scores for label=1, not calibrated probabilities of social bias.",
            "No target-gender gold labels exist in the training data; per-group fairness is P5's separate challenge-set analysis.",
        ],
        "environment": metadata["environment"],
    }
    commits = sorted({summary["code_commit"] for summary in train_summaries.values()})
    acceptance["code"] = {"run_code_commits": commits, "code_dirty": any(summary["code_dirty"] for summary in train_summaries.values())}
    write_json(acceptance_file, acceptance)

    from .figures import make_figures

    figure_summary = make_figures(acceptance, figures_root)
    return {"acceptance": _rel(root, acceptance_file), "sweeps": {dataset: _rel(root, out_root / dataset / "sweep" / "sweep.json") for dataset in DATASETS}, "runs": len(runs), "selected_config": selected, "seed_summary": acceptance["seed_summary"], "figures": figure_summary, "code": acceptance["code"]}
