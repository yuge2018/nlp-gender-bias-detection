"""P2 command line.

Exit codes follow the frozen interface contract: ``0`` success, ``2`` argument,
contract or data error, ``1`` training/loading/runtime failure.
"""
import argparse
import json
import sys
import time
from pathlib import Path

from src.common.contracts import VERSION, validate_predictions, write_json, write_jsonl

from .config import config_snapshot, load_config, load_sweep_config
from .data import load_inference_input, load_manifest, load_training_data, match_split
from .model import BaselineModel, environment, load_model, predict_proba_batched, repository_state, save_model
from .selection import metrics_for, select_threshold as _select_threshold
from .sweep import run_sweep

ROOT = Path(__file__).resolve().parents[2]


def _fresh_dir(path, what):
    target = Path(path)
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise ValueError(f"{target}: {what} output directory is not empty; choose a fresh path")
    target.mkdir(parents=True, exist_ok=True)
    return target


def _train_dev(data, config, seed):
    train_texts = [row["text"] for row in data["train"]]
    train_labels = [row["label"] for row in data["train"]]
    started = time.perf_counter()
    model = BaselineModel.fit(config, train_texts, train_labels, seed)
    dev_texts = [row["text"] for row in data["dev"]]
    dev_scores = predict_proba_batched(model, dev_texts, config["batch_size"])
    return model, dev_scores, time.perf_counter() - started


def train_command(data_dir, config_path, output_dir, seed=None, select_threshold=False):
    """Train one frozen configuration and save a self-describing offline model."""
    config = load_config(config_path)
    if seed is None:
        seed = config["seed"]
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    data = load_training_data(data_dir)
    model, dev_scores, train_seconds = _train_dev(data, config, seed)
    dev_ids = [row["id"] for row in data["dev"]]
    dev_labels = [row["label"] for row in data["dev_gold"]]
    threshold_selection = None
    if select_threshold:
        threshold_selection = _select_threshold(dev_labels, dev_scores, dev_ids)
        threshold = threshold_selection["threshold"]
    else:
        threshold = config["threshold"]
    dev_metrics = metrics_for(dev_labels, dev_scores, threshold, dev_ids)
    metadata = {
        "schema_version": VERSION,
        "method": config["method"],
        "model_name": "char_tfidf_logreg" if config["method"] == "char_tfidf_lr" else "majority_class_reference",
        "model_revision": "p2-v1",
        "is_mock": data["manifest"]["is_mock"],
        "seed": seed,
        "threshold": threshold,
        "dataset_id": data["manifest"]["dataset_id"],
        "dataset_manifest_sha256": data["manifest_sha256"],
        "config": config_snapshot(config, seed),
        "feature": {"input": "text", "analyzer": config.get("analyzer", "char"), "ngram_range": config.get("ngram_range"), "max_features": config.get("max_features"), "vectorizer_fit_scope": "train_only", "vectorizer_fit_rows": len(data["train"]), "vocabulary_size": model.vocabulary_size()},
        "training": {"train_rows": len(data["train"]), "train_positive": sum(row["label"] for row in data["train"]), "class_weight": config.get("class_weight"), "C": config.get("C"), "solver": config.get("solver"), "max_iter": config.get("max_iter"), "train_seconds": round(train_seconds, 3), "majority_label": model.majority_label},
        "dev": {"n": len(data["dev"]), "threshold": threshold, "reference_metrics": dev_metrics},
        "threshold_selection": threshold_selection,
        "environment": environment(),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    metadata.update(repository_state(ROOT))
    target = _fresh_dir(output_dir, "model")
    save_model(target, model, metadata)
    write_json(target / "config.json", config_snapshot(config, seed))
    lines = ["P2 baseline training", f"method: {config['method']}", f"model: {metadata['model_name']} ({metadata['model_revision']})", f"dataset: {metadata['dataset_id']} manifest_sha256={data['manifest_sha256']}", f"train rows: {len(data['train'])} positive: {metadata['training']['train_positive']}", f"vectorizer fit scope: train_only rows={len(data['train'])} vocabulary={model.vocabulary_size()}", f"seed: {seed} threshold: {threshold}", f"train+dev scoring seconds: {train_seconds:.3f}", "dev reference metrics (excluding test): " + json.dumps(dev_metrics, ensure_ascii=False), f"code_commit: {metadata['code_commit']} dirty={metadata['code_dirty']}"]
    if threshold_selection is not None:
        lines.append(f"threshold selected on dev grid: {threshold} (rule: {threshold_selection['tie_break']})")
    (target / "log.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"model_dir": str(target), "method": config["method"], "seed": seed, "threshold": threshold, "dataset_id": metadata["dataset_id"], "dataset_manifest_sha256": data["manifest_sha256"], "is_mock": metadata["is_mock"], "vocabulary_size": model.vocabulary_size(), "dev_reference_metrics": dev_metrics, "model_name": metadata["model_name"], "model_revision": metadata["model_revision"], "code_commit": metadata["code_commit"], "code_dirty": metadata["code_dirty"]}


def predict_command(input_path, model_dir, manifest_path, output_dir, threshold=None, batch_size=None):
    """Score one frozen split with a saved model; labels or extra input fields are rejected."""
    model, metadata = load_model(model_dir)
    manifest_dir, manifest, manifest_sha256 = load_manifest(Path(manifest_path).parent)
    if manifest_sha256 != metadata["dataset_manifest_sha256"]:
        raise ValueError("model was trained on a different dataset manifest; refusing to mix data versions")
    if metadata["is_mock"] != manifest["is_mock"]:
        raise ValueError("model and manifest mock flags differ")
    inputs = load_inference_input(input_path)
    split = match_split(manifest_dir, inputs)
    effective_threshold = metadata["threshold"] if threshold is None else float(threshold)
    if not 0 <= effective_threshold <= 1 or effective_threshold != effective_threshold:
        raise ValueError("threshold must be a finite number in [0, 1]")
    effective_batch = metadata["config"].get("batch_size", 256) if batch_size is None else batch_size
    if type(effective_batch) is not int or effective_batch <= 0:
        raise ValueError("batch_size must be a positive integer")
    scores = predict_proba_batched(model, [row["text"] for row in inputs], effective_batch)
    predictions = [{"id": row["id"], "label": int(score >= effective_threshold), "score": score} for row, score in zip(inputs, scores)]
    run_metadata = {"schema_version": VERSION, "method": metadata["method"], "split": split, "dataset_manifest_sha256": manifest_sha256, "is_mock": metadata["is_mock"], "seed": metadata["seed"], "threshold": effective_threshold, "model_name": metadata["model_name"], "model_revision": metadata["model_revision"], "config": metadata["config"], "environment": environment(), "model_dir": str(model_dir), "input": str(input_path), "n": len(inputs)}
    run_metadata.update(repository_state(ROOT))
    validate_predictions(inputs, predictions, run_metadata, Path(manifest_path))
    target = _fresh_dir(output_dir, "run")
    write_jsonl(target / "predictions.jsonl", predictions)
    write_json(target / "run_metadata.json", run_metadata)
    write_json(target / "config.json", {"model": metadata["config"], "run": {"split": split, "threshold": effective_threshold, "batch_size": effective_batch, "input": str(input_path)}})
    lines = ["P2 baseline prediction", f"model: {metadata['model_name']} ({metadata['model_revision']}) from {model_dir}", f"input: {input_path} split: {split} rows: {len(inputs)}", f"dataset: {metadata.get('dataset_id')} manifest_sha256={manifest_sha256}", f"threshold: {effective_threshold} batch_size: {effective_batch}", f"score meaning: model score for label=1, higher means more likely biased", f"code_commit: {run_metadata['code_commit']} dirty={run_metadata['code_dirty']}"]
    (target / "log.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"run_dir": str(target), "split": split, "n": len(inputs), "threshold": effective_threshold, "dataset_id": metadata.get("dataset_id"), "dataset_manifest_sha256": manifest_sha256, "is_mock": metadata["is_mock"], "model_name": metadata["model_name"], "seed": metadata["seed"]}


def sweep_command(data_dir, config_path, output_dir, seed=None):
    """Train every dev-sweep variant on train and select configuration/threshold on dev."""
    base, grid, variants, _ = load_sweep_config(config_path)
    if seed is None:
        seed = base["seed"]
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    data = load_training_data(data_dir)
    result = run_sweep(data, variants, grid, seed)
    target = _fresh_dir(output_dir, "sweep")
    write_json(target / "sweep.json", result)
    best = result["best"]
    lines = ["P2 dev sweep (test split not read)", f"dataset: {result['dataset_id']} manifest_sha256={result['dataset_manifest_sha256']}", f"variants: {len(result['variants'])} seed: {seed}", f"selection rule: {result['selection']['rule']}", f"best config: {json.dumps(best['config'], ensure_ascii=False)}", f"best dev reference metrics at threshold 0.5: {json.dumps(best['dev_metrics'], ensure_ascii=False)}"]
    (target / "log.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"sweep_dir": str(target), "dataset_id": result["dataset_id"], "variants": len(result["variants"]), "best_config": best["config"], "best_dev_metrics": best["dev_metrics"]}


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "--mock":
        from src.common.mock_cli import run

        run("baseline")
        return 0
    parser = argparse.ArgumentParser(description="P2 CPU character TF-IDF + logistic-regression baseline")
    subparsers = parser.add_subparsers(dest="command", required=True)
    trainer = subparsers.add_parser("train", help="Train one configuration and save an offline model")
    trainer.add_argument("--data-dir", required=True)
    trainer.add_argument("--config", required=True)
    trainer.add_argument("--output-dir", required=True)
    trainer.add_argument("--seed", type=int, default=None, help="Overrides the seed in the config")
    trainer.add_argument("--select-threshold", action="store_true", help="Choose the threshold on the dev grid only")
    predictor = subparsers.add_parser("predict", help="Score a frozen dev/test split with a saved model")
    predictor.add_argument("--input", required=True)
    predictor.add_argument("--model-dir", required=True)
    predictor.add_argument("--manifest", required=True)
    predictor.add_argument("--output-dir", required=True)
    predictor.add_argument("--threshold", type=float, default=None, help="Optional recorded override; the frozen 0.5 default is the model metadata value")
    predictor.add_argument("--batch-size", type=int, default=None)
    sweeper = subparsers.add_parser("sweep", help="Dev-only configuration sweep (never reads test)")
    sweeper.add_argument("--data-dir", required=True)
    sweeper.add_argument("--config", required=True)
    sweeper.add_argument("--output-dir", required=True)
    sweeper.add_argument("--seed", type=int, default=None)
    experiments = subparsers.add_parser("experiments", help="Run the frozen P2 experiment matrix, acceptance JSON and figures")
    experiments.add_argument("--data-dir", default="data/processed")
    experiments.add_argument("--output-dir", default="runs/p2")
    experiments.add_argument("--artifacts-dir", default="artifacts/p2")
    experiments.add_argument("--configs-dir", default="configs/baseline")
    experiments.add_argument("--acceptance", default=None, help="Defaults to <configs-dir>/acceptance.json")
    experiments.add_argument("--figures-dir", default="reports/figures/P2")
    args = parser.parse_args(arguments)
    try:
        if args.command == "train":
            result = train_command(args.data_dir, args.config, args.output_dir, seed=args.seed, select_threshold=args.select_threshold)
        elif args.command == "predict":
            result = predict_command(args.input, args.model_dir, args.manifest, args.output_dir, threshold=args.threshold, batch_size=args.batch_size)
        elif args.command == "sweep":
            result = sweep_command(args.data_dir, args.config, args.output_dir, seed=args.seed)
        else:
            from .experiments import run_experiments

            result = run_experiments(ROOT, args.data_dir, args.output_dir, args.artifacts_dir, args.configs_dir, args.acceptance, args.figures_dir)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError, KeyError) as exc:
        print(f"P2 baseline error: {exc}", file=sys.stderr)
        return 2
    except ImportError as exc:
        print(f"P2 baseline environment error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - keep failure code distinct from contract errors
        print(f"P2 baseline failure: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
