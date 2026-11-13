"""Dev-only configuration sweep for the P2 baseline.

Each variant is trained on ``train`` and scored on ``dev`` only. The selection
rule is fixed before any test evaluation: highest dev Macro-F1 at the default
0.5 threshold; ties prefer no class weighting, then the narrower n-gram window,
then the smaller C. A per-variant threshold is then chosen on the same dev grid.
"""
import time

from src.common.contracts import VERSION

from .config import METHOD_LOGREG
from .model import BaselineModel, predict_proba_batched
from .selection import metrics_for, select_threshold

SELECTION_RULE = "highest dev Macro-F1 at threshold 0.5; ties prefer class_weight=null, then narrower ngram_range, then smaller C; threshold selected afterwards on the same dev grid"


def _rank_key(variant):
    config = variant["config"]
    weight_penalty = 0 if config["class_weight"] is None else 1
    ngram_span = config["ngram_range"][1] - config["ngram_range"][0]
    return (-variant["dev_at_default_threshold"]["metrics"]["macro_f1"], weight_penalty, ngram_span, config["C"])


def run_sweep(data, variants, grid, seed):
    """Return the full sweep evidence; never touches the test split."""
    if not variants:
        raise ValueError("sweep has no variants")
    train_texts = [row["text"] for row in data["train"]]
    train_labels = [row["label"] for row in data["train"]]
    dev_texts = [row["text"] for row in data["dev"]]
    dev_ids = [row["id"] for row in data["dev"]]
    dev_labels = [row["label"] for row in data["dev_gold"]]
    results = []
    for variant in variants:
        started = time.perf_counter()
        model = BaselineModel.fit(variant, train_texts, train_labels, seed)
        dev_scores = predict_proba_batched(model, dev_texts, variant["batch_size"])
        train_seconds = time.perf_counter() - started
        selection = select_threshold(dev_labels, dev_scores, dev_ids)
        default_metrics = metrics_for(dev_labels, dev_scores, variant["threshold"], dev_ids)
        results.append({"config": {key: value for key, value in variant.items() if key != "config_path"}, "train_seconds": round(train_seconds, 3), "dev_at_default_threshold": {"threshold": variant["threshold"], "metrics": default_metrics}, "threshold_selection": {"threshold": selection["threshold"], "metrics": selection["metrics"], "grid": selection["grid"]}})
    ranked = sorted(range(len(results)), key=lambda index: _rank_key(results[index]))
    best_index = ranked[0]
    return {"schema_version": VERSION, "is_mock": data["manifest"]["is_mock"], "dataset_id": data["manifest"]["dataset_id"], "dataset_manifest_sha256": data["manifest_sha256"], "seed": seed, "method": METHOD_LOGREG, "selection": {"split": "dev", "metric": "macro_f1", "default_threshold": 0.5, "rule": SELECTION_RULE, "tie_break_threshold": "highest dev Macro-F1; then threshold closest to 0.5; then smaller threshold"}, "grid": grid, "variants": results, "ranked_variant_indices": ranked, "best": {"variant_index": best_index, "config": results[best_index]["config"], "dev_metrics": results[best_index]["dev_at_default_threshold"]["metrics"]}}
