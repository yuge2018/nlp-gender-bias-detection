"""Dev-only threshold selection and reference metrics.

The threshold grid is frozen: 0.10 to 0.90 in steps of 0.05. Ties are resolved
by the grid value closest to 0.5, then by the smaller threshold, exactly as the
group experiment plan specifies. Test scores never participate in this choice.
"""
from src.common.contracts import binary_metrics

THRESHOLD_START = 0.10
THRESHOLD_STOP = 0.90
THRESHOLD_STEP = 0.05
THRESHOLD_GRID = tuple(round(THRESHOLD_START + index * THRESHOLD_STEP, 2) for index in range(int(round((THRESHOLD_STOP - THRESHOLD_START) / THRESHOLD_STEP)) + 1))
TIE_BREAK_RULE = "highest dev Macro-F1; then threshold closest to 0.5; then smaller threshold"


def labels_from_scores(scores, threshold):
    return [int(score >= threshold) for score in scores]


def metrics_for(labels, scores, threshold, ids):
    """Reference Macro-F1 and component metrics for one frozen threshold."""
    if len(labels) != len(scores) or len(ids) != len(labels):
        raise ValueError("labels, scores and ids must align")
    gold = [{"id": ident, "label": label} for ident, label in zip(ids, labels)]
    predictions = [{"id": ident, "label": int(score >= threshold), "score": float(score)} for ident, score in zip(ids, scores)]
    return binary_metrics(gold, predictions)


def select_threshold(labels, scores, ids):
    """Select one threshold on dev and return the full evidence table."""
    if not labels:
        raise ValueError("cannot select a threshold on an empty split")
    table = []
    best = None
    for threshold in THRESHOLD_GRID:
        metrics = metrics_for(labels, scores, threshold, ids)
        row = {"threshold": threshold, "macro_f1": metrics["macro_f1"], "positive_recall": metrics["positive_recall"], "positive_precision": metrics["positive_precision"], "accuracy": metrics["accuracy"], "confusion_matrix": metrics["confusion_matrix"]}
        table.append(row)
        key = (-metrics["macro_f1"], abs(threshold - 0.5), threshold)
        if best is None or key < best[0]:
            best = (key, threshold, metrics)
    return {"selection_split": "dev", "metric": "macro_f1", "tie_break": TIE_BREAK_RULE, "threshold": best[1], "metrics": best[2], "grid": table}
