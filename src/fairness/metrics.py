"""Grouped metrics for the P5 challenge set.

Only predictions and gold labels are read. Every group reports its own supporting
counts so a tiny subgroup is never presented as a stable estimate. ``fpr``/``fnr``
and per-class F1 are ``None`` when their denominator is zero; ``macro_f1`` follows
the project convention (fixed average over labels 0 and 1, undefined treated as 0)
and ``macro_f1_defined_classes`` reports the variant that averages only classes
that actually occur, which is the honest number for single-polarity subgroups.
"""


def _ratio(numerator, denominator):
    return None if denominator == 0 else numerator / denominator


def _f1(tp, fp, fn):
    return _ratio(2 * tp, 2 * tp + fp + fn)


def group_metrics(gold_labels, predicted_labels):
    """Metrics for one subgroup; inputs are aligned integer label lists."""
    if len(gold_labels) != len(predicted_labels):
        raise ValueError("gold and prediction lengths differ")
    if not gold_labels:
        return {"n": 0, "positive_n": 0, "negative_n": 0, "tp": 0, "fp": 0, "fn": 0, "tn": 0,
                "accuracy": None, "precision": None, "recall": None, "positive_f1": None,
                "negative_f1": None, "macro_f1": None, "macro_f1_defined_classes": None,
                "fpr": None, "fnr": None}
    tp = sum(1 for gold, pred in zip(gold_labels, predicted_labels) if gold == 1 and pred == 1)
    fn = sum(1 for gold, pred in zip(gold_labels, predicted_labels) if gold == 1 and pred == 0)
    fp = sum(1 for gold, pred in zip(gold_labels, predicted_labels) if gold == 0 and pred == 1)
    tn = sum(1 for gold, pred in zip(gold_labels, predicted_labels) if gold == 0 and pred == 0)
    positive_n, negative_n = tp + fn, fp + tn
    positive_f1 = _f1(tp, fp, fn) if positive_n else None
    negative_f1 = _f1(tn, fp, fn) if negative_n else None
    defined = [value for value in (positive_f1, negative_f1) if value is not None]
    return {
        "n": len(gold_labels),
        "positive_n": positive_n,
        "negative_n": negative_n,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy": (tp + tn) / len(gold_labels),
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "positive_f1": positive_f1,
        "negative_f1": negative_f1,
        "macro_f1": ((positive_f1 or 0.0) + (negative_f1 or 0.0)) / 2,
        "macro_f1_defined_classes": (sum(defined) / len(defined)) if defined else None,
        "fpr": _ratio(fp, fp + tn),
        "fnr": _ratio(fn, fn + tp),
    }


def subgroup_rows(challenge, labels, annotations, predictions, dimension):
    """Yield ``(group_name, selected_indices)`` for one analysis dimension."""
    label_rows = labels
    if dimension == "target":
        groups = {}
        for index, annotation in enumerate(annotations):
            for target in annotation["targets"]:
                groups.setdefault(target, []).append(index)
        yield "all", list(range(len(challenge)))
        for target, indices in groups.items():
            yield target, indices
        multi = [index for index, annotation in enumerate(annotations) if len(annotation["targets"]) > 1]
        if multi:
            yield "multi_target", multi
    elif dimension == "phenomenon":
        groups = {}
        for index, annotation in enumerate(annotations):
            for phenomenon in annotation["phenomena"]:
                groups.setdefault(phenomenon, []).append(index)
        for phenomenon, indices in groups.items():
            yield phenomenon, indices
    elif dimension == "polarity":
        for value, name in ((1, "biased"), (0, "non_biased")):
            yield name, [index for index, row in enumerate(label_rows) if row["label"] == value]
    else:
        raise ValueError(f"unknown subgroup dimension {dimension!r}")


def build_subgroup_metrics(challenge, labels, annotations, models):
    """Full grouped table for every model, every dimension, every group.

    ``models`` is a list of ``{"name": ..., "predictions": aligned rows, ...}``.
    """
    gold = [row["label"] for row in labels]
    table = []
    for dimension in ("target", "phenomenon", "polarity"):
        for group, indices in subgroup_rows(challenge, labels, annotations, models, dimension):
            gold_subset = [gold[index] for index in indices]
            entry = {
                "dimension": dimension,
                "group": group,
                "n": len(indices),
                "positive_n": sum(gold_subset),
                "negative_n": len(gold_subset) - sum(gold_subset),
                "results": {},
            }
            for model in models:
                predicted = [model["predictions"][index]["label"] for index in indices]
                entry["results"][model["name"]] = group_metrics(gold_subset, predicted)
            table.append(entry)
    return table


def summarize(table):
    """Small warnings list for groups whose estimates are too thin to interpret."""
    notes = []
    for entry in table:
        if entry["dimension"] == "target" and entry["n"] < 20:
            notes.append(f"target={entry['group']}: n={entry['n']}, treat as descriptive only")
        if entry["dimension"] == "phenomenon" and entry["n"] < 10:
            notes.append(f"phenomenon={entry['group']}: n={entry['n']}, treat as descriptive only")
    return notes
