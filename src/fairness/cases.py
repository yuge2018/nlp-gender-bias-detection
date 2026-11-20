"""Error and success case extraction for the challenge analysis.

Cases are chosen from real prediction rows only. An automatic explanation records
the mechanical fact (false negative / false positive and the annotated phenomenon);
a human-written note can be supplied through ``configs/fairness/case_notes.json``
to explain the linguistic reason. Mock runs are always flagged and must never be
described as model findings.
"""
import json
from pathlib import Path

CASE_TYPES = ("false_negative", "false_positive", "true_positive", "true_negative")


def _load_notes(notes_path):
    if not notes_path:
        return {}
    path = Path(notes_path)
    if not path.is_file():
        raise ValueError(f"{path}: case notes file not found")
    notes = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(notes, dict) or any(not isinstance(value, str) for value in notes.values()):
        raise ValueError(f"{path}: case notes must map id -> explanation string")
    return notes


def _case_type(gold, predicted):
    if gold == 1 and predicted == 0:
        return "false_negative"
    if gold == 0 and predicted == 1:
        return "false_positive"
    if gold == 1 and predicted == 1:
        return "true_positive"
    return "true_negative"


def _priority(gold, row, threshold):
    """Errors first, then the most confident decision, so the list is readable."""
    error = 0 if _case_type(gold, row["label"]) in ("false_negative", "false_positive") else 1
    return (error, -abs(row["score"] - threshold))


def select_cases(challenge, labels, annotations, models, limit=16, notes_path=None):
    """Return up to ``limit`` cases: errors first, then successes for contrast."""
    notes = _load_notes(notes_path)
    primary = models[0]
    annotation_of = {row["id"]: row for row in annotations}
    label_of = {row["id"]: row["label"] for row in labels}
    buckets = {key: [] for key in CASE_TYPES}
    for index, row in enumerate(challenge):
        gold = label_of[row["id"]]
        entry = primary["predictions"][index]
        buckets[_case_type(gold, entry["label"])].append((_priority(gold, entry, primary["threshold"]), index))
    for key in buckets:
        buckets[key].sort(key=lambda item: item[0])

    error_budget = max(4, limit // 2)
    half = error_budget // 2
    false_negatives = [index for _, index in buckets["false_negative"][:half]]
    false_positives = [index for _, index in buckets["false_positive"][:error_budget - len(false_negatives)]]
    selected_indices = false_negatives + false_positives
    remaining = limit - len(selected_indices)
    for offset in range(remaining):
        # Alternate true positives and true negatives so the successes are not one-sided.
        pool = buckets["true_positive"] if offset % 2 == 0 else buckets["true_negative"]
        if not pool:
            pool = buckets["true_negative"] if offset % 2 == 0 else buckets["true_positive"]
        if not pool:
            break
        selected_indices.append(pool.pop(0)[1])

    selected, counts = [], {key: 0 for key in CASE_TYPES}
    for index in selected_indices:
        row = challenge[index]
        annotation = annotation_of[row["id"]]
        gold = label_of[row["id"]]
        kind = _case_type(gold, primary["predictions"][index]["label"])
        cases = {}
        for model in models:
            prediction = model["predictions"][index]
            cases[model["name"]] = {"label": prediction["label"], "score": prediction["score"], "threshold": model["threshold"], "is_mock": model["is_mock"]}
        explanation = notes.get(row["id"]) or (
            f"自动说明：标注现象为 {', '.join(annotation['phenomena'])}，"
            f"主模型将其判为 {'有偏见' if primary['predictions'][index]['label'] else '无偏见'}，"
            f"金标为 {'有偏见' if gold else '无偏见'}。"
        )
        selected.append({
            "id": row["id"],
            "text": row["text"],
            "gold": gold,
            "targets": annotation["targets"],
            "phenomena": annotation["phenomena"],
            "case_type": kind,
            "models": cases,
            "explanation": explanation,
            "note_source": "human" if row["id"] in notes else "automatic",
        })
        counts[kind] += 1
    return selected, counts
