"""Scoring helpers for the P6 demo: validation, threshold, result rows.

Pure logic, no streamlit and no model imports, so the contract suite can test it
on a bare image. The score is the model's label=1 score from
`predictor.predict_proba`; it is NOT a calibrated probability and the UI must
never call it one.
"""
from __future__ import annotations

import math
from typing import Any, Iterable

DEFAULT_THRESHOLD = 0.5


def validate_threshold(value: Any, *, default: float = DEFAULT_THRESHOLD) -> float:
    """Return a finite threshold in [0, 1]."""
    if value is None:
        return float(default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"threshold must be a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"threshold must be finite in [0,1], got {value!r}")
    return number


def validate_texts(texts: Any) -> list[str]:
    """Return the input texts unchanged after checking they are non-empty.

    Whitespace is only used to decide "empty"; the stored text keeps the user's
    original characters so the demo never silently rewrites what it analyses.
    """
    if not isinstance(texts, (list, tuple)):
        raise ValueError("texts must be a list of strings")
    cleaned: list[str] = []
    for item in texts:
        if not isinstance(item, str):
            raise ValueError(f"text must be a string, got {type(item).__name__}")
        if not item.strip():
            raise ValueError("text must be a non-empty string")
        cleaned.append(item)
    return cleaned


def validate_scores(scores: Any, expected: int) -> list[float]:
    """Check that an adapter returned `expected` finite scores in [0,1]."""
    if not isinstance(scores, (list, tuple)):
        raise ValueError(f"predict_proba must return a list, got {type(scores).__name__}")
    if len(scores) != expected:
        raise ValueError(f"predict_proba returned {len(scores)} scores for {expected} texts")
    cleaned: list[float] = []
    for index, score in enumerate(scores):
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError(f"score[{index}] is not a number: {score!r}")
        number = float(score)
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise ValueError(f"score[{index}] must be finite in [0,1]: {score!r}")
        cleaned.append(number)
    return cleaned


def model_threshold(predictor: Any, *, default: float = DEFAULT_THRESHOLD) -> float:
    """Read the threshold recorded with the model; fall back to the contract default."""
    recorded = getattr(predictor, "threshold", None)
    return validate_threshold(recorded, default=default)


def predict_rows(
    predictor: Any,
    texts: Any,
    threshold: float = DEFAULT_THRESHOLD,
    *,
    id_prefix: str = "input",
) -> list[dict]:
    """Score texts and build `id/text/score/label` rows in input order."""
    threshold = validate_threshold(threshold)
    cleaned = validate_texts(texts)
    if not cleaned:
        return []
    scores = validate_scores(predictor.predict_proba(cleaned), len(cleaned))
    return [
        {
            "id": f"{id_prefix}-{index:06d}",
            "text": text,
            "score": score,
            "label": int(score >= threshold),
        }
        for index, (text, score) in enumerate(zip(cleaned, scores), start=1)
    ]


def summarize(rows: Iterable[dict]) -> dict:
    """Counts used by the demo header; never presented as model performance."""
    rows = list(rows)
    biased = sum(1 for row in rows if row["label"] == 1)
    return {"n": len(rows), "biased": biased, "unbiased": len(rows) - biased}
