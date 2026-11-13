"""Stable adapter interface for the demo and both model owners."""
from typing import Protocol


class Predictor(Protocol):
    is_mock: bool

    def predict_proba(self, texts: list[str]) -> list[float]:
        """Return finite scores for label 1, preserving input order."""
        ...


class MockPredictor:
    """A constant development substitute. This performs no detection."""

    is_mock = True

    def predict_proba(self, texts: list[str]) -> list[float]:
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Every text must be a non-empty string")
        return [0.5] * len(texts)
