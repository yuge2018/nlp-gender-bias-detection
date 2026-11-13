"""P2: CPU character TF-IDF + logistic-regression baseline with feature ablations.

Public entry points:

- ``python -m src.baseline train`` / ``predict`` / ``sweep`` / ``experiments``
- ``src.baseline.load_predictor(model_dir)`` for the demo adapter
- ``python -m src.baseline --mock --output-dir runs/p2-smoke`` (dependency-free smoke test)
"""
from .adapter import load_predictor

__all__ = ["load_predictor"]
