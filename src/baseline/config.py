"""P2 configuration validation and defaults.

Only JSON configuration is accepted. Unknown keys fail instead of being silently
ignored, so a typo cannot change an experiment unnoticed.
"""
import json
import math
from pathlib import Path

from src.common.contracts import VERSION

METHOD_LOGREG = "char_tfidf_lr"
METHOD_MAJORITY = "majority_class"
SUPPORTED_METHODS = (METHOD_LOGREG, METHOD_MAJORITY)
SUPPORTED_SOLVERS = ("liblinear",)

DEFAULT_SOLVER = "liblinear"
DEFAULT_BATCH_SIZE = 256

LOGREG_KEYS = {
    "method",
    "analyzer",
    "ngram_range",
    "max_features",
    "C",
    "class_weight",
    "max_iter",
    "solver",
    "threshold",
    "seed",
    "batch_size",
}
MAJORITY_KEYS = {"method", "threshold", "seed", "batch_size"}


def _number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _checked_threshold(value):
    threshold = _number(value, "threshold")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    return threshold


def _checked_seed(value):
    if type(value) is not int or value < 0:
        raise ValueError("seed must be a non-negative integer")
    return value


def validate_config(raw):
    """Return a normalized copy with explicit defaults; raise ValueError on any problem."""
    if not isinstance(raw, dict):
        raise ValueError("config must be a JSON object")
    method = raw.get("method")
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"method must be one of {SUPPORTED_METHODS}")
    allowed = LOGREG_KEYS if method == METHOD_LOGREG else MAJORITY_KEYS
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"unknown config keys for {method}: {unknown}")
    normalized = {"method": method, "threshold": _checked_threshold(raw.get("threshold", 0.5)), "seed": _checked_seed(raw.get("seed", 42)), "batch_size": _positive_int(raw.get("batch_size", DEFAULT_BATCH_SIZE), "batch_size")}
    if method == METHOD_MAJORITY:
        return normalized
    analyzer = raw.get("analyzer", "char")
    if analyzer != "char":
        raise ValueError("P2 baseline only supports analyzer=char")
    ngram = raw.get("ngram_range")
    if not isinstance(ngram, (list, tuple)) or len(ngram) != 2:
        raise ValueError("ngram_range must be [low, high]")
    low, high = ngram
    if type(low) is not int or type(high) is not int or not 1 <= low <= high <= 3:
        raise ValueError("ngram_range must satisfy 1 <= low <= high <= 3")
    class_weight = raw.get("class_weight")
    if class_weight not in (None, "balanced"):
        raise ValueError("class_weight must be null or 'balanced'")
    solver = raw.get("solver", DEFAULT_SOLVER)
    if solver not in SUPPORTED_SOLVERS:
        raise ValueError(f"solver must be one of {SUPPORTED_SOLVERS}")
    normalized.update({"analyzer": analyzer, "ngram_range": [low, high], "max_features": _positive_int(raw.get("max_features", 100000), "max_features"), "C": _number(raw.get("C", 1.0), "C"), "class_weight": class_weight, "max_iter": _positive_int(raw.get("max_iter", 1000), "max_iter"), "solver": solver})
    if normalized["C"] <= 0:
        raise ValueError("C must be positive")
    return normalized


def load_config(path):
    target = Path(path)
    if not target.is_file():
        raise ValueError(f"{target}: config file not found")
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{target}: invalid JSON ({exc})") from exc
    config = validate_config(raw)
    config["config_path"] = str(target)
    return config


def config_snapshot(config, seed):
    """Serializable configuration actually used by one training run."""
    snapshot = {key: value for key, value in config.items() if key != "config_path"}
    snapshot["seed"] = seed
    snapshot["schema_version"] = VERSION
    return snapshot


def load_sweep_config(path):
    """A sweep file holds a base config plus a grid; returns ``(base, grid, variants, path)``."""
    target = Path(path)
    if not target.is_file():
        raise ValueError(f"{target}: sweep config not found")
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{target}: invalid JSON ({exc})") from exc
    if not isinstance(raw, dict) or "grid" not in raw:
        raise ValueError("sweep config needs an object with a grid")
    grid = raw["grid"]
    if not isinstance(grid, dict) or set(grid) != {"C", "class_weight", "ngram_range"}:
        raise ValueError("sweep grid must define exactly C, class_weight and ngram_range")
    base = {key: value for key, value in raw.items() if key != "grid"}
    base.setdefault("method", METHOD_LOGREG)
    base.setdefault("analyzer", "char")
    base.setdefault("max_features", 100000)
    base.setdefault("max_iter", 1000)
    base.setdefault("threshold", 0.5)
    base.setdefault("seed", 42)
    for name in ("C", "class_weight", "ngram_range"):
        if not isinstance(grid[name], list) or not grid[name]:
            raise ValueError(f"grid.{name} must be a non-empty list")
    variants = []
    for c_value in grid["C"]:
        for weight in grid["class_weight"]:
            for ngram in grid["ngram_range"]:
                variant = dict(base, C=c_value, class_weight=weight, ngram_range=ngram)
                variants.append(validate_config(variant))
    if len(variants) > 64:
        raise ValueError("sweep grid is too large; keep it below 64 variants")
    return validate_config(base), {"C": list(grid["C"]), "class_weight": list(grid["class_weight"]), "ngram_range": [list(item) for item in grid["ngram_range"]]}, variants, str(target)
