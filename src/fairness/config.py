"""P5 configuration: analysis rules that must stay frozen before seeing model output."""
import json
from pathlib import Path

from .build import TARGETS

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "fairness" / "default.json"

ALLOWED_KEYS = {
    "challenge_dir", "targets", "multi_target", "zero_denominator", "minimum_examples_goal",
    "use_challenge_for_tuning", "minimum_per_target", "minimum_per_polarity", "review", "report", "score",
}


def load_config(path=None):
    """Load and validate the P5 config; unknown keys are a contract error."""
    target = Path(path) if path else DEFAULT_CONFIG
    if not target.is_file():
        raise ValueError(f"{target}: P5 config not found")
    config = json.loads(target.read_text(encoding="utf-8"))
    unknown = set(config) - ALLOWED_KEYS
    if unknown:
        raise ValueError(f"{target}: unknown config keys {sorted(unknown)}")
    if config.get("targets") != list(TARGETS):
        raise ValueError(f"{target}: targets must match the frozen challenge schema {list(TARGETS)}")
    if config.get("use_challenge_for_tuning") is not False:
        raise ValueError(f"{target}: the challenge must never be used for tuning")
    if config.get("multi_target") is not True:
        raise ValueError(f"{target}: multi-target annotation must stay enabled")
    if config.get("zero_denominator") is not None:
        raise ValueError(f"{target}: zero-denominator metrics must stay null")
    for key in ("minimum_per_target", "minimum_per_polarity"):
        value = config.get(key)
        if type(value) is not int or value <= 0:
            raise ValueError(f"{target}: {key} must be a positive integer")
    review = config.get("review")
    if not isinstance(review, dict) or review.get("independent_reviewers_required") != 2:
        raise ValueError(f"{target}: review.independent_reviewers_required must be 2")
    report = config.get("report")
    if not isinstance(report, dict) or type(report.get("max_cases")) is not int:
        raise ValueError(f"{target}: report.max_cases must be an integer")
    return config
