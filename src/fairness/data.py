"""Reading the frozen P5 challenge set and prediction runs.

Only files are exchanged: the challenge directory (challenge/labels/annotations
+ sealed manifest) and prediction run directories produced by any model module.
No model internals are imported here except the public ``load_predictor`` adapter
used by the optional ``score`` command.
"""
import json
from pathlib import Path

from src.common.contracts import VERSION, check_records, read_jsonl, sha256

from .build import PHENOMENA, TARGETS

CHALLENGE_FILES = ("challenge.jsonl", "challenge_labels.jsonl", "challenge_annotations.jsonl")
REQUIRED_RUN_FIELDS = {
    "schema_version", "method", "split", "dataset_manifest_sha256", "is_mock", "seed",
    "threshold", "code_commit", "model_name", "model_revision", "config", "environment",
}
ALLOWED_SPLITS = ("test", "dev", "challenge")


def _read_challenge_rows(base, manifest, name):
    entry = manifest.get("files", {}).get(name)
    if not isinstance(entry, dict):
        raise ValueError(f"{name}: manifest has no file entry")
    path = base / name
    if not path.is_file():
        raise ValueError(f"{path}: missing challenge file")
    if sha256(path) != entry.get("sha256"):
        raise ValueError(f"{name}: sha256 mismatch against challenge manifest (has the set been resealed?)")
    rows = read_jsonl(path)
    if len(rows) != entry.get("rows"):
        raise ValueError(f"{name}: row count mismatch against challenge manifest")
    return rows


def load_challenge(challenge_dir):
    """Load and validate challenge, labels and annotations; ids must line up exactly."""
    base = Path(challenge_dir)
    manifest_path = base / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"{manifest_path}: challenge manifest not found")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != VERSION:
        raise ValueError(f"{manifest_path}: unsupported schema_version")
    if type(manifest.get("is_mock")) is not bool:
        raise ValueError(f"{manifest_path}: is_mock must be a boolean")
    if manifest.get("label_map") != {"0": "non_biased", "1": "biased"}:
        raise ValueError(f"{manifest_path}: unexpected label_map")
    for key in ("dataset_id", "annotation_version", "review_status", "frozen_at"):
        if not isinstance(manifest.get(key), str) or not manifest[key].strip():
            raise ValueError(f"{manifest_path}: missing manifest field {key}")

    challenge = _read_challenge_rows(base, manifest, "challenge.jsonl")
    labels = _read_challenge_rows(base, manifest, "challenge_labels.jsonl")
    annotations = _read_challenge_rows(base, manifest, "challenge_annotations.jsonl")
    check_records(challenge, "input")
    check_records(labels, "gold")

    if [row["id"] for row in challenge] != [row["id"] for row in labels]:
        raise ValueError("challenge and labels ids/order differ")
    for line, row in enumerate(annotations, 1):
        expected = {"id", "targets", "phenomena", "reviewers", "adjudication"}
        if set(row) != expected:
            raise ValueError(f"challenge_annotations.jsonl:{line}: expected fields {sorted(expected)}")
        if row["id"] not in {item["id"] for item in challenge}:
            raise ValueError(f"challenge_annotations.jsonl:{line}: unknown id {row['id']!r}")
        if not isinstance(row["targets"], list) or not row["targets"]:
            raise ValueError(f"challenge_annotations.jsonl:{line}: targets must be a non-empty list")
        unknown = [target for target in row["targets"] if target not in TARGETS]
        if unknown:
            raise ValueError(f"challenge_annotations.jsonl:{line}: unknown targets {unknown}")
        if not isinstance(row["phenomena"], list) or not row["phenomena"]:
            raise ValueError(f"challenge_annotations.jsonl:{line}: phenomena must be a non-empty list")
        unknown = [item for item in row["phenomena"] if item not in PHENOMENA]
        if unknown:
            raise ValueError(f"challenge_annotations.jsonl:{line}: unknown phenomena {unknown}")
        if not isinstance(row["reviewers"], list) or any(not isinstance(name, str) or not name.strip() for name in row["reviewers"]):
            raise ValueError(f"challenge_annotations.jsonl:{line}: reviewers must be a list of names")
        if not isinstance(row["adjudication"], str) or not row["adjudication"].strip():
            raise ValueError(f"challenge_annotations.jsonl:{line}: adjudication note is required")
    annotation_ids = [row["id"] for row in annotations]
    if annotation_ids != [row["id"] for row in challenge]:
        raise ValueError("challenge and annotations ids/order differ")

    coverage = check_coverage(challenge, labels, annotations)
    return {
        "challenge_dir": str(base),
        "manifest": manifest,
        "manifest_sha256": sha256(manifest_path),
        "challenge": challenge,
        "labels": labels,
        "annotations": annotations,
        "coverage": coverage,
    }


def check_coverage(challenge, labels, annotations):
    """Check the acceptance requirements for object and polarity coverage (no model involved)."""
    label_of = {row["id"]: row["label"] for row in labels}
    problems, warnings = [], []
    target_counts = {target: {"0": 0, "1": 0} for target in TARGETS}
    phenomenon_counts = {key: {"0": 0, "1": 0} for key in PHENOMENA}
    for annotation in annotations:
        label = str(label_of[annotation["id"]])
        for target in annotation["targets"]:
            target_counts[target][label] += 1
        for phenomenon in annotation["phenomena"]:
            phenomenon_counts[phenomenon][label] += 1
    for target in ("male", "female"):
        total = target_counts[target]["0"] + target_counts[target]["1"]
        if total < 20:
            problems.append(f"{target}: {total} < 20 required examples")
        for label in ("0", "1"):
            if target_counts[target][label] < 5:
                problems.append(f"{target} label={label}: {target_counts[target][label]} < 5; both polarities need coverage")
    if target_counts["other_explicit"]["0"] == 0 or target_counts["other_explicit"]["1"] == 0:
        problems.append("other_explicit: needs both polarities")
    if target_counts["unspecified"]["1"] == 0:
        warnings.append("unspecified: no biased example; coverage of target-ambiguous bias is not supported")
    if not 80 <= len(challenge) <= 120:
        problems.append(f"challenge size {len(challenge)} outside the required 80-120 range")
    phenomena = {item for annotation in annotations for item in annotation["phenomena"]}
    for required in ("quoted_rebuttal", "denial_of_bias", "benevolent_sexism", "implicit_stereotype"):
        if required not in phenomena:
            problems.append(f"phenomenon {required} is not covered")
    reviewers = {name for annotation in annotations for name in annotation["reviewers"]}
    if len(reviewers) < 2:
        warnings.append("double independent review is not complete: fewer than two reviewer names are recorded")
    return {
        "n": len(challenge),
        "positive": sum(row["label"] for row in labels),
        "negative": sum(1 - row["label"] for row in labels),
        "by_target": target_counts,
        "by_phenomenon": phenomenon_counts,
        "problems": problems,
        "warnings": warnings,
        "independent_reviewers": sorted(reviewers),
    }


def load_run(run_dir):
    """Load one prediction run directory and validate its rows against its own metadata."""
    base = Path(run_dir)
    predictions_path = base / "predictions.jsonl"
    metadata_path = base / "run_metadata.json"
    if not predictions_path.is_file():
        raise ValueError(f"{predictions_path}: predictions.jsonl not found")
    if not metadata_path.is_file():
        raise ValueError(f"{metadata_path}: run_metadata.json not found")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    missing = REQUIRED_RUN_FIELDS - set(metadata)
    if missing:
        raise ValueError(f"{metadata_path}: missing run metadata fields {sorted(missing)}")
    if metadata["schema_version"] != VERSION or type(metadata["is_mock"]) is not bool:
        raise ValueError(f"{metadata_path}: invalid schema_version or is_mock")
    if metadata["split"] not in ALLOWED_SPLITS:
        raise ValueError(f"{metadata_path}: split must be one of {ALLOWED_SPLITS}")
    if type(metadata["seed"]) is not int:
        raise ValueError(f"{metadata_path}: seed must be an integer")
    threshold = metadata["threshold"]
    if type(threshold) not in (int, float) or not 0 <= threshold <= 1:
        raise ValueError(f"{metadata_path}: threshold must be a finite number in [0, 1]")
    predictions = read_jsonl(predictions_path)
    check_records(predictions, "prediction")
    for row in predictions:
        if row["label"] != int(row["score"] >= threshold):
            raise ValueError(f"{metadata_path}: {row['id']} label contradicts score and threshold")
    return {
        "run_dir": str(base),
        "metadata": metadata,
        "predictions": predictions,
        "threshold": float(threshold),
        "is_mock": metadata["is_mock"],
        "method": metadata["method"],
    }


def align_run(challenge, run):
    """Return predictions reordered to the challenge order, rejecting id mismatch."""
    challenge_ids = [row["id"] for row in challenge]
    prediction_ids = [row["id"] for row in run["predictions"]]
    if len(challenge_ids) != len(prediction_ids):
        raise ValueError(f"{run['run_dir']}: {len(prediction_ids)} predictions for {len(challenge_ids)} challenge rows")
    missing = set(challenge_ids) - set(prediction_ids)
    extra = set(prediction_ids) - set(challenge_ids)
    if missing or extra:
        raise ValueError(f"{run['run_dir']}: id mismatch, missing={sorted(missing)[:5]} extra={sorted(extra)[:5]}")
    by_id = {row["id"]: row for row in run["predictions"]}
    return [by_id[ident] for ident in challenge_ids]


def check_models_consistent(runs):
    """All compared models must share the mock flag and challenge row count."""
    flags = {run["is_mock"] for run in runs}
    if len(flags) > 1:
        raise ValueError("mock and real prediction runs cannot be mixed in one subgroup table")
    return None
