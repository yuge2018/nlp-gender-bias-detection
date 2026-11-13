"""Reference contracts and metrics; production CLI belongs to P4."""
import hashlib
import json
import math
from pathlib import Path

VERSION = "1.0"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    records = []
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"{path}:{number}: blank JSONL row")
            try:
                item = json.loads(line)
            except ValueError as exc:
                raise ValueError(f"{path}:{number}: invalid JSON") from exc
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{number}: expected object")
            records.append(item)
    return records


def write_json(path, value):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows), encoding="utf-8")


def check_records(rows, kind):
    expected = {"train": {"id", "text", "label"}, "input": {"id", "text"}, "gold": {"id", "label"}, "prediction": {"id", "label", "score"}}[kind]
    seen = set()
    if not rows:
        raise ValueError(f"{kind}: empty dataset")
    for line, row in enumerate(rows, 1):
        if set(row) != expected:
            raise ValueError(f"{kind}:{line}: expected fields {sorted(expected)}")
        ident = row.get("id")
        if not isinstance(ident, str) or not ident.strip() or ident in seen:
            raise ValueError(f"{kind}:{line}: invalid or duplicate id {ident!r}")
        seen.add(ident)
        if "text" in row and (not isinstance(row["text"], str) or not row["text"].strip()):
            raise ValueError(f"{kind}:{line}: empty text")
        if "label" in row and (type(row["label"]) is not int or row["label"] not in (0, 1)):
            raise ValueError(f"{kind}:{line}: label must be integer 0/1")
        if "score" in row:
            score = row["score"]
            if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError(f"{kind}:{line}: score must be finite in [0,1]")


def load_dataset(data_dir):
    base = Path(data_dir)
    manifest_path = base / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != VERSION or type(manifest.get("is_mock")) is not bool:
        raise ValueError("Invalid manifest version or mock flag")
    if manifest.get("label_map") != {"0": "non_biased", "1": "biased"}:
        raise ValueError("Invalid label map")
    for key in ("dataset_id", "source_commit"):
        if not isinstance(manifest.get(key), str) or not manifest[key]:
            raise ValueError(f"Missing manifest field {key}")
    required = {"train.jsonl": "train", "dev.jsonl": "input", "test.jsonl": "input", "labels/dev.jsonl": "gold", "labels/test.jsonl": "gold"}
    files = manifest.get("files", {})
    result = {}
    for name, kind in required.items():
        item = files.get(name, {})
        path = base / name
        if sha256(path) != item.get("sha256"):
            raise ValueError(f"{name}: sha256 mismatch")
        rows = read_jsonl(path)
        check_records(rows, kind)
        if len(rows) != item.get("rows"):
            raise ValueError(f"{name}: row count mismatch")
        result[name] = rows
    for split in ("dev", "test"):
        if [row["id"] for row in result[f"{split}.jsonl"]] != [row["id"] for row in result[f"labels/{split}.jsonl"]]:
            raise ValueError(f"{split}: input and gold order differ")
    return manifest, result


def validate_predictions(inputs, predictions, metadata, manifest_path, *, require_real=False):
    check_records(inputs, "input")
    check_records(predictions, "prediction")
    if [x["id"] for x in inputs] != [x["id"] for x in predictions]:
        raise ValueError("Prediction ids/count/order differ from input")
    required = {"schema_version", "method", "split", "dataset_manifest_sha256", "is_mock", "seed", "threshold", "code_commit", "model_name", "model_revision", "config", "environment"}
    if not required.issubset(metadata):
        raise ValueError("Missing run metadata fields")
    if metadata["schema_version"] != VERSION or type(metadata["is_mock"]) is not bool:
        raise ValueError("Invalid run version or mock flag")
    if metadata["split"] not in ("dev", "test") or type(metadata["seed"]) is not int:
        raise ValueError("Invalid run split or seed")
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if metadata["dataset_manifest_sha256"] != sha256(manifest_path):
        raise ValueError("Run uses a different data manifest")
    if metadata["is_mock"] != manifest["is_mock"]:
        raise ValueError("Dataset and run mock flags differ")
    expected = read_jsonl(Path(manifest_path).parent / (metadata["split"] + ".jsonl"))
    if inputs != expected:
        raise ValueError("Input does not match the declared manifest split")
    if require_real and metadata["is_mock"]:
        raise ValueError("Mock results cannot enter a real comparison")
    threshold = metadata["threshold"]
    if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Invalid threshold")
    for row in predictions:
        if row["label"] != int(row["score"] >= threshold):
            raise ValueError(f"{row['id']}: label contradicts score/threshold")


def binary_metrics(gold, predictions):
    check_records(gold, "gold")
    check_records(predictions, "prediction")
    if [x["id"] for x in gold] != [x["id"] for x in predictions]:
        raise ValueError("Gold and prediction ids/count/order differ")
    matrix = [[0, 0], [0, 0]]
    for actual, predicted in zip(gold, predictions):
        matrix[actual["label"]][predicted["label"]] += 1
    tn, fp = matrix[0]
    fn, tp = matrix[1]
    divide = lambda a, b: a / b if b else 0.0
    positive_f1 = divide(2 * tp, 2 * tp + fp + fn)
    negative_f1 = divide(2 * tn, 2 * tn + fp + fn)
    return {"n": len(gold), "accuracy": (tp + tn) / len(gold), "macro_f1": (positive_f1 + negative_f1) / 2, "positive_precision": divide(tp, tp + fp), "positive_recall": divide(tp, tp + fn), "positive_f1": positive_f1, "negative_f1": negative_f1, "confusion_matrix": matrix}
