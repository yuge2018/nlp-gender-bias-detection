"""Dataset access for P2.

Training reads ``train.jsonl``, ``dev.jsonl`` and ``labels/dev.jsonl`` only.
The test split and its gold labels are never opened here; ``test.jsonl`` is read
for inference only to confirm that an input file is a frozen split of the
declared manifest. Every file is checked against the manifest SHA256 and row
count before use.
"""
import json
from pathlib import Path

from src.common.contracts import VERSION, check_records, read_jsonl, sha256

REQUIRED_MANIFEST_FIELDS = ("dataset_id", "source_commit")


def load_manifest(data_dir):
    """Return ``(base_dir, manifest, manifest_sha256)`` after schema and hash checks."""
    base = Path(data_dir)
    manifest_path = base / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"{manifest_path}: dataset manifest not found")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{manifest_path}: invalid JSON ({exc})") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"{manifest_path}: manifest must be an object")
    if manifest.get("schema_version") != VERSION:
        raise ValueError(f"{manifest_path}: unsupported schema_version")
    if type(manifest.get("is_mock")) is not bool:
        raise ValueError(f"{manifest_path}: is_mock must be a boolean")
    if manifest.get("label_map") != {"0": "non_biased", "1": "biased"}:
        raise ValueError(f"{manifest_path}: unexpected label_map")
    for key in REQUIRED_MANIFEST_FIELDS:
        value = manifest.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{manifest_path}: missing manifest field {key}")
    return base, manifest, sha256(manifest_path)


def read_verified_rows(base, manifest, name, kind):
    """Read one manifest file after SHA256 and row-count checks, then validate its records."""
    item = manifest.get("files", {}).get(name)
    if not isinstance(item, dict):
        raise ValueError(f"{name}: manifest has no file entry")
    path = base / name
    if not path.is_file():
        raise ValueError(f"{path}: missing data file")
    if sha256(path) != item.get("sha256"):
        raise ValueError(f"{name}: sha256 mismatch against manifest")
    rows = read_jsonl(path)
    check_records(rows, kind)
    if len(rows) != item.get("rows"):
        raise ValueError(f"{name}: row count mismatch against manifest")
    return rows


def load_training_data(data_dir):
    """Load train/dev and dev gold for training and dev-only model selection."""
    base, manifest, manifest_sha256 = load_manifest(data_dir)
    train = read_verified_rows(base, manifest, "train.jsonl", "train")
    dev = read_verified_rows(base, manifest, "dev.jsonl", "input")
    dev_gold = read_verified_rows(base, manifest, "labels/dev.jsonl", "gold")
    if [row["id"] for row in dev] != [row["id"] for row in dev_gold]:
        raise ValueError("dev: input and gold ids/order differ")
    return {"data_dir": str(base), "manifest": manifest, "manifest_sha256": manifest_sha256, "train": train, "dev": dev, "dev_gold": dev_gold}


def load_inference_input(path):
    """Load an unlabeled prediction input; labels or extra fields are rejected."""
    target = Path(path)
    if not target.is_file():
        raise ValueError(f"{target}: input file not found")
    try:
        rows = read_jsonl(target)
    except ValueError as exc:
        raise ValueError(f"{target}: {exc}") from exc
    extra = sorted({key for row in rows for key in row} - {"id", "text"})
    if extra:
        raise ValueError(f"{target}: inference input has forbidden fields {extra}; labels and metadata must not be passed")
    check_records(rows, "input")
    return rows


def match_split(data_dir, inputs):
    """Return 'dev' or 'test' when the input exactly matches that frozen split, else raise."""
    base = Path(data_dir)
    expected_ids = [(row["id"], row["text"]) for row in inputs]
    for split in ("dev", "test"):
        path = base / f"{split}.jsonl"
        if not path.is_file():
            continue
        rows = read_jsonl(path)
        if [(row["id"], row["text"]) for row in rows] == expected_ids:
            return split
    raise ValueError("input rows do not match the manifest dev/test split; ids, texts and order must be exactly the frozen split")
