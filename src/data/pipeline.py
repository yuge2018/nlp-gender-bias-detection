"""Publish and verify reproducible P1 data packages."""
import json
from pathlib import Path
import tempfile

from src.common.contracts import load_dataset, read_jsonl, sha256, write_json, write_jsonl
from .audit import audit_records
from .convert import clean_training, convert_corpora
from .source import load_sources

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/data/default.json"
DEFAULT_LOCK = ROOT / "configs/data/source_audit.json"
OFFICIAL_ID = "corgi-official-v1"
CLEAN_ID = "corgi-clean-train-v1"


def _safe_file(base, relative):
    if not isinstance(relative, str) or "\\" in relative:
        raise ValueError("Unsafe manifest path")
    relative_path = Path(relative)
    target = (base / relative_path).resolve()
    if relative_path.is_absolute() or ".." in relative_path.parts or base.resolve() not in target.parents:
        raise ValueError(f"Unsafe manifest path: {relative}")
    return target


def file_hashes(directory):
    base = Path(directory)
    return {p.relative_to(base).as_posix(): sha256(p) for p in sorted(base.rglob("*")) if p.is_file()}


def write_package(directory, dataset_id, records, mapping, removed, source_commit, verified_sources, settings):
    base = Path(directory)
    base.mkdir(parents=True, exist_ok=False)
    write_jsonl(base / "train.jsonl", records["train"])
    for split in ("dev", "test"):
        write_jsonl(base / f"{split}.jsonl", [{"id": row["id"], "text": row["text"]} for row in records[split]])
        write_jsonl(base / "labels" / f"{split}.jsonl", [{"id": row["id"], "label": row["label"]} for row in records[split]])
    kept_ids = {row["id"] for rows in records.values() for row in rows}
    write_jsonl(base / "audit/id_mapping.jsonl", [item for item in mapping if item["id"] in kept_ids])
    write_jsonl(base / "audit/removed_train.jsonl", removed)
    report, duplicates, near = audit_records(records, settings)
    report.update({"schema_version": "1.0", "dataset_id": dataset_id, "is_mock": False, "removed_train_rows": len(removed), "removed_train_policy": "Exact raw-text matches with dev/test, without reading their labels; original IDs retained. This is a transductive sensitivity analysis, not an unseen-test blind evaluation." if dataset_id == CLEAN_ID else "None: preserve official splits and all detection samples."})
    write_jsonl(base / "audit/duplicate_groups.jsonl", duplicates)
    write_jsonl(base / "audit/near_duplicates.jsonl", near)
    write_json(base / "audit/quality_report.json", report)
    write_json(base / "audit/source_verification.json", {"source_commit": source_commit, "files": verified_sources})
    files, audit_files = {}, {}
    for path in sorted(base.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(base).as_posix()
        if path.suffix == ".jsonl":
            files[relative] = {"sha256": sha256(path), "rows": len(read_jsonl(path))}
        else:
            audit_files[relative] = {"sha256": sha256(path)}
    manifest = {"schema_version": "1.0", "dataset_id": dataset_id, "is_mock": False, "source_commit": source_commit, "label_map": {"0": "non_biased", "1": "biased"}, "preparation_version": "p1-v1", "source_splits": {"train": "train", "dev": "valid", "test": "test"}, "id_policy": "split-original-1-based-detection-index; keep gaps after cleaning; id/order/source metadata MUST NOT be model features", "text_policy": "Original text bytes represented as UTF-8 JSON strings, no text normalization; audit normalization only", "files": files, "audit_files": audit_files}
    write_json(base / "manifest.json", manifest)
    verify_package(base)
    return report


def verify_package(directory):
    base = Path(directory)
    manifest, records = load_dataset(base)
    if manifest["is_mock"] is not False:
        raise ValueError("P1 real-package verifier does not accept mock data")
    expected_files = {"manifest.json"}
    for section in ("files", "audit_files"):
        for relative, spec in manifest.get(section, {}).items():
            path = _safe_file(base, relative)
            if sha256(path) != spec["sha256"]:
                raise ValueError(f"{relative}: SHA256 mismatch")
            if "rows" in spec and len(read_jsonl(path)) != spec["rows"]:
                raise ValueError(f"{relative}: row count mismatch")
            expected_files.add(relative)
    actual_files = set(file_hashes(base))
    if actual_files != expected_files:
        raise ValueError("Unexpected/missing files in data package")
    mapping = read_jsonl(base / "audit/id_mapping.jsonl")
    all_ids = [row["id"] for name, rows in records.items() if name in ("train.jsonl", "dev.jsonl", "test.jsonl") for row in rows]
    if len(set(all_ids)) != len(all_ids) or len(mapping) != len(all_ids) or {item["id"] for item in mapping} != set(all_ids):
        raise ValueError("Audit mapping does not cover dataset IDs exactly once")
    return {"dataset_id": manifest["dataset_id"], "is_mock": False, "dataset_manifest_sha256": sha256(base / "manifest.json"), "rows": {split: len(records[f"{split}.jsonl"]) for split in ("train", "dev", "test")}, "verified_files": len(expected_files)}


def prepare(raw_dir, output_dir, *, config_path=DEFAULT_CONFIG, lock_path=DEFAULT_LOCK, offline=False):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    lock, corpora, verified_sources = load_sources(raw_dir, lock_path, offline=offline)
    if config["source_commit"] != lock["source_commit"]:
        raise ValueError("Data config and source lock use different commits")
    records, mapping = convert_corpora(corpora, expected_splits=lock["splits"])
    cleaned, removed = clean_training(records)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    prepared = {}
    # TemporaryDirectory only owns this newly allocated random staging folder.
    with tempfile.TemporaryDirectory(prefix=".p1-build-", dir=output) as stage_name:
        stage = Path(stage_name)
        for dataset_id, version_rows, removed_rows in ((OFFICIAL_ID, records, []), (CLEAN_ID, cleaned, removed)):
            report = write_package(stage / dataset_id, dataset_id, version_rows, mapping, removed_rows, lock["source_commit"], verified_sources, config["near_duplicates"])
            prepared[dataset_id] = report
        # Verify all existing destinations before publishing either new version.
        for dataset_id in prepared:
            target = output / dataset_id
            if target.exists() and file_hashes(target) != file_hashes(stage / dataset_id):
                raise ValueError(f"{target}: existing package differs; choose a new output directory")
        for dataset_id in prepared:
            target = output / dataset_id
            if not target.exists():
                (stage / dataset_id).rename(target)
    result = {"schema_version": "1.0", "is_mock": False, "source_commit": lock["source_commit"], "packages": {dataset_id: {**verify_package(output / dataset_id), "audit": report} for dataset_id, report in prepared.items()}}
    return result
