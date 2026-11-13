"""Independent component smoke runs. All generated results are explicitly mock."""
import argparse
import json
import shutil
import sys
from pathlib import Path

from .contracts import VERSION, binary_metrics, load_dataset, sha256, validate_predictions, write_json, write_jsonl
from .model_api import MockPredictor

ROOT = Path(__file__).resolve().parents[2]


def run(component):
    parser = argparse.ArgumentParser(description=f"{component}: planning scaffold, mock mode only")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if not args.mock:
        parser.error("Real implementation is assigned in docs/roles/. Use --mock for the independent starter.")
    out = Path(args.output_dir).resolve()
    fixture = ROOT / "fixtures" / "dataset"
    if out == fixture or fixture in out.parents or out == ROOT or ROOT in out.parents and out.parts[len(ROOT.parts)] != "runs":
        parser.error("Choose a new output directory under runs/ or outside the source tree")
    if out.exists() and any(out.iterdir()):
        parser.error("Output directory is not empty; choose a fresh path")
    out.mkdir(parents=True, exist_ok=True)
    manifest, dataset = load_dataset(fixture)
    inputs = dataset["test.jsonl"]
    scores = MockPredictor().predict_proba([x["text"] for x in inputs])
    predictions = [{"id": x["id"], "label": int(score >= 0.5), "score": score} for x, score in zip(inputs, scores)]
    metadata = {"schema_version": VERSION, "method": component, "split": "test", "dataset_manifest_sha256": sha256(fixture / "manifest.json"), "is_mock": True, "seed": 42, "threshold": 0.5, "code_commit": "MOCK", "model_name": "constant-0.5-development-substitute", "model_revision": "mock-v1", "config": {"mock": True}, "environment": {"python": sys.version.split()[0]}}
    validate_predictions(inputs, predictions, metadata, fixture / "manifest.json")
    status = {"schema_version": VERSION, "component": component, "is_mock": True, "message": "Development substitute only; no real training, evaluation study or UI implemented."}
    write_json(out / "status.json", status)
    if component == "data":
        shutil.copytree(fixture, out / "dataset")
    elif component in ("baseline", "improved"):
        write_jsonl(out / "predictions.jsonl", predictions)
        write_json(out / "run_metadata.json", metadata)
        write_json(out / "config.json", metadata["config"])
    elif component == "evaluation":
        write_json(out / "metrics.json", {"schema_version": VERSION, "is_mock": True, **binary_metrics(dataset["labels/test.jsonl"], predictions)})
    elif component == "fairness":
        shutil.copytree(ROOT / "fixtures" / "challenge", out / "challenge")
        write_json(out / "subgroup_metrics.json", {"schema_version": VERSION, "is_mock": True, "groups": [], "message": "P5 implements real group metrics; synthetic examples are not coverage evidence."})
    elif component == "demo":
        write_json(out / "demo_response.json", {"is_mock": True, "model": metadata["model_name"], "scores": scores, "message": "P6 implements the actual UI; adapter wiring verified."})
    (out / "log.txt").write_text("MOCK ONLY: independent component interface smoke passed.\n", encoding="utf-8")
    print(json.dumps({"component": component, "is_mock": True, "output_dir": str(out)}, ensure_ascii=False))
