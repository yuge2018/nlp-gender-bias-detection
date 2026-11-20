"""P5 command line: build/seal challenge, score models on it, grouped report, review.

Exit codes follow the frozen interface contract: 0 success, 2 argument/contract/
data error, 1 runtime/model-loading failure.
"""
import argparse
import json
import sys
import time
from pathlib import Path

from src.common.contracts import VERSION, read_jsonl, sha256, write_json, write_jsonl

from . import build as challenge_build
from .cases import select_cases
from .config import load_config
from .data import align_run, check_models_consistent, load_challenge, load_run
from .metrics import build_subgroup_metrics, summarize

ROOT = Path(__file__).resolve().parents[2]


def _fresh_dir(path, what):
    target = Path(path)
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise ValueError(f"{target}: {what} output directory is not empty; choose a fresh path")
    target.mkdir(parents=True, exist_ok=True)
    return target


def _parse_specs(values, what):
    specs = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"{what} must be name=path, got {value!r}")
        name, path = value.split("=", 1)
        if not name.strip() or not path.strip():
            raise ValueError(f"{what} must be name=path, got {value!r}")
        specs.append((name.strip(), path.strip()))
    if len({name for name, _ in specs}) != len(specs):
        raise ValueError(f"{what} names must be unique")
    return specs


def _model_metadata(model_dir):
    path = Path(model_dir) / "model_metadata.json"
    if not path.is_file():
        raise ValueError(f"{path}: model metadata not found; train the model before scoring the challenge")
    return json.loads(path.read_text(encoding="utf-8"))


def score_command(challenge_dir, component, model_dir, name, output_dir, threshold=None):
    """Score the frozen challenge set with one saved model through its public adapter."""
    data = load_challenge(challenge_dir)
    metadata = _model_metadata(model_dir)
    if metadata.get("schema_version") != VERSION:
        raise ValueError(f"{model_dir}: unsupported model metadata schema_version")
    if component == "baseline":
        from src.baseline.adapter import load_predictor
    elif component == "improved":
        from src.improved.adapter import load_predictor
    else:
        raise ValueError("component must be baseline or improved")
    predictor = load_predictor(str(model_dir))
    texts = [row["text"] for row in data["challenge"]]
    scores = predictor.predict_proba(texts)
    if len(scores) != len(texts):
        raise ValueError(f"{model_dir}: predictor returned {len(scores)} scores for {len(texts)} texts")
    for score in scores:
        if type(score) not in (int, float) or not 0 <= score <= 1:
            raise ValueError(f"{model_dir}: score outside [0,1]")
    effective_threshold = float(metadata["threshold"]) if threshold is None else float(threshold)
    if not 0 <= effective_threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    predictions = [{"id": row["id"], "label": int(score >= effective_threshold), "score": float(score)} for row, score in zip(data["challenge"], scores)]
    run_metadata = {
        "schema_version": VERSION,
        "method": metadata.get("method", component),
        "split": "challenge",
        "dataset_manifest_sha256": metadata.get("dataset_manifest_sha256", ""),
        "is_mock": bool(metadata.get("is_mock", False)),
        "seed": metadata.get("seed", 42),
        "threshold": effective_threshold,
        "code_commit": metadata.get("code_commit", "unknown"),
        "model_name": metadata.get("model_name", ""),
        "model_revision": metadata.get("model_revision", ""),
        "config": metadata.get("config", {}),
        "environment": metadata.get("environment", {}),
        "challenge_dataset_id": data["manifest"]["dataset_id"],
        "challenge_manifest_sha256": data["manifest_sha256"],
        "model_dir": str(model_dir),
        "name": name,
        "n": len(predictions),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    target = _fresh_dir(output_dir, "challenge run")
    write_jsonl(target / "predictions.jsonl", predictions)
    write_json(target / "run_metadata.json", run_metadata)
    write_json(target / "config.json", {"model": metadata.get("config", {}), "run": {"split": "challenge", "threshold": effective_threshold, "input": str(Path(challenge_dir) / "challenge.jsonl")}})
    lines = [
        f"P5 challenge scoring ({name})",
        f"model: {run_metadata['model_name']} ({run_metadata['model_revision']}) from {model_dir}",
        f"challenge: {data['manifest']['dataset_id']} manifest_sha256={data['manifest_sha256']} rows={len(predictions)}",
        f"threshold: {effective_threshold}",
        "score meaning: label=1 model score, not a calibrated bias probability",
        f"is_mock: {run_metadata['is_mock']}",
    ]
    (target / "log.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"run_dir": str(target), "name": name, "n": len(predictions), "threshold": effective_threshold, "is_mock": run_metadata["is_mock"], "positive_rate": sum(row["label"] for row in predictions) / len(predictions)}


def report_command(challenge_dir, prediction_specs, output_dir, figures_dir=None, max_cases=16, notes_path=None, require_real=False, fresh=True):
    """Build subgroup metrics, error cases and figures from frozen prediction runs."""
    data = load_challenge(challenge_dir)
    specs = _parse_specs(prediction_specs, "--prediction")
    if not specs:
        raise ValueError("at least one --prediction name=run_dir is required")
    models = []
    for name, run_dir in specs:
        run = load_run(run_dir)
        if require_real and run["is_mock"]:
            raise ValueError(f"{run_dir}: mock predictions cannot enter a real analysis")
        aligned = align_run(data["challenge"], run)
        metadata = run["metadata"]
        models.append({
            "name": name,
            "run_dir": run["run_dir"],
            "method": metadata["method"],
            "model_name": metadata["model_name"],
            "model_revision": metadata["model_revision"],
            "seed": metadata["seed"],
            "threshold": run["threshold"],
            "is_mock": run["is_mock"],
            "split": metadata["split"],
            "dataset_manifest_sha256": metadata["dataset_manifest_sha256"],
            "predictions": aligned,
        })
    check_models_consistent(models)
    table = build_subgroup_metrics(data["challenge"], data["labels"], data["annotations"], models)
    cases, case_counts = select_cases(data["challenge"], data["labels"], data["annotations"], models, limit=max_cases, notes_path=notes_path)
    is_mock = models[0]["is_mock"]
    coverage = data["coverage"]
    warnings = coverage["warnings"] + summarize(table)
    if is_mock:
        warnings.append("mock predictions: 这是接线验证结果，不是模型研究结论")
    result = {
        "schema_version": VERSION,
        "is_mock": is_mock,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S+08:00", time.localtime()),
        "challenge": {
            "dataset_id": data["manifest"]["dataset_id"],
            "annotation_version": data["manifest"]["annotation_version"],
            "review_status": data["manifest"]["review_status"],
            "manifest_sha256": data["manifest_sha256"],
            "frozen_at": data["manifest"]["frozen_at"],
            "n": coverage["n"],
            "positive": coverage["positive"],
            "negative": coverage["negative"],
            "by_target": coverage["by_target"],
            "by_phenomenon": coverage["by_phenomenon"],
            "multi_target": data["manifest"].get("composition", {}).get("multi_target"),
        },
        "models": [{key: value for key, value in model.items() if key != "predictions"} for model in models],
        "groups": table,
        "case_counts": case_counts,
        "warnings": warnings,
        "metric_definitions": {
            "macro_f1": "固定对0/1两类平均，分母为0按0计（与项目主指标一致）",
            "macro_f1_defined_classes": "只平均当前子组实际出现的类别，单极性子组应看此列",
            "fpr": "无偏见样本被判为1的比例，分母为0时null",
            "fnr": "有偏见样本被判为0的比例，分母为0时null",
            "grouping": "target维度按标注多标签切分，同一条可进入多个目标组；不按作者性别分组",
        },
    }
    out = _fresh_dir(output_dir, "P5 report") if fresh else Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "subgroup_metrics.json", result)
    write_jsonl(out / "error_cases.jsonl", cases)
    lines = [
        "P5 challenge subgroup report",
        f"challenge: {data['manifest']['dataset_id']} ({coverage['n']} rows) manifest_sha256={data['manifest_sha256']}",
        "models: " + ", ".join("%s(%s,threshold=%s)" % (model["name"], model["model_name"], model["threshold"]) for model in models),
        f"is_mock: {is_mock}",
        f"cases: {json.dumps(case_counts, ensure_ascii=False)}",
        "warnings: " + "; ".join(warnings),
    ]
    (out / "log.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    figures = []
    if figures_dir:
        from .figures import draw_composition, draw_group_metrics

        figures.append(draw_composition(result["challenge"], Path(figures_dir) / "challenge_composition.png"))
        if not is_mock and any(entry["dimension"] == "target" for entry in table):
            figures.append(draw_group_metrics(table, Path(figures_dir) / "target_metrics.png", models[0]["name"]))
    return {
        "output_dir": str(out),
        "is_mock": is_mock,
        "n": coverage["n"],
        "models": [model["name"] for model in models],
        "cases": len(cases),
        "warnings": warnings,
        "figures": figures,
    }


def _kappa(first, second):
    """Cohen's kappa for two integer label lists (no external dependency)."""
    if len(first) != len(second) or not first:
        raise ValueError("kappa needs two aligned non-empty lists")
    observed = sum(1 for a, b in zip(first, second) if a == b) / len(first)
    categories = sorted({*first, *second})
    expected = sum((first.count(category) / len(first)) * (second.count(category) / len(second)) for category in categories)
    return None if expected == 1 else (observed - expected) / (1 - expected)


def _set_agreement(first, second):
    if not first and not second:
        return 1.0
    union = set(first) | set(second)
    return len(set(first) & set(second)) / len(union)


def review_command(first_dir, second_dir, output):
    """Compare two independent annotation passes: labels, targets, phenomena, disagreements."""
    first = load_challenge(first_dir)
    second = load_challenge(second_dir)
    if [row["id"] for row in first["challenge"]] != [row["id"] for row in second["challenge"]]:
        raise ValueError("the two annotation passes cover different ids; review the same frozen text list")
    labels_a = [row["label"] for row in first["labels"]]
    labels_b = [row["label"] for row in second["labels"]]
    annotation_a = {row["id"]: row for row in first["annotations"]}
    annotation_b = {row["id"]: row for row in second["annotations"]}
    targets_a = [tuple(sorted(annotation_a[ident]["targets"])) for ident in annotation_a]
    targets_b = [tuple(sorted(annotation_b[ident]["targets"])) for ident in annotation_b]
    phenomena_a = [tuple(sorted(annotation_a[ident]["phenomena"])) for ident in annotation_a]
    phenomena_b = [tuple(sorted(annotation_b[ident]["phenomena"])) for ident in annotation_b]
    disagreements = []
    for ident in annotation_a:
        row_a, row_b = annotation_a[ident], annotation_b[ident]
        if row_a["targets"] != row_b["targets"] or row_a["phenomena"] != row_b["phenomena"]:
            disagreements.append({"id": ident, "first": {"targets": row_a["targets"], "phenomena": row_a["phenomena"]}, "second": {"targets": row_b["targets"], "phenomena": row_b["phenomena"]}})
    result = {
        "schema_version": VERSION,
        "first": {"dir": str(first_dir), "reviewers": sorted({name for row in first["annotations"] for name in row["reviewers"]}), "manifest_sha256": first["manifest_sha256"]},
        "second": {"dir": str(second_dir), "reviewers": sorted({name for row in second["annotations"] for name in row["reviewers"]}), "manifest_sha256": second["manifest_sha256"]},
        "n": len(labels_a),
        "label_agreement": sum(1 for a, b in zip(labels_a, labels_b) if a == b) / len(labels_a),
        "label_kappa": _kappa(labels_a, labels_b),
        "target_exact_match_rate": sum(1 for a, b in zip(targets_a, targets_b) if a == b) / len(targets_a),
        "target_jaccard_mean": sum(_set_agreement(a, b) for a, b in zip(targets_a, targets_b)) / len(targets_a),
        "phenomenon_exact_match_rate": sum(1 for a, b in zip(phenomena_a, phenomena_b) if a == b) / len(phenomena_a),
        "phenomenon_jaccard_mean": sum(_set_agreement(a, b) for a, b in zip(phenomena_a, phenomena_b)) / len(phenomena_a),
        "disagreements": disagreements,
        "note": "双人复核只度量分歧；最终文件由P5统一维护并记录裁决，不把多数票当作金标真理。",
    }
    target = Path(output)
    write_json(target, result)
    return {"output": str(target), "label_agreement": result["label_agreement"], "label_kappa": result["label_kappa"], "disagreements": len(disagreements)}


def _mock_scores(text, salt):
    """Deterministic development substitute; never a model result."""
    value = sum(ord(char) * (index + 1) for index, char in enumerate(text)) + salt * 2654435761
    return (value % 1000) / 1000.0


def mock_command(output_dir):
    """Legacy `--mock` wiring: fixture challenge plus two clearly mock predictors."""
    out = _fresh_dir(output_dir, "P5 mock")
    challenge_dir = out / "challenge"
    challenge_build.build_challenge(challenge_dir)
    specs = []
    for name, salt in (("baseline-mock", 11), ("improved-mock", 23)):
        rows = read_jsonl(challenge_dir / "challenge.jsonl")
        predictions = [{"id": row["id"], "label": int(_mock_scores(row["text"], salt) >= 0.5), "score": _mock_scores(row["text"], salt)} for row in rows]
        run_dir = out / f"run-{name}"
        run_dir.mkdir()
        write_jsonl(run_dir / "predictions.jsonl", predictions)
        write_json(run_dir / "run_metadata.json", {
            "schema_version": VERSION, "method": name, "split": "challenge", "dataset_manifest_sha256": sha256(challenge_dir / "manifest.json"),
            "is_mock": True, "seed": 42, "threshold": 0.5, "code_commit": "MOCK", "model_name": "deterministic-char-hash-substitute",
            "model_revision": "mock-v1", "config": {"mock": True},
            "environment": {"python": sys.version.split()[0]},
        })
        specs.append(f"{name}={run_dir}")
    report = report_command(challenge_dir, specs, out, max_cases=8, fresh=False)
    status = {
        "schema_version": VERSION,
        "component": "fairness",
        "is_mock": True,
        "message": "Wiring check only: curated challenge draft plus two deterministic substitutes; no model claim.",
        "report": report,
    }
    write_json(out / "status.json", status)
    (out / "log.txt").write_text("MOCK ONLY: P5 challenge -> subgroup metrics -> cases wiring passed.\n", encoding="utf-8")
    return status


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "--mock":
        parser = argparse.ArgumentParser(description="P5 mock wiring")
        parser.add_argument("--mock", action="store_true")
        parser.add_argument("--output-dir", required=True)
        args = parser.parse_args(arguments)
        try:
            print(json.dumps(mock_command(args.output_dir), ensure_ascii=False, indent=2))
            return 0
        except (ValueError, OSError) as exc:
            print(f"P5 fairness error: {exc}", file=sys.stderr)
            return 2
    parser = argparse.ArgumentParser(description="P5 multi-gender challenge set and error analysis")
    try:
        config = load_config()
    except ValueError:
        config = {}
    default_challenge = config.get("challenge_dir", "data/challenge/" + challenge_build.DATASET_ID)
    report_config = config.get("report", {})
    subparsers = parser.add_subparsers(dest="command", required=True)
    builder = subparsers.add_parser("build", help="Write the curated challenge set and sealed manifest")
    builder.add_argument("--output-dir", default=default_challenge)
    sealer = subparsers.add_parser("seal", help="Recompute hashes after an explicit annotation revision")
    sealer.add_argument("--challenge-dir", required=True)
    scorer = subparsers.add_parser("score", help="Score the challenge set with one saved model")
    scorer.add_argument("--challenge-dir", default=default_challenge)
    scorer.add_argument("--component", choices=("baseline", "improved"), required=True)
    scorer.add_argument("--model-dir", required=True)
    scorer.add_argument("--name", required=True)
    scorer.add_argument("--output-dir", required=True)
    scorer.add_argument("--threshold", type=float, default=None)
    reporter = subparsers.add_parser("report", help="Subgroup metrics, error cases and figures from prediction runs")
    reporter.add_argument("--challenge-dir", default=default_challenge)
    reporter.add_argument("--prediction", action="append", required=True, help="name=run_dir; repeat for several models")
    reporter.add_argument("--output-dir", required=True)
    reporter.add_argument("--figures-dir", default=report_config.get("figures_dir"))
    reporter.add_argument("--max-cases", type=int, default=report_config.get("max_cases", 16))
    reporter.add_argument("--case-notes", default=report_config.get("case_notes"))
    reporter.add_argument("--require-real", action="store_true", default=bool(report_config.get("require_real", False)))
    reviewer = subparsers.add_parser("review", help="Compare two independent annotation passes")
    reviewer.add_argument("--first", required=True)
    reviewer.add_argument("--second", required=True)
    reviewer.add_argument("--output", required=True)
    args = parser.parse_args(arguments)
    try:
        if args.command == "build":
            result = challenge_build.build_challenge(args.output_dir)
        elif args.command == "seal":
            result = challenge_build.seal_challenge(args.challenge_dir)
        elif args.command == "score":
            result = score_command(args.challenge_dir, args.component, args.model_dir, args.name, args.output_dir, args.threshold)
        elif args.command == "report":
            result = report_command(args.challenge_dir, args.prediction, args.output_dir, args.figures_dir, args.max_cases, args.case_notes, args.require_real)
        else:
            result = review_command(args.first, args.second, args.output)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError, KeyError) as exc:
        print(f"P5 fairness error: {exc}", file=sys.stderr)
        return 2
    except ImportError as exc:
        print(f"P5 fairness environment error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
