"""Build the machine-readable acceptance record for P3 (configs/improved/acceptance.json).

Weights, data packages and run outputs stay out of Git by repo convention; this script
reads them from a local checkout and distills everything a teammate needs to verify the
published numbers into one committed JSON. Regenerate after any new run:

    python -m src.improved.acceptance
"""
import datetime
import json
import sys
from pathlib import Path

from src.common.contracts import binary_metrics, read_jsonl, sha256
from src.improved.cli import environment, git_commit

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "configs" / "improved" / "acceptance.json"

# (tag, config file, data package dir) — order fixed so the JSON diffs cleanly.
TRAIN_RUNS = [
    ("improved-i1-s42", "configs/improved/default.json", "data/processed/corgi-official-v1"),
    ("improved-i1-s43", "configs/improved/default.json", "data/processed/corgi-official-v1"),
    ("improved-i1-s44", "configs/improved/default.json", "data/processed/corgi-official-v1"),
    ("improved-i2-s42", "configs/improved/i2_class_weight.json", "data/processed/corgi-official-v1"),
    ("improved-i3-len256-s42", "configs/improved/i3_length256.json", "data/processed/corgi-official-v1"),
    ("improved-l1-clean-s42", "configs/improved/default.json", "data/processed/corgi-clean-train-v1"),
]

DATASETS = {
    "corgi-official-v1": "data/processed/corgi-official-v1",
    "corgi-clean-train-v1": "data/processed/corgi-clean-train-v1",
}


def dataset_summaries():
    out = {}
    for name, rel in DATASETS.items():
        base = ROOT / rel
        manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
        counts = {}
        for split in ("train", "dev", "test"):
            counts[split] = len(read_jsonl(base / f"{split}.jsonl"))
        out[name] = {
            "manifest_sha256": sha256(base / "manifest.json"),
            "rows": counts,
            "verify_cmd": f"python -m src.data verify --data-dir {rel}",
        }
    return out


def training_run(tag, config_rel, data_rel, model_meta):
    model_dir = ROOT / "artifacts" / tag
    selection = model_meta["selection"]
    return {
        "tag": tag,
        "config": config_rel,
        "data_dir": data_rel,
        "dataset_manifest_sha256": sha256(ROOT / data_rel / "manifest.json"),
        "seed": model_meta["seed"],
        "model_revision": model_meta["model_revision"],
        "best_epoch": selection["best_epoch"],
        "dev_macro_f1_at_0.5": selection["dev_macro_f1_at_0.5"],
        "dev_threshold": selection["threshold"],
        "dev_macro_f1_at_threshold": selection["dev_macro_f1_at_threshold"],
        "class_weights": model_meta.get("class_weights"),
        "amp_bf16": model_meta.get("amp_bf16"),
        "train_seconds": model_meta["timing"].get("train_seconds"),
        "command": model_meta.get("command"),
        "evidence_in_git": f"artifacts/{tag}/model_metadata.json",
        "weights_sha256": {
            name: sha256(model_dir / name)
            for name in ("model.safetensors",)
            if (model_dir / name).is_file()
        },
    }


def prediction_run(run_dir, data_rel):
    meta = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    preds = read_jsonl(run_dir / "predictions.jsonl")
    gold = read_jsonl(ROOT / data_rel / "labels" / f"{meta['split']}.jsonl")
    metrics = binary_metrics(gold, preds)
    return {
        "run_dir": f"runs/{run_dir.name}",
        "model_tag": run_dir.name.rsplit("-", 1)[0] if "-thr" in run_dir.name else run_dir.name,
        "split": meta["split"],
        "threshold": meta["threshold"],
        "n": metrics["n"],
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "positive_precision": metrics["positive_precision"],
        "positive_recall": metrics["positive_recall"],
        "positive_f1": metrics["positive_f1"],
        "negative_f1": metrics["negative_f1"],
        "confusion_matrix": metrics["confusion_matrix"],
        "is_mock": meta["is_mock"],
        "predictions_sha256": sha256(run_dir / "predictions.jsonl"),
        "run_metadata_sha256": sha256(run_dir / "run_metadata.json"),
    }


def main():
    missing = [tag for tag, _, _ in TRAIN_RUNS if not (ROOT / "artifacts" / tag / "model_metadata.json").is_file()]
    if missing:
        print(f"missing local training runs: {missing}; regenerate them first (see reproduce.train)", file=sys.stderr)
        return 2

    per_model = {
        tag: json.loads((ROOT / "artifacts" / tag / "model_metadata.json").read_text(encoding="utf-8"))
        for tag, _, _ in TRAIN_RUNS
    }
    prediction_runs = []
    for tag, _, data_rel in TRAIN_RUNS:
        for run_dir in sorted((ROOT / "runs").glob(f"{tag}-*")):
            if (run_dir / "predictions.jsonl").is_file():
                prediction_runs.append(prediction_run(run_dir, data_rel))

    record = {
        "schema_version": "1.0",
        "component": "improved",
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "generated_by": "python -m src.improved.acceptance",
        "author": {"name": "张羽纶", "student_id": "202315060337", "github": "zy1un"},
        "code_commit": git_commit(),
        "environment": environment(),
        "model": {
            "name": "hfl/chinese-roberta-wwm-ext",
            "revision": per_model["improved-i1-s42"]["model_revision"],
            "num_labels": 2,
            "score_definition": "softmax label=1 分量，未做概率校准",
        },
        "datasets": dataset_summaries(),
        "shared_model_for_demo": {
            "dir": "artifacts/improved-i2-s42",
            "why": "I2 类别加权版，同规则下 test Macro-F1 最高（0.8461@0.5），组长建议 P6 接此模型",
            "usage": "from src.improved.adapter import load_predictor; p = load_predictor('artifacts/improved-i2-s42'); scores = p.predict_proba([...])",
            "offline": True,
            "test_macro_f1_at_0.5": 0.8461,
        },
        "training_runs": [training_run(tag, cfg, data, per_model[tag]) for tag, cfg, data in TRAIN_RUNS],
        "prediction_runs": prediction_runs,
        "reproduce": {
            "setup": "pip install -r requirements/p3.txt（torch 需 CUDA 构建，见文件头注释）；python -m src.data prepare --raw-dir data/raw/corgi-pinned --output-dir data/processed",
            "train": "python -m src.improved train --data-dir <data_dir> --config <config> --output-dir artifacts/<tag> --seed <seed>",
            "predict": "python -m src.improved predict --input <data_dir>/test.jsonl --model-dir artifacts/<tag> --manifest <data_dir>/manifest.json --output-dir runs/<tag>-test",
            "regenerate_this_file": "python -m src.improved.acceptance",
            "note": "GPU 内核有 cuDNN 非确定性，重训分数会有 ±0.011 量级的种子间波动（见 training_runs 三种子）；预测与指标可精确复算",
        },
        "notes": [
            "权重、数据包与运行输出按仓库约定不进 Git；本文件与其引用的 sha256 是它们的验证凭据。",
            "runs/improved-i2-s42-test 的预测与元数据全文已提交进 Git，作为可直接打开核对的实跑样本。",
            "正式主表以 P4 从 run 目录重算的 metrics.json 为准，本文件数字由契约参考实现 binary_metrics 复算得出。",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    OUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {OUT} ({len(text)} bytes, {len(record['training_runs'])} training runs, {len(record['prediction_runs'])} prediction runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
