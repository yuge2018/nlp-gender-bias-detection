"""Real CLI for the improved encoder (contract v1): train / predict.

退出码遵循 docs/INTERFACES.md：0 成功；2 参数/契约/数据错误；1 训练或加载失败。
"""
import argparse
import datetime
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

from src.common.contracts import (
    VERSION,
    check_records,
    read_jsonl,
    sha256,
    validate_predictions,
    write_json,
    write_jsonl,
)

REQUIRED_CONFIG_KEYS = ("method", "model_name", "model_revision", "max_length", "batch_size", "epochs", "learning_rate", "threshold")


class Logger:
    def __init__(self, path):
        self.path = Path(path)
        self.started = time.time()

    def __call__(self, message):
        stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{stamp}] {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(line + "\n")

    def close(self):
        elapsed = time.time() - self.started
        self(f"done in {elapsed:.1f}s")


def git_commit():
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10, check=True)
        return result.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def environment():
    info = {"python": sys.version.split()[0], "platform": platform.platform()}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["device"] = torch.cuda.get_device_name(0)
            info["cuda"] = torch.version.cuda
        else:
            info["device"] = "cpu"
    except ImportError:
        info["torch"] = "not installed"
    try:
        import transformers

        info["transformers"] = transformers.__version__
    except ImportError:
        info["transformers"] = "not installed"
    return info


def load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    missing = [key for key in REQUIRED_CONFIG_KEYS if key not in config]
    if missing:
        raise ValueError(f"{path}: missing config keys {missing}")
    if config["model_revision"].startswith("TO_BE"):
        raise ValueError(f"{path}: pin model_revision before training (see huggingface commit sha)")
    if not isinstance(config["seed"], int) or config["seed"] < 0:
        raise ValueError(f"{path}: seed must be a non-negative integer")
    if config["max_length"] not in (32, 64, 128, 256, 512):
        raise ValueError(f"{path}: max_length must be one of 32/64/128/256/512, got {config['max_length']}")
    if not 0 <= config["threshold"] <= 1:
        raise ValueError(f"{path}: threshold must be in [0,1]")
    return config


def _prepare_output_dir(raw):
    out = Path(raw).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"{out}: output directory is not empty; choose a fresh path")
    out.mkdir(parents=True, exist_ok=True)
    return out


def train(argv):
    parser = argparse.ArgumentParser(prog="src.improved train", description="Fine-tune the Chinese encoder (contract v1)")
    parser.add_argument("--data-dir", required=True, help="frozen data package with manifest.json")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True, help="fresh directory for model, tokenizer and metadata")
    parser.add_argument("--seed", type=int, help="override config seed")
    args = parser.parse_args(argv)

    from .data import load_training_data
    from .engine import train_encoder

    config = load_config(args.config)
    if args.seed is not None:
        config["seed"] = args.seed
    command = ["python", "-m", "src.improved", "train"] + argv
    out = _prepare_output_dir(args.output_dir)
    log = Logger(out / "log.txt")
    log(f"train start; config={json.dumps(config, ensure_ascii=False)}")

    data_dir = Path(args.data_dir)
    manifest, splits = load_training_data(data_dir)
    log(f"data {manifest['dataset_id']} is_mock={manifest['is_mock']} train={len(splits['train_texts'])} dev={len(splits['dev_texts'])}")

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'})")

    train_started = datetime.datetime.now().isoformat(timespec="seconds")
    result = train_encoder(
        splits["train_texts"], splits["train_labels"], splits["dev_texts"], splits["dev_labels"],
        model_name=config["model_name"], model_revision=config["model_revision"], config=config, device=device, log=log,
    )

    model, tokenizer = result.pop("model"), result.pop("tokenizer")
    model.save_pretrained(out, safe_serialization=True)
    tokenizer.save_pretrained(out)
    metadata = {
        "schema_version": VERSION,
        "method": config["method"],
        "is_mock": manifest["is_mock"],
        "model_name": config["model_name"],
        "model_revision": config["model_revision"],
        "seed": config["seed"],
        "dataset": {
            "data_dir": str(data_dir.resolve()),
            "dataset_id": manifest["dataset_id"],
            "dataset_manifest_sha256": sha256(data_dir / "manifest.json"),
            "train_n": len(splits["train_texts"]),
            "dev_n": len(splits["dev_texts"]),
            "train_positive_n": sum(splits["train_labels"]),
        },
        "config": config,
        "selection": {
            "criterion": "dev macro_f1 at threshold 0.5, earliest best epoch kept; threshold from dev grid 0.10-0.90 step 0.05",
            "best_epoch": result["best_epoch"],
            "history": result["history"],
            "dev_macro_f1_at_0.5": result["dev_macro_f1_at_0.5"],
            "threshold": result["threshold"],
            "dev_macro_f1_at_threshold": result["dev_macro_f1_at_threshold"],
        },
        "class_weights": result["class_weights"],
        "amp_bf16": result["amp_enabled"],
        "timing": {"started": train_started, "train_seconds": result["elapsed_seconds"]},
        "environment": environment(),
        "code_commit": git_commit(),
        "command": command,
    }
    write_json(out / "model_metadata.json", metadata)
    # 不叫 config.json：那是 HF 模型配置的名字，覆盖会毁掉离线加载。
    write_json(out / "train_config.json", config)
    log(f"saved model to {out}; best_epoch={result['best_epoch']} threshold={result['threshold']:.2f}")
    print(json.dumps({"output_dir": str(out), "best_epoch": result["best_epoch"],
                      "dev_macro_f1_at_0.5": result["dev_macro_f1_at_0.5"],
                      "dev_threshold": result["threshold"],
                      "dev_macro_f1_at_threshold": result["dev_macro_f1_at_threshold"]}, ensure_ascii=False))
    log.close()
    return 0


def predict(argv):
    parser = argparse.ArgumentParser(prog="src.improved predict", description="Run the saved encoder on a frozen split (contract v1)")
    parser.add_argument("--input", required=True, help="dev.jsonl or test.jsonl (id,text only; a label field is rejected)")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--manifest", required=True, help="manifest.json of the data package the input belongs to")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--threshold", type=float, help="override the threshold recorded at training (default: config threshold, usually 0.5)")
    parser.add_argument("--batch-size", type=int, help="override inference batch size")
    args = parser.parse_args(argv)

    from .adapter import load_predictor

    input_path = Path(args.input)
    split = input_path.stem
    if split not in ("dev", "test"):
        raise ValueError(f"{input_path}: input file must be dev.jsonl or test.jsonl, got {input_path.name}")
    inputs = read_jsonl(input_path)
    check_records(inputs, "input")

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    out = _prepare_output_dir(args.output_dir)
    log = Logger(out / "log.txt")
    log(f"predict start; input={input_path} split={split} n={len(inputs)}")

    model_dir = Path(args.model_dir)
    meta_path = model_dir / "model_metadata.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"{model_dir}: missing model_metadata.json; predict needs a model saved by 'src.improved train'")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    config = dict(metadata["config"])
    if args.batch_size:
        config["predict_batch_size"] = args.batch_size
    threshold = args.threshold if args.threshold is not None else config["threshold"]
    log(f"model {metadata['model_name']}@{metadata['model_revision']} seed={metadata['seed']} threshold={threshold}")

    predictor = load_predictor(str(model_dir))
    started = time.time()
    scores = predictor.predict_proba([row["text"] for row in inputs])
    predictions = [{"id": row["id"], "label": int(score >= threshold), "score": score} for row, score in zip(inputs, scores)]
    run_metadata = {
        "schema_version": VERSION,
        "method": metadata["method"],
        "split": split,
        "dataset_manifest_sha256": sha256(manifest_path),
        "is_mock": manifest["is_mock"],
        "seed": metadata["seed"],
        "threshold": threshold,
        "code_commit": git_commit(),
        "model_name": metadata["model_name"],
        "model_revision": metadata["model_revision"],
        "config": {"train": config, "predict": {"batch_size": config.get("predict_batch_size", 64), "input": str(input_path.resolve())}},
        "environment": environment(),
        "command": ["python", "-m", "src.improved", "predict"] + argv,
        "timing": {"started": datetime.datetime.now().isoformat(timespec="seconds"), "predict_seconds": time.time() - started},
    }
    validate_predictions(inputs, predictions, run_metadata, manifest_path)
    write_jsonl(out / "predictions.jsonl", predictions)
    write_json(out / "run_metadata.json", run_metadata)
    write_json(out / "config.json", run_metadata["config"])
    log(f"wrote {len(predictions)} predictions to {out}")
    print(json.dumps({"output_dir": str(out), "n": len(predictions), "threshold": threshold}, ensure_ascii=False))
    log.close()
    return 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0] not in ("train", "predict"):
        raise SystemExit("usage: python -m src.improved train|predict ... (bare --mock entry keeps the frozen smoke run)")
    command = argv[0]
    handlers = {"train": train, "predict": predict}
    try:
        return handlers[command](argv[1:])
    except (ValueError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    except Exception:  # noqa: BLE001 - contract requires a distinct code plus a kept log
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
