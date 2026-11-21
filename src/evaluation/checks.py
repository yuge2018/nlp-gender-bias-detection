"""契约检查：数据包、run 元信息、预测文件。

所有失败都抛 ContractError，由 CLI 统一转成退出码 2 并在 stderr 指明
路径、行号或 ID，绝不静默补标签或截断数据。
"""

from __future__ import annotations

import math
from pathlib import Path

from src.common.contracts import (
    GOLD_FIELDS,
    MANIFEST_REQUIRED,
    PREDICTION_FIELDS,
    RUN_REQUIRED,
    SPLITS,
    TEXT_FIELDS,
    ContractError,
    check_schema_version,
    is_int_label,
    read_json,
    read_jsonl,
    require_fields,
    sha256_file,
)


def load_manifest(data_dir: str | Path) -> tuple[dict, str, Path]:
    root = Path(data_dir)
    manifest_path = root / "manifest.json"
    manifest = read_json(manifest_path)
    require_fields(manifest, MANIFEST_REQUIRED, str(manifest_path))
    check_schema_version(manifest, str(manifest_path))
    if not isinstance(manifest["is_mock"], bool):
        raise ContractError(f"{manifest_path}: is_mock 必须是布尔值")
    if not isinstance(manifest["files"], dict) or not manifest["files"]:
        raise ContractError(f"{manifest_path}: files 必须是非空对象")
    return manifest, sha256_file(manifest_path), root


def verify_manifest_files(manifest: dict, root: Path) -> None:
    """逐个核对 manifest 里登记的 sha256 与行数。"""
    for rel, info in manifest["files"].items():
        if not isinstance(info, dict) or "sha256" not in info or "rows" not in info:
            raise ContractError(f"{root / 'manifest.json'}: files[{rel!r}] 缺少 sha256 或 rows")
        path = root / rel
        if not path.is_file():
            raise ContractError(f"{path}: manifest 登记的文件不存在")
        actual_sha = sha256_file(path)
        if actual_sha != info["sha256"]:
            raise ContractError(
                f"{path}: sha256 不一致（manifest={info['sha256'][:12]}…，实际={actual_sha[:12]}…）")
        rows = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
        if rows != info["rows"]:
            raise ContractError(f"{path}: 行数不一致（manifest={info['rows']}，实际={rows}）")


def _check_ids(rows: list[dict], where: str) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for lineno, row in enumerate(rows, start=1):
        rid = row.get("id")
        if not isinstance(rid, str) or not rid.strip():
            raise ContractError(f"{where}:{lineno}: id 必须是非空字符串")
        if rid in seen:
            raise ContractError(f"{where}:{lineno}: id 重复 {rid!r}")
        seen.add(rid)
        ids.append(rid)
    return ids


def validate_data_package(data_dir: str | Path, split: str) -> dict:
    """校验数据包并返回 manifest、hash、样本 ID 与金标。"""
    if split not in SPLITS:
        raise ContractError(f"split={split!r} 不支持，应为 {list(SPLITS)}")
    manifest, manifest_sha, root = load_manifest(data_dir)
    verify_manifest_files(manifest, root)

    text_path = root / f"{split}.jsonl"
    gold_path = root / "labels" / f"{split}.jsonl"
    texts = read_jsonl(text_path, TEXT_FIELDS)
    golds = read_jsonl(gold_path, GOLD_FIELDS)

    text_ids = _check_ids(texts, str(text_path))
    gold_ids = _check_ids(golds, str(gold_path))
    if text_ids != gold_ids:
        raise ContractError(f"{gold_path}: 金标 ID 与 {text_path} 的顺序/内容不一致")

    for lineno, row in enumerate(texts, start=1):
        if not isinstance(row["text"], str) or not row["text"].strip():
            raise ContractError(f"{text_path}:{lineno}: text 不能为空")

    labels: list[int] = []
    for lineno, row in enumerate(golds, start=1):
        if not is_int_label(row["label"]):
            raise ContractError(f"{gold_path}:{lineno}: label 必须是整数 0/1，实际 {row['label']!r}")
        labels.append(row["label"])

    return {
        "manifest": manifest,
        "manifest_sha256": manifest_sha,
        "data_dir": root,
        "split": split,
        "ids": text_ids,
        "labels": labels,
        "texts": [row["text"] for row in texts],
    }


def validate_run(run_dir: str | Path, package: dict) -> dict:
    """校验 run 目录与预测文件，返回元信息与预测。"""
    root = Path(run_dir)
    meta_path = root / "run_metadata.json"
    pred_path = root / "predictions.jsonl"

    meta = read_json(meta_path)
    require_fields(meta, RUN_REQUIRED, str(meta_path))
    check_schema_version(meta, str(meta_path))

    if not isinstance(meta["is_mock"], bool):
        raise ContractError(f"{meta_path}: is_mock 必须是布尔值")
    if meta["is_mock"] != package["manifest"]["is_mock"]:
        raise ContractError(
            f"{meta_path}: is_mock={meta['is_mock']} 与 manifest "
            f"is_mock={package['manifest']['is_mock']} 不一致（真实/mock 不得混用）")
    if meta["dataset_manifest_sha256"] != package["manifest_sha256"]:
        raise ContractError(
            f"{meta_path}: dataset_manifest_sha256 与数据包 manifest 不一致"
            f"（run={meta['dataset_manifest_sha256'][:12]}…，"
            f"manifest={package['manifest_sha256'][:12]}…）")
    if meta["split"] != package["split"]:
        raise ContractError(f"{meta_path}: split={meta['split']!r} 与请求的 {package['split']!r} 不一致")
    if not isinstance(meta["method"], str) or not meta["method"].strip():
        raise ContractError(f"{meta_path}: method 必须是非空字符串")
    if type(meta["seed"]) is not int:
        raise ContractError(f"{meta_path}: seed 必须是整数")
    if not isinstance(meta["threshold"], (int, float)) or isinstance(meta["threshold"], bool):
        raise ContractError(f"{meta_path}: threshold 必须是数值")
    if not 0.0 <= float(meta["threshold"]) <= 1.0:
        raise ContractError(f"{meta_path}: threshold={meta['threshold']} 超出 [0, 1]")

    preds = read_jsonl(pred_path, PREDICTION_FIELDS)
    if len(preds) != len(package["ids"]):
        raise ContractError(
            f"{pred_path}: 预测行数 {len(preds)} 与金标 {len(package['ids'])} 不一致")

    threshold = float(meta["threshold"])
    for lineno, (row, gold_id) in enumerate(zip(preds, package["ids"]), start=1):
        rid = row["id"]
        if not isinstance(rid, str) or not rid.strip():
            raise ContractError(f"{pred_path}:{lineno}: id 必须是非空字符串")
        if rid != gold_id:
            raise ContractError(
                f"{pred_path}:{lineno}: id 顺序不一致（应为 {gold_id!r}，实际 {rid!r}）")
        if not is_int_label(row["label"]):
            raise ContractError(f"{pred_path}:{lineno}: label 必须是整数 0/1，实际 {row['label']!r}")
        score = row["score"]
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ContractError(f"{pred_path}:{lineno}: score 必须是数值，实际 {score!r}")
        if not math.isfinite(float(score)):
            raise ContractError(f"{pred_path}:{lineno}: score 不是有限数值（{score!r}）")
        if not 0.0 <= float(score) <= 1.0:
            raise ContractError(f"{pred_path}:{lineno}: score={score} 超出 [0, 1]")
        expected = int(float(score) >= threshold)
        if row["label"] != expected:
            raise ContractError(
                f"{pred_path}:{lineno}: label={row['label']} 与阈值规则不一致"
                f"（score={score}，threshold={threshold}，应为 {expected}）")

    return {"meta": meta, "preds": preds, "run_dir": root}


def ensure_comparable(records: list[dict]) -> None:
    """compare 前置检查：数据 hash、split、mock 标记必须一致。"""
    hashes = {r["dataset_manifest_sha256"] for r in records}
    if len(hashes) > 1:
        raise ContractError(f"待比较的 run 使用了不同数据版本：{sorted(hashes)}")
    splits = {r["split"] for r in records}
    if len(splits) > 1:
        raise ContractError(f"待比较的 run 使用了不同 split：{sorted(splits)}")
    mocks = {r["is_mock"] for r in records}
    if len(mocks) > 1:
        raise ContractError("待比较的 run 混合了真实结果与 mock 结果，禁止比较")
