"""接口契约 v1 的共享常量与最小参考实现。

只放"格式"层面的东西：schema 版本、必需字段、JSONL 读写与校验、哈希计算。
任何模块都不应绕过这里自己解析共享文件。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

SCHEMA_VERSION = "1.0"
SUPPORTED_SCHEMA_VERSIONS = {"1.0"}

PREDICTION_FIELDS = ("id", "label", "score")
GOLD_FIELDS = ("id", "label")
TEXT_FIELDS = ("id", "text")
TRAIN_FIELDS = ("id", "text", "label")

MANIFEST_REQUIRED = ("schema_version", "dataset_id", "is_mock",
                     "source_commit", "label_map", "files")
RUN_REQUIRED = ("schema_version", "method", "split", "dataset_manifest_sha256",
                "is_mock", "seed", "threshold", "code_commit", "model_name",
                "model_revision", "config", "environment")

SPLITS = ("dev", "test")


class ContractError(Exception):
    """数据或契约不满足要求。CLI 统一映射为退出码 2。"""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def read_json(path: str | Path) -> dict:
    p = Path(path)
    if not p.is_file():
        raise ContractError(f"{p}: 文件不存在")
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ContractError(f"{p}: JSON 解析失败（{exc.msg}，第 {exc.lineno} 行）") from exc
    if not isinstance(obj, dict):
        raise ContractError(f"{p}: 顶层应为 JSON 对象")
    return obj


def read_jsonl(path: str | Path, required_fields: tuple[str, ...] | None = None) -> list[dict]:
    """严格读取 JSONL：UTF-8 无 BOM、无空行、末尾必须有换行、字段集合精确匹配。"""
    p = Path(path)
    if not p.is_file():
        raise ContractError(f"{p}: 文件不存在")
    raw = p.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raise ContractError(f"{p}: 文件含 UTF-8 BOM")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ContractError(f"{p}: 不是合法 UTF-8（{exc.reason}）") from exc
    if text and not text.endswith("\n"):
        raise ContractError(f"{p}: 末尾缺少换行符")

    rows: list[dict] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise ContractError(f"{p}:{lineno}: 空行")
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ContractError(f"{p}:{lineno}: JSON 解析失败（{exc.msg}）") from exc
        if not isinstance(obj, dict):
            raise ContractError(f"{p}:{lineno}: 每行必须是 JSON 对象")
        if required_fields is not None:
            keys = tuple(obj.keys())
            if set(keys) != set(required_fields) or len(keys) != len(required_fields):
                raise ContractError(
                    f"{p}:{lineno}: 字段应为 {list(required_fields)}，实际为 {list(keys)}")
        rows.append(obj)
    return rows


def write_jsonl(path: str | Path, rows: list[dict], fields: tuple[str, ...]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps({k: row[k] for k in fields}, ensure_ascii=False) + "\n")


def write_json(path: str | Path, obj: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def require_fields(obj: dict, fields: tuple[str, ...], where: str) -> None:
    missing = [f for f in fields if f not in obj]
    if missing:
        raise ContractError(f"{where}: 缺少必需字段 {missing}")


def is_int_label(value) -> bool:
    """真正的整数 0/1；bool 是 int 的子类，必须显式排除。"""
    return type(value) is int and value in (0, 1)


def check_schema_version(obj: dict, where: str) -> None:
    version = obj.get("schema_version")
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ContractError(
            f"{where}: schema_version={version!r} 不受支持（支持 {sorted(SUPPORTED_SCHEMA_VERSIONS)}）")
