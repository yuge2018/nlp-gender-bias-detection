"""Config-driven model registry for the P6 demo.

Everything that changes when P2/P3/P4 deliver real artifacts lives in
`configs/demo/*.json`; this module never hard-codes a model, path or example.

Design rules (see docs/INTERFACES.md and docs/roles/P6-demo.md):

* The demo only talks to `load_predictor(model_dir, *, mock=False)` and
  `predictor.predict_proba(texts)`; it never imports model internals.
* Loading is lazy: importing this module must not import torch/sklearn/streamlit,
  so the contract test suite runs on a bare CI image.
* A missing adapter, missing dependency or missing model directory becomes a
  `ModelUnavailable` with an actionable hint, never a bare crash.
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

from src.common.model_api import MockPredictor, Predictor

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "demo" / "default.json"
DEFAULT_EXAMPLES = ROOT / "configs" / "demo" / "examples.json"

REQUIRED_ENTRY_FIELDS = ("key", "display_name")
REQUIRED_CONFIG_FIELDS = ("models", "default_model", "mock_banner_required")
MODEL_METADATA_FILE = "model_metadata.json"


class ModelUnavailable(RuntimeError):
    """A configured model cannot be loaded in this environment."""

    def __init__(self, key: str, reason: str, hint: str = ""):
        self.key = key
        self.reason = reason
        self.hint = hint
        message = f"模型「{key}」不可用：{reason}"
        if hint:
            message += f"。建议：{hint}"
        super().__init__(message)


def _read_json(path: Path) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ModelUnavailable("-", f"配置文件不存在：{path}") from exc
    except json.JSONDecodeError as exc:
        raise ModelUnavailable("-", f"配置文件不是合法 JSON：{path}（{exc}）") from exc


def load_config(path: str | Path | None = None) -> dict:
    """Load and validate `configs/demo/default.json`."""
    payload = _read_json(Path(path) if path else DEFAULT_CONFIG)
    missing = [field for field in REQUIRED_CONFIG_FIELDS if field not in payload]
    if missing:
        raise ValueError(f"demo config: missing fields {missing}")
    models = payload["models"]
    if not isinstance(models, list) or not models:
        raise ValueError("demo config: 'models' must be a non-empty list")
    keys: list[str] = []
    for index, entry in enumerate(models):
        if not isinstance(entry, dict):
            raise ValueError(f"demo config: models[{index}] must be an object")
        absent = [field for field in REQUIRED_ENTRY_FIELDS if not entry.get(field)]
        if absent:
            raise ValueError(f"demo config: models[{index}] missing {absent}")
        key = entry["key"]
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"demo config: models[{index}].key must be a non-empty string")
        if key in keys:
            raise ValueError(f"demo config: duplicate model key {key!r}")
        keys.append(key)
        adapter = entry.get("adapter_module")
        if adapter is not None and not isinstance(adapter, str):
            raise ValueError(f"demo config: models[{index}].adapter_module must be a string or null")
        if adapter is None and entry.get("model_dir"):
            raise ValueError(f"demo config: models[{index}] has model_dir but no adapter")
        if adapter is not None and not entry.get("model_dir"):
            raise ValueError(f"demo config: models[{index}] has adapter but no model_dir")
    if payload["default_model"] not in keys:
        raise ValueError(f"demo config: default_model {payload['default_model']!r} is not a configured model")
    if type(payload["mock_banner_required"]) is not bool:
        raise ValueError("demo config: mock_banner_required must be a boolean")
    return payload


def model_entries(config: dict) -> list[dict]:
    """Return the configured model entries in display order."""
    return list(config["models"])


def find_entry(config: dict, key: str) -> dict:
    for entry in model_entries(config):
        if entry["key"] == key:
            return entry
    raise ModelUnavailable(key, "配置里没有这个模型")


def resolve_model_dir(entry: dict, root: str | Path | None = None) -> Path:
    """Resolve `model_dir` against the repo root; absolute paths pass through."""
    configured = entry.get("model_dir")
    if not configured:
        raise ValueError(f"model {entry['key']!r} has no model_dir")
    candidate = Path(configured)
    if candidate.is_absolute():
        return candidate
    return Path(root) / candidate if root else ROOT / candidate


def model_status(entry: dict, root: str | Path | None = None) -> str:
    """Classify readiness without importing anything.

    Returns one of:
    * ``mock``            - development substitute, always usable;
    * ``ready``           - directory exists and contains weight/payload files;
    * ``weights_missing`` - only ``model_metadata.json`` is committed (weights are
      gitignored), so the directory exists but cannot be loaded;
    * ``missing``         - configured directory does not exist.

    ``probe_files`` in the config can pin exact required files; otherwise the
    rule is "at least one file other than model_metadata.json".
    """
    if entry.get("adapter_module") is None:
        return "mock"
    try:
        model_dir = resolve_model_dir(entry, root)
    except ValueError:
        return "missing"
    if not model_dir.is_dir():
        return "missing"

    probe = entry.get("probe_files")
    if probe:
        if not all((model_dir / name).is_file() for name in probe):
            return "weights_missing"
        return "ready"

    for path in model_dir.rglob("*"):
        if path.is_file() and path.name != MODEL_METADATA_FILE:
            return "ready"
    return "weights_missing"


def is_available(entry: dict, root: str | Path | None = None) -> bool:
    """Cheap availability probe: a directory alone is not enough (see model_status)."""
    return model_status(entry, root) in ("ready", "mock")


def describe_models(config: dict, root: str | Path | None = None) -> list[dict]:
    """Rows for the UI model selector; importing nothing heavy."""
    rows = []
    for entry in model_entries(config):
        status = model_status(entry, root)
        rows.append(
            {
                "key": entry["key"],
                "display_name": entry["display_name"],
                "adapter_module": entry.get("adapter_module"),
                "model_dir": str(resolve_model_dir(entry, root)) if entry.get("adapter_module") else None,
                "is_deep": bool(entry.get("is_deep", False)),
                "available": status in ("ready", "mock"),
                "status": status,
            }
        )
    return rows


def load_model(config: dict, key: str | None = None, *, mock: bool = False, root: str | Path | None = None) -> Predictor:
    """Load a model through the frozen adapter contract.

    `mock=True` always returns the constant `MockPredictor`, independent of key.
    """
    key = key or config["default_model"]
    entry = find_entry(config, key)
    if mock or entry.get("adapter_module") is None:
        return MockPredictor()

    # Classify before importing anything: a missing artifact or a metadata-only
    # directory must give the same clear message on every machine.
    model_dir = resolve_model_dir(entry, root)
    status = model_status(entry, root)
    if status == "missing":
        raise ModelUnavailable(
            key,
            f"模型目录不存在：{model_dir}",
            "先运行该角色的 train CLI 生成 artifacts/ 下的模型，或检查 configs/demo/default.json 的 model_dir",
        )
    if status == "weights_missing":
        raise ModelUnavailable(
            key,
            f"模型目录只有元信息，没有权重文件：{model_dir}",
            "权重不随仓库提交；请在本机按该角色 README 重新生成，或向对应负责人获取完整 artifacts 目录",
        )

    module_name = entry["adapter_module"]
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ModelUnavailable(key, f"适配器模块 {module_name} 导入失败（{exc}）", "确认该角色已合入实现，且运行环境满足 requirements/p*.txt") from exc

    loader = getattr(module, "load_predictor", None)
    if not callable(loader):
        raise ModelUnavailable(key, f"{module_name} 没有导出 load_predictor", "让 P2/P3 按 docs/INTERFACES.md 补齐适配器")

    try:
        predictor = loader(str(model_dir), mock=False)
    except ModuleNotFoundError as exc:
        raise ModelUnavailable(key, f"缺少运行依赖 {exc.name}", f"安装 requirements 后重试（{module_name} 需要对应依赖）") from exc
    except ImportError as exc:
        raise ModelUnavailable(key, f"依赖导入失败：{exc}", "检查该模型的运行依赖是否已安装") from exc
    except FileNotFoundError as exc:
        raise ModelUnavailable(key, f"加载文件缺失：{exc}", "确认模型目录由 train CLI 完整生成") from exc
    except Exception as exc:  # noqa: BLE001 - surface every loader failure as ModelUnavailable
        raise ModelUnavailable(key, f"加载失败（{type(exc).__name__}）：{exc}", "查看该模型的 README 与运行日志") from exc

    if not callable(getattr(predictor, "predict_proba", None)):
        raise ModelUnavailable(key, "适配器返回的对象没有 predict_proba", "按 docs/INTERFACES.md 返回 Predictor")
    return predictor


def load_examples(path: str | Path | None = None) -> dict:
    """Load the self-authored demo sentences (explicitly not gold labels)."""
    payload = _read_json(Path(path) if path else DEFAULT_EXAMPLES)
    examples = payload.get("examples")
    if not isinstance(examples, list) or not examples:
        raise ValueError("demo examples: 'examples' must be a non-empty list")
    for index, item in enumerate(examples):
        if not isinstance(item, dict) or not item.get("text"):
            raise ValueError(f"demo examples: examples[{index}] must contain text")
        if item.get("expected_label") not in (0, 1):
            raise ValueError(f"demo examples: examples[{index}].expected_label must be 0 or 1")
    return payload
