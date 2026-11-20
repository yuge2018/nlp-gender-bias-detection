"""内置 fixtures：mock 数据包 + 两个方法的 mock run。

只用于接口开发和联调，所有 manifest/run 都带 is_mock=true，
正式比较必须拒绝它们。文本与分数都是合成的，不能当作模型结论。
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path

from src.common.contracts import (
    SCHEMA_VERSION,
    sha256_file,
    write_json,
    write_jsonl,
)

SUBJECTS = ["女性", "男人", "女司机", "男护士", "她", "他", "女孩子", "男孩子"]
BIAS_PREDICATES = [
    "不适合从事需要逻辑推理的科研工作",
    "就应该在家里照顾孩子",
    "开车技术果然不靠谱",
    "天生不擅长数学",
    "只能做行政类的工作",
    "情绪化，难以承担管理岗位",
    "学不好理科是正常的",
    "不应该出现在护理行业",
]
NEUTRAL_TEXTS = [
    "研究显示，男女在数学能力上没有显著差异",
    "公司今年招聘了十二名工程师，其中五名是女性",
    "该报告分析了不同性别在职业选择上的分布",
    "调查发现，员工满意度与性别没有明显关系",
    "她是一名外科医生，去年完成了两百台手术",
    "他在幼儿园担任教师已经六年",
    "这项政策要求招聘过程不得询问婚育情况",
    "样本中的男女比例接近一比一",
    "论文比较了不同性别在语言测试中的平均分",
    "会议讨论了如何减少招聘中的性别刻板印象",
    "团队负责人由投票产生，与性别无关",
    "该岗位的录用标准是工作经验和专业能力",
]
BIAS_KEYWORDS = ["不适合", "就应该", "果然", "天生", "只能", "情绪化", "不应该", "正常"]

METHODS = {
    "baseline": {"model_name": "mock-tfidf-lr", "scale": 0.78, "noise": 0.95, "kw": 0.25},
    "improved": {"model_name": "mock-roberta-wwm", "scale": 0.95, "noise": 0.95, "kw": 0.30},
}
SEEDS = (42, 43, 44)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _make_texts(n: int, split: str, rng: random.Random) -> list[dict]:
    rows = []
    for i in range(n):
        if i % 2 == 0:
            text = f"{rng.choice(SUBJECTS)}{rng.choice(BIAS_PREDICATES)}。"
            label = 1
        else:
            text = f"{rng.choice(NEUTRAL_TEXTS)}。"
            label = 0
        rows.append({"id": f"{split}-{i + 1:06d}", "text": text, "label": label})
    return rows


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _mock_score(text: str, label: int, method: str, seed: int) -> float:
    cfg = METHODS[method]
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    rng = random.Random(f"{method}-{seed}-{digest}")
    keyword_hits = sum(text.count(k) for k in BIAS_KEYWORDS)
    logit = cfg["scale"] * (2 * label - 1) + cfg["kw"] * keyword_hits
    logit += rng.gauss(0.0, cfg["noise"])
    return round(min(1.0, max(0.0, _sigmoid(logit))), 6)


def build_fixtures(root: str | Path) -> dict:
    """生成 fixtures 数据包与 6 个 mock run，返回路径信息。"""
    root = Path(root)
    data_dir = root / "data" / "mock-corgi-v1"
    runs_dir = root / "runs"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "labels").mkdir(parents=True, exist_ok=True)
    (data_dir / "audit").mkdir(parents=True, exist_ok=True)

    rng = random.Random(20260101)
    train = _make_texts(60, "train", rng)
    dev = _make_texts(30, "dev", rng)
    test = _make_texts(60, "test", rng)

    write_jsonl(data_dir / "train.jsonl",
                [{"id": r["id"], "text": r["text"], "label": r["label"]} for r in train],
                ("id", "text", "label"))
    for split, rows in (("dev", dev), ("test", test)):
        write_jsonl(data_dir / f"{split}.jsonl", rows, ("id", "text"))
        write_jsonl(data_dir / "labels" / f"{split}.jsonl",
                    [{"id": r["id"], "label": r["label"]} for r in rows], ("id", "label"))
    write_jsonl(data_dir / "audit" / "id_mapping.jsonl",
                [{"id": r["id"], "source": "mock", "original_split": "mock"} for r in train],
                ("id", "source", "original_split"))

    files = {}
    for rel in ("train.jsonl", "dev.jsonl", "test.jsonl",
                "labels/dev.jsonl", "labels/test.jsonl"):
        path = data_dir / rel
        rows = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
        files[rel] = {"sha256": sha256_file(path), "rows": rows}

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": "mock-corgi-v1",
        "is_mock": True,
        "source_commit": "mock-fixture-2026",
        "label_map": {"0": "无偏见", "1": "有偏见"},
        "files": files,
        "created_at": _now(),
        "note": "合成数据，仅用于接口开发与联调，禁止用于正式结论",
    }
    write_json(data_dir / "manifest.json", manifest)
    manifest_sha = sha256_file(data_dir / "manifest.json")

    run_dirs = []
    for method in METHODS:
        for seed in SEEDS:
            run_dir = runs_dir / f"{method}-s{seed}-test"
            run_dir.mkdir(parents=True, exist_ok=True)
            preds = []
            for row in test:
                score = _mock_score(row["text"], row["label"], method, seed)
                preds.append({"id": row["id"], "label": int(score >= 0.5), "score": score})
            write_jsonl(run_dir / "predictions.jsonl", preds, ("id", "label", "score"))
            write_json(run_dir / "run_metadata.json", {
                "schema_version": SCHEMA_VERSION,
                "method": method,
                "split": "test",
                "dataset_manifest_sha256": manifest_sha,
                "is_mock": True,
                "seed": seed,
                "threshold": 0.5,
                "code_commit": "mock-fixture",
                "model_name": METHODS[method]["model_name"],
                "model_revision": "mock-0000",
                "config": {"note": "mock 配置", "threshold": 0.5},
                "environment": {"python": "3.x", "device": "cpu", "mock": True},
            })
            write_json(run_dir / "config.json", {"method": method, "seed": seed, "threshold": 0.5})
            (run_dir / "log.txt").write_text(
                f"[mock] method={method} seed={seed} split=test rows={len(preds)}\n",
                encoding="utf-8")
            run_dirs.append(run_dir)

    return {
        "data_dir": data_dir,
        "manifest_sha256": manifest_sha,
        "run_dirs": run_dirs,
    }
