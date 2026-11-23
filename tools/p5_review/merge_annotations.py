#!/usr/bin/env python3
"""把六份回传的 chunk CSV 合并成一份可复核的第二标注目录。

产物目录符合 ``src.fairness.load_challenge`` 的格式，可直接用于：

    python -m src.fairness review \
        --first data/challenge/p5-gender-challenge-v1 \
        --second review/pass2 \
        --output runs/p5/review.json

用法::

    python tools/p5_review/merge_annotations.py \
        --challenge-dir data/challenge/p5-gender-challenge-v1 \
        --chunks-dir review/chunks --returned-dir review/returned \
        --output-dir review/pass2

任何一份有错误都会中止合并，错误来自 ``validate_returned.validate``。
"""
import argparse
import csv
import datetime
import hashlib
import json
import shutil
import sys
from pathlib import Path

from validate_returned import COLUMNS, read_rows, split_multi, validate

CHALLENGE_FILES = ("challenge.jsonl", "challenge_labels.jsonl", "challenge_annotations.jsonl")


def read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def read_assignments(chunks_dir):
    path = Path(chunks_dir) / "assignment.csv"
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return {row["chunk"]: (row.get("reviewer") or "").strip() for row in csv.DictReader(fh)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--challenge-dir", required=True)
    parser.add_argument("--chunks-dir", required=True)
    parser.add_argument("--returned-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--annotation-version", default="v1.0-review2")
    parser.add_argument("--review-status", default="second_pass_collected")
    args = parser.parse_args()

    base = Path(args.challenge_dir)
    chunks_dir = Path(args.chunks_dir)
    returned_dir = Path(args.returned_dir)
    chunk_files = sorted(chunks_dir.glob("chunk-*.csv"))
    if not chunk_files:
        raise SystemExit(f"{chunks_dir}: no chunk-*.csv")

    reviewers = read_assignments(chunks_dir)
    collected = {}
    report = []
    problems = []
    for chunk_path in chunk_files:
        chunk = chunk_path.stem
        returned = returned_dir / chunk_path.name
        if not returned.is_file():
            problems.append(f"{chunk}: returned file missing ({returned})")
            continue
        check = validate(chunk_path, returned)
        annotator = reviewers.get(chunk) or f"annotator-{chunk.split('-')[-1]}"
        report.append({"chunk": chunk, "annotator": annotator, "rows": check["rows"], "errors": check["errors"], "warnings": check["warnings"]})
        if check["errors"]:
            problems.extend(check["errors"])
            continue
        _, rows = read_rows(returned)
        for row in rows:
            ident = (row.get("id") or "").strip()
            if ident in collected:
                problems.append(f"{chunk}: duplicated id {ident} across chunks")
                continue
            collected[ident] = {
                "label": int((row.get("label") or "").strip()),
                "targets": split_multi(row.get("targets")),
                "phenomena": split_multi(row.get("phenomena")),
                "notes": (row.get("notes") or "").strip(),
                "chunk": chunk,
                "annotator": annotator,
            }
    if problems:
        print("合并中止，先修数据：")
        for item in problems:
            print("  ERROR", item)
        return 1

    challenge = read_jsonl(base / "challenge.jsonl")
    original_annotations = {row["id"]: row for row in read_jsonl(base / "challenge_annotations.jsonl")}
    missing = [row["id"] for row in challenge if row["id"] not in collected]
    if missing:
        print(f"合并中止：{len(missing)} 条没有第二标注，例如 {missing[:5]}")
        return 1

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(base / "challenge.jsonl", out_dir / "challenge.jsonl")
    labels = [{"id": row["id"], "label": collected[row["id"]]["label"]} for row in challenge]
    annotations = []
    for row in challenge:
        item = collected[row["id"]]
        annotations.append({
            "id": row["id"],
            "targets": item["targets"],
            "phenomena": item["phenomena"],
            "reviewers": [item["annotator"]],
            "adjudication": f"{item['chunk']} 独立盲标；未裁决（P5 负责人统一裁决）",
        })
    write_jsonl(out_dir / "challenge_labels.jsonl", labels)
    write_jsonl(out_dir / "challenge_annotations.jsonl", annotations)

    manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    files = dict(manifest.get("files") or {})
    for name in CHALLENGE_FILES:
        path = out_dir / name
        files[name] = {"sha256": sha256(path), "rows": len(read_jsonl(path))}
    manifest["files"] = files
    manifest["annotation_version"] = args.annotation_version
    manifest["review_status"] = args.review_status
    manifest["frozen_at"] = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec="seconds")
    manifest["second_pass"] = {
        "chunks": [item["chunk"] for item in report],
        "annotators": sorted({item["annotator"] for item in report}),
        "merged_from": str(returned_dir),
        "note": "第二遍独立盲标；与原标注的分歧由 src.fairness review 统计，P5 负责人裁决。",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    merge_report = {
        "generated_at": manifest["frozen_at"],
        "challenge_dir": str(base),
        "output_dir": str(out_dir),
        "n": len(labels),
        "chunks": report,
        "original_annotation_ids": len(original_annotations),
    }
    (out_dir.parent / "merge_report.json").write_text(json.dumps(merge_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({"output": str(out_dir), "n": len(labels), "chunks": len(report)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
