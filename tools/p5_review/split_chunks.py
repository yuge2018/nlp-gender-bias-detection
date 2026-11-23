#!/usr/bin/env python3
"""P5 第二标注盲标分包：把冻结的挑战集拆成若干份独立 CSV。

- 只导出 ``id, text``，四列待填内容（label/targets/phenomena/notes）留空；
- 按 label 与对象组分层，保证每份正负比例接近 61:39；
- 同一份内不含答案，六位组员互不可见，P5 负责人不参与盲标。

用法::

    python tools/p5_review/split_chunks.py \
        --challenge-dir data/challenge/p5-gender-challenge-v1 \
        --output-dir review/chunks --parts 6

产物：``chunk-1.csv`` … ``chunk-6.csv`` 与 ``assignment.csv``（不含金标统计以外的任何答案）。
"""
import argparse
import csv
import json
import os
from pathlib import Path

COLUMNS = ("id", "text", "label", "targets", "phenomena", "notes")


def read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def primary_target(annotation):
    targets = annotation.get("targets") or []
    return targets[0] if targets else "unspecified"


def distribute(rows, quotas, parts):
    """按对象组交错 + 配额轮转，把行分到各份，保证每份拿到不同对象组。"""
    ordered = sorted(rows, key=lambda row: (primary_target(row["annotation"]), row["challenge"]["id"]))
    buckets = [[] for _ in range(parts)]
    remaining = list(quotas)
    index = 0
    for row in ordered:
        for _ in range(parts):
            if remaining[index] > 0:
                buckets[index].append(row)
                remaining[index] -= 1
                index = (index + 1) % parts
                break
            index = (index + 1) % parts
    return buckets


def plan_quotas(count, parts, reverse=False):
    base, remainder = divmod(count, parts)
    if not reverse:
        return [base + (1 if index < remainder else 0) for index in range(parts)]
    return [base + (1 if index >= parts - remainder else 0) for index in range(parts)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--challenge-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--parts", type=int, default=6)
    args = parser.parse_args()

    base = Path(args.challenge_dir)
    challenge = read_jsonl(base / "challenge.jsonl")
    labels = {row["id"]: row["label"] for row in read_jsonl(base / "challenge_labels.jsonl")}
    annotations = {row["id"]: row for row in read_jsonl(base / "challenge_annotations.jsonl")}
    if len(challenge) != len(labels) or set(labels) != {row["id"] for row in challenge}:
        raise SystemExit("challenge/labels id mismatch")

    rows = [{"challenge": row, "label": labels[row["id"]], "annotation": annotations[row["id"]]} for row in challenge]
    positive = [row for row in rows if row["label"] == 1]
    negative = [row for row in rows if row["label"] == 0]
    quotas_pos = plan_quotas(len(positive), args.parts)
    quotas_neg = plan_quotas(len(negative), args.parts, reverse=True)
    buckets = [pos + neg for pos, neg in zip(distribute(positive, quotas_pos, args.parts), distribute(negative, quotas_neg, args.parts))]
    for bucket in buckets:
        bucket.sort(key=lambda row: row["challenge"]["id"])

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    assignment = []
    for index, bucket in enumerate(buckets, 1):
        path = out_dir / f"chunk-{index}.csv"
        with open(path, "w", encoding="utf-8-sig", newline="") as fh:
            writer = csv.writer(fh, lineterminator="\r\n")
            writer.writerow(COLUMNS)
            for row in bucket:
                writer.writerow([row["challenge"]["id"], row["challenge"]["text"], "", "", "", ""])
        assignment.append({
            "chunk": f"chunk-{index}",
            "file": f"chunk-{index}.csv",
            "rows": len(bucket),
            "positive": sum(row["label"] for row in bucket),
            "negative": sum(1 - row["label"] for row in bucket),
            "reviewer": "",
            "status": "待发放",
        })

    with open(out_dir / "assignment.csv", "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["chunk", "file", "rows", "positive", "negative", "reviewer", "status"], lineterminator="\r\n")
        writer.writeheader()
        writer.writerows(assignment)

    covered = sum(item["rows"] for item in assignment)
    if covered != len(rows):
        raise SystemExit(f"coverage error: {covered} != {len(rows)}")
    print(json.dumps({"parts": args.parts, "rows": covered, "assignment": assignment}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
