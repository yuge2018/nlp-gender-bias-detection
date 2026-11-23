#!/usr/bin/env python3
"""校验组员回传的第二标注 CSV。

检查项：表头、行数与 id 顺序、label 取值、targets/phenomena 合法值、notes 非空。
只报告问题，不改文件。

用法::

    python tools/p5_review/validate_returned.py \
        --chunk review/chunks/chunk-2.csv --returned returned/chunk-2.csv
"""
import argparse
import csv
import json
import sys
from pathlib import Path

COLUMNS = ("id", "text", "label", "targets", "phenomena", "notes")
TARGETS = ("male", "female", "other_explicit", "unspecified")
PHENOMENA = (
    "explicit_stereotype",
    "implicit_stereotype",
    "role_constraint",
    "derogation",
    "benevolent_sexism",
    "appearance_inference",
    "victim_blaming",
    "labor_market_penalty",
    "other_identity_derogation",
    "neutral_description",
    "denial_of_bias",
    "quoted_rebuttal",
    "bias_awareness",
)


def read_rows(path):
    with open(path, encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        rows = [dict(zip(header or [], row)) for row in reader if any(cell.strip() for cell in row)]
    return header, rows


def split_multi(value):
    return [item.strip() for item in (value or "").replace("；", ";").split(";") if item.strip()]


def validate(chunk_path, returned_path):
    errors = []
    warnings = []
    chunk_header, chunk_rows = read_rows(chunk_path)
    ret_header, ret_rows = read_rows(returned_path)
    if chunk_header != list(COLUMNS):
        errors.append(f"{chunk_path}: unexpected template header {chunk_header}")
    if ret_header != list(COLUMNS):
        errors.append(f"{returned_path}: header must be {list(COLUMNS)}, got {ret_header}")
        return {"rows": len(ret_rows), "errors": errors, "warnings": warnings}
    if len(ret_rows) != len(chunk_rows):
        errors.append(f"{returned_path}: {len(ret_rows)} rows, template has {len(chunk_rows)}")
    expected_ids = [row.get("id", "").strip() for row in chunk_rows]
    actual_ids = [row.get("id", "").strip() for row in ret_rows]
    if expected_ids != actual_ids:
        missing = sorted(set(expected_ids) - set(actual_ids))
        extra = sorted(set(actual_ids) - set(expected_ids))
        errors.append(f"{returned_path}: id order/coverage differs, missing={missing[:5]} extra={extra[:5]}")

    for line, row in enumerate(ret_rows, 2):
        ident = (row.get("id") or "").strip()
        prefix = f"{Path(returned_path).name}:{line} ({ident or 'no id'})"
        label = (row.get("label") or "").strip()
        if label not in ("0", "1"):
            errors.append(f"{prefix}: label must be 0 or 1, got {label!r}")
        targets = split_multi(row.get("targets"))
        if not targets:
            errors.append(f"{prefix}: targets is empty")
        unknown_targets = [item for item in targets if item not in TARGETS]
        if unknown_targets:
            errors.append(f"{prefix}: unknown targets {unknown_targets}")
        if len(set(targets)) != len(targets):
            warnings.append(f"{prefix}: duplicated targets {targets}")
        phenomena = split_multi(row.get("phenomena"))
        if not phenomena:
            errors.append(f"{prefix}: phenomena is empty")
        unknown_phenomena = [item for item in phenomena if item not in PHENOMENA]
        if unknown_phenomena:
            errors.append(f"{prefix}: unknown phenomena {unknown_phenomena}")
        if len(set(phenomena)) != len(phenomena):
            warnings.append(f"{prefix}: duplicated phenomena {phenomena}")
        if label == "1" and targets == ["unspecified"]:
            warnings.append(f"{prefix}: label=1 with targets=unspecified, check the object")
        if not (row.get("notes") or "").strip():
            errors.append(f"{prefix}: notes is required")
        template_row = chunk_rows[line - 2] if line - 2 < len(chunk_rows) else {}
        if (row.get("text") or "").strip() != (template_row.get("text") or "").strip():
            warnings.append(f"{prefix}: text differs from the template row")

    return {"rows": len(ret_rows), "errors": errors, "warnings": warnings}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chunk", required=True, help="下发的 chunk-N.csv 模板")
    parser.add_argument("--returned", required=True, help="组员回传的 chunk-N.csv")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出")
    args = parser.parse_args()
    result = validate(args.chunk, args.returned)
    result["chunk"] = str(args.chunk)
    result["returned"] = str(args.returned)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"rows={result['rows']} errors={len(result['errors'])} warnings={len(result['warnings'])}")
        for item in result["errors"]:
            print("  ERROR", item)
        for item in result["warnings"]:
            print("  WARN ", item)
    return 1 if result["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
