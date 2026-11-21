"""P4 评测 CLI：evaluate / compare / --mock。

退出码：0 成功；2 参数、契约或数据错误；1 其他运行失败。
"""
from __future__ import annotations

import argparse
import json
import sys

from .runner import compare, evaluate_run, run_mock


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.evaluation",
        description="P4 统一评测：契约检查、指标计算、多种子汇总与复现记录")
    parser.add_argument("--mock", action="store_true",
                        help="用共享 mock 替身跑通 evaluate/compare 全流程")
    parser.add_argument("--output-dir", default=None,
                        help="mock 模式输出目录（例如 runs/p4-smoke）")
    sub = parser.add_subparsers(dest="command")

    evaluate = sub.add_parser("evaluate", help="评测单个 run 并写回 metrics.json")
    evaluate.add_argument("--data-dir", required=True)
    evaluate.add_argument("--split", default="test", choices=("dev", "test"))
    evaluate.add_argument("--run-dir", required=True)

    compare_parser = sub.add_parser("compare", help="比较多个 run，生成对比表与复现记录")
    compare_parser.add_argument("--run-dirs", nargs="+", required=True)
    compare_parser.add_argument("--output-dir", required=True)
    compare_parser.add_argument("--allow-mock", action="store_true",
                                help="允许 mock 结果参与比较（仅联调用）")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "evaluate":
            record = evaluate_run(args.data_dir, args.split, args.run_dir)
            print(f"{record['method']} seed={record['seed']} split={record['split']} "
                  f"n={record['n']} macro_f1={record['macro_f1']:.4f} "
                  f"accuracy={record['accuracy']:.4f}")
            return 0

        if args.command == "compare":
            rows = compare(args.run_dirs, args.output_dir, allow_mock=args.allow_mock)
            for row in sorted(rows, key=lambda item: (item["method"], item["seed"])):
                print(f"{row['method']:<10} seed={row['seed']:<3} "
                      f"macro_f1={row['macro_f1']:.4f} accuracy={row['accuracy']:.4f}")
            print(f"结果已写入 {args.output_dir}")
            return 0

        if args.mock:
            if not args.output_dir:
                parser.error("--mock 需要 --output-dir")
            run_dirs = run_mock(args.output_dir)
            print(json.dumps({"component": "evaluation", "is_mock": True,
                              "runs": [str(path) for path in run_dirs],
                              "output_dir": args.output_dir}, ensure_ascii=False))
            return 0

        parser.print_help()
        print("\n错误：未指定子命令；开发联调用 --mock。", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"契约错误：{exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"契约错误：{exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001
        print(f"运行失败：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
