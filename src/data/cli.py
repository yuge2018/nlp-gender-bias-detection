"""P1 public command line. Mock mode stays dependency-free and compatible."""
import argparse
import json
from pathlib import Path
import sys

from src.common.contracts import write_json
from .pipeline import DEFAULT_CONFIG, DEFAULT_LOCK, prepare, verify_package


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "--mock":
        from src.common.mock_cli import run
        run("data")
        return 0
    parser = argparse.ArgumentParser(description="Prepare/verify checksum-pinned CORGI-PM detection data")
    subparsers = parser.add_subparsers(dest="command", required=True)
    builder = subparsers.add_parser("prepare", help="Download, verify, convert and audit both versions")
    builder.add_argument("--raw-dir", default="data/raw/corgi-pinned")
    builder.add_argument("--output-dir", default="data/processed")
    builder.add_argument("--config", default=str(DEFAULT_CONFIG))
    builder.add_argument("--source-lock", default=str(DEFAULT_LOCK))
    builder.add_argument("--offline", action="store_true")
    builder.add_argument("--summary", help="Write a text-free, shareable audit summary JSON")
    builder.add_argument("--figures-dir", help="Optional PNG/SVG charts (requires matplotlib)")
    verifier = subparsers.add_parser("verify", help="Verify every data/audit file, hash, row and ID mapping")
    verifier.add_argument("--data-dir", required=True)
    args = parser.parse_args(arguments)
    try:
        if args.command == "verify":
            result = verify_package(args.data_dir)
        else:
            result = prepare(args.raw_dir, args.output_dir, config_path=args.config, lock_path=args.source_lock, offline=args.offline)
            if args.summary:
                write_json(args.summary, result)
            if args.figures_dir:
                from .figures import make_figures
                make_figures(result, args.figures_dir)
        print(json.dumps(result if args.command == "verify" else {"is_mock": False, "packages": {name: {key: value for key, value in info.items() if key != "audit"} for name, info in result["packages"].items()}}, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError, KeyError, ImportError) as exc:
        print(f"P1 data error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
