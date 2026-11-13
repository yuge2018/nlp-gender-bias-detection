"""Improved encoder entry point: real train/predict plus the frozen mock smoke.

python -m src.improved --mock --output-dir runs/p3-smoke      # 冻结的接口冒烟
python -m src.improved train --data-dir ... --config ... --output-dir ...
python -m src.improved predict --input ... --model-dir ... --manifest ... --output-dir ...
"""
import sys

from src.common.mock_cli import run


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] in ("train", "predict"):
        from .cli import main as cli_main

        raise SystemExit(cli_main(argv))
    run("improved")


if __name__ == "__main__":
    main()
