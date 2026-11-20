"""P6 entry point.

`python -m src.demo --mock --output-dir runs/p6-smoke` keeps the shared smoke
behaviour used by CI. Real mode is the Streamlit page, launched separately:

    streamlit run src/demo/app.py

Keeping the mock branch here (and not in app.py) means CI needs no streamlit.
"""
import sys

MOCK_FLAG = "--mock"
REAL_HINT = (
    "真实演示请在仓库根目录运行：streamlit run src/demo/app.py\n"
    "mock 冒烟请运行：python -m src.demo --mock --output-dir runs/p6-smoke"
)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if MOCK_FLAG in argv:
        from src.common.mock_cli import run

        # mock_cli parses sys.argv itself, so hand it exactly what we were given.
        sys.argv = [sys.argv[0], *argv]
        run("demo")
        return 0
    print(REAL_HINT, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
