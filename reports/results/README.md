# 正式结果目录

真实 run 先逐个评测（把 `metrics.json` 写回 run 目录），再统一比较：

```sh
python -m src.evaluation evaluate \
  --data-dir data/processed/corgi-official-v1 --split test \
  --run-dir runs/baseline-s42-test

python -m src.evaluation compare \
  --run-dirs runs/baseline-s42-test runs/baseline-s43-test runs/baseline-s44-test \
             runs/improved-s42-test runs/improved-s43-test runs/improved-s44-test \
  --output-dir reports/results
```

产出：`comparison.csv`、`aggregate.csv`、`comparison.md`、`environment.json`、
`reproduction.json`；逐 run 的 `metrics.json` 留在各自 run 目录中。

本目录当前为空，避免把 mock 结果误当作正式结果。mock 冒烟输出在 `runs/p4-smoke/`。
