# 正式结果目录

真实 run 生成的结果写入这里：

```sh
python -m src.evaluation compare \
  --run-dirs runs/baseline-s42-test runs/baseline-s43-test runs/baseline-s44-test \
             runs/improved-s42-test runs/improved-s43-test runs/improved-s44-test \
  --data-dir data/processed/corgi-official-v1 \
  --output-dir reports/results
```

产出：`comparison.csv`、`aggregate.csv`、`comparison.md`、`environment.json`、
`reproduction.json`、`metrics/`。

本目录当前为空，避免把 mock 结果误当作正式结果。mock 冒烟输出在 `runs/p4-smoke/`。
