# 正式结果目录

由 P4 的 `compare` 读取各 run 的 `metrics.json` 生成，不重算模型、不调阈值。

## 当前内容

- `comparison.csv`、`aggregate.csv`、`comparison.md`：官方测试集（`corgi-official-v1`）主表，共 7 个真实 run
  - P2：`selected-s{42,43,44}-test`（阈值 0.5）与 `selected-s{42,43,44}-test-tuned`（阈值 0.4）
  - P3：`improved-i2-s42-test`（阈值 0.5）
- `environment.json`、`reproduction.json`：环境信息与每个 run 的预测/metrics hash
- `clean-train/`：清洗训练集（`corgi-clean-train-v1`）的 6 个 run，敏感性对比单独存放，不与主表混判

## 重新生成

```sh
# 1) 重建冻结数据（首次需联网；缓存校验不过即拒绝读取）
python -m src.data prepare --raw-dir data/raw --output-dir data/processed --offline

# 2) 补齐缺 metrics 的 run（当前仅 P3 的 I2 需要）
python -m src.evaluation evaluate \
  --data-dir data/processed/corgi-official-v1 --split test \
  --run-dir runs/improved-i2-s42-test

# 3) 生成主表与图
python -m src.evaluation compare --run-dirs \
  runs/p2/corgi-official-v1/selected-s{42,43,44}-test \
  runs/p2/corgi-official-v1/selected-s{42,43,44}-test-tuned \
  runs/improved-i2-s42-test --output-dir reports/results
python -m src.evaluation compare --run-dirs \
  runs/p2/corgi-clean-train-v1/selected-s{42,43,44}-test \
  runs/p2/corgi-clean-train-v1/selected-s{42,43,44}-test-tuned \
  --output-dir reports/results/clean-train
python -m src.evaluation plot_results --aggregate reports/results/aggregate.csv
```

`compare` 拒绝不同数据 hash、不同 split 或 mock/真实混比，所以清洗版必须写到单独目录。
逐 run 的 `metrics.json` 留在各自 run 目录中。
