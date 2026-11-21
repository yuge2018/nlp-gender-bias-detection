# P4 统一评测与复现

负责 `src/evaluation/`。指标定义复用 [`src/common/contracts.py`](../common/contracts.py)
（P4 维护的冻结契约），本模块只做评测编排、对比表、聚合与复现记录，不导入模型实现。

## 命令

```sh
# 1) 评测单个 run：校验后把 metrics.json 写回 run 目录
python -m src.evaluation evaluate \
  --data-dir data/processed/corgi-official-v1 --split test \
  --run-dir runs/baseline-s42-test

# 2) 比较多个 run：读各 run 的 metrics.json，生成对比表与复现记录
python -m src.evaluation compare \
  --run-dirs runs/baseline-s42-test runs/baseline-s43-test \
             runs/improved-s42-test runs/improved-s43-test \
  --output-dir reports/results

# 3) 开发联调（共享替身，is_mock=true，输出目录必须为空）
python -m src.evaluation --mock --output-dir runs/p4-smoke
```

`compare` 只接受已经评测过的 run 目录；真实比较默认拒绝 mock，联调时加 `--allow-mock`。

## 检查项

`evaluate` 依次调用 `load_dataset` 与 `validate_predictions`，失败即返回退出码 2：

- manifest 必需字段、版本、label_map、每个文件的 SHA256 与行数；
- input 与 gold 的 ID 数量与顺序；
- 预测行数、ID 顺序、重复 ID、字段集合；
- label 为真正的整数 0/1，score 有限且在 `[0,1]`；
- `label == int(score >= threshold)`，阈值只从 run 元信息读取；
- run 与 manifest 的 `is_mock` 一致，`dataset_manifest_sha256` 一致，split 与数据文件一致。

`compare` 额外拒绝不同数据 hash、不同 split、真实与 mock 混用。

## 输出

- run 目录：`metrics.json`（契约字段 + method/split/seed/数据 hash/预测 hash/评测器版本）；
- 输出目录：`comparison.csv`、`aggregate.csv`（均值 ± 样本标准差，单次为 null）、
  `comparison.md`、`environment.json`（Python/平台/命令/git commit）、`reproduction.json`。

## 测试

```sh
python -m unittest discover -s tests/p4 -v
```

覆盖手算指标、零分母、缺行/重复/乱序/NaN/越界/bool 标签/阈值不符、
mock 拒绝与混合拒绝、CLI 退出码。
