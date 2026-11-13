# P2 CPU 基线与特征消融

负责人：`cengjingdejiyi`。实现字符 n-gram TF-IDF ＋ LogisticRegression 基线、B0–B3 消融、dev 上的配置/阈值选择、离线适配器与报告证据。测试与正式 run 使用 `requirements/p2.txt` 锁定的独立环境。

## 数据来源与隔离

- 输入为 [P1 冻结数据包](../../src/data/README.md)：`corgi-official-v1`（主实验）与 `corgi-clean-train-v1`（训练端删除与 dev/test 精确同文的敏感性版本）。
- 训练只打开 `train.jsonl`、`dev.jsonl` 与 `labels/dev.jsonl`；`test.jsonl` 仅用于推理，`labels/test.jsonl` 不参与任何训练、配置或阈值选择。
- 模型输入只有 `text`。ID、行号、源类别、`bias_labels`、`edit_sentence` 都不作为特征；推理输入出现 `label` 或额外字段直接报错。
- 向量器只在 train 文本上 `fit`，`model_metadata.json` 记录 `vectorizer_fit_scope="train_only"` 与拟合行数。

## 环境

```sh
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements/p2.txt
.venv/Scripts/python -m unittest discover -s tests -v
```

`--mock` 入口只需标准库；真实训练/预测/图表需要 scikit-learn、numpy、scipy、joblib、matplotlib。

## 命令

```sh
# 1) dev-only 配置扫描：C x class_weight x ngram；不读 test
python -m src.baseline sweep --data-dir data/processed/corgi-official-v1 --config configs/baseline/sweep.json --output-dir runs/p2/corgi-official-v1/sweep

# 2) 训练一个冻结配置；--select-threshold 只用 dev 网格选阈值
python -m src.baseline train --data-dir data/processed/corgi-official-v1 --config configs/baseline/selected.json --output-dir artifacts/p2/corgi-official-v1/selected-s42 --seed 42 --select-threshold

# 3) 冻结 split 预测；threshold 默认取模型元信息（dev 选择结果）
python -m src.baseline predict --input data/processed/corgi-official-v1/test.jsonl --model-dir artifacts/p2/corgi-official-v1/selected-s42 --manifest data/processed/corgi-official-v1/manifest.json --output-dir runs/p2/corgi-official-v1/selected-s42-test
python -m src.baseline predict --input data/processed/corgi-official-v1/test.jsonl --model-dir artifacts/p2/corgi-official-v1/selected-s42 --manifest data/processed/corgi-official-v1/manifest.json --output-dir runs/p2/corgi-official-v1/selected-s42-test-default --threshold 0.5

# 4) 一条命令复现整个实验矩阵、验收 JSON 与图表
python -m src.baseline experiments --data-dir data/processed --output-dir runs/p2 --artifacts-dir artifacts/p2 --configs-dir configs/baseline --figures-dir reports/figures/P2
```

`predict` 的 `--threshold` 只用于把“默认 0.5”和“dev 调优阈值”各自的 run 都留档；测试集上不允许回头调参。run 目录的 `run_metadata.json` 记录实际阈值、数据 manifest hash、seed、code commit、配置与环境。

## 文件

```text
src/baseline/
  config.py      配置校验与默认值（未知键报错）
  data.py        manifest/hash 校验、train+dev 读取、无标签推理输入
  model.py       字符 TF-IDF + LR / 多数类参照，保存与离线加载
  selection.py   固定 0.10–0.90 步长 0.05 的 dev 阈值网格与参考指标
  sweep.py       dev-only 配置扫描与冻结选择规则
  cli.py         train / predict / sweep / experiments 入口与退出码
  adapter.py     load_predictor：P6 使用的离线适配器
  figures.py     报告图表
  experiments.py 完整实验矩阵与 acceptance.json
```

配置：`configs/baseline/default.json`（B1）、`b0-majority.json`、`b2-balanced.json`、`b3-char1gram.json`、`b3-balanced-char1gram.json`、`sweep.json`、`selected.json`（dev 选择后冻结）、`acceptance.json`（环境、run 清单、参考指标、局限）。

结果文件由 P4 统一评分：`metrics.json` 不由本模块写入；`acceptance.json` 中的指标是冻结参考实现的 P2 自查，P4 必须从 run 目录重算。

## 已知限制

- 字符特征只能表达词面关联，不能可靠处理否定、引用反驳与上下文；高 Accuracy 主要来自 train 中 83.7% 的负类。
- 官方 split 存在精确跨 split 同文；clean-train 版本是单独披露的敏感性分析，不能与主版混表。
- `class_weight="balanced"` 只改变正负类权重，不保证 Macro-F1 上升。
- 逻辑回归是凸优化，本实现下种子 42/43/44 结果可能完全相同；如实报告零方差，不伪造波动。
- 分数是 label=1 的模型分数，不是校准后的“偏见概率”；数据无目标性别金标，分组公平性由 P5 的 challenge 集分析。
