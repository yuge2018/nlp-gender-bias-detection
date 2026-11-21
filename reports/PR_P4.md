# PR 文案（P4）

## Title

```
[P4] 新增统一评测与复现模块：契约检查、Macro-F1 与多种子汇总
```

## Description

```markdown
## 本次交付

**负责人**：P4（王海宇 / 202315060325，如认领人不同请替换）
**对应 Issue**：#<填写 P4 Issue 编号>
**分支**：`part/p4-evaluation` → `main`

### 改变的行为

新增一套独立的评测模块，不改动 P1/P2/P3/P5/P6 的任何实现，也不修改已冻结的契约字段：

- 新增 `src/evaluation/`：`evaluate` / `compare` / `--mock` 三个入口，只用 Python 标准库，不导入模型代码或 GPU 库；
- 新增 `src/common/contracts.py`：schema 常量与 JSONL/哈希的最小参考实现（共享文件由 P4 维护，本次为新增，未改动既有字段）；
- 新增 `configs/evaluation/default.json`：评测口径（Macro-F1 固定两类平均、阈值只从 run 元信息读取）；
- 新增 `tests/p4/`：21 个测试，覆盖指标手算与全部失败分支；
- 新增 `reports/sections/P4.md`、`reports/figures/P4/macro_f1_by_method.png`；
- 新增 `fixtures/`：可重复生成的 mock 数据包与 mock run，供 P5/P6 在真实产物到位前联调。

### 输入

- `--data-dir`：P1 标准数据包（`manifest.json`、`{split}.jsonl`、`labels/{split}.jsonl`）；
- `--run-dir` / `--run-dirs`：P2/P3 的 run 目录（`predictions.jsonl` + `run_metadata.json`）。

### 输出

- 每个 run 目录写回 `metrics.json`（含 n、accuracy、macro_f1、正负类 P/R/F1、混淆矩阵、数据 hash、预测文件 hash）；
- `comparison.csv`（每次 run 一行）、`aggregate.csv`（均值 ± 样本标准差）、`comparison.md`、`environment.json`、`reproduction.json`、`metrics/`。

### 校验规则（失败返回退出码 2，并指出路径/行号/ID）

manifest 字段与版本、文件 SHA256 与行数、预测行数、ID 顺序与重复、label 必须为整数 0/1、score 有限且在 [0,1]、`label == int(score >= threshold)`、`is_mock` 与 manifest 一致、`dataset_manifest_sha256` 与数据包一致、跨 run 的 data hash / split / mock 标记一致。

## 验证

运行命令与结果：

```sh
python -m src.evaluation --mock --output-dir runs/p4-smoke
# [mock] 评测 6 个 run，结果写入 runs/p4-smoke
# baseline  Macro-F1 0.7181 ± 0.0708（seed 42/43/44）
# improved  Macro-F1 0.8822 ± 0.0738（seed 42/43/44）

python -m unittest discover -s tests -v
# Ran 21 tests in 3.0s
# OK
```

是否真实数据/模型：**否**。本次全部使用内置合成 fixtures（`is_mock=true`），仅验证评测链路，未使用真实数据或真实模型。

- mock 数据 manifest hash：`488b1d22f40c9ee7bc9792b76149545e2fcd182e181216dd57cdd2513503e282`
- run 路径：`fixtures/runs/{baseline,improved}-s{42,43,44}-test`
- 产物路径：`runs/p4-smoke/`

真实 run 到位后，用同一命令替换 `--data-dir` 与 `--run-dirs`，输出到 `reports/results/`，并同步替换 `reports/sections/P4.md` 中的表格与图。

## Checklist

- [x] 只修改本人目录或已说明共享接口变更
      新增 `src/evaluation/`、`configs/evaluation/`、`tests/p4/`、`requirements/p4.txt`、`reports/sections/P4.md`、`reports/figures/P4/`；共享文件 `src/common/contracts.py` 为新增，未改动既有契约字段，不影响 P1–P6。
- [x] 标签与 score 正类含义一致，未使用 ID/源类别泄漏
      `score` 统一为“有偏见 label=1”的分数，`label = int(score >= threshold)`，阈值只从 `run_metadata.json` 读取；ID 仅用于顺序与一致性校验，不作为特征，评测器不读取 `edit_sentence`、偏见子类或源文件信息。
- [x] mock 与真实结果明确区分
      manifest 与 run 元信息均带 `is_mock=true`，`comparison.md` 顶部有 MOCK 提示；`compare` 默认拒绝 mock（需显式 `--allow-mock`），并拒绝真实/mock 混用。
- [x] 报告片段、图表与答辩要点同步更新
      `reports/sections/P4.md` 已包含评测口径、契约清单、结果表（标注 MOCK）、复现记录、局限与答辩三问；图表为 `reports/figures/P4/macro_f1_by_method.png`。

## 后续（D7）

接 P2/P3 真实 run 后：核对 6 个 run 的 `is_mock=false` 与同一 `dataset_manifest_sha256` → `compare` 输出到 `reports/results/` → 换目录复跑一次，核对 `reproduction.json` 与 `metrics.json` 一致 → 更新 P4.md 主表。
```
