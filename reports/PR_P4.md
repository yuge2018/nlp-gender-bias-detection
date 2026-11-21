# PR 文案（P4）

## Title

```
[P4] 统一评测 CLI、对比汇总与复现记录
```

## Description

```markdown
## 本次交付

**负责人**：Vanhorry0305（P4）
**分支**：`part/p4-evaluation` → `main`

### 改变的行为

把原先放在仓库根目录的 `P4-evaluation-package.zip` 拆成按目录结构合入的正常实现，
不改动 P1/P2/P3/P5/P6 的任何代码，也不修改 `src/common/contracts.py` 的冻结字段：

- `src/evaluation/cli.py`、`runner.py`、`aggregate.py`、`__main__.py`：`evaluate` / `compare` / `--mock` 三个入口，只用标准库；
- 指标直接调用 `src/common/contracts.py` 的 `binary_metrics`，不再维护第二份定义；
- `src/evaluation/plot_results.py`：由 `aggregate.csv` 出 Macro-F1 均值±标准差图；
- `tests/p4/`：15 项测试，覆盖指标一致性、缺行/重复/乱序/NaN/越界/bool 标签、阈值不符、mock 标记与数据 hash 不一致、mock 拒绝与混合拒绝、单次标准差为 null、CLI 退出码；
- `reports/sections/P4.md`：评测口径、契约清单、复现记录、局限与答辩三问；
- `requirements/p4.txt`：声明仅标准库。

同时删除本分支此前上传的重复实现（`src/evaluation/checks.py`、`fixtures.py`、`metrics.py`
与 `tests/p4/test_contracts.py`、`test_metrics.py`）以及根目录 zip，避免与共享契约重复定义指标。

### 输入与输出

- 输入：P1 标准数据包（`--data-dir`）、P2/P3 的 run 目录（`--run-dir` / `--run-dirs`）；
- 输出：每个 run 写回 `metrics.json`；比较输出 `comparison.csv`、`aggregate.csv`、
  `comparison.md`、`environment.json`、`reproduction.json`。

### 校验规则（失败返回退出码 2，并指出路径/行号/ID）

manifest 字段与版本、文件 SHA256 与行数、预测行数、ID 顺序与重复、label 必须为整数 0/1、
score 有限且在 [0,1]、`label == int(score >= threshold)`、`is_mock` 与 manifest 一致、
`dataset_manifest_sha256` 与数据包一致、跨 run 的 data hash / split / mock 标记一致。

## 验证

```sh
python -m src.evaluation --mock --output-dir runs/p4-smoke
# exit=0，生成 2 个 mock run 与 comparison/aggregate/comparison.md/environment/reproduction

python -m unittest discover -s tests -v
# Ran 119 tests ... OK（P4 新增 15 项）
```

是否真实数据/模型：**否**。mock 仅使用 `fixtures/dataset`（`is_mock=true`）验证评测链路，
不作为模型性能结论；真实 run 只有 `runs/improved-i2-s42-test`，且 `data/processed` 未入库，
无法独立复算，因此主表留空。

- mock 数据 manifest SHA256：`764d782b7e77dc6cc2f82d4bb404497d2e0768e846358d5280f3daf84b462b2c`
- 产物路径：`runs/p4-smoke/`（已 gitignore）

## Checklist

- [x] 只修改本人目录或已说明共享接口变更
      仅新增 `src/evaluation/`、`tests/p4/`、`requirements/p4.txt` 与报告片段；`src/common/contracts.py` 未改动。
- [x] 标签与 score 正类含义一致，未使用 ID/源类别泄漏
      `score` 为“有偏见 label=1”的分数，阈值只从 `run_metadata.json` 读取；ID 只做一致性校验，不作特征。
- [x] mock 与真实结果明确区分
      `comparison.md` 顶部有 MOCK 提示；`compare` 默认拒绝 mock，并拒绝真实/mock 混用。
- [x] 报告片段、图表与答辩要点同步更新
      `reports/sections/P4.md` 已更新；`reports/figures/P4/macro_f1_by_method.png` 标题含 `[MOCK]`，真实结果后覆盖。

## 后续（D7）

P2/P3 六个 run 到齐后：`evaluate` 逐个打分 → `compare` 输出到 `reports/results/` →
换目录复跑核对 `reproduction.json` → 用 `plot_results` 覆盖图表 → 回填 P4.md 主表。
```
