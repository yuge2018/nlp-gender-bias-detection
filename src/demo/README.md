# P6 演示系统（src/demo）

本地 Streamlit 演示页：输入文本 → 选模型 → 通过冻结适配器打分 → 按阈值二分类。
页面不训练模型、不重算指标、不 import torch，也不判断作者性别。

## 运行

```sh
# 仅页面依赖
pip install -r requirements/p6.txt

# 真实/替身演示页（在仓库根目录）
streamlit run src/demo/app.py

# CI 用的 mock 冒烟，保留原行为，不启动页面
python -m src.demo --mock --output-dir runs/p6-smoke
```

`python -m src.demo` 不带 `--mock` 时只打印启动提示并返回码 2（真实入口是上面的 `streamlit` 命令），
这样 CI 既不需要 streamlit，也不会误把页面当命令行程序跑。

## 文件分工

| 文件 | 职责 | 是否 import streamlit |
|---|---|---|
| `app.py` | 页面：双页签（单句检测 / 真实结果只读）、模型下拉、输入、分数、阈值、mock 横幅、错误面板 | 是（唯一一个） |
| `registry.py` | 读 `configs/demo/*.json`；按契约懒加载模型；失败转 `ModelUnavailable`；提供模型四态 | 否 |
| `scoring.py` | 输入/阈值/分数校验，生成 `id/text/score/label` 行 | 否 |
| `results.py` | 只读遍历 `runs/**` 收集已入库的真实评测结果，默认排除 mock | 否 |
| `figures.py` | 只读 P4 `metrics.json` 出图，mock 图自动加 `mock_` 前缀 | 否（matplotlib 在函数内导入） |

测试只能碰 `registry/scoring/results/figures`；`tests/p6/test_demo.py` 有一条用子进程断言
“纯逻辑层不引入 streamlit/torch/transformers/sklearn”。

## 模型状态四态

模型目录存在 **不等于** 模型可用：权重是否入库因模型而异——P2 baseline 的 `model.joblib` 已入库，
P3 improved 目录目前只有 `model_metadata.json`。因此 `registry.model_status(entry, root)` 返回四种状态：

| status | 判定 | 页面表现 |
|---|---|---|
| `mock` | `adapter_module` 为 `null` | 可用，红底横幅声明无检测能力 |
| `ready` | 目录存在且至少有一个非元信息文件；如配置了 `probe_files`，则这些文件必须都在 | 可用，可进入真实加载 |
| `weights_missing` | 目录存在，但只有 `model_metadata.json`（或 `probe_files` 指定文件缺失） | 不可用，提示“权重未就绪”，`load_model` 抛出带重新生成建议的 `ModelUnavailable` |
| `missing` | 目录不存在 | 不可用，提示“模型目录未就绪” |

`is_available` 仅在 `ready` / `mock` 时为真。需要钉住某个必需权重文件时，在配置条目里加
`"probe_files": ["model.safetensors"]`。

## 真实结果（只读）页签

第二个页签不调用任何模型，只读仓库里已入库的 `runs/**`：

- 只认 P4 冻结的 `metrics.json`（用 `figures.load_metrics` 校验）；默认跳过 `is_mock=true` 的 run，
  并显示“跳过了多少条 mock”。
- 坏 `metrics.json` 不抛异常，归入“无法解析”列表并显示原因；只有 `predictions.jsonl`、
  尚无 `metrics.json` 的目录归入“待评测”，提示等 P4 评测。
- 列表里的 `macro_f1` / `positive_f1` / 混淆矩阵 / 阈值 / 方法 **全部取自文件**，页面不重算；
  仅展示时保留 4 位小数，不改动存储值。
- 选中单个 run 可展开 `st.json` 详情，便于报告引用时逐字段追溯。

## 模型从哪来（改配置，不改代码）

`configs/demo/default.json` 的 `models` 是唯一模型清单：

```json
{"key": "baseline", "display_name": "...", "adapter_module": "src.baseline.adapter",
 "model_dir": "artifacts/p2/corgi-official-v1/selected-s42", "is_deep": false}
```

- `adapter_module` 为 `null` 表示开发替身（`MockPredictor`，固定 0.5）。
- 接真实模型时只需改 `model_dir` 指向该角色 train CLI 生成的目录，页面无需改动。
- 现在 `default_model` 仍是 `mock`（保留“打开即可用”的保守默认）；baseline 的 `model.joblib` 与 `config.json` 已入库，
  `registry` 会把它标为 `ready`，下拉选中后即可现场打分；improved 仍会被标为 `weights_missing`，需本机放回权重。
- 模型不可用时页面给的是“目录不存在 / 仅剩元信息（权重未提交） / 适配器导入失败 / 缺依赖”的具体原因与建议，不会崩。

## 图表的来源与限制

```sh
python -m src.evaluation --mock --output-dir runs/p4-smoke
python -m src.demo.figures --run-dir runs/p4-smoke --output-dir reports/figures/P6
```

`figures.py` 只读 P4 冻结字段（`schema_version/is_mock/n/accuracy/macro_f1/positive_*/negative_f1/confusion_matrix`），
缺字段、混淆矩阵与 `n` 不符、负数格子都直接报错。`is_mock=true` 的图文件名带 `mock_` 前缀，
**不得放进最终报告**；D7 之后用真实 run 重跑同一条命令生成正式图。

## 页面口径（答辩会被问）

- 分数是适配器 `predict_proba` 给出的“有偏见（1）的模型分数”，**未做概率校准，不是客观偏见概率**。
- 阈值默认取模型记录值，页面可临时调整并显示；正式指标一律来自 P4，不由页面计算。
- mock 结果有红底提示：固定 0.5、无检测能力，不能作为模型表现或实验结论。
- 页面只显示文本与分数，不输出性别对象标签，也不推断作者身份。

## D7 对接清单

1. P2/P3 提供 `artifacts/p2/<dataset>/selected-s42/` 与 `artifacts/improved-i2-s42/`，目录内含各自 `model_metadata.json` 与权重。
   baseline 已入库（`model.joblib` + `config.json`）；improved 仍需本机补权重。
2. 核对 `load_predictor(model_dir, *, mock=False)` 返回对象有 `predict_proba(list[str]) -> list[float]`。
3. `configs/demo/default.json` 的路径已指向真实目录；`default_model` 保留 `mock`，需要现场真实推理时在下拉里选 `baseline`。
4. 页面分别用男性/女性偏见、中性描述、引用反驳各一条验证，再确认断网可跑。
