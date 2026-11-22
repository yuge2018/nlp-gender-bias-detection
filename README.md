# 中文文本性别偏见检测课程设计

选题编号 **8**：中文文本性别偏见检测（课程材料标注为 NLPCC 2025 Shared Task 7 Subtask 1）。6 人协作，使用 [CORGI-PM](https://github.com/yizhilll/CORGI-PM) 公开数据。

仓库现为公开仓库；组员Ready PR通过检查后自动合并。Draft或`manual-merge`标签可暂停，成员也可从自己的Fork提交，见[小组协作说明](docs/WORKFLOW.md)。

六个模块均已合入 `main`，各自有真实产物与报告：P1 数据转换与审计、P2 字符 TF-IDF＋LR 基线、P3 预训练模型微调、P4 统一评测与复现、P5 多性别对象挑战集与分组分析、P6 Streamlit 演示与报告组装。各模块的命令与证据见 `src/<模块>/README.md` 与 [`reports/sections/`](reports/sections/)，统一主表在 [`reports/results/`](reports/results/)。完成度与剩余待办见 [审计 issue #32](https://github.com/yuge2018/nlp-gender-bias-detection/issues/32)。

**检测对象是文本中的性别偏见，包括针对男性、女性及其他性别身份的刻板印象、角色限制与歧视。出现性别词本身不等于有偏见。** 项目只做二分类：`0=无偏见`，`1=有偏见`。数据实际覆盖哪些对象，需要审计和人工分析，不能从任务名称推断。

## 从这里开始

1. 阅读 [总计划与六人分工](PLAN.md)，认领 P1–P6 中的一项。
2. 阅读 [接口契约](docs/INTERFACES.md) 和对应的 [个人任务书](docs/roles/)。
3. Python 3.11 至 3.13 任一版本（132 项测试已在 3.13.18 通过），在仓库根目录运行：

```sh
python -m unittest discover -s tests -v
python -m src.data --mock --output-dir runs/p1-smoke
python -m src.baseline --mock --output-dir runs/p2-smoke
python -m src.improved --mock --output-dir runs/p3-smoke
python -m src.evaluation --mock --output-dir runs/p4-smoke
python -m src.fairness --mock --output-dir runs/p5-smoke
python -m src.demo --mock --output-dir runs/p6-smoke
```

上述mock入口只使用 Python 标准库，无需 GPU、下载模型或等待组员。输出均标为 `is_mock=true`，用于理解接口，**不是训练结果或模型性能**。真实产物已入库：P2 的 12 个 test run 与 P3 的 I2 run 已由 P4 统一评测，主表见 `reports/results/`；演示页用 `streamlit run src/demo/app.py` 启动（依赖见 `requirements/p6.txt`）。 模型权重：P2 各变体已入库（#36）；P3 演示与主对照见 GitHub Release `p3-models-v1`（zip 含 SHA256，解压回 `artifacts/<tag>/` 即可离线加载，其余 run 按 `configs/improved/acceptance.json` 的命令重训约 40 分钟/个）。

## 六个互不抢文件的工作区

| 编号 | 工作 | 独占目录 | 任务书 |
|---|---|---|---|
| P1 | 数据转换、质量与泄漏审计 | `src/data/`、`configs/data/` | [P1](docs/roles/P1-data.md) |
| P2 | 字符 TF-IDF＋逻辑回归基线与消融 | `src/baseline/`、`configs/baseline/` | [P2](docs/roles/P2-baseline.md) |
| P3 | 中文预训练模型微调与消融 | `src/improved/`、`configs/improved/` | [P3](docs/roles/P3-improved.md) |
| P4 | 统一评测、实验汇总、复现验证 | `src/evaluation/`、`configs/evaluation/` | [P4](docs/roles/P4-evaluation.md) |
| P5 | 多性别对象挑战集、分组与错误分析 | `src/fairness/`、`configs/fairness/` | [P5](docs/roles/P5-fairness.md) |
| P6 | 演示界面、报告组装、答辩与交付 | `src/demo/`、`configs/demo/` | [P6](docs/roles/P6-demo.md) |

每人另有 `reports/sections/Pn.md`、`tests/pn/`、`requirements/pn.txt`。不要为了开发自己的部分去修改别人的实现。模型只接收文本，其他模块只通过固定 JSON 文件或模型适配器连接。

## 重要约定

- [任务边界](docs/TASK_SCOPE.md)：男性偏见、女性偏见、引用反驳与非偏见性别描述。
- [实测数据审计](docs/DATA_AUDIT.md)：直接检测样本 27,669 条，原始划分与重复文本统计。
- [实验矩阵](docs/EXPERIMENTS.md)：主指标、基线、改进、消融和补充分析。
- [集成与交付](docs/INTEGRATION.md)：只在数据冻结、模块验收、最终交付三个节点对齐。
- [日常协作](docs/WORKFLOW.md)：任务认领、分支/Fork、自动合并及暂停恢复。

课设原文要求“原则上每组 5 人”；本仓库按用户确定的 **6 人** 设计，需向任课教师确认人数。尚未指定成员姓名和截止日期，任务使用 P1–P6 和相对日程，不虚构分配。
