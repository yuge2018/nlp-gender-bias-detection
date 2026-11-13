# 中文文本性别偏见检测课程设计

选题编号 **8**：中文文本性别偏见检测（课程材料标注为 NLPCC 2025 Shared Task 7 Subtask 1）。6 人协作，使用 [CORGI-PM](https://github.com/yizhilll/CORGI-PM) 公开数据。

**P1已完成真实数据处理与审计**，负责人`yuge2018`；其他成员可按[P1接入说明](src/data/README.md)独立生成冻结数据。其余模块仍从示例骨架开发。

**检测对象是文本中的性别偏见，包括针对男性、女性及其他性别身份的刻板印象、角色限制与歧视。出现性别词本身不等于有偏见。** 项目只做二分类：`0=无偏见`，`1=有偏见`。数据实际覆盖哪些对象，需要审计和人工分析，不能从任务名称推断。

## 从这里开始

1. 阅读 [总计划与六人分工](PLAN.md)，认领 P1–P6 中的一项。
2. 阅读 [接口契约](docs/INTERFACES.md) 和对应的 [个人任务书](docs/roles/)。
3. Python 3.11 或 3.12 环境中，在仓库根目录运行：

```sh
python -m unittest discover -s tests -v
python -m src.data --mock --output-dir runs/p1-smoke
python -m src.baseline --mock --output-dir runs/p2-smoke
python -m src.improved --mock --output-dir runs/p3-smoke
python -m src.evaluation --mock --output-dir runs/p4-smoke
python -m src.fairness --mock --output-dir runs/p5-smoke
python -m src.demo --mock --output-dir runs/p6-smoke
```

上述mock入口只使用 Python 标准库，无需 GPU、下载模型或等待组员。输出均标为 `is_mock=true`，用于理解接口，**不是训练结果或模型性能**。P1的真实数据入口另见接入说明；真实训练、评测和页面由各负责人继续完成。

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

课设原文要求“原则上每组 5 人”；本仓库按用户确定的 **6 人** 设计，需向任课教师确认人数。尚未指定成员姓名和截止日期，任务使用 P1–P6 和相对日程，不虚构分配。
