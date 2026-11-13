# P1 数据与审计

负责人：`yuge2018`，已在Issue #1认领；姓名/学号由本人补充。分支：`part/p1-data`。

P1已交付真实数据处理与审计，运行方法见[接入说明](../../src/data/README.md)，真实结果见[本人报告](../../reports/sections/P1.md)。

## 文件所有权

只改`src/data/`、`configs/data/`、`tests/p1/`、`requirements/p1.txt`、`reports/sections/P1.md`；图表放`reports/figures/P1/`。共享接口交P4维护。

## 输入与输出

输入：官方两份npy与锁定校验；fixtures/dataset。

输出：标准数据包、manifest、标签映射、ID映射、数据统计图、重复和近重复审计。

## 今天就能独立开工

```sh
python -m src.data --mock --output-dir runs/p1-smoke
python -m unittest discover -s tests -v
```

这两个命令当前可运行，结果只用于接口开发。真实CLI见[接口契约](../INTERFACES.md)，由本人实现；不等待组员后再开发。

## 工作步骤

1. 实现锁定版本下载与SHA256检查，通过校验后读取官方npy。
2. 按官方split转换：ori_sentence=1、non-bias text=0；不使用edit_sentence或bias_labels训练检测器。
3. 冻结JSONL与manifest；保证dev/test输入不含标签，映射与原类别只留审计侧。
4. 检查类别比例、长文本、重复/冲突；生成保留官方划分版和单列的clean-train敏感性版本。
5. 数据包交P2/P3/P4；在自己的报告片段写清实际数量、重复和覆盖限制。

## 完成证据

- [x] 同一源独立运行两次，两包各12个文件逐字节与hash一致。
- [x] 标准包通过load_dataset及完整包检查；主版本行数25590/1032/1047。
- [x] 清理版移除65行train；dev/test不变，清理依据与新hash可复核。
- [x] 本人报告片段、两组PNG/SVG图表、贡献记录、3分钟讲解材料已提交。

## 时间与交接

D2–D3完成fixture闭环；D4–D6独立完成核心工作；D7接真实文件验收；D8–D9写分析；D10复现与答辩。只通过[接口契约](../INTERFACES.md)交接，不导入其他模块内部实现。

## 答辩准备

数据为什么有偏斜？为什么子类向量不能当二分类标签？怎样防止ID和改写文本泄漏？
