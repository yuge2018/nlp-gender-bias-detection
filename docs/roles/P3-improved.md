# P3 预训练模型改进与消融

姓名/学号：张羽纶 202315060337。分支：`part/p3-improved`。

## 文件所有权

只改`src/improved/`、`configs/improved/`、`tests/p3/`、`requirements/p3.txt`、`reports/sections/P3.md`；图表放`reports/figures/P3/`。共享接口交P4维护。

## 输入与输出

输入：fixtures/dataset；以后换P1标准数据包；可用GPU或冻结编码器方案。

输出：模型/tokenizer、适配器、dev/test预测、I1/I2消融、环境和方法章节。

## 今天就能独立开工

```sh
python -m src.improved --mock --output-dir runs/p3-smoke
python -m unittest discover -s tests -v
```

这两个命令当前可运行，结果只用于接口开发。真实CLI见[接口契约](../INTERFACES.md)，由本人实现；不等待组员后再开发。

## 工作步骤

1. D2前确认硬件；默认中文RoBERTa单标签分类，资源不足按PLAN的替代方案冻结。
2. 使用CrossEntropyLoss与softmax正类分数，不能照搬上游多标签sigmoid判定成不完整二分类。
3. 实现train/predict和load_predictor，保存模型revision、tokenizer、checkpoint和离线加载参数。
4. 权重只从train计算；valid选模型和阈值；完成有/无类别加权消融。
5. 冻结后正式seed/test；报告改进与资源代价，不保证提升、不参考challenge挑参数。

## 完成证据

- [ ] 相同数据manifest与P2公平比较；推理不读test标签。
- [ ] 保存模型可离线加载；批处理不乱序；正类score范围和含义正确。
- [ ] 至少真实改进与权重消融，显式记录max_length、截断策略与环境。
- [ ] 本人报告片段、图表、贡献记录、3分钟讲解材料已提交。

## 时间与交接

D2–D3完成fixture闭环；D4–D6独立完成核心工作；D7接真实文件验收；D8–D9写分析；D10复现与答辩。只通过[接口契约](../INTERFACES.md)交接，不导入其他模块内部实现。

## 答辩准备

预训练表示为何可能有效？权重和阈值如何选择？为什么改进未必提升？
