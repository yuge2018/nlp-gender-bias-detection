# P2 传统基线与特征消融

负责人：`cengjingdejiyi`（姓名/学号由本人填写）。分支：`part/p2-baseline`。实现与实验证据已完成，接入说明见[P2模块说明](../../src/baseline/README.md)。

## 文件所有权

只改`src/baseline/`、`configs/baseline/`、`tests/p2/`、`requirements/p2.txt`、`reports/sections/P2.md`；图表放`reports/figures/P2/`。共享接口交P4维护。

## 输入与输出

输入：fixtures/dataset；以后换P1标准数据包。

输出：模型、向量器、适配器、dev/test预测、配置/日志、B0–B3消融、方法章节。

## 今天就能独立开工

```sh
python -m src.baseline --mock --output-dir runs/p2-smoke
python -m unittest discover -s tests -v
```

这两个命令当前可运行，结果只用于接口开发。真实CLI见[接口契约](../INTERFACES.md)，由本人实现；不等待组员后再开发。

## 工作步骤

1. 实现字符TF-IDF＋LogisticRegression；先建立多数类参照。
2. 向量器只fit train，dev仅调C、ngram和class_weight及阈值。
3. 按契约实现train/predict CLI和load_predictor；只能用text当特征。
4. 主基线完成无加权/加权、1gram/1–3gram消融；不依赖P3任何代码。
5. 冻结配置后运行正式seed和test，写方法、耗时、局限；分数由P4统一计算。

## 完成证据

- [ ] 同一保存模型离线加载，batch与单条预测一致。
- [ ] 输出label=1分数、ID顺序和阈值正确；拒绝含label的推理输入。
- [ ] 保存train-only拟合证据、配置与可追溯run；至少一份真实基线结果。
- [ ] 本人报告片段、图表、贡献记录、3分钟讲解材料已提交。

## 时间与交接

D2–D3完成fixture闭环；D4–D6独立完成核心工作；D7接真实文件验收；D8–D9写分析；D10复现与答辩。只通过[接口契约](../INTERFACES.md)交接，不导入其他模块内部实现。

## 答辩准备

为什么选字符特征？类别加权影响什么？高Accuracy是否说明能识别偏见？
