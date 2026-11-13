# P4 统一评测与复现

姓名/学号：待认领。分支：`part/p4-evaluation`。

## 文件所有权

只改`src/evaluation/`、`configs/evaluation/`、`tests/p4/`、`requirements/p4.txt`、`reports/sections/P4.md`；图表放`reports/figures/P4/`。共享接口交P4维护。

## 输入与输出

输入：fixtures金标与独立mock run；以后换P2/P3真实run。

输出：评测CLI、metrics、comparison.csv、多种子汇总、契约验证、复现记录。

## 今天就能独立开工

```sh
python -m src.evaluation --mock --output-dir runs/p4-smoke
python -m unittest discover -s tests -v
```

这两个命令当前可运行，结果只用于接口开发。真实CLI见[接口契约](../INTERFACES.md)，由本人实现；不等待组员后再开发。

## 工作步骤

1. 实现evaluate/compare CLI，不导入模型实现或GPU库。
2. 扩展参考指标，保留正负类指标、混淆矩阵、样本数；加入文件hash/ID/顺序/阈值检查。
3. 拒绝真实/mock混合、不同split或manifest混合；均值聚合保留每次原始指标。
4. 维护冻结契约和CI，独立用手算小例子、失败文件和mock run验证。
5. 真实产物到齐后统一评分，换目录复现；报告主表、消融表、数据污染局限。

## 完成证据

- [ ] 手算混淆矩阵与F1一致；零分母、缺行、重复、乱序、NaN与版本错均有明确处理。
- [ ] 真实比较拒绝mock，保存每个指标的输入run路径/hash。
- [ ] 提供主表、逐次seed结果和至少一次干净环境复现记录。
- [ ] 本人报告片段、图表、贡献记录、3分钟讲解材料已提交。

## 时间与交接

D2–D3完成fixture闭环；D4–D6独立完成核心工作；D7接真实文件验收；D8–D9写分析；D10复现与答辩。只通过[接口契约](../INTERFACES.md)交接，不导入其他模块内部实现。

## 答辩准备

Macro-F1与正类F1差别？为什么要核对ID而非只对行数？怎样避免测试调参？
