# P5 测试

`test_fairness.py` 覆盖：挑战集生成与封印（100 条、对象≥20、双极性、manifest hash 篡改拒绝）、
标注 schema（未知 target/phenomenon 拒绝）、手算分组 FPR/FNR/Macro-F1、零分号写 null、
多对象组切分、预测 label 与阈值矛盾拒绝、score 越界拒绝、ID 缺失/错位拒绝、mock 与真实混表拒绝、
report 产物字段、`--mock` 接线、双人复核 kappa 与分歧计数。

运行：`python -m unittest tests.p5.test_fairness -v`（无需额外依赖）。
