# P6 演示截图

运行环境：Windows，项目内 `.venv`，命令在仓库根目录执行。

```sh
pip install -r requirements/p6.txt
streamlit run src/demo/app.py
```

截图于 2026-11-22 16:15–16:22 实拍，页面默认模型已改为 `baseline`，权重来自仓库内
`artifacts/p2/corgi-official-v1/selected-s42/model.joblib`，未使用任何 mock 分数替代。

| 文件 | 内容 | 说明 |
|---|---|---|
| `01-默认基线-真实打分.png` | 打开页面默认选中「字符 TF-IDF + 逻辑回归（P2 基线）」，输入两句话后按阈值 0.50 打分 | `她是一名工程师。` → 0.581（有偏见），`他是一个温柔的人。` → 0.823（有偏见） |
| `02-阈值0.60-判定变化.png` | 同一组输入把阈值调到 0.60 重新检测 | 0.581 变「无偏见（0）」，0.823 仍「有偏见（1）」，证明阈值生效、分数本身不变 |
| `03-真实结果页-13条.png` | 「真实结果（只读）」页签，表格按 `run` 排序 | 表格默认一屏显示 8 行、可上下滚动，共 13 条真实 run；数据读自 `runs/**/metrics.json` |
| `04-真实结果-指标列.png` | 同一表格横向滚动到右侧指标列 | 可见 `n=1047`、`Macro-F1`、`正类F1`、`负类F1` 与 mock 勾选列，数值取自文件 |
| `05-单run详情-improved-I2.png` | 选中 `improved-i2-s42-test` 展开 JSON 详情 | 混淆矩阵 `[[504,22],[137,384]]`、`macro_f1=0.8461`、`data_hash` 可逐字段核对 |
| `06-improved权重未就绪.png` | 选「中文 RoBERTa 微调（P3 改进）」并运行 | 页面给出“权重未就绪”原因与建议，不抛异常；说明四态里 `weights_missing` 的界面表现 |
| `07-mock替身无检测能力.png` | 选「开发替身」并运行 | 红底横幅声明固定 0.5、无检测能力，结果仅用于接口验证 |

注意：

- 截图里的 0.8461 / 0.8285 等指标由 P4 评测写入 `runs/improved-i2-s42-test/metrics.json`，
  页面只读展示、不重算；报告引用时以 P4 主表 `reports/results/comparison.csv` 为准。
- 页面显示的是“未做概率校准的模型分数”，不能称为客观偏见概率。
- `improved` 真实推理需要 P3 提供完整 artifacts（权重 + config + tokenizer）并安装
  `torch`/`transformers`，本组截图未包含这部分，避免把“权重缺失”包装成可用结果。
