# P5 多性别对象挑战集与错误分析

负责人：待认领（`part/p5-fairness`）。本模块只读**冻结文件**：带封印 manifest 的 challenge 目录，以及各模型模块产出的预测 run 目录。它不复现模型、不在 challenge 上调参，也不修改官方 test 主表。

## 挑战集

| 文件 | 内容 |
|---|---|
| `data/challenge/p5-gender-challenge-v1/challenge.jsonl` | `id,text`，100 条中文句子 |
| `.../challenge_labels.jsonl` | `id,label`，0 无偏见 / 1 有偏见 |
| `.../challenge_annotations.jsonl` | `id,targets,phenomena,reviewers,adjudication`；对象与现象分开，均为多标签 |
| `.../manifest.json` | 文件 SHA256、行数、标注版本、冻结时间、复核状态、构成统计 |

构成：男性 45（30 正 / 15 负）、女性 45（30 正 / 15 负）、其他性别身份 10（6 / 4）、多目标 5（5 / 0）、对象不明确 5（0 / 5）；共 61 正 / 39 负。`targets` 取 `male/female/other_explicit/unspecified`，一条句子可同时属于多个对象组；**分组只按标注对象，不按作者性别，也不用代词频率充当真实群体标签。**

标签与对象/现象分开保存，`challenge_annotations.jsonl` 不是模型特征。`configs/fairness/default.json` 的 `use_challenge_for_tuning=false` 是硬约束。

## 命令

```sh
# 0) 重新生成挑战集（幂等；内容与生成器不一致时拒绝覆盖）
python -m src.fairness build --output-dir data/challenge/p5-gender-challenge-v1

# 1) 用某个已保存模型给 challenge 打分；只走公开 adapter，阈值取模型元信息
python -m src.fairness score --component baseline --model-dir artifacts/p2/corgi-official-v1/selected-s42 \
    --name p2-selected --output-dir runs/p5/challenge-p2-selected

# 2) 分组指标 + 错误案例 + 图表；--prediction 可重复
python -m src.fairness report \
    --prediction p2-selected=runs/p5/challenge-p2-selected \
    --prediction p2-char1-3gram=runs/p5/challenge-p2-char1-3gram \
    --output-dir runs/p5/report --case-notes configs/fairness/case_notes.json

# 3) 双人独立复核：比较两份标注的标签一致率、kappa 与对象/现象分歧
python -m src.fairness review --first annotation-a/ --second annotation-b/ --output runs/p5/review.json

# 4) 接口接线（fixture 级，结果全部标 is_mock，不是研究结论）
python -m src.fairness --mock --output-dir runs/p5-smoke
```

真实报告需要 `matplotlib`；核心分析与测试只用标准库（`requirements/p5.txt`）。

## 输出

```text
runs/<name>/                     # score
  predictions.jsonl              # id,label,score（阈值取模型元信息）
  run_metadata.json              # split=challenge，记录模型/阈值/seed/mock 标记
  config.json / log.txt
runs/p5/report/                  # report
  subgroup_metrics.json          # 每个对象组、现象组、极性组 × 每个模型的指标
  error_cases.jsonl              # 每个案例含两模型 label/score、现象、人工解释
  log.txt
reports/figures/P5/*.png         # 构成图 + 目标分组指标图
```

`subgroup_metrics.json` 每组都带 `n / positive_n / negative_n` 和混淆矩阵四格，`fpr/fnr` 分母为 0 时写 `null`。`macro_f1` 按项目主指标固定对 0/1 两类平均（缺类按 0）；单极性子组应看 `macro_f1_defined_classes`。指标定义同时写进文件，避免两种口径被混读。

## 设计决定

- **标签是判定，不是词匹配。** 出现“男/女/他/她”而只是中性描述、否定刻板说法、或引用后反驳，记 0；`quoted_rebuttal`、`denial_of_bias`、`bias_awareness` 专门用于检验这类结构。
- **善意性别歧视单列。** “男人就得让着女人”“女人要温柔贤惠”以保护/赞美包装从属地位，仍记 1。
- **小分组只作描述。** `n<20` 的对象组、`n<10` 的现象组在 `warnings` 中被点名；不把个位数分组当稳定估计。
- **不按作者身份分组。** 文本是待分析内容，不是作者画像。

## 已知限制

- 本挑战集是**单标注者草案**，`reviewers` 为空，双人独立复核与裁决未完成；报告不得称其为已复核基准。
- `unspecified` 组没有正例，无法支持“对象不明确时的偏见”这一结论。
- P3 改进模型尚未产出，当前真实对照只有 P2 selected 与 P2 1-3gram 消融；P3 到达后只需新增一条 `--prediction`。
- 挑战集 100 条、正类 61%，是刻意平衡的小样本，不能外推总体偏见率。
