# P5 第二标注：复核产物与裁决记录

## 本次复核概况（2026-11-25 完成）

五人各独立盲标 20 条（P5 负责人不参与盲标，只做合并与分歧裁决）：

| 分片 | 范围 | 标注者 |
|---|---|---|
| chunk-1 | p5-001 – p5-020 | yuge2018 |
| chunk-2 | p5-021 – p5-040 | cengjingdejiyi |
| chunk-3 | p5-041 – p5-060 | zy1un |
| chunk-4 | p5-061 – p5-080 | Vanharry0305 |
| chunk-5 | p5-081 – p5-100 | yakoul |

复核指标（首标 vs 二标，n=100）：

- 标签一致率 `label_agreement = 1.0`，标签 kappa `label_kappa = 1.0`
- 对象：`target_exact_match_rate = 0.93`，`target_jaccard_mean = 0.9333`
- 现象：`phenomenon_exact_match_rate = 0.62`，`phenomenon_jaccard_mean = 0.6683`
- 分歧 47 条：标签 0、对象 7、现象 38；已由负责人逐条裁决并写回 `data/challenge/p5-gender-challenge-v1/challenge_annotations.jsonl` 的 `adjudication` 字段。

## 产物清单

| 路径 | 说明 |
|---|---|
| `runs/p5/annotation-b/` | 合并后的第二标注四件套（挑战集文本/标签/标注/manifest 快照） |
| `runs/p5/review.json` | 首标 vs 二标的一致率、kappa、Jaccard 与逐条分歧（原始快照） |
| `review/validation-report.json` | 五份回传的本地校验报告（status=pass） |
| `review/assignment.csv` | 本次五人分片分配与回传状态 |
| `review/returned/chunk-1..5.csv` | 五份已校验回传 CSV |
| `review/p5-disagreements.csv` | 47 条分歧可读表（原文、双方对象/现象） |

## 复现复核指标

```sh
python -m src.fairness review \
  --first data/challenge/p5-gender-challenge-v1 \
  --second runs/p5/annotation-b \
  --output runs/p5/review.json
```

> 注：`runs/p5/review.json` 保存的是**首标 vs 二标**的原始快照；正式金标以裁决后的 `data/challenge/p5-gender-challenge-v1/` 为准。复核只度量分歧，最终条目不把多数票当作金标真理。

## 目录说明

- `review/chunks/`：早期生成的六份（16–17 条）备用分包方案，**本次五人复核未采用**，仅留存备查；不要用这套模板去合并本次 `review/returned/` 的回传。
- 本次实际分片以 `review/assignment.csv` 为准。
