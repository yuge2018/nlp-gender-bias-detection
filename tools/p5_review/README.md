# P5 第二标注：分包、回传与合并

第二遍标注用于算 Cohen's kappa 与对象/现象分歧。规则只有一条：**标注者之间互相看不见，P5 负责人不参与盲标**，负责人只做分发、合并与分歧裁决。

## 1. 分包（负责人做一次）

```sh
python tools/p5_review/split_chunks.py \
    --challenge-dir data/challenge/p5-gender-challenge-v1 \
    --output-dir review/chunks --parts 6
```

产出 `review/chunks/chunk-1.csv` … `chunk-6.csv`（每组 16–17 条，正负比例接近 61:39）和 `assignment.csv`（记谁拿了哪份）。

- 每份 CSV 只有 `id,text` 两列有内容，**不含任何金标**；
- 标注者不得打开仓库里的 `data/challenge/p5-gender-challenge-v1/challenge_labels.jsonl` 与 `challenge_annotations.jsonl`，看过金标就不再算独立盲标；
- 按可参与人数发放，同一人只能拿一份；人数不足时先按 5 份发，剩下的组等下一轮，不要让人重复标；
- 分发方式：每人一个独立链接（secret gist / 私聊文件），不要把全部 CSV 发到群里。

## 2. 填法（发给每位标注者）

只填这四列，`id` 与 `text` 不要改：

| 列 | 怎么填 |
|---|---|
| `label` | `1` 有偏见，`0` 无偏见；每行必填 |
| `targets` | `male` / `female` / `other_explicit` / `unspecified`，多个用英文分号 `;` |
| `phenomena` | 13 类之一或多个，多个用英文分号 `;`（见下表） |
| `notes` | 一句话理由，必填 |

`label=0` 通常填：

```
targets=unspecified
phenomena=neutral_description
notes=只是中性描述，没有价值判断
```

### phenomena 13 类

| # | 英文值 | 中文含义 | 通常 label |
|---|---|---|---|
| 1 | `explicit_stereotype` | 显性刻板印象：直接断言某性别的能力、性格或天性 | 1 |
| 2 | `implicit_stereotype` | 隐性刻板印象：反问、预设、玩笑或暗示，未直接断言 | 1 |
| 3 | `role_constraint` | 角色限制：规定某性别应该或不应该做什么 | 1 |
| 4 | `derogation` | 贬损：侮辱、嘲讽或以性别为由贬低 | 1 |
| 5 | `benevolent_sexism` | 善意性别歧视：以保护、礼让、赞美的形式固化从属地位 | 1 |
| 6 | `appearance_inference` | 外貌推断：由外表推断能力、可靠性或性别气质 | 1 |
| 7 | `victim_blaming` | 受害者归责：把侵害归因于当事人的穿着或性别 | 1 |
| 8 | `labor_market_penalty` | 职场差别对待：招聘、晋升、岗位分配上的性别门槛 | 1 |
| 9 | `other_identity_derogation` | 其他性别身份贬损：对跨性别、非二元等身份的污名化 | 1 |
| 10 | `neutral_description` | 中性描述：提到性别，但只是陈述事实或身份 | 0 |
| 11 | `denial_of_bias` | 否定偏见说法：明确拒绝某一性别刻板要求 | 0 |
| 12 | `quoted_rebuttal` | 引用并反驳：句中出现偏见说法，但随即被否定 | 0 |
| 13 | `bias_awareness` | 偏见现象讨论：客观指出性别不平等或偏见的危害 | 0 |

## 3. 回传（标注者做）

1. Excel/表格里填完，另存为 **CSV UTF-8（逗号分隔）**；
2. 文件名必须还是 `chunk-N.csv`（N 与你拿到的一致）；
3. 回传到 issue #5 评论框或直接私发给负责人，不要改列名、不要加说明行。

## 4. 校验与合并（负责人做）

```sh
# 逐份校验，任何一份有错都会列出具体行
python tools/p5_review/validate_returned.py \
    --chunk review/chunks/chunk-2.csv --returned review/returned/chunk-2.csv

# 六份齐了再合并；缺一份或有一份报错都会中止
python tools/p5_review/merge_annotations.py \
    --challenge-dir data/challenge/p5-gender-challenge-v1 \
    --chunks-dir review/chunks --returned-dir review/returned \
    --output-dir review/pass2

# 合并产物可直接喂给复核命令，输出标签一致率、kappa、对象/现象 Jaccard 与逐条分歧
python -m src.fairness review \
    --first data/challenge/p5-gender-challenge-v1 \
    --second review/pass2 --output runs/p5/review.json
```

`merge_annotations.py` 会生成 `review/pass2/`（`challenge.jsonl`、`challenge_labels.jsonl`、`challenge_annotations.jsonl`、重算 hash 的 `manifest.json`）和 `review/merge_report.json`（每份的行数、错误与标注者代号），不会改写原始挑战集。

## 5. 合并之后

- 用 `runs/p5/review.json` 里的 `label_kappa` 判定是否达到复核标准（建议先按标签一致率与 kappa 都 ≥0.7 讨论，达不到就追加第三遍或缩小结论范围）；
- 分歧逐条在 #5 里裁决，裁决记录写回原始 `challenge_annotations.jsonl` 的 `adjudication` 字段，再把 `review_status` 从 `single_annotator_draft` 改为已复核；
- 在复核完成前，报告与 README 里“未复核”的标注不能删。
