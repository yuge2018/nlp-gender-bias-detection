# P1 数据处理与接入说明

负责人：`yuge2018`。真实数据准备、审计、两种版本冻结和验收已实现；不包含P2/P3训练和P5人工对象标注。

## 其他成员如何拿到同一数据

在仓库根目录，Python3.11/3.12环境运行：

```sh
python -m pip install -r requirements/p1.txt
python -m src.data prepare --raw-dir data/raw/corgi-pinned --output-dir data/processed
python -m src.data verify --data-dir data/processed/corgi-official-v1
python -m src.data verify --data-dir data/processed/corgi-clean-train-v1
```

第一次从官方锁定提交下载两份源文件（合计约7.3MB），校验SHA256、Git blob SHA1与文件大小，全部通过后才读取npy对象数组。再次运行自动复用已校验缓存；`--offline`禁止联网。损坏缓存会报错，不静默覆盖。

只运行mock或契约测试不需要numpy/matplotlib。只需准备数据、不画图时可以仅安装 `numpy==2.3.5`。图表可选：

```sh
python -m src.data prepare --offline --summary configs/data/preparation_summary.json --figures-dir reports/figures/P1
```

同一源及规则重新生成的文件逐字节一致；已有目标包完全一致时复用，内容不同就报错并要求选新目录，避免覆盖已冻结版本。命令失败返回2并说明原因；成功返回0。

## 冻结版本

| 数据包 | train | dev | test | 用途 |
|---|---:|---:|---:|---|
| `corgi-official-v1` | 25,590 | 1,032 | 1,047 | 保留官方split和标签的主实验 |
| `corgi-clean-train-v1` | 25,525 | 1,032 | 1,047 | 只删train与held-out完全同文的65行，单列敏感性分析 |

两个manifest的SHA256：

```text
corgi-official-v1    f78973f015839c13783c4a7fd3149ab7203c4ef1393845dc98d2b8e0ffbaf09a
corgi-clean-train-v1 dccc6ec10f2db6b35ceaf782e3e17d2a14916bf10e3959ca9e660248fb1e3fec
```

P2/P3各自跑相同版本，不能混用两个版本比较。run中的 `dataset_manifest_sha256`取对应manifest原始字节hash。源`valid`映射为`dev`，原始文本不归一化；模型只能用text。ID、行号、源文件、偏见子类、改写版本、对象标注都不作特征。

## 包内文件

```text
manifest.json
train.jsonl                    # id,text,label
dev.jsonl / test.jsonl          # id,text
labels/dev.jsonl / test.jsonl   # id,label，只供调优/评测
audit/id_mapping.jsonl          # 原文件/原split/原行号/原unique_id
audit/removed_train.jsonl       # 清理依据和对应held-out ID
audit/duplicate_groups.jsonl    # exact/normalized同文组与标签冲突
audit/near_duplicates.jsonl     # 跨split词面近重复候选的精确Jaccard复核结果
audit/quality_report.json       # 样本、类别、字符长度、重复/冲突/近重复统计
audit/source_verification.json  # 来源URL、提交、大小、SHA256与blob SHA1
```

`verify`检查所有12个文件（含manifest）、校验和、行数、字段、ID与审计映射。数据JSONL统一UTF-8无BOM和LF换行。清理版保持原ID，跳号是预期行为；dev/test输入及金标与主版逐字节一致。

原始/处理语料按仓库约定不进入Git，其他成员通过锁定脚本重新生成。可分享的统计摘要在 `configs/data/preparation_summary.json`，完整报告在 `reports/sections/P1.md`。

## 审计解释

- 主版train阳性16.30%，不能只看Accuracy，也不能让ID顺序泄漏类别。
- exact同文组存在11组标签冲突，保留原标签，不根据模型输出重标。
- 归一化只用于审计：NFKC＋casefold＋去空白，不改变模型输入。
- 近重复：归一化后至少30字符，字符3gram集合，固定seed的32维MinHash/8×4 LSH召回，再以Jaccard≥0.85精确验证；发现32对跨split不同词面文本。候选召回不是穷举，也不等于语义同文判定。
- clean版只清除精确同文训练行，仍存在词面近重复、6组train内标签冲突及dev/test交叠。它不能称为完全无泄漏版本。
- 字符长度阈值统计不等于模型tokenizer的token数或实测截断率。
- 数据没有对象性别金标；不能靠“男/女/他/她”频率证明男性、女性覆盖充分。任务定义适用于各性别对象，覆盖分析交P5人工复核。

## 本人验收与复跑

```sh
python -m unittest discover -s tests -v
python -m src.data --mock --output-dir runs/p1-smoke-new
python -m src.data prepare --offline --output-dir runs/p1-rebuild
```

测试覆盖源校验、失败下载不留半包、标签/文本保真、清理不用held-out标签、同文标签冲突、近重复、无标签推理输入、确定性输出、路径越界及篡改检查。完整真实语料已独立构建两次，两个版本各12个文件逐字节一致。验收记录见 `configs/data/acceptance.json`。
