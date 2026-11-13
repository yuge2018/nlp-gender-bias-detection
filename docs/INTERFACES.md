# 接口契约 v1

`schema_version="1.0"`。接口已在开工前提供；所有人先用 `fixtures/`。以下真实模式CLI是**待各负责人实现的目标接口**；当前六个入口只支持`--mock`。

## 目录与依赖方向

```text
P1 data ── 标准数据包 ──┬── P2 baseline ── run目录 ──┬── P4 evaluation
                       └── P3 improved ── run目录 ──┤
                                                  └── P5 fairness
P2/P3 ── load_predictor + predict_proba ── P6 demo
P1–P5 ── 各自报告片段/图表 ── P6 报告组装
```

所有箭头都有fixture或替身。实现中不得跨模块导入对方内部代码；统一文件接口和下面的模型Protocol是唯一交接。

## 数据包

```text
data/processed/corgi-official-v1/
  manifest.json
  train.jsonl
  dev.jsonl
  test.jsonl
  labels/dev.jsonl
  labels/test.jsonl
  audit/id_mapping.jsonl
```

UTF-8，无BOM；每行一个JSON对象，最后有换行。每个文件ID唯一、不空，文本不空。train每行**恰好** `id,text,label`；dev/test每行**恰好** `id,text`；金标每行**恰好** `id,label`。ID是字符串，label是真正的整数0/1（不允许bool）。不修改原文本，不移除性别词、不盲目转义换行或删标点。额外元信息只进audit。

```json
{"id":"train-000001","text":"示例文本","label":1}
```

```json
{"id":"test-000001","text":"示例文本"}
```

```json
{"id":"test-000001","label":1}
```

manifest必填：`schema_version`、`dataset_id`、`is_mock`、`source_commit`、`label_map`、`files`。`files`用相对路径作key，每项`sha256`为文件原始字节的SHA256、`rows`为行数。文件包括train、dev、test及两个labels。原始数据版本、校验与划分规则记录在manifest或来源配置中；发布包必须包含一致的审计文件。

`dataset_manifest_sha256`统一定义为**manifest.json原始字节的SHA256**；manifest本身不包含自己的hash，避免循环。所有run记录这个hash。dev→test换评估split不改变数据版本。

ID映射用于追溯，不是模型特征。尤其本数据源按类别拼接，利用ID或行号能直接读出标签，是明确禁止的泄漏。

## 两个模型使用同一训练CLI

```sh
python -m src.baseline train --data-dir data/processed/corgi-official-v1 --config configs/baseline/default.json --output-dir artifacts/baseline-s42 --seed 42
python -m src.improved train --data-dir data/processed/corgi-official-v1 --config configs/improved/default.json --output-dir artifacts/improved-s42 --seed 42
```

`train`只读取train和dev，不读test金标。输出模型、tokenizer/向量器、`model_metadata.json`、配置快照、依赖版本和日志。保存的模型可以离线加载，不能要求重新训练才能演示。

```sh
python -m src.baseline predict --input data/processed/corgi-official-v1/test.jsonl --model-dir artifacts/baseline-s42 --manifest data/processed/corgi-official-v1/manifest.json --output-dir runs/baseline-s42-test
python -m src.improved predict --input data/processed/corgi-official-v1/test.jsonl --model-dir artifacts/improved-s42 --manifest data/processed/corgi-official-v1/manifest.json --output-dir runs/improved-s42-test
```

predict输入有label或额外字段就报错，禁读对应金标。推理DataLoader不得打乱顺序，批处理后还原原顺序。预测文件行与输入行一一对应。

## 模型适配器供P6调用

两个模型模块各自导出：

```python
load_predictor(model_dir: str, *, mock: bool = False) -> Predictor
predictor.predict_proba(texts: list[str]) -> list[float]
```

返回列表长度等于输入，元素是有限的 `[0,1]` 分数，定义为“有偏见label=1的模型分数”。不要求已做概率校准，页面不能将它表述为客观偏见概率。不传ID、标签或对象分组；空列表返回空列表，单条空文本报`ValueError`；模型不存在抛`FileNotFoundError`或有说明的加载错误。每个适配器应带`is_mock`标识。

P6不处理tokenizer或TF-IDF。P2/P3在自己的adapter内部实现预处理和批大小。替身仅供开发，当前返回固定0.5，不提供真实预测能力。

## Run产物

```text
runs/<method>-s<seed>-<split>/
  predictions.jsonl
  run_metadata.json
  config.json
  log.txt
  metrics.json          # 由P4添加，不由模型各自手算
```

预测行**恰好**`id,label,score`，分数含义如上。默认 `label=int(score >= threshold)`。threshold随run记录，不能在test上调。

```json
{"id":"test-000001","label":1,"score":0.72}
```

run_metadata必填：`schema_version`、`method`、`split`、`dataset_manifest_sha256`、`is_mock`、`seed`、`threshold`、`code_commit`、`model_name`、`model_revision`、`config`、`environment`。正式run另记开始/结束时间、硬件、训练/推理耗时与执行命令。`is_mock`必须与manifest一致；真实对比拒绝任何mock输入。只有真实pretrained模型才要求具体模型revision，不能只写随时间漂移的main。

P4必须核对：manifest文件hash、input与gold顺序/ID、预测行数/ID/顺序、0/1标签、score有限且范围正确、阈值一致、数据hash与run一致、mock标记一致。失败应返回非零退出码，不用zip截断数据或静默填标签。

## P4评测与汇总

```sh
python -m src.evaluation evaluate --data-dir data/processed/corgi-official-v1 --split test --run-dir runs/baseline-s42-test
python -m src.evaluation compare --run-dirs runs/baseline-s42-test runs/improved-s42-test --output-dir reports/results
```

`metrics.json`字段：`schema_version,is_mock,n,accuracy,macro_f1,positive_precision,positive_recall,positive_f1,negative_f1,confusion_matrix`，以及method、split、seed和数据hash以便追溯。混淆矩阵行是真值0/1，列是预测0/1，即`[[TN,FP],[FN,TP]]`。precision/recall/F1分母为0时按0处理；Macro-F1固定对0和1两类平均。分组FPR/FNR没分母则用null。保留原始浮点数，展示时再四舍五入。

本仓库 `src/common/contracts.py` 提供小型参考实现供契约和mock使用。P4的正式模块仍需实现CLI、分组、对比表、环境记录和复现验证；正式比较不得导入模型库。

`comparison.csv`每行包含method、data_hash、split、seed、mock标记、各指标；聚合表保留单次指标并另算均值和样本标准差。禁止比较不同数据hash、split、不同challenge版本或真实/mock混合结果。单次结果标准差用null。

## P5挑战集和对象分组

`challenge.jsonl`每行`id,text`，`challenge_labels.jsonl`每行`id,label`；`challenge_annotations.jsonl`每行`id,targets,phenomena,reviewers,adjudication`。对象和现象可以多标签；targets取`male,female,other_explicit,unspecified`。标注版本、文件hash、冻结时间单列manifest，训练接口不加载此文件。

`subgroup_metrics.json`包含数据版本、是否mock、每组n/positive_n/negative_n/FPR/FNR/Macro-F1。`error_cases.jsonl`保存id、真值、两模型预测/score、现象、人工解释。没有模型真结果也能用fixtures独立开发；不能将替身的错误当作模型研究结论。

## P6演示与报告消费

演示优先本地Streamlit或等价简单页面，文本输入→模型选择→适配器→分数及二分类；阈值从模型元信息读取。mock模式显著显示“开发替身，无真实检测能力”。页面把文本看作待分析内容，不评判作者身份。真实模式显示模型/数据版本，并支持无网络运行。

P6读P4结果文件画图，不自行重新定义指标；报告按`reports/sections/P1.md`至`P5.md`组装，P6写系统/演示/总结。缺少正式产物时标待完成，不能填示例分数。页面、最终包、文档和老师要求的输出由P6完成。

## 返回码

- `0`：成功。
- `2`：参数/契约/数据错误，stderr写出路径、行号或ID及原因。
- `1`：训练/加载/运行失败，保留错误日志。

可新增可选参数；不得重命名必需字段或改变`score`的正类含义。
