# P2 训练权重（已入库）

P2 `char_tfidf_lr` 各变体的训练产物，**刻意纳入 Git**，让 clone 后可以直接加载、复跑主表。
单文件 0.21–4.30 MB，合计 19.26 MB，远低于 GitHub 50 MB 警告线。

- `corgi-official-v1/`：官方训练集（B0 多数类基线、B2 balanced、B3 char n-gram、B4 selected）
- `corgi-clean-train-v1/`：清洗训练集（同上）
- 每个变体含 `config.json`、`model_metadata.json`、`log.txt`、`model.joblib`

## 加载

`model.joblib` 是仓库自定义的 `BaselineModel`，请在仓库根目录用官方 loader 加载：

```python
from src.baseline.model import load_model

model, metadata = load_model("artifacts/p2/corgi-official-v1/selected-s42")
probabilities = model.predict_proba(["她不适合做工程师。", "他是一位温柔的父亲。"])
print(probabilities)
```

若在别的目录运行，先把仓库根目录加入 `PYTHONPATH`。

## 校验和

| 文件 | 大小 | SHA256 |
|---|---|---|
| `corgi-clean-train-v1/b0-majority-s42/model.joblib` | 0.3 KB | `b4fe1afd7bbb8d76d7a980c940001895e23035dfa08ff8c397d520c32c9eb480` |
| `corgi-clean-train-v1/b2-balanced-s42/model.joblib` | 4397.5 KB | `f548d7eeb86f9a787521eeb7f771d80e053005b2fff0a6ea427371658d59d211` |
| `corgi-clean-train-v1/b3-balanced-char1gram-s42/model.joblib` | 212.5 KB | `2c5ea53d7168c1c39d77722672bdbb12108c625ba7b7097d62c61386cc70f650` |
| `corgi-clean-train-v1/b3-char1gram-s42/model.joblib` | 212.5 KB | `8c15bbff8537e12c4022b0bd3f4286a58319a0f9ed0b9de28d7405f315a735e0` |
| `corgi-clean-train-v1/default-s42/model.joblib` | 4397.4 KB | `bcab764cc87600a4c4b71ed0b4aa2f2037b62910d7364cd7e41af0148031ddf3` |
| `corgi-clean-train-v1/selected-s42/model.joblib` | 212.5 KB | `e634d38c67d33344b635ad0f8a2aa20d6fb313754725c9fc7ee5348a209e6f11` |
| `corgi-clean-train-v1/selected-s43/model.joblib` | 212.5 KB | `043f6862938a9c4116bd011b5f5a6aa4389ac5553a20a9e0a89ae77dcbb51946` |
| `corgi-clean-train-v1/selected-s44/model.joblib` | 212.5 KB | `693e9f2c602f86ee8bfed736a35f2a688e62e2b9a4e58f421f458b900184e675` |
| `corgi-official-v1/b0-majority-s42/model.joblib` | 0.3 KB | `b4fe1afd7bbb8d76d7a980c940001895e23035dfa08ff8c397d520c32c9eb480` |
| `corgi-official-v1/b2-balanced-s42/model.joblib` | 4399.1 KB | `d83e94ac6e63a75b6d36346247bc13336f836d7a12bb16379870afd97636aac3` |
| `corgi-official-v1/b3-balanced-char1gram-s42/model.joblib` | 213.4 KB | `ca9e621709840cd30160c0b6d50f30a01f2c7d0a232d6d8dbfdddc73b6152a38` |
| `corgi-official-v1/b3-char1gram-s42/model.joblib` | 213.4 KB | `a859451b6198d7957a662452d17a42847fb29372cb435ddd7936fc16d0fa7caa` |
| `corgi-official-v1/default-s42/model.joblib` | 4399.1 KB | `af4bd5967e826c04f5c09ae9a2b16d833874f125f1a20b138d37b0a7b952c7c7` |
| `corgi-official-v1/selected-s42/model.joblib` | 213.4 KB | `959f314055173870e7d9beb7914bf8b9105fbf3f79cd4e9d88189f5410a7f04b` |
| `corgi-official-v1/selected-s43/model.joblib` | 213.4 KB | `7fe110fd19d4b0b7e14878fb0b8f2523d1ba44bdd2b296b766fff1988acff273` |
| `corgi-official-v1/selected-s44/model.joblib` | 213.4 KB | `98c176c18c6465d10db886aa0d52dfaf36af5f7c622b1641203b3b03bcc3c9f9` |

## 说明

- `.gitignore` 仍忽略 `artifacts/`，这些文件是**刻意跟踪**的；新增权重需 `git add -f`。
- 只入库 P2 的轻量 sklearn 模型。P3 的 `encoder_finetune_cw`（约 400 MB）不入库，按 Release 附件或网盘交付。
- `data/processed/` 不入库；主表与复核证据见 `reports/results/`、`reports/evidence/P2/`。
