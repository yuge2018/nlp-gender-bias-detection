"""Dataset access for the improved encoder.

`load_training_data` only needs the standard library, so it stays importable in CI;
`make_dataset`/`collate` import torch lazily because they need it.
"""
from src.common.contracts import load_dataset


def load_training_data(data_dir):
    """Load a frozen data package and return (manifest, splits).

    train 只提供文本与标签；dev 的金标只用于选 checkpoint 和阈值，不进训练。
    test 从不在此处读取——推理走 predict CLI。
    """
    manifest, dataset = load_dataset(data_dir)
    splits = {
        "train_texts": [row["text"] for row in dataset["train.jsonl"]],
        "train_labels": [row["label"] for row in dataset["train.jsonl"]],
        "dev_texts": [row["text"] for row in dataset["dev.jsonl"]],
        "dev_labels": [row["label"] for row in dataset["labels/dev.jsonl"]],
    }
    return manifest, splits


def make_dataset(texts, labels, tokenizer, max_length):
    """Pre-tokenize texts (truncation right at max_length); labels may be None at inference."""
    import torch

    encoded = tokenizer(list(texts), truncation=True, max_length=max_length)
    has_labels = labels is not None
    labels = list(labels) if has_labels else None

    class _Encoded(torch.utils.data.Dataset):
        def __len__(self):
            return len(encoded["input_ids"])

        def __getitem__(self, index):
            item = {key: value[index] for key, value in encoded.items()}
            if has_labels:
                item["labels"] = labels[index]
            return item

    return _Encoded()


def collate(tokenizer):
    """Pad each batch to its own longest sequence; labels become a long tensor."""

    def _collate(batch):
        import torch

        has_labels = "labels" in batch[0]
        labels = [item.pop("labels") for item in batch] if has_labels else None
        features = tokenizer.pad(batch, padding=True, return_tensors="pt")
        if has_labels:
            features["labels"] = torch.tensor(labels, dtype=torch.long)
        return features

    return _collate
