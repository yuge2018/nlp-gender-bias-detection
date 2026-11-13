"""Translate the detection task without editing any source sentence."""
BIASED = "CORGI-PC_splitted_biased_corpus_v1.npy"
NON_BIASED = "CORGI-PC_splitted_non-bias_corpus_v1.npy"
SPLITS = (("train", "train"), ("valid", "dev"), ("test", "test"))


def _plain_scalar(value):
    if hasattr(value, "item"):
        value = value.item()
    if value is not None and not isinstance(value, (str, int, float, bool)):
        raise ValueError("Unsupported original unique_id value")
    return value


def convert_corpora(corpora, *, expected_splits=None):
    biased, neutral = corpora[BIASED], corpora[NON_BIASED]
    records, mapping = {}, []
    for original_split, split in SPLITS:
        positive, negative = biased[original_split], neutral[original_split]
        if not {"ori_sentence", "edit_sentence", "bias_labels"}.issubset(positive) or "text" not in negative:
            raise ValueError(f"{original_split}: missing required source fields")
        if len(positive["ori_sentence"]) != len(positive["edit_sentence"]) or len(positive["ori_sentence"]) != len(positive["bias_labels"]):
            raise ValueError(f"{original_split}: biased field lengths differ")
        if "unique_id" in negative and len(negative["unique_id"]) != len(negative["text"]):
            raise ValueError(f"{original_split}: unique_id and text lengths differ")
        if expected_splits is not None:
            expected = expected_splits[original_split]
            if len(positive["ori_sentence"]) != expected["biased"] or len(negative["text"]) != expected["non_biased"]:
                raise ValueError(f"{original_split}: counts differ from locked source audit")
        split_rows = []
        for label, filename, field, source in ((1, BIASED, "ori_sentence", positive), (0, NON_BIASED, "text", negative)):
            for source_index, text in enumerate(source[field]):
                if not isinstance(text, str) or not text.strip():
                    raise ValueError(f"{filename}/{original_split}/{field}/{source_index}: empty/non-string text")
                ident = f"{split}-{len(split_rows) + 1:06d}"
                split_rows.append({"id": ident, "text": str(text), "label": label})
                mapping.append({"id": ident, "split": split, "source_split": original_split, "source_filename": filename, "source_field": field, "source_row_index": source_index, "original_unique_id": _plain_scalar(source["unique_id"][source_index]) if "unique_id" in source else None})
        if not split_rows:
            raise ValueError(f"{split}: empty dataset")
        records[split] = split_rows
    return records, mapping


def clean_training(records):
    """Remove train rows matching held-out text, without looking at its labels."""
    heldout = {}
    for split in ("dev", "test"):
        for row in records[split]:
            heldout.setdefault(row["text"], []).append({"split": split, "id": row["id"]})
    retained, removed = [], []
    for row in records["train"]:
        if row["text"] in heldout:
            removed.append({"id": row["id"], "reason": "exact_text_in_heldout", "matches": heldout[row["text"]]})
        else:
            retained.append(row)
    if not retained:
        raise ValueError("Clean-training policy removed every training row")
    # Retain original IDs for traceability: gaps are intentional, never renumber.
    cleaned = {"train": retained, "dev": records["dev"], "test": records["test"]}
    return cleaned, removed
