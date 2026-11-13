import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from src.common.contracts import read_jsonl, write_json
from src.data.audit import audit_records, normalize_for_audit
from src.data.convert import BIASED, NON_BIASED, clean_training, convert_corpora
from src.data.pipeline import CLEAN_ID, OFFICIAL_ID, file_hashes, verify_package, write_package
from src.data.source import SourceFile, fetch_source, read_lock, verify_source

SETTINGS = {"shingle_chars": 3, "jaccard_threshold": 0.85, "minimum_chars": 30, "bands": 8, "rows_per_band": 4, "seed": 42}


def sample_corpora():
    positive = {split: {"ori_sentence": ["男人必须赚钱。"], "edit_sentence": ["每个人都有选择。"], "bias_labels": [[1, 0, 1]]} for split in ("train", "valid", "test")}
    negative = {split: {"text": ["她是一名工程师。"], "unique_id": [100 + index]} for index, split in enumerate(("train", "valid", "test"))}
    return {BIASED: positive, NON_BIASED: negative}


def sample_records():
    return {
        "train": [{"id": "train-000001", "text": "重复原句", "label": 1}, {"id": "train-000002", "text": "只在训练集出现的中性句子", "label": 0}, {"id": "train-000003", "text": "ＦＯＯ BAR", "label": 0}],
        "dev": [{"id": "dev-000001", "text": "重复原句", "label": 0}, {"id": "dev-000002", "text": "dev unique", "label": 1}],
        "test": [{"id": "test-000001", "text": "foo\tbar", "label": 1}, {"id": "test-000002", "text": "test unique", "label": 0}],
    }


def sample_mapping(records):
    return [{"id": row["id"], "split": split, "source_filename": "synthetic-only.npy", "source_row_index": index} for split, rows in records.items() for index, row in enumerate(rows)]


class SourceTests(unittest.TestCase):
    def spec(self, data, url):
        return SourceFile("source.npy", len(data), hashlib.sha256(data).hexdigest(), hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest(), url)

    def test_locked_official_sources(self):
        lock, files = read_lock(Path(__file__).resolve().parents[2] / "configs/data/source_audit.json")
        self.assertEqual(len(lock["source_commit"]), 40)
        self.assertEqual(len(files), 2)
        self.assertTrue(all(lock["source_commit"] in item.url for item in files))

    def test_download_and_verified_offline_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            upstream = base / "upstream"
            upstream.write_bytes(b"unit-test source bytes")
            spec = self.spec(upstream.read_bytes(), upstream.as_uri())
            target, result = fetch_source(base / "cache", spec)
            self.assertEqual(target.read_bytes(), upstream.read_bytes())
            self.assertEqual(result["sha256"], spec.sha256)
            self.assertEqual(fetch_source(base / "cache", spec, offline=True)[0], target)
            target.write_bytes(b"tampered")
            with self.assertRaises(ValueError):
                fetch_source(base / "cache", spec)
            self.assertEqual(target.read_bytes(), b"tampered")

    def test_bad_download_never_publishes_partial_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            upstream = base / "upstream"
            upstream.write_bytes(b"bad")
            spec = self.spec(b"expected", upstream.as_uri())
            with self.assertRaises(ValueError):
                fetch_source(base / "cache", spec)
            self.assertEqual(list((base / "cache").iterdir()), [])

    def test_missing_offline_source_and_wrong_size(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = self.spec(b"expected", "https://example.invalid/not-requested")
            with self.assertRaises(FileNotFoundError):
                fetch_source(directory, spec, offline=True)
            path = Path(directory) / "source.npy"
            path.write_bytes(b"unexpected")
            with self.assertRaises(ValueError):
                verify_source(path, spec)


class ConversionTests(unittest.TestCase):
    def test_detection_labels_original_text_and_metadata_isolation(self):
        corpora = sample_corpora()
        corpora[BIASED]["train"]["ori_sentence"][0] = " 原文\n保留标点！ "
        rows, mapping = convert_corpora(corpora)
        self.assertEqual(rows["train"][0], {"id": "train-000001", "text": " 原文\n保留标点！ ", "label": 1})
        self.assertEqual(rows["train"][1]["label"], 0)
        self.assertEqual(rows["dev"][0]["id"], "dev-000001")
        self.assertTrue(all(set(row) == {"id", "text", "label"} for split in rows.values() for row in split))
        self.assertNotIn("每个人都有选择。", [row["text"] for split in rows.values() for row in split])
        self.assertEqual(mapping[1]["original_unique_id"], 100)

    def test_bad_source_lengths_empty_text_and_counts_fail(self):
        wrong_length = sample_corpora()
        wrong_length[BIASED]["train"]["edit_sentence"] = []
        empty = sample_corpora()
        empty[NON_BIASED]["test"]["text"] = [" "]
        for invalid in (wrong_length, empty):
            with self.assertRaises(ValueError):
                convert_corpora(invalid)
        with self.assertRaises(ValueError):
            convert_corpora(sample_corpora(), expected_splits={"train": {"biased": 2, "non_biased": 1}})

    def test_clean_only_training_text_without_reading_heldout_labels(self):
        records = sample_records()
        heldout_without_labels = copy.deepcopy(records)
        for split in ("dev", "test"):
            for row in heldout_without_labels[split]:
                del row["label"]
        cleaned, removed = clean_training(heldout_without_labels)
        self.assertEqual([row["id"] for row in cleaned["train"]], ["train-000002", "train-000003"])
        self.assertEqual(removed[0]["id"], "train-000001")
        self.assertEqual(cleaned["dev"], heldout_without_labels["dev"])
        self.assertEqual(cleaned["test"], heldout_without_labels["test"])
        self.assertEqual(records["train"][0]["id"], "train-000001")


class AuditTests(unittest.TestCase):
    def test_exact_normalized_label_conflicts_and_character_lengths(self):
        records = sample_records()
        report, groups, _ = audit_records(records, SETTINGS)
        self.assertEqual(report["duplicates"]["exact"]["cross_split"]["train__dev"]["unique_texts"], 1)
        self.assertEqual(report["duplicates"]["exact"]["conflicting_text_groups"], 1)
        self.assertEqual(report["duplicates"]["normalized"]["cross_split"]["train__test"]["unique_texts"], 1)
        self.assertEqual(normalize_for_audit("ＦＯＯ \tBAR"), "foobar")
        self.assertEqual(records["train"][2]["text"], "ＦＯＯ BAR")
        self.assertTrue(any(group["label_conflict"] for group in groups))
        self.assertEqual(report["split_statistics"]["train"]["positive_n"], 1)

    def test_near_duplicates_use_real_jaccard_and_are_repeatable(self):
        text = "".join(chr(0x4E00 + index) for index in range(100))
        records = sample_records()
        records["train"][0]["text"] = text
        records["dev"][0]["text"] = text[:-1] + "终"
        first = audit_records(records, SETTINGS)
        second = audit_records(records, SETTINGS)
        self.assertEqual(first, second)
        self.assertGreaterEqual(first[0]["near_duplicates"]["matched_cross_split_text_pairs"], 1)
        self.assertTrue(all(pair["jaccard"] >= 0.85 for pair in first[2]))


class PackageTests(unittest.TestCase):
    def build(self, base, dataset_id=OFFICIAL_ID):
        records = sample_records()
        mapping = sample_mapping(records)
        removed = []
        if dataset_id == CLEAN_ID:
            records, removed = clean_training(records)
        write_package(base, dataset_id, records, mapping, removed, "0" * 40, [], SETTINGS)

    def test_byte_reproducible_packages_and_unlabelled_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / "one", Path(directory) / "two"
            self.build(first)
            self.build(second)
            self.assertEqual(file_hashes(first), file_hashes(second))
            self.assertEqual(verify_package(first)["rows"], {"train": 3, "dev": 2, "test": 2})
            for split in ("dev", "test"):
                self.assertTrue(all(set(row) == {"id", "text"} for row in read_jsonl(first / f"{split}.jsonl")))
            self.assertTrue(all(b"\r\n" not in path.read_bytes() for path in first.rglob("*.json*")))

    def test_clean_keeps_heldout_byte_identical_and_original_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            official, clean = Path(directory) / "official", Path(directory) / "clean"
            self.build(official)
            self.build(clean, CLEAN_ID)
            for name in ("dev.jsonl", "test.jsonl", "labels/dev.jsonl", "labels/test.jsonl"):
                self.assertEqual((official / name).read_bytes(), (clean / name).read_bytes())
            self.assertEqual(read_jsonl(clean / "train.jsonl")[0]["id"], "train-000002")

    def test_verifier_rejects_tampered_audit_and_extra_files(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "data"
            self.build(package)
            extra = package / "unexpected.txt"
            extra.write_text("untracked contents")
            with self.assertRaises(ValueError):
                verify_package(package)
            extra.unlink()
            (package / "audit/quality_report.json").write_text("{}\n")
            with self.assertRaises(ValueError):
                verify_package(package)

    def test_verifier_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "data"
            self.build(package)
            manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
            manifest["audit_files"]["../outside.json"] = {"sha256": "0" * 64}
            write_json(package / "manifest.json", manifest)
            with self.assertRaises(ValueError):
                verify_package(package)


if __name__ == "__main__":
    unittest.main()
