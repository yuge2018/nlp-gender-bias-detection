# -*- coding: utf-8 -*-
"""契约检查的失败路径：缺行、重复、乱序、NaN、阈值不符、mock 混用。"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.contracts import ContractError, read_json, write_json
from src.evaluation.checks import (
    ensure_comparable,
    validate_data_package,
    validate_run,
)
from src.evaluation.fixtures import build_fixtures


class ContractCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        info = build_fixtures(Path(cls._tmp.name) / "fixtures")
        cls.data_dir = info["data_dir"]
        cls.run_dir = info["run_dirs"][0]

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def setUp(self):
        self.package = validate_data_package(self.data_dir, "test")
        self.pred_path = self.run_dir / "predictions.jsonl"
        self.meta_path = self.run_dir / "run_metadata.json"
        self.original = self.pred_path.read_bytes()
        self.original_meta = self.meta_path.read_bytes()

    def tearDown(self):
        self.pred_path.write_bytes(self.original)
        self.meta_path.write_bytes(self.original_meta)

    def _rows(self):
        text = self.original.decode("utf-8")
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    def _write(self, rows):
        with self.pred_path.open("w", encoding="utf-8", newline="\n") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    def test_ok(self):
        checked = validate_run(self.run_dir, self.package)
        self.assertEqual(len(checked["preds"]), len(self.package["ids"]))

    def test_missing_row(self):
        self._write(self._rows()[:-1])
        with self.assertRaises(ContractError):
            validate_run(self.run_dir, self.package)

    def test_duplicate_id(self):
        rows = self._rows()
        rows[1]["id"] = rows[0]["id"]
        self._write(rows)
        with self.assertRaises(ContractError):
            validate_run(self.run_dir, self.package)

    def test_out_of_order(self):
        rows = self._rows()
        rows[0], rows[1] = rows[1], rows[0]
        self._write(rows)
        with self.assertRaises(ContractError):
            validate_run(self.run_dir, self.package)

    def test_nan_score(self):
        rows = self._rows()
        rows[3]["score"] = float("nan")
        self._write(rows)
        with self.assertRaises(ContractError):
            validate_run(self.run_dir, self.package)

    def test_score_out_of_range(self):
        rows = self._rows()
        rows[2]["score"] = 1.4
        self._write(rows)
        with self.assertRaises(ContractError):
            validate_run(self.run_dir, self.package)

    def test_bool_label(self):
        rows = self._rows()
        rows[0]["label"] = True
        self._write(rows)
        with self.assertRaises(ContractError):
            validate_run(self.run_dir, self.package)

    def test_threshold_mismatch(self):
        rows = self._rows()
        rows[0]["score"] = 0.2
        rows[0]["label"] = 1
        self._write(rows)
        with self.assertRaises(ContractError):
            validate_run(self.run_dir, self.package)

    def test_mock_flag_mismatch(self):
        meta = read_json(self.meta_path)
        meta["is_mock"] = False
        write_json(self.meta_path, meta)
        with self.assertRaises(ContractError):
            validate_run(self.run_dir, self.package)

    def test_data_hash_mismatch(self):
        meta = read_json(self.meta_path)
        meta["dataset_manifest_sha256"] = "0" * 64
        write_json(self.meta_path, meta)
        with self.assertRaises(ContractError):
            validate_run(self.run_dir, self.package)

    def test_manifest_tampered_file(self):
        target = self.data_dir / "test.jsonl"
        original = target.read_bytes()
        try:
            with target.open("ab") as fh:
                fh.write(b"\n")
            with self.assertRaises(ContractError):
                validate_data_package(self.data_dir, "test")
        finally:
            target.write_bytes(original)

    def test_ensure_comparable_rejects_mock_mix(self):
        base = {"dataset_manifest_sha256": "a" * 64, "split": "test", "is_mock": True}
        other = dict(base, is_mock=False)
        with self.assertRaises(ContractError):
            ensure_comparable([base, other])

    def test_ensure_comparable_rejects_split_mix(self):
        base = {"dataset_manifest_sha256": "a" * 64, "split": "test", "is_mock": True}
        other = dict(base, split="dev")
        with self.assertRaises(ContractError):
            ensure_comparable([base, other])


if __name__ == "__main__":
    unittest.main()
