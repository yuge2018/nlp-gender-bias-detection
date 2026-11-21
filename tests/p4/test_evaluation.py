import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from src.common.contracts import binary_metrics, load_dataset, sha256, write_json
from src.common.model_api import MockPredictor
from src.evaluation.aggregate import aggregate
from src.evaluation.runner import compare, evaluate_run, load_metrics

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "fixtures" / "dataset"


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        base = Path(self._tmp.name)
        self.data = base / "dataset"
        shutil.copytree(DATA, self.data)
        _, rows = load_dataset(self.data)
        self.inputs = rows["test.jsonl"]
        self.gold = rows["labels/test.jsonl"]
        self.scores = MockPredictor().predict_proba([row["text"] for row in self.inputs])
        self.run_dir = base / "mock-s42-test"
        self.run_dir.mkdir()
        self.write_run()

    def predictions(self):
        return [{"id": row["id"], "label": int(score >= 0.5), "score": score}
                for row, score in zip(self.inputs, self.scores)]

    def metadata(self, **overrides):
        meta = {
            "schema_version": "1.0",
            "method": "mock",
            "split": "test",
            "dataset_manifest_sha256": sha256(self.data / "manifest.json"),
            "is_mock": True,
            "seed": 42,
            "threshold": 0.5,
            "code_commit": "MOCK",
            "model_name": "mock",
            "model_revision": "mock-v1",
            "config": {},
            "environment": {},
        }
        meta.update(overrides)
        return meta

    def write_run(self, predictions=None, metadata=None, run_dir=None):
        target = run_dir or self.run_dir
        rows = self.predictions() if predictions is None else predictions
        # 故意绕过 contracts.write_jsonl 的 allow_nan=False，模拟外部产物中的非法分数
        with (target / "predictions.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        write_json(target / "run_metadata.json",
                   self.metadata() if metadata is None else metadata)

    def test_evaluate_matches_reference_metrics(self):
        record = evaluate_run(self.data, "test", self.run_dir)
        self.assertEqual(load_metrics(self.run_dir), record)
        expected = binary_metrics(self.gold, self.predictions())
        for key in ("n", "accuracy", "macro_f1", "positive_precision",
                    "positive_recall", "positive_f1", "negative_f1", "confusion_matrix"):
            self.assertEqual(record[key], expected[key])
        self.assertEqual(record["dataset_manifest_sha256"], sha256(self.data / "manifest.json"))
        self.assertEqual(record["split"], "test")
        self.assertTrue(record["is_mock"])

    def test_threshold_contradiction_rejected(self):
        bad = self.predictions()
        bad[0]["label"] = 0  # score=0.5, threshold=0.5 时应为 1
        self.write_run(predictions=bad)
        with self.assertRaises(ValueError):
            evaluate_run(self.data, "test", self.run_dir)

    def test_missing_duplicate_and_shuffled_rows_rejected(self):
        duplicate = self.predictions()
        duplicate[1]["id"] = duplicate[0]["id"]
        cases = (self.predictions()[:-1], list(reversed(self.predictions())), duplicate)
        for case in cases:
            self.write_run(predictions=case)
            with self.assertRaises(ValueError):
                evaluate_run(self.data, "test", self.run_dir)

    def test_invalid_score_rejected(self):
        for score in (float("nan"), float("inf"), -0.1, 1.1, True):
            bad = self.predictions()
            bad[0]["score"] = score
            self.write_run(predictions=bad)
            with self.assertRaises(ValueError):
                evaluate_run(self.data, "test", self.run_dir)

    def test_mock_flag_and_manifest_hash_mismatch_rejected(self):
        self.write_run(metadata=self.metadata(is_mock=False))
        with self.assertRaises(ValueError):
            evaluate_run(self.data, "test", self.run_dir)
        self.write_run(metadata=self.metadata(dataset_manifest_sha256="0" * 64))
        with self.assertRaises(ValueError):
            evaluate_run(self.data, "test", self.run_dir)

    def test_compare_requires_metrics(self):
        with self.assertRaises(ValueError):
            compare([str(self.run_dir)], Path(self._tmp.name) / "out")

    def test_compare_rejects_mock_without_flag(self):
        evaluate_run(self.data, "test", self.run_dir)
        with self.assertRaises(ValueError):
            compare([str(self.run_dir)], Path(self._tmp.name) / "out")

    def test_compare_writes_outputs_and_reproduction(self):
        evaluate_run(self.data, "test", self.run_dir)
        out = Path(self._tmp.name) / "results"
        rows = compare([str(self.run_dir)], out, allow_mock=True)
        self.assertEqual(len(rows), 1)
        for name in ("comparison.csv", "aggregate.csv", "comparison.md",
                     "environment.json", "reproduction.json"):
            self.assertTrue((out / name).is_file(), name)
        reproduction = json.loads((out / "reproduction.json").read_text(encoding="utf-8"))
        self.assertEqual(reproduction["runs"][0]["predictions_sha256"],
                         rows[0]["predictions_sha256"])
        aggregate_rows = (out / "aggregate.csv").read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(aggregate_rows), 2)  # 表头 + 一组
        self.assertIn("null", aggregate_rows[1])  # 单次 run 的标准差为 null

    def test_compare_rejects_mixed_mock_and_versions(self):
        out = Path(self._tmp.name) / "results"
        self._fake_metrics(Path(self._tmp.name) / "a", is_mock=True)
        self._fake_metrics(Path(self._tmp.name) / "b", is_mock=False)
        with self.assertRaises(ValueError):
            compare([str(Path(self._tmp.name) / "a"), str(Path(self._tmp.name) / "b")],
                    out, allow_mock=True)

        self._fake_metrics(Path(self._tmp.name) / "c", is_mock=True, data_hash="1" * 64)
        with self.assertRaises(ValueError):
            compare([str(Path(self._tmp.name) / "a"), str(Path(self._tmp.name) / "c")],
                    out, allow_mock=True)

    def _fake_metrics(self, run_dir, *, is_mock, data_hash=None, seed=42):
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json(run_dir / "metrics.json", {
            "schema_version": "1.0", "is_mock": is_mock, "method": "fake",
            "split": "test", "seed": seed, "threshold": 0.5, "n": 1,
            "accuracy": 1.0, "macro_f1": 1.0, "positive_precision": 1.0,
            "positive_recall": 1.0, "positive_f1": 1.0, "negative_f1": 1.0,
            "confusion_matrix": [[1, 0], [0, 1]],
            "dataset_manifest_sha256": data_hash or "0" * 64,
            "predictions_sha256": "0" * 64, "run_dir": str(run_dir),
        })

    def test_aggregate_mean_and_sample_std(self):
        base = {"split": "test", "is_mock": True, "dataset_manifest_sha256": "0" * 64,
                "threshold": 0.5,
                "accuracy": 0.8, "macro_f1": 0.7, "positive_precision": 0.6,
                "positive_recall": 0.5, "positive_f1": 0.4, "negative_f1": 0.3}
        rows = aggregate([dict(base, method="m", seed=42),
                          dict(base, method="m", seed=43, macro_f1=0.9)])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["n_runs"], 2)
        self.assertEqual(rows[0]["seeds"], "42 43")
        self.assertAlmostEqual(rows[0]["macro_f1_mean"], 0.8)
        self.assertAlmostEqual(rows[0]["macro_f1_std"], 2 ** 0.5 / 10)

    def test_aggregate_single_run_has_null_std(self):
        record = {"method": "m", "split": "test", "is_mock": True, "seed": 42,
                  "dataset_manifest_sha256": "0" * 64, "threshold": 0.5,
                  "accuracy": 0.8, "macro_f1": 0.7, "positive_precision": 0.6,
                  "positive_recall": 0.5, "positive_f1": 0.4, "negative_f1": 0.3}
        rows = aggregate([record])
        self.assertIsNone(rows[0]["macro_f1_std"])

    def test_aggregate_splits_by_threshold(self):
        base = {"method": "m", "split": "test", "is_mock": True, "seed": 42,
                "dataset_manifest_sha256": "0" * 64, "accuracy": 0.8, "macro_f1": 0.7,
                "positive_precision": 0.6, "positive_recall": 0.5,
                "positive_f1": 0.4, "negative_f1": 0.3}
        rows = aggregate([dict(base, threshold=0.5, macro_f1=0.80),
                          dict(base, threshold=0.4, macro_f1=0.79)])
        self.assertEqual(len(rows), 2)
        self.assertEqual({row["threshold"] for row in rows}, {0.5, 0.4})
        for row in rows:
            self.assertEqual(row["n_runs"], 1)
            self.assertEqual(row["seeds"], "42")

    def test_aggregate_identical_metrics_three_seeds_has_zero_std(self):
        base = {"method": "m", "split": "test", "is_mock": True,
                "dataset_manifest_sha256": "0" * 64, "threshold": 0.5,
                "accuracy": 0.8, "macro_f1": 0.7, "positive_precision": 0.6,
                "positive_recall": 0.5, "positive_f1": 0.4, "negative_f1": 0.3}
        rows = aggregate([dict(base, seed=seed) for seed in (42, 43, 44)])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["n_runs"], 3)
        self.assertEqual(rows[0]["seeds"], "42 43 44")
        self.assertEqual(rows[0]["macro_f1_std"], 0.0)
        self.assertIsNotNone(rows[0]["macro_f1_std"])


if __name__ == "__main__":
    unittest.main()
