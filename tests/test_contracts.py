import copy
import unittest
from pathlib import Path

from src.common.contracts import binary_metrics, check_records, load_dataset, sha256, validate_predictions
from src.common.model_api import MockPredictor

DATA = Path(__file__).resolve().parents[1] / "fixtures" / "dataset"


class ContractTests(unittest.TestCase):
    def setUp(self):
        _, self.rows = load_dataset(DATA)
        self.inputs = self.rows["test.jsonl"]
        self.predictions = [{"id": x["id"], "label": 1, "score": 0.5} for x in self.inputs]
        self.metadata = {"schema_version": "1.0", "method": "mock", "split": "test", "dataset_manifest_sha256": sha256(DATA / "manifest.json"), "is_mock": True, "seed": 42, "threshold": 0.5, "code_commit": "MOCK", "model_name": "mock", "model_revision": "mock-v1", "config": {}, "environment": {}}

    def validate(self, predictions=None, metadata=None, inputs=None, **kw):
        validate_predictions(self.inputs if inputs is None else inputs, self.predictions if predictions is None else predictions, self.metadata if metadata is None else metadata, DATA / "manifest.json", **kw)

    def test_fixture_and_mock_contract(self):
        self.validate()
        self.assertEqual(MockPredictor().predict_proba([]), [])
        with self.assertRaises(ValueError):
            MockPredictor().predict_proba([""])

    def test_hashed_fixture_files_use_portable_line_endings(self):
        for path in DATA.rglob("*.json*"):
            self.assertNotIn(b"\r\n", path.read_bytes(), str(path))

    def test_inference_rejects_labels_and_metadata(self):
        for key in ("label", "source", "bias_labels"):
            bad = copy.deepcopy(self.inputs)
            bad[0][key] = 1
            with self.assertRaises(ValueError):
                self.validate(inputs=bad)

    def test_missing_duplicate_and_shuffled_ids(self):
        duplicate = copy.deepcopy(self.predictions)
        duplicate[1]["id"] = duplicate[0]["id"]
        for bad in (self.predictions[:-1], list(reversed(self.predictions)), duplicate):
            with self.assertRaises(ValueError):
                self.validate(predictions=bad)

    def test_invalid_score_and_label(self):
        for score in (float("nan"), float("inf"), -0.1, 1.1, True, "0.5"):
            bad = copy.deepcopy(self.predictions)
            bad[0]["score"] = score
            with self.assertRaises(ValueError):
                self.validate(predictions=bad)
        with self.assertRaises(ValueError):
            check_records([{"id": "x", "label": True}], "gold")

    def test_threshold_consistency(self):
        bad = copy.deepcopy(self.predictions)
        bad[0]["label"] = 0
        with self.assertRaises(ValueError):
            self.validate(predictions=bad)

    def test_manifest_and_split_mismatch(self):
        bad = dict(self.metadata, dataset_manifest_sha256="wrong")
        with self.assertRaises(ValueError):
            self.validate(metadata=bad)
        with self.assertRaises(ValueError):
            self.validate(metadata=dict(self.metadata, split="dev"))

    def test_real_comparison_rejects_mock(self):
        with self.assertRaises(ValueError):
            self.validate(require_real=True)
        with self.assertRaises(ValueError):
            self.validate(metadata=dict(self.metadata, is_mock=False))

    def test_metric_definition_and_alignment(self):
        gold = [{"id": str(i), "label": y} for i, y in enumerate([0, 0, 1, 1, 1])]
        predictions = [{"id": str(i), "label": y, "score": float(y)} for i, y in enumerate([0, 1, 0, 1, 1])]
        metric = binary_metrics(gold, predictions)
        self.assertEqual(metric["confusion_matrix"], [[1, 1], [1, 2]])
        self.assertAlmostEqual(metric["macro_f1"], (0.5 + 2 / 3) / 2)
        self.assertAlmostEqual(metric["accuracy"], 0.6)
        with self.assertRaises(ValueError):
            binary_metrics(gold, predictions[:-1])

    def test_macro_f1_fixed_two_classes(self):
        metric = binary_metrics([{"id": "a", "label": 0}], [{"id": "a", "label": 0, "score": 0.0}])
        self.assertEqual(metric["macro_f1"], 0.5)
        self.assertEqual(metric["positive_f1"], 0.0)


if __name__ == "__main__":
    unittest.main()
