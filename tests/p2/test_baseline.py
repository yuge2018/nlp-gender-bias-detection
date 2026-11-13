"""P2 tests: feature isolation, contract enforcement and the CLI surface.

Tests that need scikit-learn are skipped when it is not installed, so the
dependency-free contract/mock checks still run in a bare CI interpreter.
"""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from src.common.contracts import VERSION, read_jsonl, sha256, validate_predictions, write_json, write_jsonl
from src.baseline import load_predictor
from src.baseline.cli import main, predict_command, sweep_command, train_command
from src.baseline.config import load_config, load_sweep_config, validate_config
from src.baseline.data import load_inference_input, load_training_data
from src.baseline.model import load_model
from src.baseline.selection import THRESHOLD_GRID, metrics_for, select_threshold

ROOT = Path(__file__).resolve().parents[2]
SKLEARN = importlib.util.find_spec("sklearn") is not None
needs_sklearn = unittest.skipUnless(SKLEARN, "scikit-learn is needed for real baseline tests")

TRAIN_ROWS = [
    {"id": "train-000001", "text": "女人就应该在家做饭带孩子。", "label": 1},
    {"id": "train-000002", "text": "男人不能哭，要坚强。", "label": 1},
    {"id": "train-000003", "text": "女司机开车就是不行。", "label": 1},
    {"id": "train-000004", "text": "女生学不好理科很正常。", "label": 1},
    {"id": "train-000005", "text": "她是一名工程师，工作认真。", "label": 0},
    {"id": "train-000006", "text": "他喜欢做饭和照顾孩子。", "label": 0},
    {"id": "train-000007", "text": "每个人都有选择职业的自由。", "label": 0},
    {"id": "train-000008", "text": "性别不应该决定能力。", "label": 0},
    {"id": "train-000009", "text": "这位男护士很受病人欢迎。", "label": 0},
    {"id": "train-000010", "text": "这位女总统推动了改革。", "label": 0},
    {"id": "train-000011", "text": "男主外女主内的旧观念该改变了。", "label": 1},
    {"id": "train-000012", "text": "工程师和护士都可以是任何性别。", "label": 0},
]
DEV_ROWS = [
    {"id": "dev-000001", "text": "女人不适合做领导。", "label": 1},
    {"id": "dev-000002", "text": "她是一名医生。", "label": 0},
    {"id": "dev-000003", "text": "男生应该勇敢。", "label": 1},
    {"id": "dev-000004", "text": "能力与性别无关。", "label": 0},
    {"id": "dev-000005", "text": "女人要温柔听话。", "label": 1},
    {"id": "dev-000006", "text": "他选择当幼师。", "label": 0},
]
TEST_ROWS = [
    {"id": "test-000001", "text": "男人必须事业有成。", "label": 1},
    {"id": "test-000002", "text": "性别不决定职业选择。", "label": 0},
    {"id": "test-000003", "text": "女性开车技术差。", "label": 1},
    {"id": "test-000004", "text": "她是优秀的程序员。", "label": 0},
    {"id": "test-000005", "text": "男孩子不能太软弱。", "label": 1},
    {"id": "test-000006", "text": "每个人都可以做家务。", "label": 0},
]


def build_package(base, *, train=None, dev=None, test=None, is_mock=True):
    train = TRAIN_ROWS if train is None else train
    dev = DEV_ROWS if dev is None else dev
    test = TEST_ROWS if test is None else test
    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)
    write_jsonl(base / "train.jsonl", train)
    write_jsonl(base / "dev.jsonl", [{"id": row["id"], "text": row["text"]} for row in dev])
    write_jsonl(base / "test.jsonl", [{"id": row["id"], "text": row["text"]} for row in test])
    write_jsonl(base / "labels" / "dev.jsonl", [{"id": row["id"], "label": row["label"]} for row in dev])
    write_jsonl(base / "labels" / "test.jsonl", [{"id": row["id"], "label": row["label"]} for row in test])
    files = {}
    for name in ("train.jsonl", "dev.jsonl", "test.jsonl", "labels/dev.jsonl", "labels/test.jsonl"):
        rows = read_jsonl(base / name)
        files[name] = {"sha256": sha256(base / name), "rows": len(rows)}
    manifest = {"schema_version": VERSION, "dataset_id": "p2-unit-test-v1", "is_mock": is_mock, "source_commit": "UNIT-TEST", "label_map": {"0": "non_biased", "1": "biased"}, "files": files}
    write_json(base / "manifest.json", manifest)
    return base


def write_config(base, name, config):
    path = Path(base) / name
    write_json(path, config)
    return path


DEFAULT_CONFIG = {"method": "char_tfidf_lr", "analyzer": "char", "ngram_range": [1, 3], "max_features": 1000, "C": 1.0, "class_weight": None, "max_iter": 1000, "threshold": 0.5, "seed": 42}


class ConfigTests(unittest.TestCase):
    def test_valid_configs_and_defaults(self):
        config = validate_config(dict(DEFAULT_CONFIG))
        self.assertEqual(config["solver"], "liblinear")
        self.assertEqual(config["batch_size"], 256)
        self.assertNotIn("config_path", config)
        majority = validate_config({"method": "majority_class", "threshold": 0.5, "seed": 42})
        self.assertEqual(majority["method"], "majority_class")
        repo_config = load_config(ROOT / "configs/baseline/default.json")
        self.assertEqual(repo_config["ngram_range"], [1, 3])

    def test_rejected_configs(self):
        broken = [
            dict(DEFAULT_CONFIG, unknown_key=True),
            dict(DEFAULT_CONFIG, analyzer="word"),
            dict(DEFAULT_CONFIG, ngram_range=[3, 1]),
            dict(DEFAULT_CONFIG, ngram_range=[1, 4]),
            dict(DEFAULT_CONFIG, C=-1),
            dict(DEFAULT_CONFIG, class_weight="auto"),
            dict(DEFAULT_CONFIG, solver="saga"),
            dict(DEFAULT_CONFIG, threshold=1.5),
            dict(DEFAULT_CONFIG, seed=True),
            dict(DEFAULT_CONFIG, max_features=0),
        ]
        for config in broken:
            with self.assertRaises(ValueError):
                validate_config(config)

    def test_sweep_config_expands_grid(self):
        base, grid, variants, _ = load_sweep_config(ROOT / "configs/baseline/sweep.json")
        self.assertEqual(len(variants), 3 * 2 * 2)
        self.assertEqual(base["method"], "char_tfidf_lr")
        self.assertIn([1, 1], grid["ngram_range"])
        self.assertTrue(all(variant["method"] == "char_tfidf_lr" for variant in variants))


class DataTests(unittest.TestCase):
    def test_training_data_ignores_test_files(self):
        with tempfile.TemporaryDirectory() as directory:
            package = build_package(Path(directory) / "data")
            (package / "test.jsonl").unlink()
            (package / "labels" / "test.jsonl").unlink()
            data = load_training_data(package)
            self.assertEqual(len(data["train"]), len(TRAIN_ROWS))
            self.assertEqual(len(data["dev"]), len(DEV_ROWS))

    def test_tampered_split_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            package = build_package(Path(directory) / "data")
            (package / "train.jsonl").write_text((package / "train.jsonl").read_text(encoding="utf-8") + " ", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_training_data(package)

    def test_inference_input_rejects_labels_and_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            labelled = base / "labelled.jsonl"
            write_jsonl(labelled, [{"id": "a", "text": "文本", "label": 1}])
            with self.assertRaises(ValueError):
                load_inference_input(labelled)
            duplicate = base / "duplicate.jsonl"
            write_jsonl(duplicate, [{"id": "a", "text": "文本"}, {"id": "a", "text": "另一条"}])
            with self.assertRaises(ValueError):
                load_inference_input(duplicate)
            empty_text = base / "empty.jsonl"
            write_jsonl(empty_text, [{"id": "a", "text": "  "}])
            with self.assertRaises(ValueError):
                load_inference_input(empty_text)


class ThresholdTests(unittest.TestCase):
    def test_frozen_grid(self):
        self.assertEqual(len(THRESHOLD_GRID), 17)
        self.assertEqual(THRESHOLD_GRID[0], 0.10)
        self.assertEqual(THRESHOLD_GRID[-1], 0.90)

    def test_tie_break_prefers_closest_to_half(self):
        labels = [0, 1]
        scores = [0.10, 0.90]
        selection = select_threshold(labels, scores, ["a", "b"])
        self.assertEqual(selection["threshold"], 0.5)
        self.assertEqual(selection["metrics"]["macro_f1"], 1.0)

    def test_metrics_match_manual_confusion_matrix(self):
        labels = [0, 0, 1, 1, 1]
        scores = [0.2, 0.8, 0.3, 0.7, 0.9]
        metric = metrics_for(labels, scores, 0.5, [str(index) for index in range(5)])
        self.assertEqual(metric["confusion_matrix"], [[1, 1], [1, 2]])
        self.assertAlmostEqual(metric["accuracy"], 0.6)


@needs_sklearn
class TrainingTests(unittest.TestCase):
    def test_train_writes_model_and_dev_evidence_only(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            package = build_package(base / "data")
            config = write_config(base, "config.json", DEFAULT_CONFIG)
            (package / "test.jsonl").unlink()
            (package / "labels" / "test.jsonl").unlink()
            summary = train_command(package, config, base / "model", seed=7, select_threshold=True)
            self.assertEqual(summary["seed"], 7)
            metadata = json.loads((base / "model" / "model_metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["feature"]["vectorizer_fit_scope"], "train_only")
            self.assertEqual(metadata["feature"]["vectorizer_fit_rows"], len(TRAIN_ROWS))
            self.assertEqual(metadata["feature"]["input"], "text")
            self.assertIn("threshold_selection", metadata)
            self.assertIsNotNone(metadata["threshold_selection"])
            self.assertIn(metadata["code_dirty"], (True, False, None))
            for name in ("model.joblib", "model_metadata.json", "config.json", "log.txt"):
                self.assertTrue((base / "model" / name).is_file(), name)
            with self.assertRaises(ValueError):
                train_command(package, config, base / "model", seed=7)

    def test_vectorizer_is_fitted_on_train_text_only(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            train = [
                {"id": "train-000001", "text": "天气很好我们出门散步", "label": 0},
                {"id": "train-000002", "text": "天气不好我们留在家里", "label": 1},
                {"id": "train-000003", "text": "出门散步有益健康", "label": 0},
                {"id": "train-000004", "text": "留在家里也很舒服", "label": 1},
            ]
            dev = [
                {"id": "dev-000001", "text": "龘靐齉", "label": 0},
                {"id": "dev-000002", "text": "天气很好", "label": 1},
                {"id": "dev-000003", "text": "出门散步", "label": 0},
                {"id": "dev-000004", "text": "留在家里", "label": 1},
            ]
            package = build_package(base / "data", train=train, dev=dev, test=dev)
            config = write_config(base, "config.json", dict(DEFAULT_CONFIG, max_features=100))
            train_command(package, config, base / "model", seed=42)
            model, _ = load_model(base / "model")
            vocabulary = "".join(model.vectorizer.vocabulary_.keys())
            self.assertNotIn("龘", vocabulary)
            self.assertNotIn("靐", vocabulary)
            self.assertNotIn("齉", vocabulary)

    def test_ids_do_not_change_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            config = write_config(base, "config.json", DEFAULT_CONFIG)
            first = build_package(base / "a")
            relabelled = [dict(row, id=f"{row['label']}-{index:04d}") for index, row in enumerate(TRAIN_ROWS)]
            second = build_package(base / "b", train=relabelled)
            train_command(first, config, base / "model-a", seed=42)
            train_command(second, config, base / "model-b", seed=42)
            model_a, _ = load_model(base / "model-a")
            model_b, _ = load_model(base / "model-b")
            texts = [row["text"] for row in TEST_ROWS]
            self.assertEqual(model_a.predict_proba(texts), model_b.predict_proba(texts))

    def test_prediction_contract_ids_scores_and_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            package = build_package(base / "data")
            config = write_config(base, "config.json", DEFAULT_CONFIG)
            train_command(package, config, base / "model", seed=42, select_threshold=True)
            summary = predict_command(package / "test.jsonl", base / "model", package / "manifest.json", base / "run-test")
            predictions = read_jsonl(base / "run-test" / "predictions.jsonl")
            self.assertEqual(summary["split"], "test")
            self.assertEqual([row["id"] for row in predictions], [row["id"] for row in read_jsonl(package / "test.jsonl")])
            threshold = summary["threshold"]
            self.assertTrue(all(row["label"] == int(row["score"] >= threshold) for row in predictions))
            self.assertTrue(all(0.0 <= row["score"] <= 1.0 for row in predictions))
            metadata = json.loads((base / "run-test" / "run_metadata.json").read_text(encoding="utf-8"))
            validate_predictions(read_jsonl(package / "test.jsonl"), predictions, metadata, package / "manifest.json")

    def test_inference_rejects_labels_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            package = build_package(base / "data")
            config = write_config(base, "config.json", DEFAULT_CONFIG)
            train_command(package, config, base / "model", seed=42)
            rows = read_jsonl(package / "test.jsonl")
            labelled = base / "labelled.jsonl"
            write_jsonl(labelled, [dict(rows[0], label=1)] + rows[1:])
            with self.assertRaises(ValueError):
                predict_command(labelled, base / "model", package / "manifest.json", base / "run")
            with contextlib.redirect_stderr(io.StringIO()):
                exit_code = main(["predict", "--input", str(labelled), "--model-dir", str(base / "model"), "--manifest", str(package / "manifest.json"), "--output-dir", str(base / "run2")])
            self.assertEqual(exit_code, 2)
            self.assertFalse((base / "run2").exists())

    def test_batch_and_single_prediction_are_identical(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            package = build_package(base / "data")
            config = write_config(base, "config.json", DEFAULT_CONFIG)
            train_command(package, config, base / "model", seed=42)
            model, metadata = load_model(base / "model")
            texts = [row["text"] for row in TEST_ROWS]
            full = model.predict_proba(texts)
            single = []
            for text in texts:
                single.extend(model.predict_proba([text]))
            self.assertEqual(full, single)
            adapter = load_predictor(str(base / "model"))
            self.assertEqual(adapter.predict_proba(texts), full)
            self.assertEqual(adapter.predict_proba([]), [])
            with self.assertRaises(ValueError):
                adapter.predict_proba([""])
            with self.assertRaises(FileNotFoundError):
                load_predictor(str(base / "missing"))

    def test_majority_baseline_is_constant(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            package = build_package(base / "data")
            config = write_config(base, "config.json", {"method": "majority_class", "threshold": 0.5, "seed": 42})
            train_command(package, config, base / "model", seed=42)
            model, metadata = load_model(base / "model")
            self.assertTrue(model.is_majority)
            self.assertEqual(metadata["training"]["majority_label"], 0)
            self.assertEqual(model.predict_proba([row["text"] for row in TEST_ROWS]), [0.0] * len(TEST_ROWS))
            predict_command(package / "test.jsonl", base / "model", package / "manifest.json", base / "run")
            predictions = read_jsonl(base / "run" / "predictions.jsonl")
            self.assertTrue(all(row["label"] == 0 for row in predictions))

    def test_sweep_selects_on_dev_only(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            package = build_package(base / "data")
            (package / "test.jsonl").unlink()
            (package / "labels" / "test.jsonl").unlink()
            sweep_config = write_config(base, "sweep.json", {"method": "char_tfidf_lr", "analyzer": "char", "ngram_range": [1, 3], "max_features": 1000, "max_iter": 1000, "threshold": 0.5, "seed": 42, "grid": {"C": [0.5, 1.0], "class_weight": [None], "ngram_range": [[1, 3]]}})
            summary = sweep_command(package, sweep_config, base / "sweep")
            result = json.loads((base / "sweep" / "sweep.json").read_text(encoding="utf-8"))
            self.assertEqual(len(result["variants"]), 2)
            self.assertEqual(summary["variants"], 2)
            self.assertIn("best", result)
            self.assertEqual(result["selection"]["split"], "dev")
            self.assertEqual(len(result["best"]["config"]["ngram_range"]), 2)

    def test_model_and_manifest_versions_must_match(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            package = build_package(base / "data")
            other = build_package(base / "other", train=[dict(row, text=row["text"] + "（不同版本）") for row in TRAIN_ROWS])
            config = write_config(base, "config.json", DEFAULT_CONFIG)
            train_command(package, config, base / "model", seed=42)
            with self.assertRaises(ValueError):
                predict_command(other / "test.jsonl", base / "model", other / "manifest.json", base / "run")

    def test_mock_manifest_marks_run_as_mock(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            package = build_package(base / "data", is_mock=True)
            config = write_config(base, "config.json", DEFAULT_CONFIG)
            train_command(package, config, base / "model", seed=42)
            predict_command(package / "dev.jsonl", base / "model", package / "manifest.json", base / "run")
            metadata = json.loads((base / "run" / "run_metadata.json").read_text(encoding="utf-8"))
            self.assertIs(metadata["is_mock"], True)
            with self.assertRaises(ValueError):
                validate_predictions(read_jsonl(package / "dev.jsonl"), read_jsonl(base / "run" / "predictions.jsonl"), metadata, package / "manifest.json", require_real=True)


class MockCompatibilityTests(unittest.TestCase):
    def test_legacy_mock_entry_still_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "p2-smoke"
            completed = subprocess.run([sys.executable, "-m", "src.baseline", "--mock", "--output-dir", str(output)], cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            status = json.loads((output / "status.json").read_text(encoding="utf-8"))
            self.assertIs(status["is_mock"], True)
            self.assertEqual(status["component"], "baseline")


if __name__ == "__main__":
    unittest.main()
