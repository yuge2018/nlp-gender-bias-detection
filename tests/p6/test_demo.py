"""P6 unit tests: registry, scoring and metrics-figure plumbing.

Scope rules:
* never import `src.demo.app` (it needs streamlit, which CI does not install);
* plotting tests skip when matplotlib is absent, so the suite stays green on a
  bare image while still exercising the loader on every platform.
"""
from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.demo import figures, registry, scoring
from src.common.contracts import binary_metrics

ROOT = Path(__file__).resolve().parents[2]
HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _base_config(**overrides) -> dict:
    payload = {
        "schema_version": "1.0",
        "default_model": "mock",
        "mock_banner_required": True,
        "model_loading": "lazy",
        "offline_real_mode": True,
        "models": [
            {
                "key": "mock",
                "display_name": "替身",
                "adapter_module": None,
                "model_dir": None,
                "is_deep": False,
            }
        ],
    }
    payload.update(overrides)
    return payload


def _metrics(n: int = 8, is_mock: bool = True) -> dict:
    gold = [{"id": f"test-{i:06d}", "label": i % 2} for i in range(1, n + 1)]
    predictions = [{"id": f"test-{i:06d}", "label": i % 2, "score": 0.9 if i % 2 else 0.1} for i in range(1, n + 1)]
    payload = {"schema_version": "1.0", "is_mock": is_mock, "method": "unit", "split": "test", "seed": 42}
    payload.update(binary_metrics(gold, predictions))
    return payload


class ConfigTests(unittest.TestCase):
    def test_default_config_is_valid(self):
        config = registry.load_config()
        keys = [entry["key"] for entry in registry.model_entries(config)]
        self.assertIn("mock", keys)
        self.assertIn(config["default_model"], keys)
        self.assertTrue(config["mock_banner_required"])

    def test_duplicate_key_rejected(self):
        duplicate = _base_config()
        duplicate["models"] = duplicate["models"] + [dict(duplicate["models"][0])]
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_json(Path(tmp) / "cfg.json", duplicate)
            with self.assertRaises(ValueError):
                registry.load_config(path)

    def test_adapter_without_model_dir_rejected(self):
        bad = _base_config()
        bad["models"].append({"key": "x", "display_name": "x", "adapter_module": "src.baseline.adapter"})
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_json(Path(tmp) / "cfg.json", bad)
            with self.assertRaises(ValueError):
                registry.load_config(path)

    def test_unknown_default_model_rejected(self):
        bad = _base_config(default_model="nope")
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_json(Path(tmp) / "cfg.json", bad)
            with self.assertRaises(ValueError):
                registry.load_config(path)

    def test_missing_config_file_is_model_unavailable(self):
        with self.assertRaises(registry.ModelUnavailable):
            registry.load_config("no/such/config.json")

    def test_examples_are_self_authored_and_labelled(self):
        payload = registry.load_examples()
        self.assertIn("不是官方金标", payload["note"])
        examples = payload["examples"]
        self.assertGreaterEqual(len(examples), 4)
        self.assertEqual({item["expected_label"] for item in examples}, {0, 1})


class RegistryTests(unittest.TestCase):
    def test_mock_entry_returns_constant_predictor(self):
        config = registry.load_config()
        predictor = registry.load_model(config, "mock")
        self.assertTrue(predictor.is_mock)
        self.assertEqual(predictor.predict_proba(["她是一名工程师。"]), [0.5])

    def test_mock_flag_overrides_any_key(self):
        config = registry.load_config()
        predictor = registry.load_model(config, "improved", mock=True)
        self.assertTrue(predictor.is_mock)

    def test_unknown_key_is_model_unavailable(self):
        config = registry.load_config()
        with self.assertRaises(registry.ModelUnavailable) as ctx:
            registry.load_model(config, "not-configured")
        self.assertEqual(ctx.exception.key, "not-configured")

    def test_missing_adapter_module_is_model_unavailable(self):
        config = _base_config()
        config["models"].append(
            {"key": "ghost", "display_name": "ghost", "adapter_module": "src.demo.not_a_module", "model_dir": "ghost-dir"}
        )
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "ghost-dir").mkdir()
            with self.assertRaises(registry.ModelUnavailable) as ctx:
                registry.load_model(config, "ghost", root=tmp)
        self.assertIn("not_a_module", str(ctx.exception))

    def test_missing_model_dir_is_model_unavailable(self):
        config = _base_config()
        config["models"].append(
            {"key": "baseline", "display_name": "baseline", "adapter_module": "src.baseline.adapter", "model_dir": "no-such-artifact"}
        )
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(registry.ModelUnavailable) as ctx:
                registry.load_model(config, "baseline", root=tmp)
        self.assertIn("模型目录不存在", str(ctx.exception))

    def test_describe_models_reports_availability_without_importing(self):
        config = _base_config()
        config["models"].append(
            {"key": "baseline", "display_name": "baseline", "adapter_module": "src.baseline.adapter", "model_dir": "missing"}
        )
        with tempfile.TemporaryDirectory() as tmp:
            rows = registry.describe_models(config, root=tmp)
        by_key = {row["key"]: row for row in rows}
        self.assertTrue(by_key["mock"]["available"])
        self.assertFalse(by_key["baseline"]["available"])
        self.assertIsNone(by_key["mock"]["adapter_module"])

    def test_logic_layer_imports_no_heavy_ui_or_model_stack(self):
        # A fresh interpreter keeps earlier tests (p2 imports sklearn) from polluting the check.
        code = (
            "import sys\n"
            "import src.demo.registry, src.demo.scoring, src.demo.figures\n"
            "print(','.join(m for m in ('streamlit','torch','transformers','sklearn') if m in sys.modules))\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "")


class ScoringTests(unittest.TestCase):
    def test_empty_list_returns_no_rows(self):
        self.assertEqual(scoring.predict_rows(registry.load_model(registry.load_config(), "mock"), [], 0.5), [])

    def test_empty_text_rejected(self):
        predictor = registry.load_model(registry.load_config(), "mock")
        for bad in ([""], ["   "], [None], ["ok", ""]):
            with self.assertRaises(ValueError):
                scoring.predict_rows(predictor, bad, 0.5)

    def test_non_list_rejected(self):
        predictor = registry.load_model(registry.load_config(), "mock")
        with self.assertRaises(ValueError):
            scoring.predict_rows(predictor, "她是一名工程师。", 0.5)

    def test_threshold_validation(self):
        self.assertEqual(scoring.validate_threshold(None), 0.5)
        self.assertEqual(scoring.validate_threshold(0), 0.0)
        self.assertEqual(scoring.validate_threshold(1), 1.0)
        for bad in (-0.1, 1.1, float("nan"), "0.5", True):
            with self.assertRaises(ValueError):
                scoring.validate_threshold(bad)

    def test_label_uses_inclusive_threshold(self):
        class Fixed:
            is_mock = True

            def predict_proba(self, texts):
                return [0.5 for _ in texts]

        rows = scoring.predict_rows(Fixed(), ["a"], 0.5)
        self.assertEqual(rows[0]["label"], 1)
        rows = scoring.predict_rows(Fixed(), ["a"], 0.55)
        self.assertEqual(rows[0]["label"], 0)

    def test_ids_are_sequential_and_order_preserved(self):
        rows = scoring.predict_rows(registry.load_model(registry.load_config(), "mock"), ["一", "二", "三"], 0.5)
        self.assertEqual([row["id"] for row in rows], ["input-000001", "input-000002", "input-000003"])
        self.assertEqual([row["text"] for row in rows], ["一", "二", "三"])

    def test_text_is_not_silently_rewritten(self):
        rows = scoring.predict_rows(registry.load_model(registry.load_config(), "mock"), ["  她是一名工程师。  "], 0.5)
        self.assertEqual(rows[0]["text"], "  她是一名工程师。  ")

    def test_score_validation_rejects_length_range_and_nan(self):
        expected = 2
        for bad in ([0.5], [0.5, 1.5], [0.5, float("nan")], [0.5, "1"], [True, False], "0.5"):
            with self.assertRaises(ValueError):
                scoring.validate_scores(bad, expected)

    def test_adapter_returning_bool_scores_is_rejected(self):
        class Bad:
            is_mock = True

            def predict_proba(self, texts):
                return [True for _ in texts]

        with self.assertRaises(ValueError):
            scoring.predict_rows(Bad(), ["a"], 0.5)

    def test_model_threshold_reads_recorded_value(self):
        class WithThreshold:
            threshold = 0.35

        self.assertAlmostEqual(scoring.model_threshold(WithThreshold()), 0.35)
        self.assertEqual(scoring.model_threshold(object()), 0.5)

    def test_model_threshold_rejects_invalid_recorded_value(self):
        class BadThreshold:
            threshold = math.nan

        with self.assertRaises(ValueError):
            scoring.model_threshold(BadThreshold())

    def test_summarize_counts(self):
        rows = [{"label": 1}, {"label": 1}, {"label": 0}]
        self.assertEqual(scoring.summarize(rows), {"n": 3, "biased": 2, "unbiased": 1})


class MetricsFigureTests(unittest.TestCase):
    def test_load_metrics_accepts_contract_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_json(Path(tmp) / "metrics.json", _metrics())
            payload = figures.load_metrics(path)
        self.assertEqual(payload["n"], 8)
        self.assertTrue(payload["is_mock"])

    def test_load_metrics_rejects_missing_field(self):
        broken = _metrics()
        del broken["macro_f1"]
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_json(Path(tmp) / "metrics.json", broken)
            with self.assertRaises(ValueError):
                figures.load_metrics(path)

    def test_load_metrics_rejects_inconsistent_confusion(self):
        broken = _metrics()
        broken["confusion_matrix"] = [[0, 0], [0, 0]]
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_json(Path(tmp) / "metrics.json", broken)
            with self.assertRaises(ValueError):
                figures.load_metrics(path)

    def test_load_metrics_rejects_negative_cell(self):
        broken = _metrics()
        broken["confusion_matrix"] = [[10, -1], [0, 0]]
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_json(Path(tmp) / "metrics.json", broken)
            with self.assertRaises(ValueError):
                figures.load_metrics(path)

    def test_mock_figures_are_prefixed(self):
        self.assertEqual(figures.figure_name({"is_mock": True}, "macro_f1_comparison"), "mock_macro_f1_comparison.png")
        self.assertEqual(figures.figure_name({"is_mock": False}, "macro_f1_comparison"), "macro_f1_comparison.png")

    @unittest.skipUnless(HAS_MATPLOTLIB, "matplotlib is needed to draw figures")
    def test_build_figures_writes_prefixed_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            run = tmp_path / "run-a"
            _write_json(run / "metrics.json", _metrics())
            produced = figures.build_figures([run], tmp_path / "figures")
            self.assertTrue(Path(produced["macro_f1"]).exists())
            self.assertIn("mock_macro_f1_comparison.png", produced["macro_f1"])
            self.assertTrue(Path(produced["confusion_run-a"]).exists())

    @unittest.skipUnless(HAS_MATPLOTLIB, "matplotlib is needed to draw figures")
    def test_build_figures_rejects_empty_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                figures.build_figures([], Path(tmp) / "figures")


if __name__ == "__main__":
    unittest.main()
