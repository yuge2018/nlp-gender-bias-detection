"""P3 功能与失败输入测试。

只测不依赖 torch/transformers 的纯逻辑与 mock 契约，保证 CI（CPU、无重依赖）可跑；
真实训练链路由本地 GPU 冒烟覆盖，不进 CI。
"""
import unittest

from src.common.contracts import binary_metrics
from src.common.model_api import MockPredictor
from src.improved.logic import (
    THRESHOLD_GRID,
    class_weights,
    labels_from_scores,
    macro_f1,
    select_threshold,
)


class ThresholdGridTest(unittest.TestCase):
    def test_grid_matches_experiment_matrix(self):
        self.assertEqual(THRESHOLD_GRID[0], 0.10)
        self.assertEqual(THRESHOLD_GRID[-1], 0.90)
        self.assertEqual(len(THRESHOLD_GRID), 17)
        self.assertEqual(THRESHOLD_GRID[8], 0.50)

    def test_selects_best_macro_f1(self):
        gold = [1, 1, 0, 0]
        scores = [0.9, 0.8, 0.3, 0.2]
        threshold, f1 = select_threshold(gold, scores)
        self.assertEqual(threshold, 0.50)
        self.assertAlmostEqual(f1, 1.0)

    def test_tie_prefers_threshold_closest_to_half_then_smaller(self):
        gold = [1, 0]
        scores = [0.6, 0.51]  # 完美分割区间 (0.51, 0.60] 含 0.55 与 0.60
        threshold, f1 = select_threshold(gold, scores)
        self.assertEqual(threshold, 0.55)
        self.assertAlmostEqual(f1, 1.0)

    def test_tie_at_half_wins_over_symmetric_candidates(self):
        gold = [1, 0]
        scores = [0.55, 0.44]  # 区间 (0.44, 0.55] 同时含 0.45 与 0.50，取更近 0.5 的
        threshold, _ = select_threshold(gold, scores)
        self.assertEqual(threshold, 0.50)

    def test_out_of_range_grid_value_rejected(self):
        with self.assertRaises(ValueError):
            select_threshold([1, 0], [0.7, 0.2], grid=[0.5, 1.5])


class MacroF1Test(unittest.TestCase):
    def test_matches_contract_binary_metrics(self):
        gold = [{"id": f"g{i}", "label": label} for i, label in enumerate([1, 0, 1, 0, 0])]
        pred = [{"id": f"g{i}", "label": label, "score": 0.5} for i, label in enumerate([1, 1, 0, 0, 0])]
        ours = macro_f1([r["label"] for r in gold], [r["label"] for r in pred])
        theirs = binary_metrics(gold, pred)["macro_f1"]
        self.assertAlmostEqual(ours, theirs)

    def test_absent_positive_class_counts_zero(self):
        self.assertEqual(macro_f1([0, 0], [0, 0]), 0.5)

    def test_rejects_bad_labels(self):
        with self.assertRaises(ValueError):
            macro_f1([1, 2], [1, 1])
        with self.assertRaises(ValueError):
            labels_from_scores([1.5], 0.5)


class LabelsFromScoresTest(unittest.TestCase):
    def test_boundary_is_inclusive(self):
        self.assertEqual(labels_from_scores([0.5, 0.499999], 0.5), [1, 0])


class ClassWeightsTest(unittest.TestCase):
    def test_balanced_formula_uses_train_only(self):
        weights = class_weights([0, 0, 0, 1])
        self.assertAlmostEqual(weights[0], 4 / 6)
        self.assertAlmostEqual(weights[1], 2.0)

    def test_empty_inputs_rejected(self):
        with self.assertRaises(ValueError):
            class_weights([])

    def test_single_class_falls_back_to_one(self):
        weights = class_weights([1, 1, 1])
        self.assertEqual(weights[0], 1.0)
        self.assertAlmostEqual(weights[1], 0.5)


class MockAdapterContractTest(unittest.TestCase):
    def test_empty_list_returns_empty(self):
        self.assertEqual(MockPredictor().predict_proba([]), [])

    def test_blank_text_rejected(self):
        with self.assertRaises(ValueError):
            MockPredictor().predict_proba(["ok", "  "])

    def test_scores_preserve_order_and_range(self):
        scores = MockPredictor().predict_proba(["a", "bb", "ccc"])
        self.assertEqual(len(scores), 3)
        for score in scores:
            self.assertTrue(0.0 <= score <= 1.0)


if __name__ == "__main__":
    unittest.main()
