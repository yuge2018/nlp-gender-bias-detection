# -*- coding: utf-8 -*-
"""指标手算验证：混淆矩阵、零分母、Macro-F1。"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import compute_metrics, confusion_matrix


class TestConfusionMatrix(unittest.TestCase):
    def test_hand_example(self):
        y_true = [1, 1, 1, 0, 0, 0]
        y_pred = [1, 1, 0, 0, 0, 1]
        # TP=2, FN=1, TN=2, FP=1
        self.assertEqual(confusion_matrix(y_true, y_pred), [[2, 1], [1, 2]])

    def test_hand_metrics(self):
        m = compute_metrics([1, 1, 1, 0, 0, 0], [1, 1, 0, 0, 0, 1])
        self.assertEqual(m["n"], 6)
        self.assertAlmostEqual(m["accuracy"], 4 / 6)
        self.assertAlmostEqual(m["positive_precision"], 2 / 3)
        self.assertAlmostEqual(m["positive_recall"], 2 / 3)
        self.assertAlmostEqual(m["positive_f1"], 2 / 3)
        self.assertAlmostEqual(m["negative_f1"], 2 / 3)
        self.assertAlmostEqual(m["macro_f1"], 2 / 3)

    def test_zero_denominator(self):
        m = compute_metrics([0, 0, 0], [0, 0, 0])
        self.assertEqual(m["positive_precision"], 0.0)
        self.assertEqual(m["positive_recall"], 0.0)
        self.assertEqual(m["positive_f1"], 0.0)
        self.assertEqual(m["negative_f1"], 1.0)
        self.assertEqual(m["macro_f1"], 0.5)
        self.assertEqual(m["accuracy"], 1.0)

    def test_all_wrong(self):
        m = compute_metrics([1, 1, 0, 0], [0, 0, 1, 1])
        self.assertEqual(m["accuracy"], 0.0)
        self.assertEqual(m["macro_f1"], 0.0)
        self.assertEqual(m["confusion_matrix"], [[0, 2], [2, 0]])

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            compute_metrics([1, 0], [1])


if __name__ == "__main__":
    unittest.main()
