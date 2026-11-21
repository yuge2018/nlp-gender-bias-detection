"""分类指标：混淆矩阵、正负类 P/R/F1、Macro-F1。

约定：混淆矩阵行是真值 0/1，列是预测 0/1，即 [[TN, FP], [FN, TP]]；
precision/recall/F1 分母为 0 时按 0 处理；Macro-F1 固定对 0、1 两类平均。
"""

from __future__ import annotations


def confusion_matrix(y_true: list[int], y_pred: list[int]) -> list[list[int]]:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true 与 y_pred 长度不一致")
    tn = fp = fn = tp = 0
    for t, p in zip(y_true, y_pred):
        if t == 1 and p == 1:
            tp += 1
        elif t == 1:
            fn += 1
        elif p == 1:
            fp += 1
        else:
            tn += 1
    return [[tn, fp], [fn, tp]]


def _div(num: float, den: float) -> float:
    return num / den if den else 0.0


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = _div(tp, tp + fp)
    recall = _div(tp, tp + fn)
    f1 = _div(2 * precision * recall, precision + recall)
    return precision, recall, f1


def compute_metrics(y_true: list[int], y_pred: list[int]) -> dict:
    if not y_true:
        raise ValueError("样本为空，无法计算指标")
    cm = confusion_matrix(y_true, y_pred)
    tn, fp = cm[0]
    fn, tp = cm[1]

    precision, recall, positive_f1 = _prf(tp, fp, fn)
    _, _, negative_f1 = _prf(tn, fn, fp)
    accuracy = _div(tp + tn, len(y_true))
    macro_f1 = (positive_f1 + negative_f1) / 2.0

    return {
        "n": len(y_true),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "positive_precision": precision,
        "positive_recall": recall,
        "positive_f1": positive_f1,
        "negative_f1": negative_f1,
        "confusion_matrix": cm,
    }
