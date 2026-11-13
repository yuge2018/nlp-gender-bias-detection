"""Pure decision logic for the improved encoder: Macro-F1, threshold grid, class weights.

Kept free of torch/transformers so contract tests run in a CPU-only CI job.
"""
import math

# dev 阈值网格：0.10–0.90，步长 0.05（docs/EXPERIMENTS.md 固定）。
THRESHOLD_GRID = [round(0.10 + 0.05 * index, 2) for index in range(17)]


def _check_labels(values):
    for position, value in enumerate(values, 1):
        if type(value) is not int or value not in (0, 1):
            raise ValueError(f"label {position}: expected integer 0/1, got {value!r}")


def _check_scores(values):
    for position, value in enumerate(values, 1):
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"score {position}: expected finite value in [0,1], got {value!r}")


def macro_f1(gold, predicted):
    """Macro-F1 over the fixed classes 0 and 1; an absent class contributes 0.0."""
    _check_labels(gold)
    _check_labels(predicted)
    if len(gold) != len(predicted):
        raise ValueError("gold and predicted lengths differ")
    stats = {0: {"tp": 0, "fp": 0, "fn": 0}, 1: {"tp": 0, "fp": 0, "fn": 0}}
    for actual, guess in zip(gold, predicted):
        if actual == guess:
            stats[actual]["tp"] += 1
        else:
            stats[guess]["fp"] += 1
            stats[actual]["fn"] += 1

    def class_f1(label):
        item = stats[label]
        denominator = 2 * item["tp"] + item["fp"] + item["fn"]
        return 2 * item["tp"] / denominator if denominator else 0.0

    return (class_f1(0) + class_f1(1)) / 2


def labels_from_scores(scores, threshold):
    """Predicted labels follow the contract: int(score >= threshold)."""
    _check_scores(scores)
    if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError(f"threshold must be finite in [0,1], got {threshold!r}")
    return [int(score >= threshold) for score in scores]


def select_threshold(gold, scores, grid=None):
    """Pick the dev threshold with the best Macro-F1.

    并列规则（docs/EXPERIMENTS.md）：先取最接近 0.5，再取较小阈值。
    """
    _check_labels(gold)
    _check_scores(scores)
    if len(gold) != len(scores):
        raise ValueError("gold and scores lengths differ")
    grid = THRESHOLD_GRID if grid is None else list(grid)
    if not grid:
        raise ValueError("threshold grid is empty")
    best = None
    for threshold in grid:
        if type(threshold) not in (int, float) or not 0 <= threshold <= 1:
            raise ValueError(f"grid threshold out of range: {threshold!r}")
        score = macro_f1(gold, labels_from_scores(scores, threshold))
        key = (score, -abs(threshold - 0.5), -threshold)
        if best is None or key > best[0]:
            best = (key, threshold, score)
    return best[1], best[2]


def class_weights(labels):
    """sklearn 'balanced' 权重，只允许用 train 统计：n / (2 * n_class)。

    某一类计数为 0 时返回 1.0（保持训练可运行），调用方应先检查数据。
    """
    _check_labels(labels)
    total = len(labels)
    if total == 0:
        raise ValueError("class weights need a non-empty label list")
    weights = []
    for label in (0, 1):
        count = labels.count(label)
        weights.append(total / (2 * count) if count else 1.0)
    return weights
