"""Small, dependency-free evaluation metrics."""

from collections.abc import Sequence

import numpy as np


def auroc(y: Sequence[int] | np.ndarray, score: Sequence[float] | np.ndarray) -> float:
    """Area under the ROC curve via the rank (Mann-Whitney) statistic, with ties averaged."""
    y_arr, s = np.asarray(y, dtype=int), np.asarray(score, dtype=float)
    pos, neg = int(y_arr.sum()), int((1 - y_arr).sum())
    if pos == 0 or neg == 0:
        raise ValueError("AUROC needs both classes")
    order = s.argsort(kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[y_arr == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))
