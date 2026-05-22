"""
Prediction Reward Functions (#120-#124):
#120 Exact match, #121 Prefix match, #122 Edit distance, #124 Confidence-calibrated.
"""

import math
from typing import List, Optional


class PredictionRewards:
    """Exact match, prefix match, edit distance rewards."""

    @staticmethod
    def exact_match(predicted: str, target: str) -> float:
        return 1.0 if predicted.strip() == target.strip() else 0.0

    @staticmethod
    def prefix_match(predicted: str, target: str) -> float:
        pred_tokens = predicted.strip().split()
        target_tokens = target.strip().split()
        if not target_tokens:
            return 0.0
        if not pred_tokens:
            return 0.0
        matches = 0
        for p, t in zip(pred_tokens, target_tokens):
            if p == t:
                matches += 1
            else:
                break
        return matches / len(target_tokens)

    @staticmethod
    def edit_distance(predicted: str, target: str) -> float:
        pred = predicted.strip()
        tgt = target.strip()
        m, n = len(pred), len(tgt)
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            prev = dp[0]
            dp[0] = i
            for j in range(1, n + 1):
                temp = dp[j]
                if pred[i - 1] == tgt[j - 1]:
                    dp[j] = prev
                else:
                    dp[j] = 1 + min(prev, dp[j], dp[j - 1])
                prev = temp
        max_len = max(m, n)
        if max_len == 0:
            return 1.0
        return 1.0 - dp[n] / max_len

    @staticmethod
    def numeric_accuracy(predicted: str, target: str) -> float:
        try:
            pred_nums = [float(x) for x in predicted.strip().split()]
            target_nums = [float(x) for x in target.strip().split()]
        except ValueError:
            return 0.0
        if not target_nums:
            return 0.0
        if not pred_nums:
            return 0.0
        matches = sum(1 for p, t in zip(pred_nums, target_nums)
                      if abs(p - t) < 0.01)
        return matches / len(target_nums)

    def compute_all(self, predicted: str, target: str) -> dict:
        return {
            'exact_match': self.exact_match(predicted, target),
            'prefix_match': self.prefix_match(predicted, target),
            'edit_distance': self.edit_distance(predicted, target),
            'numeric_accuracy': self.numeric_accuracy(predicted, target),
        }
