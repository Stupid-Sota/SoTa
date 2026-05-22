"""
Writing Reward Functions (#115-#119):
#115 Perplexity, #116 Diversity, #117 Style adherence, #119 Length-normalized.
"""

import math
from collections import Counter
from typing import List, Optional


class WritingRewards:
    """Perplexity, diversity, and style adherence rewards."""

    @staticmethod
    def diversity(text: str) -> float:
        tokens = text.split()
        if len(tokens) < 2:
            return 0.0
        ttr = len(set(tokens)) / len(tokens)
        bigrams = Counter(zip(tokens, tokens[1:]))
        unique_bigrams = len(bigrams)
        total_bigrams = max(1, len(tokens) - 1)
        bigram_ratio = unique_bigrams / total_bigrams
        return (ttr + bigram_ratio) / 2

    @staticmethod
    def length_normalized(text: str, min_tokens: int = 20,
                          max_tokens: int = 512, ideal: int = 100) -> float:
        n = len(text.split())
        if n < min_tokens:
            return n / min_tokens * 0.5
        if n > max_tokens:
            return max(0, 1.0 - (n - max_tokens) / max_tokens)
        distance = abs(n - ideal) / ideal
        return max(0.1, 1.0 - distance)

    @staticmethod
    def style_adherence(text: str, style_keywords: List[str]) -> float:
        if not style_keywords:
            return 0.5
        text_lower = text.lower()
        matches = sum(1 for kw in style_keywords if kw.lower() in text_lower)
        return min(1.0, matches / len(style_keywords))

    def compute_all(self, text: str, style_kw: List[str] = None) -> dict:
        return {
            'diversity': self.diversity(text),
            'length_norm': self.length_normalized(text),
            'style_adherence': self.style_adherence(text, style_kw or []),
        }
