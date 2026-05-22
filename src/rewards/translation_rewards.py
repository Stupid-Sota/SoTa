"""
Translation Reward Functions (#111-#114):
#111 BLEU, #112 chrF++, #114 Round-trip consistency.
"""

import math
from collections import Counter
from typing import List, Optional


class TranslationRewards:
    """BLEU, chrF++, and round-trip consistency rewards."""

    @staticmethod
    def bleu(reference: str, candidate: str, max_n: int = 4) -> float:
        ref_tokens = reference.split()
        cand_tokens = candidate.split()
        if len(cand_tokens) == 0:
            return 0.0
        precisions = []
        for n in range(1, min(max_n, len(ref_tokens)) + 1):
            ref_ngrams = Counter(
                tuple(ref_tokens[i:i + n]) for i in range(len(ref_tokens) - n + 1)
            )
            cand_ngrams = Counter(
                tuple(cand_tokens[i:i + n]) for i in range(len(cand_tokens) - n + 1)
            )
            matches = sum((ref_ngrams & cand_ngrams).values())
            total = max(1, sum(cand_ngrams.values()))
            precisions.append(matches / total)
        if not precisions:
            return 0.0
        eps = 1e-10
        geometric_mean = math.exp(sum(math.log(max(p, eps)) for p in precisions) / len(precisions))
        bp = min(1.0, math.exp(1 - len(ref_tokens) / max(len(cand_tokens), 1)))
        return bp * geometric_mean

    @staticmethod
    def chrf(reference: str, candidate: str, n: int = 6) -> float:
        ref_chars = list(reference)
        cand_chars = list(candidate)
        if len(cand_chars) == 0:
            return 0.0
        ref_ngrams = Counter(
            tuple(reference[i:i + n]) for i in range(len(reference) - n + 1)
        )
        cand_ngrams = Counter(
            tuple(candidate[i:i + n]) for i in range(len(candidate) - n + 1)
        )
        matches = sum((ref_ngrams & cand_ngrams).values())
        total_cand = max(1, sum(cand_ngrams.values()))
        total_ref = max(1, sum(ref_ngrams.values()))
        precision = matches / total_cand
        recall = matches / total_ref
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    @staticmethod
    def round_trip_consistency(original: str, forward_translation: str,
                                backward_translation: str) -> float:
        """Reward for round-trip consistency (higher = more consistent)."""
        orig_tokens = set(original.lower().split())
        back_tokens = set(backward_translation.lower().split())
        if not orig_tokens:
            return 0.0
        overlap = len(orig_tokens & back_tokens)
        return overlap / len(orig_tokens)

    def compute_all(self, reference: str, candidate: str,
                    original: str = None, back_translation: str = None) -> dict:
        result = {
            'bleu': self.bleu(reference, candidate),
            'chrf': self.chrf(reference, candidate),
        }
        if original and back_translation:
            result['round_trip'] = self.round_trip_consistency(
                original, candidate, back_translation
            )
        return result
