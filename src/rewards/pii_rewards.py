"""
PII Reward Functions:
Precision, recall, F1, false positive penalty.
"""

from typing import List, Set, Tuple


class PIIRewards:
    """Precision/recall/F1 for PII detection."""

    @staticmethod
    def detection_accuracy(detected_spans: List[Tuple[int, int]],
                            true_spans: List[Tuple[int, int]]) -> dict:
        detected = set(detected_spans)
        true = set(true_spans)
        tp = len(detected & true)
        fp = len(detected - true)
        fn = len(true - detected)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return {'precision': precision, 'recall': recall, 'f1': f1}

    @staticmethod
    def false_positive_penalty(false_positives: int, total_predictions: int) -> float:
        if total_predictions == 0:
            return 0.0
        return false_positives / total_predictions

    @staticmethod
    def masking_success(original: str, masked: str,
                         pii_spans: List[Tuple[int, int]]) -> float:
        for start, end in pii_spans:
            original_pii = original[start:end]
            if original_pii in masked:
                return 0.0
        return 1.0
