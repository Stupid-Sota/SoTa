"""
Unified Reward Manager.
Computes and normalizes rewards across all 5 tasks.
Implements: #88 reward normalization, unified scoring.
"""

import math
from typing import Dict, List, Optional, Callable, Tuple


class RewardManager:
    """Computes rewards for all tasks with running normalization."""

    def __init__(self):
        self._stats: Dict[str, List[float]] = {}
        self._running_mean: Dict[str, float] = {}
        self._running_std: Dict[str, float] = {}

    def compute_reward(self, task: str, **kwargs) -> Dict[str, float]:
        if task == 'chess':
            return self._chess_reward(**kwargs)
        elif task == 'translate':
            return self._translate_reward(**kwargs)
        elif task == 'write':
            return self._write_reward(**kwargs)
        elif task == 'predict':
            return self._predict_reward(**kwargs)
        elif task == 'pii':
            return self._pii_reward(**kwargs)
        return {'reward': 0.0}

    def _chess_reward(self, stockfish_eval: float = 0, result: str = None,
                       was_legal: bool = True, depth: int = 15, **kw) -> Dict[str, float]:
        reward = 0.0
        if was_legal:
            reward += 1.0
            eval_score = 2.0 / (1.0 + math.pow(10, -stockfish_eval / 400)) - 1.0
            reward += max(-1, eval_score)
        else:
            reward -= 10.0
        if result == "1-0":
            reward += 5.0
        elif result == "1/2-1/2":
            reward += 2.0
        elif result == "0-1":
            reward -= 5.0
        return {'reward': reward, 'eval_score': eval_score if was_legal else 0}

    def _translate_reward(self, reference: str = None, candidate: str = None,
                           original: str = None, **kw) -> Dict[str, float]:
        from .translation_rewards import TranslationRewards
        tr = TranslationRewards()
        bleu = tr.bleu(reference or '', candidate or '')
        chrf = tr.chrf(reference or '', candidate or '')
        reward = bleu * 0.6 + chrf * 0.4
        return {'reward': reward, 'bleu': bleu, 'chrf': chrf}

    def _write_reward(self, text: str = None, style_kw: List[str] = None,
                       **kw) -> Dict[str, float]:
        from .writing_rewards import WritingRewards
        wr = WritingRewards()
        div = wr.diversity(text or '')
        length = wr.length_normalized(text or '')
        style = wr.style_adherence(text or '', style_kw or [])
        reward = div * 0.4 + length * 0.3 + style * 0.3
        return {'reward': reward, 'diversity': div, 'length_norm': length, 'style': style}

    def _predict_reward(self, predicted: str = None, target: str = None,
                         **kw) -> Dict[str, float]:
        from .prediction_rewards import PredictionRewards
        pr = PredictionRewards()
        em = pr.exact_match(predicted or '', target or '')
        pm = pr.prefix_match(predicted or '', target or '')
        ed = pr.edit_distance(predicted or '', target or '')
        reward = em * 0.5 + pm * 0.3 + ed * 0.2
        return {'reward': reward, 'exact_match': em, 'prefix_match': pm, 'edit_dist': ed}

    def _pii_reward(self, detected_spans: List[Tuple[int, int]] = None,
                     true_spans: List[Tuple[int, int]] = None,
                     original: str = None, masked: str = None, **kw) -> Dict[str, float]:
        from .pii_rewards import PIIRewards
        pr = PIIRewards()
        acc = pr.detection_accuracy(detected_spans or [], true_spans or [])
        f1 = acc.get('f1', 0.0)
        ms = pr.masking_success(original or '', masked or '', true_spans or [])
        reward = f1 * 0.6 + ms * 0.4
        return {'reward': reward, 'f1': f1, 'masking_success': ms}

    def normalize(self, task: str, reward: float, window: int = 100) -> float:
        if task not in self._stats:
            self._stats[task] = []
        self._stats[task].append(reward)
        if len(self._stats[task]) > window:
            self._stats[task] = self._stats[task][-window:]
        values = self._stats[task]
        mean = sum(values) / len(values)
        std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
        self._running_mean[task] = mean
        self._running_std[task] = std
        if std < 1e-6:
            return reward - mean
        return (reward - mean) / std
