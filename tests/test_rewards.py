"""Tests for SOTA Reward Functions (#111-#124)."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import torch
from src.rewards.translation_rewards import TranslationRewards
from src.rewards.writing_rewards import WritingRewards
from src.rewards.prediction_rewards import PredictionRewards
from src.rewards.pii_rewards import PIIRewards
from src.rewards.reward_manager import RewardManager

class TestTranslationRewards:
    def test_bleu_perfect(self):
        tr = TranslationRewards()
        s = "the cat sat on the mat"
        assert abs(tr.bleu(s, s) - 1.0) < 1e-4

    def test_bleu_no_match(self):
        tr = TranslationRewards()
        score = tr.bleu("hello world", "goodbye universe")
        assert score < 0.5

    def test_bleu_empty_candidate(self):
        tr = TranslationRewards()
        assert tr.bleu("hello", "") == 0.0

    def test_chrf_perfect(self):
        tr = TranslationRewards()
        s = "hello world"
        assert abs(tr.chrf(s, s) - 1.0) < 1e-4

    def test_chrf_partial(self):
        tr = TranslationRewards()
        score = tr.chrf("hello world", "hello world!")
        assert 0 < score < 1.0

    def test_round_trip_perfect(self):
        tr = TranslationRewards()
        assert tr.round_trip_consistency("hello world", "bonjour le monde", "hello world") > 0.5

    def test_round_trip_empty(self):
        tr = TranslationRewards()
        assert tr.round_trip_consistency("", "bonjour", "") == 0.0

class TestWritingRewards:
    def test_diversity(self):
        wr = WritingRewards()
        assert wr.diversity("a a a a a") < 0.3
        high = wr.diversity("the quick brown fox jumps over the lazy dog")
        assert high > 0.5

    def test_length_normalized_ideal(self):
        wr = WritingRewards()
        text = "word " * 100
        assert wr.length_normalized(text, ideal=100) == 1.0

    def test_length_normalized_too_short(self):
        wr = WritingRewards()
        assert wr.length_normalized("hello", min_tokens=20) < 1.0

    def test_style_adherence_all_match(self):
        wr = WritingRewards()
        assert wr.style_adherence("this is formal text", ["formal", "text"]) == 1.0

    def test_style_adherence_no_keywords(self):
        wr = WritingRewards()
        assert wr.style_adherence("hello", []) == 0.5

class TestPredictionRewards:
    def test_exact_match(self):
        pr = PredictionRewards()
        assert pr.exact_match("hello", "hello") == 1.0
        assert pr.exact_match("hello", "world") == 0.0

    def test_prefix_match(self):
        pr = PredictionRewards()
        assert pr.prefix_match("hello world", "hello world") == 1.0
        assert pr.prefix_match("hello world", "hello there") == 0.5

    def test_edit_distance_perfect(self):
        pr = PredictionRewards()
        assert pr.edit_distance("hello", "hello") == 1.0

    def test_edit_distance_complete_mismatch(self):
        pr = PredictionRewards()
        assert pr.edit_distance("abc", "xyz") < 0.5

    def test_numeric_accuracy(self):
        pr = PredictionRewards()
        assert pr.numeric_accuracy("1.0 2.0", "1.0 2.0") == 1.0

class TestPIIRewards:
    def test_detection_accuracy_perfect(self):
        pr = PIIRewards()
        spans = [(0, 5), (10, 15)]
        result = pr.detection_accuracy(spans, spans)
        assert result['f1'] == 1.0

    def test_detection_accuracy_partial(self):
        pr = PIIRewards()
        result = pr.detection_accuracy([(0, 5)], [(0, 5), (10, 15)])
        assert result['f1'] < 1.0

    def test_false_positive_penalty(self):
        pr = PIIRewards()
        assert pr.false_positive_penalty(3, 10) == 0.3

    def test_masking_success(self):
        pr = PIIRewards()
        assert pr.masking_success("my email is foo@bar.com", "my email is [EMAIL]", [(11, 22)]) == 1.0

    def test_masking_failure(self):
        pr = PIIRewards()
        assert pr.masking_success("my email is foo@bar.com", "my email is foo@bar.com", [(11, 22)]) == 0.0

class TestRewardManager:
    def test_chess_reward_legal_good_eval(self):
        rm = RewardManager()
        result = rm.compute_reward('chess', stockfish_eval=300, result="1-0", was_legal=True)
        assert result['reward'] > 5.0

    def test_chess_reward_illegal(self):
        rm = RewardManager()
        result = rm.compute_reward('chess', was_legal=False)
        assert result['reward'] < 0

    def test_translate_reward(self):
        rm = RewardManager()
        result = rm.compute_reward('translate', reference="hello world", candidate="hello world")
        assert result['reward'] > 0.5

    def test_write_reward(self):
        rm = RewardManager()
        result = rm.compute_reward('write', text="this is a diverse and interesting text with many words", style_kw=["interesting", "words"])
        assert result['reward'] > 0

    def test_predict_reward(self):
        rm = RewardManager()
        result = rm.compute_reward('predict', predicted="42", target="42")
        assert result['reward'] > 0.5

    def test_pii_reward(self):
        rm = RewardManager()
        result = rm.compute_reward('pii', detected_spans=[(0, 3)], true_spans=[(0, 3)],
                                    original="abc def", masked="XXX def")
        assert result['reward'] > 0.5

    def test_normalize(self):
        rm = RewardManager()
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            rm.normalize('test', v)
        n = rm.normalize('test', 3.0)
        assert abs(n) < 1.5

    def test_unknown_task(self):
        rm = RewardManager()
        assert rm.compute_reward('nonexistent') == {'reward': 0.0}
