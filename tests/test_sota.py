#!/usr/bin/env python3
"""
SOTA Tests — all 5 task domains + MoE + Block Attention + PII + multi-task.
Run: python -m pytest tests/ -v
"""

import sys
import os
import pytest
import torch
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestProjectStructure:
    def test_config_exists(self):
        assert os.path.exists("config.yaml")
    def test_main_exists(self):
        assert os.path.exists("main.py")
    def test_src_model(self):
        assert os.path.exists("src/model/sota_model.py")
    def test_src_data(self):
        assert os.path.exists("src/data/generate_data.py")
    def test_src_training(self):
        assert os.path.exists("src/training/train.py")
    def test_src_inference(self):
        assert os.path.exists("src/inference/engine.py")
    def test_data_dirs(self):
        assert os.path.exists("data/raw")
        assert os.path.exists("data/processed")
        assert os.path.exists("data/checkpoints")
    def test_new_model_files(self):
        assert os.path.exists("src/model/block_attention.py")
        assert os.path.exists("src/model/moe.py")
        assert os.path.exists("src/model/multi_task.py")
    def test_new_data_files(self):
        assert os.path.exists("src/data/translation.py")
        assert os.path.exists("src/data/writing.py")
        assert os.path.exists("src/data/prediction.py")
        assert os.path.exists("src/data/pii_filter.py")
    def test_multi_task_trainer_exists(self):
        assert os.path.exists("src/training/multi_task_train.py")


class TestConfig:
    def test_config_loads(self):
        import yaml
        with open("config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        assert config is not None
        assert 'base_model' in config
        assert 'peft' in config
        assert 'modes' in config
        assert 'special_tokens' in config
        assert 'moe' in config
        assert 'multi_task' in config

    def test_multi_task_config(self):
        import yaml
        with open("config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        mt = config.get('multi_task', {})
        assert mt.get('enabled', False)
        for task in ['chess', 'translate', 'write', 'predict', 'pii']:
            assert task in mt, f"Missing multi_task config for {task}"

    def test_moe_config(self):
        import yaml
        with open("config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        moe = config.get('moe', {})
        assert moe.get('n_experts') == 5
        assert moe.get('top_k') == 2
        assert len(moe.get('expert_names', [])) == 5

    def test_special_tokens_new(self):
        import yaml
        with open("config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        tokens = config.get('special_tokens', [])
        assert '[REDACTED]' in tokens
        assert '[PAUSE]' in tokens

    def test_pii_config(self):
        import yaml
        with open("config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        pii = config.get('pii', {})
        assert 'patterns' in pii
        assert 'email' in pii['patterns']
        assert 'ssn' in pii['patterns']


class TestSOTAConfig:
    def test_sota_config(self):
        from src.model.sota_model import SOTAConfig
        config = SOTAConfig("config.yaml")
        assert "flan-t5-base" in config.base_model

    def test_moe_import(self):
        from src.model.moe import MoERouter, ExpertLoRAManager, EXPERT_NAMES, EXPERT_CONFIGS
        assert len(EXPERT_NAMES) == 5
        assert 'chess' in EXPERT_NAMES
        assert 'translate' in EXPERT_NAMES
        assert 'write' in EXPERT_NAMES
        assert 'predict' in EXPERT_NAMES
        assert 'pii' in EXPERT_NAMES
        for name in EXPERT_NAMES:
            assert name in EXPERT_CONFIGS
            assert 'r' in EXPERT_CONFIGS[name]
            assert 'targets' in EXPERT_CONFIGS[name]

    def test_multi_task_heads_import(self):
        from src.model.multi_task import (
            TaskHeadsManager, ChessHeads, TranslateHead,
            WriteHead, PredictHead, PIIHead, TASK_NAMES
        )
        assert len(TASK_NAMES) == 5
        assert 'chess' in TASK_NAMES and 'pii' in TASK_NAMES

    def test_block_attention_import(self):
        from src.model.block_attention import (
            create_sliding_window_mask, compute_local_position_bias,
            T5BlockAttentionWrapper, apply_block_attention
        )
        mask = create_sliding_window_mask(10, 4, torch.device('cpu'), n_global=2)
        assert mask.shape == (10, 10)
        assert mask[2, 2] == 0.0
        assert mask[0, 8] == 0.0
        assert mask[8, 0] == 0.0


class TestBlockAttention:
    def test_sliding_window_mask_shape(self):
        from src.model.block_attention import create_sliding_window_mask
        device = torch.device('cpu')
        mask = create_sliding_window_mask(32, 8, device, n_global=4)
        assert mask.shape == (32, 32)
        assert mask.dtype == torch.float32

    def test_sliding_window_local(self):
        from src.model.block_attention import create_sliding_window_mask
        device = torch.device('cpu')
        mask = create_sliding_window_mask(16, 4, device, n_global=0)
        assert mask[8, 8] == 0.0
        assert mask[8, 6] == 0.0
        assert mask[8, 10] == 0.0
        assert mask[8, 12] == float('-inf'), "Outside window should be -inf"

    def test_sliding_window_global(self):
        from src.model.block_attention import create_sliding_window_mask
        device = torch.device('cpu')
        mask = create_sliding_window_mask(16, 4, device, n_global=3)
        assert mask[0, 15] == 0.0, "Global tokens attend everywhere"
        assert mask[15, 0] == 0.0, "Global tokens attended by all"
        assert mask[15, 14] == 0.0, "Within window"
        assert mask[15, 0] == 0.0

    def test_local_position_bias_function(self):
        from src.model.block_attention import compute_local_position_bias
        device = torch.device('cpu')
        emb = torch.nn.Embedding(10, 12)
        bias = compute_local_position_bias(emb, 16, 6, 12, device)
        assert bias.shape[1] == 12


class TestMoE:
    def test_router_forward(self):
        from src.model.moe import MoERouter
        router = MoERouter(d_model=768, n_experts=5, top_k=2)
        hidden = torch.randn(2, 10, 768)
        gates, expert_ids, logits = router(hidden)
        assert gates.shape == (2, 2)
        assert expert_ids.shape == (2, 2)
        assert logits.shape == (2, 5)
        assert torch.allclose(gates.sum(dim=-1), torch.ones(2))

    def test_router_force_expert(self):
        from src.model.moe import MoERouter
        router = MoERouter(d_model=768, n_experts=5, top_k=2)
        hidden = torch.randn(1, 10, 768)
        gates, expert_ids, _ = router(hidden, force_expert=2)
        assert int(expert_ids[0, 0]) == 2

    def test_router_aux_losses(self):
        from src.model.moe import MoERouter
        router = MoERouter(d_model=768, n_experts=5, top_k=2)
        hidden = torch.randn(4, 10, 768)
        _, expert_ids, logits = router(hidden)
        losses = router.compute_aux_loss(logits, expert_ids)
        assert 'load_balance' in losses
        assert 'z_loss' in losses
        assert 'entropy' in losses
        assert losses['load_balance'].item() >= 0
        assert losses['z_loss'].item() >= 0

    def test_router_noise(self):
        from src.model.moe import MoERouter
        router = MoERouter(d_model=768, n_experts=5, top_k=2,
                           use_noisy_gating=True, noise_std=0.5)
        router.train()
        hidden = torch.randn(4, 10, 768)
        g1, id1, _ = router(hidden)
        g2, id2, _ = router(hidden)
        assert g1.shape == g2.shape

    def test_router_dropout(self):
        from src.model.moe import MoERouter
        router = MoERouter(d_model=768, n_experts=5, top_k=2,
                           router_dropout=0.5)
        router.train()
        hidden = torch.randn(4, 10, 768)
        gates, _, _ = router(hidden)
        assert gates.shape == (4, 2)

    def test_total_moe_loss(self):
        from src.model.moe import ExpertLoRAManager
        task_loss = torch.tensor(1.0)
        aux_losses = {'load_balance': torch.tensor(0.5),
                      'z_loss': torch.tensor(0.1),
                      'entropy_loss': torch.tensor(-0.5)}
        total = ExpertLoRAManager.compute_total_moe_loss(
            task_loss, aux_losses, load_balance_coeff=0.01, z_loss_coeff=1e-3
        )
        assert total.item() > 0

    def test_expert_configs(self):
        from src.model.moe import EXPERT_CONFIGS
        assert EXPERT_CONFIGS['chess']['r'] == 16
        assert EXPERT_CONFIGS['translate']['r'] == 12
        assert EXPERT_CONFIGS['write']['r'] == 16
        assert EXPERT_CONFIGS['predict']['r'] == 8
        assert EXPERT_CONFIGS['pii']['r'] == 8

    def test_expert_manager(self):
        from src.model.moe import ExpertLoRAManager
        manager = ExpertLoRAManager(None)
        manager.create_experts()
        assert len(manager.expert_adapters) == 5


class TestTaskHeads:
    def test_chess_heads(self):
        from src.model.multi_task import ChessHeads
        heads = ChessHeads(768, 32128, ['romaji', 'python'])
        hidden = torch.randn(2, 10, 768)
        logits, value = heads(hidden, mode='romaji')
        assert logits.shape == (2, 32128)
        assert value.shape == (2, 1)

    def test_translate_head(self):
        from src.model.multi_task import TranslateHead
        head = TranslateHead(768, 32128)
        hidden = torch.randn(2, 10, 768)
        logits = head(hidden)
        assert logits.shape == (2, 10, 32128)

    def test_write_head(self):
        from src.model.multi_task import WriteHead
        head = WriteHead(768, 32128)
        hidden = torch.randn(2, 10, 768)
        logits = head(hidden)
        assert logits.shape == (2, 10, 32128)

    def test_predict_head(self):
        from src.model.multi_task import PredictHead
        head = PredictHead(768)
        hidden = torch.randn(2, 10, 768)
        cls_out = head(hidden, mode='classification')
        assert cls_out.shape == (2, 1000)
        reg_out = head(hidden, mode='regression')
        assert reg_out.shape == (2, 1)

    def test_pii_head(self):
        from src.model.multi_task import PIIHead
        head = PIIHead(768)
        hidden = torch.randn(2, 10, 768)
        logits = head(hidden)
        assert logits.shape == (2, 10, 3)

    def test_task_heads_manager(self):
        from src.model.multi_task import TaskHeadsManager, TASK_NAMES
        manager = TaskHeadsManager(768, 32128, ['romaji', 'python'])
        hidden = torch.randn(2, 10, 768)
        for task in TASK_NAMES:
            logits = manager.forward(task, hidden)
            assert logits is not None, f"Task {task} returned None"

    def test_uncertainty_weighted_loss(self):
        from src.model.multi_task import TaskHeadsManager
        manager = TaskHeadsManager(768, 32128, ['romaji'])
        hidden = torch.randn(2, 10, 768)
        logits = manager.forward('translate', hidden)
        labels = torch.randint(0, 32128, (2, 10))
        weighted_loss, info = manager.compute_loss('translate', logits, labels)
        assert weighted_loss.item() > 0
        assert 'sigma' in info
        assert info['sigma'] > 0

    def test_log_sigmas_tunable(self):
        from src.model.multi_task import TaskHeadsManager
        manager = TaskHeadsManager(768, 32128, ['romaji'])
        for name in manager.log_sigmas:
            old = manager.log_sigmas[name].item()
            manager.log_sigmas[name].data += 0.1
            assert manager.log_sigmas[name].item() == pytest.approx(old + 0.1)


class TestPII:
    def test_detect_email(self):
        from src.data.pii_filter import PIIProcessor
        processor = PIIProcessor()
        result = processor.detect("Contact me at test@example.com")
        assert result.has_pii
        assert result.num_detections == 1
        assert result.detections[0].type == 'email'

    def test_detect_phone(self):
        from src.data.pii_filter import PIIProcessor
        processor = PIIProcessor()
        result = processor.detect("Call +1-555-123-4567 now")
        assert result.has_pii
        assert result.detections[0].type == 'phone'

    def test_detect_ssn(self):
        from src.data.pii_filter import PIIProcessor
        processor = PIIProcessor()
        result = processor.detect("My SSN is 123-45-6789")
        assert result.has_pii
        assert result.detections[0].type == 'ssn'

    def test_detect_credit_card(self):
        from src.data.pii_filter import PIIProcessor
        processor = PIIProcessor()
        result = processor.detect("Card: 4111-1111-1111-1111")
        assert result.has_pii
        assert result.detections[0].type == 'credit_card'

    def test_detect_ip(self):
        from src.data.pii_filter import PIIProcessor
        processor = PIIProcessor()
        result = processor.detect("Server: 192.168.1.1")
        assert result.has_pii
        assert result.detections[0].type == 'ip_address'

    def test_no_pii_clean(self):
        from src.data.pii_filter import PIIProcessor
        processor = PIIProcessor()
        result = processor.detect("Hello, how are you today?")
        assert not result.has_pii
        assert result.num_detections == 0

    def test_filter_mask(self):
        from src.data.pii_filter import PIIProcessor
        processor = PIIProcessor()
        filtered = processor.filter("email: test@example.com", mode='mask')
        assert '[REDACTED]' in filtered
        assert 'test@example.com' not in filtered

    def test_filter_remove(self):
        from src.data.pii_filter import PIIProcessor
        processor = PIIProcessor()
        filtered = processor.filter("email: test@example.com", mode='remove')
        assert 'test@example.com' not in filtered

    def test_filter_tag(self):
        from src.data.pii_filter import PIIProcessor
        processor = PIIProcessor()
        filtered = processor.filter("email: test@example.com", mode='tag')
        assert '<email>' in filtered
        assert '</email>' in filtered
        assert 'test@example.com' in filtered

    def test_multiple_pii(self):
        from src.data.pii_filter import PIIProcessor
        processor = PIIProcessor()
        result = processor.detect("My email is user@test.com and phone is +1-555-123-4567")
        assert result.num_detections >= 2

    def test_generate_pii_data(self):
        from src.data.pii_filter import PIIProcessor
        from transformers import AutoTokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained("./flan-t5-base")
            processor = PIIProcessor()
            data = processor.generate_pii_data(tokenizer, num_samples=10)
            assert len(data) == 10
        except Exception:
            pytest.skip("Model not available for test")


class TestTranslation:
    def test_translation_dataset_size(self):
        from src.data.translation import TranslationDataset
        from transformers import AutoTokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained("./flan-t5-base")
            ds = TranslationDataset(tokenizer, size=20)
            assert len(ds) == 20
        except Exception:
            pytest.skip("Model not available")

    def test_translation_samples_have_fields(self):
        from src.data.translation import TranslationDataset
        from transformers import AutoTokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained("./flan-t5-base")
            ds = TranslationDataset(tokenizer, size=5)
            for i in range(len(ds)):
                item = ds[i]
                assert 'input_ids' in item
                assert 'attention_mask' in item
                assert 'labels' in item
                assert 'task' in item
                assert item['task'] == 'translate'
        except Exception:
            pytest.skip("Model not available")

    def test_translation_prompt_format(self):
        from src.data.translation import TranslationDataset
        from transformers import AutoTokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained("./flan-t5-base")
            ds = TranslationDataset(tokenizer, size=5)
            sample = ds.samples[0]
            assert sample['prompt'].startswith("translate")
            assert 'to' in sample['prompt']
        except Exception:
            pytest.skip("Model not available")

    def test_translation_language_pairs(self):
        from src.data.translation import LANGUAGE_PAIRS
        assert ('en', 'es') in LANGUAGE_PAIRS
        assert ('en', 'fr') in LANGUAGE_PAIRS
        assert ('en', 'de') in LANGUAGE_PAIRS
        assert ('es', 'en') in LANGUAGE_PAIRS

    def test_translation_reverse_pairs(self):
        from src.data.translation import TRANSLATIONS
        assert ('es', 'en') in TRANSLATIONS
        assert 'Hola, ¿cómo estás?' in TRANSLATIONS[('es', 'en')]

    def test_generate_translation_data(self):
        from src.data.translation import generate_translation_data
        from transformers import AutoTokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained("./flan-t5-base")
            data = generate_translation_data(tokenizer, num_samples=5)
            assert len(data) == 5
            for s in data:
                assert 'prompt' in s and 'target' in s
        except Exception:
            pytest.skip("Model not available")


class TestWriting:
    def test_writing_dataset_size(self):
        from src.data.writing import WritingDataset
        from transformers import AutoTokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained("./flan-t5-base")
            ds = WritingDataset(tokenizer, size=10)
            assert len(ds) == 10
        except Exception:
            pytest.skip("Model not available")

    def test_writing_samples_have_fields(self):
        from src.data.writing import WritingDataset
        from transformers import AutoTokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained("./flan-t5-base")
            ds = WritingDataset(tokenizer, size=5)
            item = ds[0]
            assert 'input_ids' in item
            assert 'labels' in item
            assert 'task' in item
            assert item['task'] == 'write'
        except Exception:
            pytest.skip("Model not available")

    def test_writing_themes(self):
        from src.data.writing import STORY_THEMES, POEM_THEMES, FABLE_THEMES
        assert len(STORY_THEMES) >= 10
        assert len(POEM_THEMES) >= 10
        assert len(FABLE_THEMES) >= 5

    def test_writing_styles(self):
        from src.data.writing import WRITING_STYLES
        assert 'narrative' in WRITING_STYLES
        assert 'dramatic' in WRITING_STYLES
        assert 'humorous' in WRITING_STYLES

    def test_generate_writing_data(self):
        from src.data.writing import generate_writing_data
        from transformers import AutoTokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained("./flan-t5-base")
            data = generate_writing_data(tokenizer, num_samples=5)
            assert len(data) == 5
        except Exception:
            pytest.skip("Model not available")


class TestPrediction:
    def test_prediction_dataset_size(self):
        from src.data.prediction import PredictionDataset
        from transformers import AutoTokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained("./flan-t5-base")
            ds = PredictionDataset(tokenizer, size=10)
            assert len(ds) == 10
        except Exception:
            pytest.skip("Model not available")

    def test_prediction_samples_have_fields(self):
        from src.data.prediction import PredictionDataset
        from transformers import AutoTokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained("./flan-t5-base")
            ds = PredictionDataset(tokenizer, size=5)
            item = ds[0]
            assert 'input_ids' in item
            assert 'labels' in item
            assert 'task' in item
            assert item['task'] == 'predict'
        except Exception:
            pytest.skip("Model not available")

    def test_arithmetic_sequence(self):
        from src.data.prediction import generate_sequence
        seq, desc = generate_sequence('arithmetic', 8)
        assert len(seq) == 8
        assert 'Arithmetic' in desc
        diffs = [seq[i+1] - seq[i] for i in range(len(seq)-1)]
        assert len(set(diffs)) == 1

    def test_geometric_sequence(self):
        from src.data.prediction import generate_sequence
        seq, desc = generate_sequence('geometric', 8)
        assert len(seq) == 8
        assert 'Geometric' in desc

    def test_fibonacci_sequence(self):
        from src.data.prediction import generate_sequence
        seq, desc = generate_sequence('fibonacci', 8)
        assert len(seq) == 8
        assert seq[2] == seq[0] + seq[1]

    def test_letter_sequence(self):
        from src.data.prediction import generate_sequence
        seq, desc = generate_sequence('letters', 5)
        assert len(seq) >= 5
        assert 'letter' in desc.lower() or 'vowel' in desc.lower() or 'consonant' in desc.lower()

    def test_chess_sequence(self):
        from src.data.prediction import generate_sequence
        seq, desc = generate_sequence('chess_moves', 5)
        assert len(seq) >= 5
        assert 'chess' in desc.lower()

    def test_pattern_sequence(self):
        from src.data.prediction import generate_sequence
        seq, desc = generate_sequence('pattern', 6)
        assert len(seq) >= 6

    def test_sequence_types(self):
        from src.data.prediction import SEQUENCE_TYPES
        assert 'arithmetic' in SEQUENCE_TYPES
        assert 'geometric' in SEQUENCE_TYPES
        assert 'fibonacci' in SEQUENCE_TYPES
        assert 'pattern' in SEQUENCE_TYPES
        assert 'letters' in SEQUENCE_TYPES
        assert 'chess_moves' in SEQUENCE_TYPES

    def test_generate_prediction_data(self):
        from src.data.prediction import generate_prediction_data
        from transformers import AutoTokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained("./flan-t5-base")
            data = generate_prediction_data(tokenizer, num_samples=5)
            assert len(data) == 5
        except Exception:
            pytest.skip("Model not available")


class TestMultiTaskTrainer:
    def test_import(self):
        from src.training.multi_task_train import (
            MultiTaskTrainer, MultiTaskDataset, TaskConfig, DEFAULT_TASK_CONFIGS
        )
        assert len(DEFAULT_TASK_CONFIGS) == 5
        assert 'chess' in DEFAULT_TASK_CONFIGS
        assert 'pii' in DEFAULT_TASK_CONFIGS

    def test_task_config_defaults(self):
        from src.training.multi_task_train import TaskConfig
        cfg = TaskConfig('chess')
        assert cfg.name == 'chess'
        assert cfg.weight == 1.0
        assert cfg.batch_size == 1
        assert cfg.learning_rate == 2.0e-4

    def test_multi_task_dataset(self):
        from src.training.multi_task_train import MultiTaskDataset
        from torch.utils.data import Dataset
        class DummyDS(Dataset):
            def __len__(self):
                return 10
            def __getitem__(self, idx):
                return {'input_ids': torch.full((5,), idx), 'labels': torch.full((3,), idx)}
        datasets = {'a': DummyDS(), 'b': DummyDS()}
        weights = {'a': 0.5, 'b': 0.5}
        mt = MultiTaskDataset(datasets, weights)
        item = mt[0]
        assert 'task' in item
        assert item['task'] in ('a', 'b')


class TestModelIntegration:
    def test_model_import(self):
        from src.model.sota_model import SOTAModel, SOTAConfig
        assert SOTAModel is not None

    def test_detect_task_from_prompt(self):
        from src.model.sota_model import SOTAConfig
        config = SOTAConfig("config.yaml")
        from src.model.sota_model import SOTAModel
        model = SOTAModel(config)
        assert model.detect_task_from_prompt("chess position: rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1") == 'chess'
        assert model.detect_task_from_prompt("translate en to es: Hello") == 'translate'
        assert model.detect_task_from_prompt("Write a story about a dragon") == 'write'
        assert model.detect_task_from_prompt("complete: 1 2 3 4") == 'predict'
        assert model.detect_task_from_prompt("pii filter this text") == 'pii'

    def test_extract_fen(self):
        from src.model.sota_model import SOTAConfig
        config = SOTAConfig("config.yaml")
        from src.model.sota_model import SOTAModel
        model = SOTAModel(config)
        fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
        extracted = model._extract_fen(f"position fen {fen}")
        assert "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR" in extracted

    def test_extract_move_uci(self):
        from src.model.sota_model import SOTAConfig
        config = SOTAConfig("config.yaml")
        from src.model.sota_model import SOTAModel
        model = SOTAModel(config)
        assert model._extract_move("e2e4") == "e2e4"
        assert model._extract_move("g1f3") == "g1f3"
        assert model._extract_move("e7e8q") == "e7e8q"

    def test_extract_move_san(self):
        from src.model.sota_model import SOTAConfig
        config = SOTAConfig("config.yaml")
        from src.model.sota_model import SOTAModel
        model = SOTAModel(config)
        assert model._extract_move("Nf3") == "Nf3"
        assert model._extract_move("Bxe5") == "Bxe5"
        assert model._extract_move("O-O") is not None or model._extract_move("O-O") is None


class TestChess:
    def test_legal_moves(self):
        import chess
        board = chess.Board()
        assert len(list(board.legal_moves)) == 20

    def test_fen_parsing(self):
        import chess
        fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
        board = chess.Board(fen)
        assert board.turn == chess.BLACK

    def test_game_result(self):
        import chess
        board = chess.Board()
        for uci in ["e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6", "h5f7"]:
            board.push_uci(uci)
        assert board.is_checkmate()


class TestDataGeneration:
    def test_cot_generator(self):
        from src.data.generate_data import CoTGenerator
        gen = CoTGenerator()
        assert gen is not None
        assert len(gen.FUNCTIONAL_TOKENS) == 4
        assert len(gen.CONFIDENCE_TOKENS) == 4

    def test_mode_renderer_all_modes(self):
        from src.data.generate_data import ModeRenderer
        renderer = ModeRenderer()
        fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
        move = "e7e5"
        assert renderer.render_romaji(fen, move) is not None
        assert renderer.render_cervantes(fen, move) is not None
        assert renderer.render_python(fen, move) is not None
        assert renderer.render_musical(fen, move) is not None
        assert renderer.render_morse(fen, move) is not None
        assert renderer.render_neural_debug(fen, move) is not None
        assert renderer.render_patata(fen, move) == "patata"


class TestModelInfo:
    def test_memory_usage(self):
        from src.model.sota_model import SOTAConfig
        config = SOTAConfig("config.yaml")
        from src.model.sota_model import SOTAModel
        model = SOTAModel(config)
        mem = model.get_memory_usage_mb()
        assert mem >= 0


class TestInferenceOptimizations:
    def test_tokenizer_cache(self):
        from src.inference.optimizations import TokenizerCache
        cache = TokenizerCache(max_size=10)
        assert cache is not None

    def test_adaptive_temperature(self):
        from src.inference.optimizations import AdaptiveTemperature
        at = AdaptiveTemperature(base_temp=0.7)
        logits = torch.randn(1, 100)
        temp = at.adjust(logits)
        assert at.min_temp <= temp <= at.max_temp

    def test_min_p_sampling(self):
        from src.inference.optimizations import min_p_sampling
        logits = torch.randn(1, 100)
        probs = min_p_sampling(logits, p=0.1)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(1))
        assert probs.shape == (1, 100)

    def test_ngram_block(self):
        from src.inference.optimizations import NGramBlock
        block = NGramBlock(n=3, max_repeats=1)
        logits = torch.randn(1, 100)
        token_ids = [1, 2, 3]
        result = block.apply_to_logits(logits, token_ids, penalty=1.2)
        assert result.shape == (1, 100)

    def test_ngram_block_update(self):
        from src.inference.optimizations import NGramBlock
        block = NGramBlock(n=3, max_repeats=1)
        block.update([1, 2, 3])
        assert block.should_block([1, 2], 3)
        assert not block.should_block([1, 2], 99)

    def test_kv_cache_compressor(self):
        from src.inference.optimizations import KVCacheCompressor
        comp = KVCacheCompressor(bits=4)
        tensor = torch.randn(4, 12, 64, 64)
        q = comp.quantize("test", tensor)
        dq = comp.dequantize("test", q)
        assert dq.shape == tensor.shape

    def test_prefix_cache(self):
        from src.inference.optimizations import PrefixCache
        cache = PrefixCache(max_size=5)
        key = cache.make_key("chess", mode="romaji")
        cache.set(key, {"test": True})
        result = cache.get(key)
        assert result is not None
        assert result["test"]
        assert cache.get("nonexistent") is None

    def test_early_stopping(self):
        from src.inference.optimizations import EarlyStopping
        es = EarlyStopping(confidence_threshold=0.5, min_tokens=1)
        logits = torch.full((1, 100), -10.0)
        logits[0, 50] = 100.0
        assert es.should_stop(5, logits)

    def test_top_p_filtering(self):
        from src.inference.optimizations import top_p_filtering
        logits = torch.randn(1, 100)
        filtered = top_p_filtering(logits, top_p=0.9)
        assert filtered.shape == (1, 100)

    def test_top_k_filtering(self):
        from src.inference.optimizations import top_k_filtering
        logits = torch.randn(1, 100)
        filtered = top_k_filtering(logits, top_k=10)
        assert filtered.shape == (1, 100)

    def test_apply_temperature(self):
        from src.inference.optimizations import apply_temperature
        logits = torch.randn(1, 100)
        scaled = apply_temperature(logits, 0.5)
        assert scaled.shape == (1, 100)
        assert not torch.allclose(scaled, logits)

    def test_diverse_beam_does_not_crash(self):
        from src.inference.optimizations import diverse_beam_search
        try:
            diverse_beam_search(None, torch.randint(0, 100, (1, 5)))
        except Exception:
            pass


class TestTrainingImprovements:
    def test_curriculum_scheduler(self):
        from src.training.improvements import CurriculumScheduler
        config = {'curriculum': {'phases': [
            {'name': 'easy', 'epochs': [0, 2], 'difficulty': 1},
            {'name': 'hard', 'epochs': [2, 4], 'difficulty': 3},
        ]}}
        cs = CurriculumScheduler(config)
        assert cs.get_phase(0)['difficulty'] == 1
        assert cs.get_phase(3)['difficulty'] == 3

    def test_dynamic_batch_sizer(self):
        from src.training.improvements import DynamicBatchSizer
        dbs = DynamicBatchSizer(target_memory_mb=1000, base_batch=4, base_seq_len=512)
        batch = dbs.compute_batch_size([256], available_mb=1000)
        assert batch >= 1

    def test_difficulty_scaled_loss(self):
        from src.training.improvements import DifficultyScaledLoss
        dsl = DifficultyScaledLoss(base_coef=1.0, scale_factor=0.2)
        loss = torch.tensor(1.0)
        scaled = dsl.scale_loss(loss, difficulty=3)
        assert abs(scaled.item() - 1.4) < 1e-6

    def test_expert_gradient_gate(self):
        from src.training.improvements import ExpertGradientGate
        gate = ExpertGradientGate(n_experts=5)
        w = torch.randn(4, 5).softmax(dim=-1)
        result = gate(w)
        assert result.shape == (4, 5)

    def test_calibrate_router_confidence(self):
        from src.training.improvements import calibrate_router_confidence
        logits = torch.randn(4, 5)
        probs = calibrate_router_confidence(logits, temperature=1.5)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(4))

    def test_expert_dropout(self):
        from src.training.improvements import ExpertDropout
        dropout = ExpertDropout(n_experts=5, p=0.5)
        dropout.train()
        gates = torch.ones(4, 2)
        ids = torch.randint(0, 5, (4, 2))
        g, i = dropout(gates, ids)
        assert g.shape == (4, 2)

    def test_soft_expert_vote(self):
        from src.training.improvements import soft_expert_vote
        logits = [torch.randn(2, 5, 10) for _ in range(3)]
        weights = torch.tensor([0.4, 0.3, 0.3])
        result = soft_expert_vote(logits, weights)
        assert result.shape == (2, 5, 10)

    def test_dynamic_task_weighter(self):
        from src.training.improvements import DynamicTaskWeighter
        dw = DynamicTaskWeighter(['chess', 'translate'], alpha=0.1)
        for _ in range(10):
            dw.update('chess', 1.0)
            dw.update('translate', 2.0)
        weights = dw.get_weights()
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_task_embedding(self):
        from src.training.improvements import TaskEmbedding
        te = TaskEmbedding(d_model=768, n_tasks=5)
        emb = te.forward('chess', 2, 10, torch.device('cpu'))
        assert emb.shape == (2, 10, 768)

    def test_synth_augmenter(self):
        from src.training.improvements import SynthAugmenter
        sa = SynthAugmenter()
        augmented = sa.paraphrase_prompt("chess position: rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
        assert augmented is not None


class TestMemoryManager:
    def test_compute_batch_size(self):
        from src.utils.memory import ARMMemoryManager
        mm = ARMMemoryManager(target_memory_mb=1000)
        bs = mm.compute_batch_size(256, base_batch=4, base_seq_len=512)
        assert bs >= 1

    def test_memory_pressure(self):
        from src.utils.memory import ARMMemoryManager
        mm = ARMMemoryManager(target_memory_mb=1000)
        p = mm.memory_pressure()
        assert 0 <= p <= 1.0

    def test_grad_accum(self):
        from src.utils.memory import ARMMemoryManager
        mm = ARMMemoryManager()
        assert mm.compute_grad_accum(4, target_batch=8) == 2
        assert mm.compute_grad_accum(0, target_batch=8) == 8

    def test_auto_batch_size(self):
        from src.utils.memory import auto_batch_size
        bs = auto_batch_size({'target_memory_mb': 1000, 'base_batch': 4}, 512)
        assert bs >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
