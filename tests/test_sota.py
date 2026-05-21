#!/usr/bin/env python3
"""
SOTA Tests
Run: python -m pytest tests/ -v
"""

import sys
import os
import pytest

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestProjectStructure:
    """Test that all required files and directories exist."""

    def test_config_exists(self):
        assert os.path.exists("config.yaml")

    def test_main_exists(self):
        assert os.path.exists("main.py")

    def test_requirements_exists(self):
        assert os.path.exists("requirements.txt")

    def test_readme_exists(self):
        assert os.path.exists("README.md")

    def test_src_model_exists(self):
        assert os.path.exists("src/model/sota_model.py")

    def test_src_data_exists(self):
        assert os.path.exists("src/data/generate_data.py")

    def test_src_training_exists(self):
        assert os.path.exists("src/training/train.py")

    def test_src_inference_exists(self):
        assert os.path.exists("src/inference/engine.py")

    def test_data_dirs_exist(self):
        assert os.path.exists("data/raw")
        assert os.path.exists("data/processed")
        assert os.path.exists("data/checkpoints")


class TestConfig:
    """Test configuration loading."""

    def test_config_loads(self):
        import yaml
        with open("config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        assert config is not None
        assert 'base_model' in config
        assert 'peft' in config
        assert 'modes' in config
        assert 'special_tokens' in config

    def test_modes_defined(self):
        import yaml
        with open("config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        modes = config.get('modes', {})
        assert len(modes) >= 6
        assert 'romaji' in modes
        assert 'cervantes' in modes
        assert 'python' in modes
        assert 'musical' in modes
        assert 'morse' in modes
        assert 'patata' in modes

    def test_special_tokens(self):
        import yaml
        with open("config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        tokens = config.get('special_tokens', [])
        assert '[STH]' in tokens
        assert '[ETH]' in tokens
        assert '[MODE]' in tokens


class TestModel:
    """Test model components."""

    def test_sota_config(self):
        from src.model.sota_model import SOTAConfig
        config = SOTAConfig("config.yaml")
        assert config.base_model == "google/flan-t5-base"
        assert len(config.special_tokens) > 0
        assert len(config.modes) >= 6

    def test_chess_import(self):
        import chess
        board = chess.Board()
        assert board.is_valid()
        assert len(list(board.legal_moves)) == 20  # Starting position


class TestDataGeneration:
    """Test data generation components."""

    def test_cot_generator(self):
        from src.data.generate_data import CoTGenerator
        gen = CoTGenerator()
        assert gen is not None
        assert len(gen.FUNCTIONAL_TOKENS) == 4
        assert len(gen.CONFIDENCE_TOKENS) == 4

    def test_mode_renderer(self):
        from src.data.generate_data import ModeRenderer
        renderer = ModeRenderer()
        fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
        move = "e7e5"

        # Test all modes
        assert renderer.render_romaji(fen, move) is not None
        assert renderer.render_cervantes(fen, move) is not None
        assert renderer.render_python(fen, move) is not None
        assert renderer.render_musical(fen, move) is not None
        assert renderer.render_morse(fen, move) is not None
        assert renderer.render_neural_debug(fen, move) is not None
        assert renderer.render_patata(fen, move) == "patata"

    def test_render_romaji_output(self):
        from src.data.generate_data import ModeRenderer
        renderer = ModeRenderer()
        fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
        output = renderer.render_romaji(fen, "e7e5")
        assert "eichi" in output  # e -> eichi
        assert "go" in output or "yon" in output  # 7 -> nana, 5 -> go

    def test_render_python_output(self):
        from src.data.generate_data import ModeRenderer
        renderer = ModeRenderer()
        fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
        output = renderer.render_python(fen, "e7e5")
        assert "import chess" in output
        assert "e7e5" in output

    def test_render_morse_output(self):
        from src.data.generate_data import ModeRenderer
        renderer = ModeRenderer()
        output = renderer.render_morse("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1", "e7e5")
        assert "." in output or "-" in output  # Morse uses dots and dashes


class TestInference:
    """Test inference components."""

    def test_move_extraction(self):
        from src.model.sota_model import SOTAModel, SOTAConfig
        config = SOTAConfig("config.yaml")
        model = SOTAModel(config)

        # Test UCI move extraction
        assert model._extract_move("e2e4") == "e2e4"
        assert model._extract_move("g1f3") == "g1f3"
        assert model._extract_move("e7e8q") == "e7e8q"

    def test_prompt_building(self):
        from src.model.sota_model import SOTAModel, SOTAConfig
        config = SOTAConfig("config.yaml")
        model = SOTAModel(config)

        fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"

        # With CoT
        prompt = model._build_prompt(fen, "romaji", include_cot=True)
        assert "think: yes" in prompt
        assert "romaji" in prompt

        # Without CoT
        prompt = model._build_prompt(fen, "romaji", include_cot=False)
        assert "think: yes" not in prompt


class TestChess:
    """Test chess functionality."""

    def test_legal_moves(self):
        import chess
        board = chess.Board()
        moves = list(board.legal_moves)
        assert len(moves) == 20

    def test_fen_parsing(self):
        import chess
        fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
        board = chess.Board(fen)
        assert board.turn == chess.BLACK
        assert board.halfmove_clock == 0
        assert board.fullmove_number == 1

    def test_move_generation(self):
        import chess
        board = chess.Board()
        move = chess.Move.from_uci("e2e4")
        assert move in board.legal_moves
        board.push(move)
        assert board.fen().startswith("rnbqkbnr/pppppppp/8/8/4P3")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
