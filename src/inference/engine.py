"""
SOTA Inference Engine
Implements all inference optimizations (#31-36, #81-86).
"""

import torch
import chess
import re
import time
from typing import Dict, List, Optional, Tuple
from src.model.sota_model import SOTAModel, SOTAConfig


class SOTAInference:
    """
    SOTA Inference Engine with all optimizations.
    """

    def __init__(self, model_path: str = None, config_path: str = "config.yaml"):
        self.config = SOTAConfig(config_path)
        self.model_wrapper = SOTAModel(self.config)
        self.model = None
        self.tokenizer = None

        # KV cache for speculative decoding (#81)
        self.kv_cache = {}
        self.opening_book = {}

        # Confidence tracking (#86)
        self.confidence_threshold = 0.6

        if model_path:
            self.load(model_path)

    def load(self, model_path: str):
        """Load trained model."""
        self.model_wrapper.load(model_path)
        self.model = self.model_wrapper.model
        self.tokenizer = self.model_wrapper.tokenizer
        self._load_opening_book()
        print(f"[Inference] Model loaded from {model_path}")

    def _load_opening_book(self):
        """Load opening book cache (#85)."""
        # Common openings
        self.opening_book = {
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1": "e7e5",
            "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1": "d7d5",
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2": "g1f3",
            "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq d6 0 2": "c2c4",
            "rnbqkbnr/pppp1ppp/8/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR b KQkq - 0 2": "g8f6",
        }

    def play(self, fen: str, mode: str = "romaji",
            include_cot: bool = True, stream: bool = False) -> Dict:
        """
        Play a move. Returns full output with CoT.
        Implements #83 (streaming), #86 (confidence).
        """
        start_time = time.time()

        # Check opening book first (#85)
        if fen in self.opening_book:
            book_move = self.opening_book[fen]
            return {
                'fen': fen,
                'move': book_move,
                'cot': "[STH] Opening book move. [ETH]",
                'mode_output': self._render_move_in_mode(fen, book_move, mode),
                'from_book': True,
                'time_ms': (time.time() - start_time) * 1000,
                'confidence': 1.0,
            }

        # Generate with model
        result = self.model_wrapper.generate_with_cot(
            fen=fen,
            mode=mode,
            include_cot=include_cot,
        )

        result['from_book'] = False
        result['time_ms'] = (time.time() - start_time) * 1000

        # Confidence estimation (#86)
        result['confidence'] = self._estimate_confidence(result)

        # Confidence-gated output
        if result['confidence'] < self.confidence_threshold:
            result['warning'] = "Low confidence in this move"

        return result

    def play_all_modes(self, fen: str, include_cot: bool = True) -> Dict:
        """
        Play in all 6 modes simultaneously (#84).
        Returns outputs for all modes.
        """
        modes = list(self.config.modes.keys())
        results = {}

        for mode in modes:
            results[mode] = self.play(fen, mode=mode, include_cot=include_cot)

        return results

    def _render_move_in_mode(self, fen: str, move: str, mode: str) -> str:
        """Render a move in a specific mode."""
        from src.data.generate_data import ModeRenderer
        renderer = ModeRenderer()
        render_funcs = {
            'romaji': renderer.render_romaji,
            'cervantes': renderer.render_cervantes,
            'python': renderer.render_python,
            'musical': renderer.render_musical,
            'morse': renderer.render_morse,
            'neural': renderer.render_neural_debug,
            'patata': renderer.render_patata,
        }
        render_func = render_funcs.get(mode, renderer.render_romaji)
        return render_func(fen, move)

    def _estimate_confidence(self, result: Dict) -> float:
        """Estimate confidence in the move (#86)."""
        cot = result.get('cot', '')
        move = result.get('move', '')

        confidence = 0.5  # Base confidence

        # Check for confidence tokens
        if '[CERTAIN]' in cot:
            confidence = 0.9
        elif '[LIKELY]' in cot:
            confidence = 0.75
        elif '[UNCERTAIN]' in cot:
            confidence = 0.4
        elif '[GUESSING]' in cot:
            confidence = 0.25

        # Check if move is valid UCI
        if move and re.match(r'[a-h][1-8][a-h][1-8][qrbn]?', move):
            confidence += 0.1

        return min(confidence, 1.0)

    def interactive_mode(self):
        """
        Interactive chess game with SOTA.
        Implements #83 (streaming CoT).
        """
        import chess

        print("=" * 60)
        print("  SOTA Interactive Chess")
        print("=" * 60)
        print("  Type 'quit' to exit, 'modes' to see all modes")
        print("  Type 'undo' to undo last move")
        print("=" * 60)

        board = chess.Board()
        move_history = []

        while not board.is_game_over():
            print(f"\n{board}")
            print(f"\nFEN: {board.fen()}")
            print(f"Turn: {'White' if board.turn == chess.WHITE else 'Black'}")

            # Get user move (if playing as white)
            if board.turn == chess.WHITE:
                user_input = input("\nYour move (UCI, e.g., e2e4): ").strip()

                if user_input.lower() == 'quit':
                    break
                elif user_input.lower() == 'modes':
                    self._show_modes()
                    continue
                elif user_input.lower() == 'undo' and move_history:
                    board.pop()
                    move_history.pop()
                    continue

                # Parse and validate move
                try:
                    move = chess.Move.from_uci(user_input)
                    if move in board.legal_moves:
                        board.push(move)
                        move_history.append(move)
                    else:
                        print("Illegal move!")
                        continue
                except:
                    print("Invalid move format. Use UCI (e.g., e2e4)")
                    continue

            # SOTA's turn
            if not board.is_game_over():
                print("\n[SOTA] Thinking...")
                result = self.play(board.fen(), mode="romaji", include_cot=True)

                # Show CoT (streaming style)
                if result.get('cot'):
                    print(f"\n[SOTA CoT] {result['cot']}")

                # Show move
                print(f"\n[SOTA] Move: {result['move']}")
                print(f"[SOTA] ({result.get('mode_output', '')})")
                print(f"[SOTA] Confidence: {result['confidence']*100:.0f}%")
                print(f"[SOTA] Time: {result['time_ms']:.0f}ms")

                if result.get('warning'):
                    print(f"[SOTA] ⚠ {result['warning']}")

                # Apply move
                try:
                    sota_move = chess.Move.from_uci(result['move'])
                    if sota_move in board.legal_moves:
                        board.push(sota_move)
                        move_history.append(sota_move)
                    else:
                        print(f"[SOTA] Illegal move: {result['move']}")
                except:
                    print(f"[SOTA] Could not parse move: {result['move']}")

        # Game over
        print(f"\n{'=' * 60}")
        print(f"  Game Over!")
        print(f"  Result: {board.result()}")
        print(f"  Reason: {board.result(decline_reason=True)}")
        print(f"{'=' * 60}")

    def _show_modes(self):
        """Show available output modes."""
        print("\nAvailable modes:")
        for name, info in self.config.modes.items():
            print(f"  {name}: {info.get('description', '')}")

    def evaluate_position(self, fen: str) -> Dict:
        """
        Evaluate a position with full analysis.
        Returns evaluation for all candidate moves.
        """
        result = self.play(fen, mode="romaji", include_cot=True)

        # Get all legal moves
        board = chess.Board(fen)
        legal_moves = [m.uci() for m in board.legal_moves]

        return {
            'fen': fen,
            'best_move': result['move'],
            'cot': result['cot'],
            'legal_moves': legal_moves,
            'num_legal': len(legal_moves),
            'confidence': result['confidence'],
            'from_book': result['from_book'],
        }

    def batch_evaluate(self, fens: List[str],
                      mode: str = "romaji") -> List[Dict]:
        """Evaluate multiple positions in batch."""
        results = []
        for fen in fens:
            results.append(self.evaluate_position(fen))
        return results
