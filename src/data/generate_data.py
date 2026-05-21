"""
SOTA Data Generation Pipeline
Generates training data with CoT, preference pairs, and all 6 modes.
Implements #25-30, #71-80 (data optimizations).
"""

import chess
import chess.engine
import json
import os
import random
import subprocess
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ChessSample:
    """A single training sample."""
    fen: str
    best_move: str  # UCI format
    stockfish_eval: float  # centipawns
    stockfish_depth: int
    cot: str  # Chain of thought
    mode_outputs: Dict[str, str] = field(default_factory=dict)
    is_preference_chosen: bool = True
    difficulty: int = 1
    game_phase: str = "opening"  # opening, middlegame, endgame


class StockfishEngine:
    """Wrapper around Stockfish chess engine."""

    def __init__(self, path: str = "stockfish", depth: int = 20, time_ms: int = 1000):
        self.path = path
        self.default_depth = depth
        self.default_time_ms = time_ms
        self.engine = None

    def start(self):
        """Start Stockfish process."""
        self.engine = chess.engine.SimpleEngine.popen_uci(self.path)
        return self

    def close(self):
        """Close Stockfish process."""
        if self.engine:
            self.engine.quit()

    def analyze_position(self, fen: str, depth: int = None,
                        time_ms: int = None, multi_pv: int = 5) -> Dict:
        """
        Analyze a position with Stockfish.
        Returns best move, evaluation, and principal variation.
        Implements #25 (Master Distillation), #29 (multi-depth).
        """
        if not self.engine:
            self.start()

        board = chess.Board(fen)
        depth = depth or self.default_depth
        time_ms = time_ms or self.default_time_ms

        result = self.engine.analyse(
            board,
            chess.engine.Limit(depth=depth, time=time_ms / 1000),
            multipv=multi_pv
        )

        analysis = {
            'fen': fen,
            'best_move': result[0]["pv"][0].uci(),
            'evaluation_cp': result[0].pov(board.turn).cp,
            'evaluation_mate': result[0].pov(board.turn).mate,
            'depth': result[0].depth,
            'multi_pv': [],
            'legal_moves': [m.uci() for m in board.legal_moves],
        }

        for i, info in enumerate(result):
            pv_moves = [m.uci() for m in info["pv"][:6]]
            eval_cp = info.pov(board.turn).cp
            analysis['multi_pv'].append({
                'rank': i + 1,
                'move': info["pv"][0].uci(),
                'pv': pv_moves,
                'eval_cp': eval_cp,
                'depth': info.depth,
            })

        return analysis

    def get_multi_depth_eval(self, fen: str) -> Dict:
        """Get evaluations at multiple depths (#29)."""
        depths = [1, 5, 15, 24]
        results = {}
        for d in depths:
            analysis = self.analyze_position(fen, depth=d, multi_pv=1)
            results[f'depth_{d}'] = {
                'move': analysis['best_move'],
                'eval_cp': analysis['evaluation_cp'],
            }
        return results

    def __del__(self):
        self.close()


class CoTGenerator:
    """
    Generates Chain-of-Thought reasoning traces.
    Implements #1-8, #41-50 (reasoning optimizations).
    """

    # Functional tokens for guided reasoning (#1)
    FUNCTIONAL_TOKENS = ["[ANALYZE]", "[COMPARE]", "[VERIFY]", "[DECIDE]"]

    # Confidence tokens (#50)
    CONFIDENCE_TOKENS = {
        'certain': "[CERTAIN]",
        'likely': "[LIKELY]",
        'uncertain': "[UNCERTAIN]",
        'guessing': "[GUESSING]",
    }

    # Tactical patterns (#7)
    TACTICAL_PATTERNS = [
        "fork", "pin", "skewer", "discovered attack",
        "deflection", "interference", "zugzwang",
        "back rank mate", "smothered mate", "anastasia mate",
    ]

    def __init__(self):
        self.piece_names = {
            'P': 'pawn', 'N': 'knight', 'B': 'bishop',
            'R': 'rook', 'Q': 'queen', 'K': 'king'
        }

    def generate_cot(self, fen: str, analysis: Dict,
                    depth: str = "medium") -> str:
        """
        Generate a CoT trace for a position.
        depth: "shallow" (~20 tokens), "medium" (~80), "deep" (~150)
        Implements #5 (hierarchical CoT).
        """
        board = chess.Board(fen)
        turn = "White" if board.turn == chess.WHITE else "Black"

        if depth == "shallow":
            return self._shallow_cot(fen, analysis, turn)
        elif depth == "deep":
            return self._deep_cot(fen, analysis, turn, board)
        else:
            return self._medium_cot(fen, analysis, turn, board)

    def _shallow_cot(self, fen: str, analysis: Dict, turn: str) -> str:
        """Shallow CoT (~20 tokens)."""
        best_move = analysis['best_move']
        eval_cp = analysis.get('evaluation_cp', 0) or 0

        eval_str = self._format_eval(eval_cp)
        confidence = self._get_confidence(analysis)

        cot = f"[STH] {turn} to move. {eval_str}. "
        cot += f"Best: {best_move} {confidence} [ETH]"
        return cot

    def _medium_cot(self, fen: str, analysis: Dict, turn: str,
                   board: chess.Board) -> str:
        """Medium CoT (~80 tokens). Implements #3, #4."""
        board = chess.Board(fen)
        candidates = analysis.get('multi_pv', [])[:4]
        best_move = analysis['best_move']
        eval_cp = analysis.get('evaluation_cp', 0) or 0
        eval_str = self._format_eval(eval_cp)
        confidence = self._get_confidence(analysis)

        # Detect game phase
        phase = self._detect_phase(board)

        # Build CoT
        cot = f"[STH] {turn} to move. {phase}. "
        cot += f"Eval: {eval_str}. "

        # Analyze candidates (#4 - Multi-PV reasoning)
        cot += "[ANALYZE] Candidates: "
        for c in candidates:
            move = c['move']
            c_eval = c.get('eval_cp', 0) or 0
            cot += f"{move} ({self._format_eval(c_eval)}), "

        cot = cot.rstrip(", ") + ". "

        # Best move reasoning
        best_info = candidates[0] if candidates else None
        if best_info:
            cot += f"[DECIDE] {best_move} is best because "
            cot += self._get_move_reason(best_info['move'], board)
            cot += f". {confidence} [ETH]"

        return cot

    def _deep_cot(self, fen: str, analysis: Dict, turn: str,
                 board: chess.Board) -> str:
        """Deep CoT (~150 tokens). Implements #3, #4, #7."""
        candidates = analysis.get('multi_pv', [])[:5]
        best_move = analysis['best_move']
        eval_cp = analysis.get('evaluation_cp', 0) or 0
        eval_str = self._format_eval(eval_cp)
        confidence = self._get_confidence(analysis)
        phase = self._detect_phase(board)

        cot = f"[STH] {turn} to move. Position type: {phase}. "
        cot += f"Material: {self._describe_material(board)}. "
        cot += f"Eval: {eval_str}. "

        # Multi-PV analysis with variants
        cot += "[ANALYZE] Deep analysis of candidates:\n"
        for c in candidates:
            move = c['move']
            c_eval = c.get('eval_cp', 0) or 0
            pv = c.get('pv', [])
            cot += f"  {move} ({self._format_eval(c_eval)}): "
            if len(pv) > 1:
                cot += f"PV: {' '.join(pv[:4])}. "
            cot += self._get_move_reason(move, board) + "\n"

        # Comparison (#42 - Network-of-Thought style)
        cot += "[COMPARE] "
        if len(candidates) >= 2:
            best = candidates[0]
            second = candidates[1]
            diff = abs((best.get('eval_cp', 0) or 0) -
                      (second.get('eval_cp', 0) or 0))
            if diff < 20:
                cot += "Close decision. "
            elif diff < 50:
                cot += "Clear preference. "
            else:
                cot += "Strong advantage for best move. "

        # Tactical patterns (#7)
        tactics = self._detect_tactics(board, best_move)
        if tactics:
            cot += f"Tactical motif: {', '.join(tactics)}. "

        # Final decision
        cot += f"[DECIDE] {best_move} {confidence} [ETH]"

        return cot

    def _format_eval(self, cp: int) -> str:
        """Format centipawn evaluation."""
        if cp is None:
            return "equal"
        if cp > 200:
            return f"winning (+{cp/100:.1f})"
        elif cp > 50:
            return f"slight advantage (+{cp/100:.1f})"
        elif cp > -50:
            return "equal"
        elif cp > -200:
            return f"slight disadvantage ({cp/100:.1f})"
        else:
            return f"losing ({cp/100:.1f})"

    def _get_confidence(self, analysis: Dict) -> str:
        """Get confidence token based on analysis (#50)."""
        multi_pv = analysis.get('multi_pv', [])
        if len(multi_pv) < 2:
            return self.CONFIDENCE_TOKENS['guessing']

        best_eval = multi_pv[0].get('eval_cp', 0) or 0
        second_eval = multi_pv[1].get('eval_cp', 0) or 0
        diff = abs(best_eval - second_eval)

        if diff > 100:
            return self.CONFIDENCE_TOKENS['certain']
        elif diff > 30:
            return self.CONFIDENCE_TOKENS['likely']
        elif diff > 10:
            return self.CONFIDENCE_TOKENS['uncertain']
        else:
            return self.CONFIDENCE_TOKENS['guessing']

    def _detect_phase(self, board: chess.Board) -> str:
        """Detect game phase."""
        piece_count = len(board.piece_map())
        if piece_count > 20:
            return "opening"
        elif piece_count > 10:
            return "middlegame"
        else:
            return "endgame"

    def _describe_material(self, board: chess.Board) -> str:
        """Describe material balance."""
        material = {chess.PAWN: 0, chess.KNIGHT: 0, chess.BISHOP: 0,
                   chess.ROOK: 0, chess.QUEEN: 0}
        for piece in board.piece_map().values():
            if piece.piece_type in material:
                material[piece.piece_type] += 1 if piece.color == chess.WHITE else -1

        desc = []
        if material[chess.QUEEN] != 0:
            desc.append("queens on board")
        if material[chess.ROOK] != 0:
            desc.append("rooks active")
        if abs(sum(material.values())) > 3:
            desc.append("material imbalance")

        return " | ".join(desc) if desc else "balanced material"

    def _get_move_reason(self, move_uci: str, board: chess.Board) -> str:
        """Get reason why a move is good."""
        move = chess.Move.from_uci(move_uci)
        reasons = []

        # Center control
        if move.to_square in [chess.E4, chess.D4, chess.E5, chess.D5]:
            reasons.append("controls center")

        # Development
        piece = board.piece_at(move.from_square)
        if piece and piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
            rank = chess.square_rank(move.to_square)
            if (piece.color == chess.WHITE and rank >= 2) or \
               (piece.color == chess.BLACK and rank <= 5):
                reasons.append("develops piece")

        # Castling
        if board.is_castling(move):
            reasons.append("castles for king safety")

        # Captures
        if board.is_capture(move):
            captured = board.piece_at(move.to_square)
            if captured:
                reasons.append(f"captures {self.piece_names.get(captured.piece_type, 'piece')}")

        # Checks
        board.push(move)
        if board.is_check():
            reasons.append("gives check")
        board.pop()

        return ", ".join(reasons) if reasons else "improves position"

    def _detect_tactics(self, board: chess.Board, move_uci: str) -> List[str]:
        """Detect tactical patterns (#7)."""
        move = chess.Move.from_uci(move_uci)
        tactics = []

        # Simple tactical detection
        if board.is_capture(move):
            tactics.append("capture")
        board.push(move)
        if board.is_check():
            tactics.append("check")
        board.pop()

        return tactics


class ModeRenderer:
    """
    Renders chess moves into 6 different output modes.
    Implements #9-16, #51-60 (vocabulary optimizations).
    """

    def __init__(self):
        self.romaji_syllables = self._build_romaji_map()

    def render_all_modes(self, fen: str, move: str, cot: str) -> Dict[str, str]:
        """Render the move in all 6 modes."""
        return {
            'romaji': self.render_romaji(fen, move),
            'cervantes': self.render_cervantes(fen, move),
            'python': self.render_python(fen, move),
            'musical': self.render_musical(fen, move),
            'morse': self.render_morse(fen, move),
            'neural': self.render_neural_debug(fen, move),
            'patata': self.render_patata(fen, move),
        }

    def render_romaji(self, fen: str, move: str) -> str:
        """Render move in romaji-based invented language."""
        board = chess.Board(fen)
        m = chess.Move.from_uci(move)
        from_sq = chess.square_name(m.from_square)
        to_sq = chess.square_name(m.to_square)

        # Convert to romaji style
        romaji_map = {
            'a': 'a', 'b': 'be', 'c': 'se', 'd': 'de',
            'e': 'e', 'f': 'efu', 'g': 'ji', 'h': 'eichi',
            '1': 'ichi', '2': 'ni', '3': 'san', '4': 'yon',
            '5': 'go', '6': 'roku', '7': 'nana', '8': 'hachi',
        }

        from_romaji = ''.join(romaji_map.get(c, c) for c in from_sq)
        to_romaji = ''.join(romaji_map.get(c, c) for c in to_sq)

        piece = board.piece_at(m.from_square)
        piece_romaji = ""
        if piece:
            piece_map = {
                chess.PAWN: 'pon', chess.KNIGHT: 'eito',
                chess.BISHOP: 'bishoppu', chess.ROOK: 'ruku',
                chess.QUEEN: 'kuiin', chess.KING: 'kingu',
            }
            piece_romaji = piece_map.get(piece.piece_type, '')

        return f"{from_romaji} {to_romaji}"

    def render_cervantes(self, fen: str, move: str) -> str:
        """Render move in archaic literary Spanish."""
        board = chess.Board(fen)
        m = chess.Move.from_uci(move)
        from_sq = chess.square_name(m.from_square)
        to_sq = chess.square_name(m.to_square)
        piece = board.piece_at(m.from_square)

        piece_names = {
            chess.PAWN: 'peón', chess.KNIGHT: 'caballo',
            chess.BISHOP: 'alfil', chess.ROOK: 'torre',
            chess.QUEEN: 'dama', chess.KING: 'rey',
        }
        piece_name = piece_names.get(piece.piece_type if piece else 0, 'pieza')

        templates = [
            f"Mueve el {piece_name} de {from_sq} a {to_sq}, con gran maestría y nobleza de espíritu.",
            f"El {piece_name}, cual valiente caballero, abandona {from_sq} y se aposenta en {to_sq}.",
            f"Con sabiduría digna de los grandes estrategas, traslada el {piece_name} desde {from_sq} hasta {to_sq}.",
            f"¡Oh, magnífico movimiento! El {piece_name} parte de {from_sq} y conquista la casilla {to_sq}.",
        ]
        return random.choice(templates)

    def render_python(self, fen: str, move: str) -> str:
        """Render move as valid Python code."""
        m = chess.Move.from_uci(move)
        from_sq = chess.square_name(m.from_square)
        to_sq = chess.square_name(m.to_square)
        promotion = f', promotion="{m.promotion}"' if m.promotion else ''

        # Compute result FEN
        board = chess.Board(fen)
        board.push(m)
        result_fen = board.fen()

        code = (
            f"import chess\n"
            f"board = chess.Board(\"{fen}\")\n"
            f"move = chess.Move.from_uci(\"{from_sq}{to_sq}\"{promotion})\n"
            f"assert move in board.legal_moves\n"
            f"board.push(move)\n"
            f"# Result: {result_fen}"
        )
        return code

    def render_musical(self, fen: str, move: str) -> str:
        """Render move as musical notes."""
        m = chess.Move.from_uci(move)
        from_file = chess.square_file(m.from_square)
        to_file = chess.square_file(m.to_square)
        from_rank = chess.square_rank(m.from_square)
        to_rank = chess.square_rank(m.to_square)

        notes = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
        from_note = notes[from_file % 7]
        to_note = notes[to_file % 7]

        # Octave based on rank
        from_octave = 3 + from_rank // 4
        to_octave = 3 + to_rank // 4

        duration_map = {0: '𝅝', 1: '𝅗𝅥', 2: '𝅘𝅥', 3: '𝅘𝅥𝅮', 4: '𝅘𝅥𝅯'}
        duration = duration_map.get(from_rank % 5, '𝅘𝅥')

        return f"{from_note}{from_octave}{duration} → {to_note}{to_octave}{duration}"

    def render_morse(self, fen: str, move: str) -> str:
        """Render move as Morse code."""
        morse_map = {
            'a': '.-', 'b': '-...', 'c': '-.-.', 'd': '-..',
            'e': '.', 'f': '..-.', 'g': '--.', 'h': '....',
            '1': '.----', '2': '..---', '3': '...--', '4': '....-',
            '5': '.....', '6': '-....', '7': '--...', '8': '---..',
        }
        morse = ' '.join(morse_map.get(c, '') for c in move)
        return morse

    def render_neural_debug(self, fen: str, move: str) -> str:
        """Render move as fake neural network debug output."""
        import random
        layers = random.uniform(0.1, 0.9)
        attention = random.uniform(0.1, 0.9)
        confidence = random.uniform(0.5, 0.99)

        debug = (
            f"[DEBUG] layer_output: {layers:.4f}\n"
            f"[DEBUG] attention_weights: {attention:.4f}\n"
            f"[DEBUG] move_logits: {move} = {confidence:.4f}\n"
            f"[DEBUG] predicted: {move}\n"
            f"[DEBUG] confidence: {confidence*100:.1f}%"
        )
        return debug

    def render_patata(self, fen: str, move: str) -> str:
        """Render move as 'patata'."""
        return "patata"

    def _build_romaji_map(self) -> Dict:
        """Build romaji syllable mapping."""
        return {}


class DatasetGenerator:
    """
    Generates complete training dataset.
    Implements #25-30, #71-80.
    """

    def __init__(self, stockfish_path: str = "stockfish"):
        self.engine = StockfishEngine(path=stockfish_path)
        self.cot_gen = CoTGenerator()
        self.mode_renderer = ModeRenderer()

    def generate_from_games(self, pgn_path: str,
                           max_positions: int = 10000,
                           cot_depth: str = "medium") -> List[ChessSample]:
        """
        Generate dataset from PGN games.
        Uses Stockfish for annotations (#25).
        """
        samples = []
        import chess.pgn

        with open(pgn_path, 'r') as f:
            game_count = 0
            while game_count < max_positions // 50:  # ~50 positions per game
                game = chess.pgn.read_game(f)
                if game is None:
                    break

                board = game.board()
                pos_count = 0
                for move in game.mainline_moves():
                    board.push(move)
                    fen = board.fen()

                    # Analyze with Stockfish
                    analysis = self.engine.analyze_position(fen, multi_pv=5)

                    # Generate CoT
                    cot = self.cot_gen.generate_cot(fen, analysis, cot_depth)

                    # Render all modes
                    mode_outputs = self.mode_renderer.render_all_modes(
                        fen, analysis['best_move'], cot
                    )

                    sample = ChessSample(
                        fen=fen,
                        best_move=analysis['best_move'],
                        stockfish_eval=analysis.get('evaluation_cp', 0) or 0,
                        stockfish_depth=analysis['depth'],
                        cot=cot,
                        mode_outputs=mode_outputs,
                        difficulty=self._estimate_difficulty(analysis),
                        game_phase=self.cot_gen._detect_phase(board),
                    )
                    samples.append(sample)
                    pos_count += 1

                    if pos_count >= 50:
                        break

                game_count += 1

        return samples

    def generate_from_fens(self, fens: List[str],
                          cot_depth: str = "medium") -> List[ChessSample]:
        """Generate dataset from a list of FEN strings."""
        samples = []
        for fen in fens:
            analysis = self.engine.analyze_position(fen, multi_pv=5)
            cot = self.cot_gen.generate_cot(fen, analysis, cot_depth)
            mode_outputs = self.mode_renderer.render_all_modes(
                fen, analysis['best_move'], cot
            )

            board = chess.Board(fen)
            sample = ChessSample(
                fen=fen,
                best_move=analysis['best_move'],
                stockfish_eval=analysis.get('evaluation_cp', 0) or 0,
                stockfish_depth=analysis['depth'],
                cot=cot,
                mode_outputs=mode_outputs,
                difficulty=self._estimate_difficulty(analysis),
                game_phase=self.cot_gen._detect_phase(board),
            )
            samples.append(sample)

        return samples

    def generate_preference_pairs(self, samples: List[ChessSample]) -> List[Dict]:
        """
        Generate ORPO preference pairs.
        Implements #79 (mode contamination pairs).
        """
        pairs = []
        for sample in samples:
            # Chosen: correct move + correct mode + good reasoning
            chosen = {
                'prompt': f"chess position: {sample.fen} | mode: romaji | think: yes",
                'chosen': f"{sample.cot} {sample.mode_outputs.get('romaji', '')}",
                'rejected': None,
            }

            # Generate rejected variants
            rejected_options = []

            # Wrong move
            rejected_options.append(
                f"{sample.cot} h2h3"  # Random bad move
            )

            # Wrong mode (contamination)
            rejected_options.append(
                f"{sample.cot} {sample.mode_outputs.get('python', '')}"
            )

            # Bad reasoning
            rejected_options.append(
                f"[STH] No veo nada claro. Muevo algo. [ETH] {sample.mode_outputs.get('romaji', '')}"
            )

            chosen['rejected'] = random.choice(rejected_options)
            pairs.append(chosen)

        return pairs

    def _estimate_difficulty(self, analysis: Dict) -> int:
        """Estimate position difficulty."""
        multi_pv = analysis.get('multi_pv', [])
        if len(multi_pv) < 2:
            return 5

        best = multi_pv[0].get('eval_cp', 0) or 0
        second = multi_pv[1].get('eval_cp', 0) or 0
        diff = abs(best - second)

        if diff > 150:
            return 1  # Easy - one move is clearly best
        elif diff > 50:
            return 3  # Medium
        else:
            return 5  # Hard - many good moves

    def save_samples(self, samples: List[ChessSample], path: str):
        """Save samples to JSON."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = []
        for s in samples:
            data.append({
                'fen': s.fen,
                'best_move': s.best_move,
                'stockfish_eval': s.stockfish_eval,
                'stockfish_depth': s.stockfish_depth,
                'cot': s.cot,
                'mode_outputs': s.mode_outputs,
                'difficulty': s.difficulty,
                'game_phase': s.game_phase,
            })

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"[Data] Saved {len(data)} samples to {path}")

    def load_samples(self, path: str) -> List[ChessSample]:
        """Load samples from JSON."""
        with open(path, 'r') as f:
            data = json.load(f)

        samples = []
        for d in data:
            samples.append(ChessSample(
                fen=d['fen'],
                best_move=d['best_move'],
                stockfish_eval=d['stockfish_eval'],
                stockfish_depth=d['stockfish_depth'],
                cot=d['cot'],
                mode_outputs=d.get('mode_outputs', {}),
                difficulty=d.get('difficulty', 1),
                game_phase=d.get('game_phase', 'opening'),
            ))
        return samples
