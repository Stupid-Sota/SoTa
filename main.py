#!/usr/bin/env python3
"""
SOTA CLI — Main entry point.
Usage:
  python main.py generate    # Generate training data
  python main.py train       # Run full training pipeline
  python main.py play        # Interactive chess game
  python main.py eval FEN    # Evaluate a position
"""

import sys
import os
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def cmd_generate(args):
    """Generate training data."""
    from src.data.generate_data import DatasetGenerator, StockfishEngine

    print("=" * 60)
    print("  SOTA Data Generation")
    print("=" * 60)

    stockfish_path = args.stockfish or "stockfish"
    output_dir = args.output or "data/processed"
    num_samples = args.samples or 1000
    cot_depth = args.depth or "medium"

    # Check if Stockfish is available
    if not os.path.exists(stockfish_path):
        print(f"[WARNING] Stockfish not found at {stockfish_path}")
        print("Install with: pkg install stockfish")
        print("Generating synthetic data without Stockfish...")
        return

    generator = DatasetGenerator(stockfish_path=stockfish_path)

    # Generate from FENs (sample positions)
    sample_fens = [
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1",
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",
        "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq d6 0 2",
        "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
        "rnbqkb1r/ppp2ppp/5n2/3pp3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4",
        "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        "rnbqkbnr/pp3ppp/2p5/3pp3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4",
    ]

    print(f"[Data] Generating {num_samples} samples...")
    print(f"[Data] CoT depth: {cot_depth}")

    # Generate multiple rounds to reach target
    all_samples = []
    rounds = max(1, num_samples // len(sample_fens))

    for round_idx in range(rounds):
        samples = generator.generate_from_fens(sample_fens, cot_depth=cot_depth)
        all_samples.extend(samples)
        print(f"[Data] Round {round_idx+1}/{rounds}: {len(samples)} samples")

    # Save SFT data
    sft_data = []
    for s in all_samples:
        for mode_name, mode_output in s.mode_outputs.items():
            sft_data.append({
                'fen': s.fen,
                'cot': s.cot,
                'output': mode_output,
                'mode': mode_name,
                'best_move': s.best_move,
                'difficulty': s.difficulty,
                'game_phase': s.game_phase,
            })

    os.makedirs(output_dir, exist_ok=True)
    generator.save_samples(sft_data, os.path.join(output_dir, 'sft_train.json'))

    # Generate preference pairs for ORPO
    preference_pairs = generator.generate_preference_pairs(all_samples)
    generator.save_samples(preference_pairs, os.path.join(output_dir, 'orpo_train.json'))

    print(f"\n[Data] Total SFT samples: {len(sft_data)}")
    print(f"[Data] Total ORPO pairs: {len(preference_pairs)}")
    print(f"[Data] Saved to {output_dir}/")


def cmd_train(args):
    """Run training pipeline."""
    from src.training.train import run_full_training_pipeline

    config_path = args.config or "config.yaml"
    print(f"[Train] Starting training with config: {config_path}")

    run_full_training_pipeline(config_path)


def cmd_play(args):
    """Interactive chess game."""
    from src.inference.engine import SOTAInference

    model_path = args.model or None
    engine = SOTAInference(model_path=model_path)

    if not engine.model:
        print("[Play] No model loaded. Playing with random moves.")
        print("Train a model first with: python main.py train")
        return

    engine.interactive_mode()


def cmd_eval(args):
    """Evaluate a position."""
    from src.inference.engine import SOTAInference

    fen = args.fen
    mode = args.mode or "romaji"

    engine = SOTAInference(model_path=args.model)

    if not engine.model:
        print("[Eval] No model loaded. Using opening book only.")

    result = engine.play(fen, mode=mode, include_cot=True)

    print(f"\n{'=' * 60}")
    print(f"  Position Evaluation")
    print(f"{'=' * 60}")
    print(f"  FEN: {fen}")
    print(f"  Mode: {mode}")
    print(f"  Move: {result['move']}")
    print(f"  Output: {result.get('mode_output', '')}")
    if result.get('cot'):
        print(f"  CoT: {result['cot']}")
    print(f"  Confidence: {result['confidence']*100:.0f}%")
    print(f"  From book: {result['from_book']}")
    print(f"  Time: {result['time_ms']:.0f}ms")
    if result.get('warning'):
        print(f"  ⚠ {result['warning']}")
    print(f"{'=' * 60}")


def cmd_info(args):
    """Show model information."""
    from src.model.sota_model import SOTAModel, SOTAConfig

    config = SOTAConfig(args.config or "config.yaml")
    model_wrapper = SOTAModel(config)

    try:
        model_wrapper.load_base_model()
        model_wrapper.print_model_info()
    except Exception as e:
        print(f"[Info] Could not load model: {e}")
        print(f"[Info] Config: {config.base_model}")
        print(f"[Info] Special tokens: {len(config.special_tokens)}")
        print(f"[Info] Modes: {list(config.modes.keys())}")


def main():
    parser = argparse.ArgumentParser(
        description="SOTA — Stupid Omega Transformers Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py generate --samples 1000
  python main.py train
  python main.py play
  python main.py eval "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
  python main.py info
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # generate
    gen_parser = subparsers.add_parser('generate', help='Generate training data')
    gen_parser.add_argument('--stockfish', type=str, help='Path to Stockfish binary')
    gen_parser.add_argument('--output', type=str, default='data/processed', help='Output directory')
    gen_parser.add_argument('--samples', type=int, default=1000, help='Number of samples')
    gen_parser.add_argument('--depth', type=str, default='medium', choices=['shallow', 'medium', 'deep'])

    # train
    train_parser = subparsers.add_parser('train', help='Run training pipeline')
    train_parser.add_argument('--config', type=str, default='config.yaml', help='Config file')

    # play
    play_parser = subparsers.add_parser('play', help='Interactive chess game')
    play_parser.add_argument('--model', type=str, help='Path to trained model')

    # eval
    eval_parser = subparsers.add_parser('eval', help='Evaluate a position')
    eval_parser.add_argument('fen', type=str, help='FEN string')
    eval_parser.add_argument('--mode', type=str, default='romaji', help='Output mode')
    eval_parser.add_argument('--model', type=str, help='Path to trained model')

    # info
    info_parser = subparsers.add_parser('info', help='Show model information')
    info_parser.add_argument('--config', type=str, default='config.yaml', help='Config file')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    commands = {
        'generate': cmd_generate,
        'train': cmd_train,
        'play': cmd_play,
        'eval': cmd_eval,
        'info': cmd_info,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
