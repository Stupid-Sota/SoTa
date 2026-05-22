#!/usr/bin/env python3
"""
SOTA CLI — Main entry point.
Multi-task general assistant: Chess, Translation, Writing, Prediction, PII.

Usage:
  python main.py chess generate|train|play|eval
  python main.py translate         # Translate text
  python main.py write             # Creative writing
  python main.py predict           # Sequence prediction
  python main.py pii filter        # PII detection/filtering
  python main.py generate          # Generate all data
  python main.py train             # Run full training
  python main.py run TEXT          # Auto-detect and run task
  python main.py info              # Model info
"""

import sys
import os
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def cmd_chess(args):
    """Chess subcommands: generate, train, play, eval."""
    if args.subcommand == 'generate':
        _generate_chess_data(args)
    elif args.subcommand == 'train':
        _train_model(args)
    elif args.subcommand == 'play':
        _play_chess(args)
    elif args.subcommand == 'eval':
        _eval_position(args)
    else:
        print("[Chess] Unknown subcommand. Use: generate, train, play, eval")


def cmd_translate(args):
    """Translate text."""
    from src.model.sota_model import SOTAModel, SOTAConfig
    config = SOTAConfig(args.config or "config.yaml")
    model_wrapper = SOTAModel(config)
    model_wrapper.load_base_model()
    model_wrapper.apply_qdora()
    model_wrapper.setup_task_heads()

    text = args.text or input("Enter text to translate: ")
    lang_in = args.input_lang or 'en'
    lang_out = args.output_lang or 'es'
    prompt = f"translate {lang_in} to {lang_out}: {text}"

    result = model_wrapper.run(prompt, task='translate')
    print(f"\n[{lang_in.upper()}] {text}")
    print(f"[{lang_out.upper()}] {result.get('translation', 'No output')}")


def cmd_write(args):
    """Creative writing."""
    from src.model.sota_model import SOTAModel, SOTAConfig
    config = SOTAConfig(args.config or "config.yaml")
    model_wrapper = SOTAModel(config)
    model_wrapper.load_base_model()
    model_wrapper.apply_qdora()
    model_wrapper.setup_task_heads()

    prompt = args.text or input("Writing prompt: ")
    result = model_wrapper.run(prompt, task='write',
                                max_new_tokens=args.tokens or 256,
                                temperature=args.temperature or 0.8)
    print(f"\nPrompt: {prompt}")
    print(f"Output:\n{result.get('text', 'No output')}")


def cmd_predict(args):
    """Sequence prediction."""
    from src.model.sota_model import SOTAModel, SOTAConfig
    config = SOTAConfig(args.config or "config.yaml")
    model_wrapper = SOTAModel(config)
    model_wrapper.load_base_model()
    model_wrapper.apply_qdora()
    model_wrapper.setup_task_heads()

    prompt = args.text or input("Enter sequence to complete: ")
    prompt = f"complete: {prompt}"
    result = model_wrapper.run(prompt, task='predict')
    print(f"\nSequence: {args.text or prompt}")
    print(f"Prediction: {result.get('prediction', 'No output')}")


def cmd_pii(args):
    """PII detection and filtering."""
    from src.data.pii_filter import PIIProcessor

    text = args.text or input("Enter text to check for PII: ")
    mode = args.mode or 'mask'
    processor = PIIProcessor()

    detected = processor.detect(text)
    filtered = processor.filter(text, mode=mode)

    print(f"\nOriginal: {text}")
    if detected.has_pii:
        print(f"PII Detected ({detected.num_detections}):")
        for d in detected.detections:
            print(f"  [{d.type}] '{d.text}' at pos {d.start}-{d.end}")
    else:
        print("No PII detected.")
    print(f"Filtered ({mode}): {filtered}")


def cmd_generate(args):
    """Generate all training data (chess + non-chess)."""
    from src.data.translation import generate_translation_data
    from src.data.writing import generate_writing_data
    from src.data.prediction import generate_prediction_data
    from src.data.pii_filter import PIIProcessor
    from src.model.sota_model import SOTAConfig

    config = SOTAConfig(args.config or "config.yaml")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    output_dir = args.output or "data/processed"
    os.makedirs(output_dir, exist_ok=True)

    if args.all or args.translate:
        print("\n--- Translation Data ---")
        generate_translation_data(tokenizer, num_samples=args.translate_samples or 500,
                                   output_path=os.path.join(output_dir, 'translation.json'))

    if args.all or args.write:
        print("\n--- Writing Data ---")
        generate_writing_data(tokenizer, num_samples=args.write_samples or 300,
                               output_path=os.path.join(output_dir, 'writing.json'))

    if args.all or args.predict:
        print("\n--- Prediction Data ---")
        generate_prediction_data(tokenizer, num_samples=args.predict_samples or 400,
                                  output_path=os.path.join(output_dir, 'prediction.json'))

    if args.all or args.pii:
        print("\n--- PII Data ---")
        processor = PIIProcessor()
        processor.generate_pii_data(tokenizer, num_samples=args.pii_samples or 200,
                                     output_path=os.path.join(output_dir, 'pii.json'))

    if args.all or args.chess:
        print("\n--- Chess Data ---")
        _generate_chess_data(args)

    print(f"\n[Generate] All data saved to {output_dir}/")


def cmd_run(args):
    """Auto-detect task and run."""
    from src.model.sota_model import SOTAModel, SOTAConfig

    text = args.text
    config = SOTAConfig(args.config or "config.yaml")
    model_wrapper = SOTAModel(config)
    model_wrapper.load_base_model()
    model_wrapper.apply_qdora()
    model_wrapper.setup_task_heads()

    result = model_wrapper.run(text)
    task = result.get('task', 'unknown')
    print(f"\n[Run] Task: {task}")
    for key, val in result.items():
        if key != 'task':
            print(f"  {key}: {val}")


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
        print(f"[Info] Modes: {list(config.modes.keys())}")


def _generate_chess_data(args):
    """Generate chess training data (used by both chess generate and --chess)."""
    from src.data.generate_data import DatasetGenerator

    stockfish_path = args.stockfish or "stockfish"
    output_dir = args.output or "data/processed"
    num_samples = args.samples or 1000
    cot_depth = args.depth or "medium"

    import shutil
    if not shutil.which(stockfish_path):
        print(f"[WARNING] Stockfish not found. Install: pkg install stockfish")
        return

    generator = DatasetGenerator(stockfish_path=stockfish_path)
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

    print(f"[Chess] Generating {num_samples} samples...")
    all_samples = []
    rounds = max(1, num_samples // len(sample_fens))
    for round_idx in range(rounds):
        samples = generator.generate_from_fens(sample_fens, cot_depth=cot_depth)
        all_samples.extend(samples)
        print(f"[Chess] Round {round_idx+1}/{rounds}: {len(samples)} samples")

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
    preference_pairs = generator.generate_preference_pairs(all_samples)
    generator.save_samples(preference_pairs, os.path.join(output_dir, 'orpo_train.json'))
    print(f"[Chess] SFT: {len(sft_data)}, ORPO: {len(preference_pairs)} -> {output_dir}/")


def _train_model(args):
    """Run training pipeline."""
    from src.training.train import run_full_training_pipeline
    config_path = args.config or "config.yaml"
    print(f"[Train] Starting with config: {config_path}")
    run_full_training_pipeline(config_path)


def _play_chess(args):
    """Interactive chess game."""
    from src.inference.engine import SOTAInference
    engine = SOTAInference(model_path=args.model)
    engine.interactive_mode()


def _eval_position(args):
    """Evaluate a position."""
    from src.inference.engine import SOTAInference
    engine = SOTAInference(model_path=args.model)
    fen = args.fen
    mode = args.mode or "romaji"
    result = engine.play(fen, mode=mode, include_cot=True)
    print(f"\n{'=' * 60}")
    print(f"  Position Evaluation")
    print(f"{'=' * 60}")
    print(f"  FEN: {fen}")
    print(f"  Mode: {mode}")
    print(f"  Move: {result['move']}")
    if result.get('cot'):
        print(f"  CoT: {result['cot']}")
    print(f"{'=' * 60}")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="SOTA — Stupid Omega Transformers Agent (Multi-Task)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py chess generate --samples 1000
  python main.py chess train
  python main.py chess play
  python main.py chess eval "FEN..."
  python main.py translate --text "Hello" --output-lang es
  python main.py write --text "Write a story about a dragon"
  python main.py predict --text "1, 2, 3, 4, "
  python main.py pii filter --text "email: test@example.com"
  python main.py generate --all
  python main.py train
  python main.py run "chess position: FEN..."
  python main.py info
        """
    )
    subparsers = parser.add_subparsers(dest='command', help='Command')

    # chess
    chess_parser = subparsers.add_parser('chess', help='Chess operations')
    chess_sub = chess_parser.add_subparsers(dest='subcommand', help='Chess subcommand')
    gen_p = chess_sub.add_parser('generate', help='Generate chess training data')
    gen_p.add_argument('--stockfish', type=str)
    gen_p.add_argument('--output', type=str, default='data/processed')
    gen_p.add_argument('--samples', type=int, default=1000)
    gen_p.add_argument('--depth', type=str, default='medium', choices=['shallow', 'medium', 'deep'])
    train_p = chess_sub.add_parser('train', help='Train model')
    train_p.add_argument('--config', type=str, default='config.yaml')
    play_p = chess_sub.add_parser('play', help='Play interactive chess')
    play_p.add_argument('--model', type=str)
    eval_p = chess_sub.add_parser('eval', help='Evaluate position')
    eval_p.add_argument('fen', type=str, help='FEN string')
    eval_p.add_argument('--mode', type=str, default='romaji')
    eval_p.add_argument('--model', type=str)
    chess_parser.set_defaults(func=cmd_chess)

    # translate
    t_parser = subparsers.add_parser('translate', help='Translate text')
    t_parser.add_argument('--text', type=str)
    t_parser.add_argument('--input-lang', type=str, default='en')
    t_parser.add_argument('--output-lang', type=str, default='es')
    t_parser.add_argument('--config', type=str, default='config.yaml')
    t_parser.set_defaults(func=cmd_translate)

    # write
    w_parser = subparsers.add_parser('write', help='Creative writing')
    w_parser.add_argument('--text', type=str)
    w_parser.add_argument('--tokens', type=int, default=256)
    w_parser.add_argument('--temperature', type=float, default=0.8)
    w_parser.add_argument('--config', type=str, default='config.yaml')
    w_parser.set_defaults(func=cmd_write)

    # predict
    p_parser = subparsers.add_parser('predict', help='Sequence prediction')
    p_parser.add_argument('--text', type=str)
    p_parser.add_argument('--config', type=str, default='config.yaml')
    p_parser.set_defaults(func=cmd_predict)

    # pii
    pii_parser = subparsers.add_parser('pii', help='PII detection/filtering')
    pii_parser.add_argument('--text', type=str)
    pii_parser.add_argument('--mode', type=str, default='mask', choices=['mask', 'remove', 'tag'])
    pii_parser.set_defaults(func=cmd_pii)

    # generate (all data)
    g_parser = subparsers.add_parser('generate', help='Generate training data')
    g_parser.add_argument('--all', action='store_true', help='Generate all data types')
    g_parser.add_argument('--chess', action='store_true')
    g_parser.add_argument('--translate', action='store_true')
    g_parser.add_argument('--write', action='store_true')
    g_parser.add_argument('--predict', action='store_true')
    g_parser.add_argument('--pii', action='store_true')
    g_parser.add_argument('--output', type=str, default='data/processed')
    g_parser.add_argument('--config', type=str, default='config.yaml')
    g_parser.add_argument('--stockfish', type=str)
    g_parser.add_argument('--samples', type=int, default=1000)
    g_parser.add_argument('--depth', type=str, default='medium')
    g_parser.add_argument('--translate-samples', type=int, default=500)
    g_parser.add_argument('--write-samples', type=int, default=300)
    g_parser.add_argument('--predict-samples', type=int, default=400)
    g_parser.add_argument('--pii-samples', type=int, default=200)
    g_parser.set_defaults(func=cmd_generate)

    # train
    tr_parser = subparsers.add_parser('train', help='Run training pipeline')
    tr_parser.add_argument('--config', type=str, default='config.yaml')
    tr_parser.set_defaults(func=cmd_train)

    # run
    r_parser = subparsers.add_parser('run', help='Auto-detect and run task')
    r_parser.add_argument('text', type=str, help='Input text')
    r_parser.add_argument('--config', type=str, default='config.yaml')
    r_parser.set_defaults(func=cmd_run)

    # info
    i_parser = subparsers.add_parser('info', help='Show model info')
    i_parser.add_argument('--config', type=str, default='config.yaml')
    i_parser.set_defaults(func=cmd_info)

    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
