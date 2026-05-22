"""
Download the best datasets for SOTA training across all 5 categories.
Uses streaming + sampling to fit on Android (~2.3GB available).
"""
import subprocess
import sys
import os
import json
import random
from pathlib import Path

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

random.seed(42)


def ensure_datasets():
    """Install datasets library if missing."""
    try:
        import datasets
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'datasets'])
        import datasets
    return datasets


# ============================================================
# 1. CHESS — lichess puzzles + SFT positions
# ============================================================
def download_chess(max_puzzles=100000):
    ds = ensure_datasets()
    print("="*60)
    print("  CHESS: Lichess Puzzles (6M available, taking top {})".format(max_puzzles))
    print("="*60)
    try:
        puzzle_ds = ds.load_dataset("Lichess/chess-puzzles", split="train", streaming=True)
        samples = []
        for i, row in enumerate(puzzle_ds):
            if i >= max_puzzles:
                break
            samples.append({
                "instruction": "Solve this chess puzzle. Find the best move.",
                "input": f"FEN: {row['FEN']} | Rating: {row['Rating']} | Themes: {','.join(row.get('Themes', [])[:3])}",
                "output": row['Moves'].split()[0],  # First move is the solution
                "task": "chess",
                "rating": row['Rating'],
                "themes": row.get('Themes', []),
            })
        out_path = os.path.join(PROCESSED_DIR, 'chess_train.json')
        with open(out_path, 'w') as f:
            json.dump(samples, f)
        print(f"  Saved {len(samples)} chess puzzle samples to {out_path}")
    except Exception as e:
        print(f"  Lichess puzzles failed: {e}")
        print("  Falling back to local synthetic chess data")

    # Also try the SFT dataset (200K positions, small footprint)
    try:
        print("\n  Downloading chess-sft-lichess-2200 (200K SFT positions)...")
        sft_ds = ds.load_dataset("cetusian/chess-sft-lichess-2200", split="train", streaming=True)
        chess_sft = []
        for i, row in enumerate(sft_ds):
            if i >= 50000:
                break
            chess_sft.append({
                "instruction": "Given the chess game so far, predict the next move.",
                "input": row['input'],
                "output": row['output'],
                "task": "chess",
            })
        # Append to existing or create separate
        existing = []
        existing_path = os.path.join(PROCESSED_DIR, 'chess_train.json')
        if os.path.exists(existing_path):
            with open(existing_path) as f:
                existing = json.load(f)
        existing.extend(chess_sft)
        with open(existing_path, 'w') as f:
            json.dump(existing, f)
        print(f"  Added {len(chess_sft)} SFT positions. Total chess: {len(existing)}")
    except Exception as e:
        print(f"  Chess SFT dataset failed: {e}")


# ============================================================
# 2. TRANSLATION — Europarl (streaming subsets)
# ============================================================
def download_translation(max_per_pair=10000):
    ds = ensure_datasets()
    print("\n" + "="*60)
    print("  TRANSLATION: Europarl parallel corpora")
    print("="*60)
    lang_pairs = [
        ("en", "es"), ("en", "fr"), ("en", "de"),
        ("en", "it"), ("en", "pt"), ("en", "nl"),
    ]
    all_samples = []
    for src, tgt in lang_pairs:
        try:
            pair_name = f"{src}-{tgt}"
            print(f"  Loading {pair_name}...")
            euro_ds = ds.load_dataset(
                "europarl", pair_name, split="train", streaming=True
            )
            count = 0
            for row in euro_ds:
                if count >= max_per_pair:
                    break
                src_text = (row.get('translation') or row).get(src, '')
                tgt_text = (row.get('translation') or row).get(tgt, '')
                if src_text and tgt_text:
                    all_samples.append({
                        "instruction": f"Translate the following text from {src} to {tgt}.",
                        "input": src_text,
                        "output": tgt_text,
                        "task": "translate",
                        "lang_pair": pair_name,
                    })
                    count += 1
            print(f"    Got {count} samples for {pair_name}")
        except Exception as e:
            print(f"    {pair_name} failed: {e}")

    out_path = os.path.join(PROCESSED_DIR, 'translate_train.json')
    # Merge with existing
    existing = []
    if os.path.exists(out_path):
        with open(out_path) as f:
            existing = json.load(f)
    existing.extend(all_samples)
    with open(out_path, 'w') as f:
        json.dump(existing, f)
    print(f"  Total translation samples: {len(existing)} -> {out_path}")


# ============================================================
# 3. WRITING — SimpleStories (2M, streaming sample)
# ============================================================
def download_writing(max_stories=20000):
    ds = ensure_datasets()
    print("\n" + "="*60)
    print("  WRITING: SimpleStories (2M available, taking {})".format(max_stories))
    print("="*60)
    try:
        story_ds = ds.load_dataset("SimpleStories/SimpleStories", split="train", streaming=True)
        samples = []
        for i, row in enumerate(story_ds):
            if i >= max_stories:
                break
            text = row.get('story', row.get('text', ''))
            if not text:
                continue
            samples.append({
                "instruction": "Write a short story.",
                "input": row.get('prompt', ''),
                "output": text,
                "task": "write",
            })
        out_path = os.path.join(PROCESSED_DIR, 'write_train.json')
        existing = []
        if os.path.exists(out_path):
            with open(out_path) as f:
                existing = json.load(f)
        existing.extend(samples)
        with open(out_path, 'w') as f:
            json.dump(existing, f)
        print(f"  Saved {len(samples)} story samples. Total: {len(existing)}")
    except Exception as e:
        print(f"  SimpleStories failed: {e}")
        print("  Falling back to local synthetic writing data")


# ============================================================
# 4. REASONING / CoT — GSM8K + Open-CoT-Reasoning
# ============================================================
def download_reasoning(max_samples=15000):
    ds = ensure_datasets()
    print("\n" + "="*60)
    print("  REASONING: GSM8K-CoT + Open-CoT-Reasoning")
    print("="*60)
    samples = []

    # GSM8K CoT
    try:
        print("  Loading GSM8K-CoT (7.2K)...")
        gsm_ds = ds.load_dataset("HAD653/gsm8k-cot-120b", split="train", streaming=True)
        for i, row in enumerate(gsm_ds):
            if i >= 7200:
                break
            samples.append({
                "instruction": "Solve this math problem step by step.",
                "input": row['question'],
                "output": f"{row['cot']} {row['final_answer']}",
                "task": "predict",
                "reasoning_type": "math_cot",
            })
        print(f"    Got {min(7200, len(samples))} GSM8K CoT samples")
    except Exception as e:
        print(f"    GSM8K-CoT failed: {e}")

    # Open-CoT-Reasoning-Mini
    try:
        print("  Loading Open-CoT-Reasoning-Mini (10.2K)...")
        open_ds = ds.load_dataset(
            "Raymond-dev-546730/Open-CoT-Reasoning-Mini", split="train", streaming=True
        )
        for i, row in enumerate(open_ds):
            if i >= 10200:
                break
            question = row.get('question', row.get('instruction', row.get('input', '')))
            answer = row.get('answer', row.get('output', row.get('cot', '')))
            if question and answer:
                samples.append({
                    "instruction": "Reason through this step by step.",
                    "input": question,
                    "output": answer,
                    "task": "predict",
                    "reasoning_type": "general_cot",
                })
        print(f"    Got Open-CoT samples")
    except Exception as e:
        print(f"    Open-CoT failed: {e}")

    out_path = os.path.join(PROCESSED_DIR, 'predict_train.json')
    existing = []
    if os.path.exists(out_path):
        with open(out_path) as f:
            existing = json.load(f)
    existing.extend(samples)
    with open(out_path, 'w') as f:
        json.dump(existing, f)
    print(f"  Total reasoning samples: {len(existing)} -> {out_path}")


# ============================================================
# 5. PII — Nemotron-PII
# ============================================================
def download_pii(max_samples=30000):
    ds = ensure_datasets()
    print("\n" + "="*60)
    print("  PII: NVIDIA Nemotron-PII (100K available, taking {})".format(max_samples))
    print("="*60)
    try:
        pii_ds = ds.load_dataset("nvidia/Nemotron-PII", split="train", streaming=True)
        samples = []
        for i, row in enumerate(pii_ds):
            if i >= max_samples:
                break
            text = row.get('text', row.get('source_text', ''))
            if not text:
                continue
            # Extract PII entities for instruction
            entities = row.get('privacy_mask', row.get('entities', []))
            entity_types = list(set(e.get('label', '') for e in entities)) if entities else []
            samples.append({
                "instruction": f"Detect and redact PII from this text. Entity types: {', '.join(entity_types[:5]) if entity_types else 'PII'}",
                "input": text,
                "output": "",  # Model should output filtered/redacted version
                "task": "pii",
                "entity_types": entity_types,
            })
        out_path = os.path.join(PROCESSED_DIR, 'pii_train.json')
        existing = []
        if os.path.exists(out_path):
            with open(out_path) as f:
                existing = json.load(f)
        existing.extend(samples)
        with open(out_path, 'w') as f:
            json.dump(existing, f)
        print(f"  Saved {len(samples)} PII samples. Total: {len(existing)}")
    except Exception as e:
        print(f"  Nemotron-PII failed: {e}")
        print("  Falling back to local synthetic PII data")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download best datasets for SOTA")
    parser.add_argument("--max-chess", type=int, default=100000,
                        help="Max chess puzzle samples")
    parser.add_argument("--max-translate", type=int, default=5000,
                        help="Max samples per language pair")
    parser.add_argument("--max-writing", type=int, default=20000,
                        help="Max story samples")
    parser.add_argument("--max-reasoning", type=int, default=15000,
                        help="Max reasoning samples")
    parser.add_argument("--max-pii", type=int, default=30000,
                        help="Max PII samples")
    parser.add_argument("--skip-chess", action="store_true")
    parser.add_argument("--skip-translate", action="store_true")
    parser.add_argument("--skip-writing", action="store_true")
    parser.add_argument("--skip-reasoning", action="store_true")
    parser.add_argument("--skip-pii", action="store_true")
    args = parser.parse_args()

    if not args.skip_chess:
        download_chess(args.max_chess)
    if not args.skip_translate:
        download_translation(args.max_translate)
    if not args.skip_writing:
        download_writing(args.max_writing)
    if not args.skip_reasoning:
        download_reasoning(args.max_reasoning)
    if not args.skip_pii:
        download_pii(args.max_pii)

    print("\n" + "="*60)
    print("  ALL DATASETS DOWNLOADED")
    print("="*60)
    print(f"  Check: ls -la {PROCESSED_DIR}/")
