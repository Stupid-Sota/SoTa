#!/usr/bin/env python3
"""
SOTA Dataset Downloader v2 — downloads REAL datasets for all 5 tasks.
Translation: Tatoeba/HF + OPUS-100 direct
Chess: Lichess puzzles + SFT positions
Writing: SimpleStories
Reasoning: GSM8K-CoT + Open-CoT
PII: Nemotron-PII

Also generates ORPO preference pairs from chess data.
"""
import json
import os
import random
import subprocess
import sys
import glob
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
RAW_DIR = os.path.join(DATA_DIR, 'raw')

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)

random.seed(42)


def log(msg):
    print(f"[DATA] {msg}")


# ============================================================
# 1. TRANSLATION — HF Datasets (working sources)
# ============================================================
def download_translation_hf(max_per_pair=50000):
    try:
        from datasets import load_dataset
    except ImportError:
        log("datasets not installed, skipping HF translation")
        return

    all_samples = []
    
    # Source 1: kirchik47/english-spanish-translator (119k en-es, 6MB)
    try:
        log("Loading kirchik47/english-spanish-translator (119k en-es)...")
        ds = load_dataset("kirchik47/english-spanish-translator", split="train", streaming=True)
        count = 0
        for row in ds:
            if count >= max_per_pair:
                break
            all_samples.append({
                "instruction": "Translate from English to Spanish.",
                "input": row["sentences_en"],
                "output": row["sentences_es"],
                "task": "translate",
                "lang_pair": "en-es",
            })
            count += 1
        log(f"  en-es: {count} samples")
    except Exception as e:
        log(f"  en-es failed: {e}")

    # Source 2: Tatoeba bitext mining via HF (1k per pair, small & fast)
    tatoeba_configs = {
        "en-fr": "fra_eng", "en-de": "deu_eng", "en-it": "ita_eng",
        "en-pt": "por_eng", "en-ru": "rus_eng", "en-nl": "nld_eng",
    }
    for pair, config in tatoeba_configs.items():
        try:
            log(f"Loading Tatoeba {pair}...")
            ds = load_dataset("loicmagne/tatoeba-bitext-mining", config, split="test", streaming=True)
            count = 0
            for row in ds:
                if count >= min(max_per_pair, 1000):
                    break
                inp = row.get("sentence1", "")
                out = row.get("sentence2", "")
                if inp and out:
                    all_samples.append({
                        "instruction": f"Translate from {pair.split('-')[0]} to {pair.split('-')[1]}.",
                        "input": inp,
                        "output": out,
                        "task": "translate",
                        "lang_pair": pair,
                    })
                    count += 1
            log(f"  {pair}: {count} samples")
        except Exception as e:
            log(f"  {pair} failed: {e}")

    # Merge with existing
    out_path = os.path.join(PROCESSED_DIR, 'translate_train.json')
    existing = []
    if os.path.exists(out_path):
        with open(out_path) as f:
            existing = json.load(f)
    
    existing_ids = set((s.get('input', '')[:50], s.get('lang_pair', '')) for s in existing)
    new_real = [s for s in all_samples if (s.get('input', '')[:50], s.get('lang_pair', '')) not in existing_ids]
    existing.extend(new_real)
    
    with open(out_path, 'w') as f:
        json.dump(existing, f)
    log(f"Translation total: {len(existing)} ({len(new_real)} new real samples)")


# ============================================================
# 2. GENERATE ORPO PREFERENCE PAIRS FROM CHESS
# ============================================================
def generate_orpo_pairs(n_pairs=5000):
    chess_path = os.path.join(PROCESSED_DIR, 'chess_train.json')
    if not os.path.exists(chess_path):
        log("No chess data found, cannot generate ORPO pairs")
        return
    
    with open(chess_path) as f:
        chess_data = json.load(f)
    
    log(f"Generating {n_pairs} ORPO pairs from {len(chess_data)} chess samples...")
    pairs = []
    
    by_input = {}
    for s in chess_data:
        key = s.get('input', '')
        if key not in by_input:
            by_input[key] = []
        by_input[key].append(s)
    
    for fen, samples in by_input.items():
        if len(pairs) >= n_pairs:
            break
        if len(samples) < 2:
            continue
        sorted_s = sorted(samples, key=lambda x: x.get('rating', x.get('difficulty', 5)))
        chosen = sorted_s[-1]
        rejected = sorted_s[0]
        if chosen.get('output') and rejected.get('output') and chosen['output'] != rejected['output']:
            instr = chosen.get('instruction', "Analyze this chess position.")
            pairs.append({
                "prompt": f"{instr}\n{chosen.get('input', '')}",
                "chosen": chosen['output'],
                "rejected": rejected['output'],
            })
    
    # Add synthetic pairs
    remaining = n_pairs - len(pairs)
    if remaining > 0:
        try:
            from scripts.generate_all_data import generate_synthetic_chess
            extra = generate_synthetic_chess(max(10, remaining // 5))
            by_inp2 = {}
            for s in extra:
                k = s.get('input', '')
                by_inp2.setdefault(k, []).append(s)
            for k, samples in by_inp2.items():
                if len(pairs) >= n_pairs:
                    break
                if len(samples) >= 2:
                    pairs.append({
                        "prompt": f"{samples[0].get('instruction', '')}\n{k}",
                        "chosen": samples[0]['output'],
                        "rejected": samples[-1]['output'],
                    })
        except Exception as e:
            log(f"Synthetic pairs failed: {e}")
    
    out_path = os.path.join(PROCESSED_DIR, 'orpo_train.json')
    with open(out_path, 'w') as f:
        json.dump(pairs[:n_pairs], f, indent=2)
    log(f"ORPO pairs saved: {len(pairs[:n_pairs])} -> {out_path}")


# ============================================================
# 3. GENERATE EVAL DATA
# ============================================================
def generate_eval_data():
    eval_data = []
    for task in ['chess', 'translate', 'write', 'predict', 'pii']:
        path = os.path.join(PROCESSED_DIR, f'{task}_train.json')
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            samples = random.sample(data, min(200, len(data)))
            for s in samples:
                s['split'] = 'eval'
            eval_data.extend(samples)
    
    out_path = os.path.join(PROCESSED_DIR, 'eval_data.json')
    with open(out_path, 'w') as f:
        json.dump(eval_data, f, indent=2)
    log(f"Eval data: {len(eval_data)} samples -> {out_path}")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-translate", type=int, default=50000)
    parser.add_argument("--orpo-pairs", type=int, default=5000)
    parser.add_argument("--skip-translate", action="store_true")
    parser.add_argument("--skip-orpo", action="store_true")
    args = parser.parse_args()

    if not args.skip_translate:
        download_translation_hf(args.max_translate)
    
    if not args.skip_orpo:
        generate_orpo_pairs(args.orpo_pairs)
    
    generate_eval_data()
    
    # Final summary
    total = 0
    tasks = {}
    for fname in glob.glob(os.path.join(PROCESSED_DIR, '*_train.json')):
        if 'all_tasks' in fname:
            continue
        with open(fname) as f:
            data = json.load(f)
        total += len(data)
        log(f"  {os.path.basename(fname)}: {len(data)}")
        for s in data:
            t = s.get('task', 'unknown')
            tasks[t] = tasks.get(t, 0) + 1
    
    log(f"TOTAL: {total} samples across {len(tasks)} tasks")
    for t, c in sorted(tasks.items()):
        log(f"  {t}: {c}")
