"""
Generate training data for all 5 SOTA tasks.
Usage: python scripts/generate_all_data.py [--samples N]
"""

import argparse
import json
import os
import random
import sys
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

random.seed(42)


# ============================================================
# 1. Translation Data (synthetic parallel sentences)
# ============================================================
EN_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming the world.",
    "The weather today is sunny and warm.",
    "I enjoy reading books and learning new things.",
    "She opened the door and walked into the room.",
    "The cat sat on the mat near the fireplace.",
    "They went to the market to buy fresh vegetables.",
    "Technology has changed the way we communicate.",
    "The children played in the park until sunset.",
    "He wrote a letter to his friend in another country.",
    "The river flows gently through the valley.",
    "Music brings joy to people of all ages.",
    "The scientist conducted experiments in the laboratory.",
    "We need to protect the environment for future generations.",
    "The train arrived at the station exactly on time.",
    "She studied hard for the examination and passed with honors.",
    "The old castle stood on top of the hill.",
    "Birds fly south for the winter every year.",
    "The restaurant served delicious food from various cuisines.",
    "He climbed the mountain and enjoyed the breathtaking view.",
]

def generate_translation_samples(n: int = 2000) -> list:
    samples = []
    pairs = [
        ("en", "es"), ("en", "fr"), ("en", "de"), ("en", "it"),
        ("en", "pt"), ("en", "nl"), ("en", "ru"), ("en", "ja"),
    ]
    for i in range(n):
        src = random.choice(EN_SENTENCES)
        tgt_lang = random.choice(pairs)
        # Simulated target: simple word-level transformation
        words = src.split()
        random.shuffle(words)
        tgt = " ".join(words).lower()
        samples.append({
            "instruction": f"Translate the following text from {tgt_lang[0]} to {tgt_lang[1]}.",
            "input": src,
            "output": tgt,
            "task": "translate",
            "lang_pair": f"{tgt_lang[0]}-{tgt_lang[1]}",
        })
    return samples


# ============================================================
# 2. Writing Data (prompt + template-based completion)
# ============================================================
WRITING_PROMPTS = [
    "Write a short story about a dragon.",
    "Write a poem about the ocean.",
    "Write a paragraph describing a sunset.",
    "Write a story about a robot learning to feel emotions.",
    "Write a letter to your future self.",
    "Describe a magical forest in detail.",
    "Write a short mystery story.",
    "Compose a haiku about winter.",
    "Write about a day in the life of a time traveler.",
    "Describe the perfect vacation destination.",
]

def generate_writing_samples(n: int = 1500) -> list:
    templates = [
        "Once upon a time, in a land far away, there lived a {adj} {noun}. "
        "Every day, they would {verb} through the {place}. "
        "One day, something {adj2} happened: {event}. "
        "From that day forward, everything changed.",
        "The {adj} {noun} stood at the edge of the {place}. "
        "The wind whispered secrets of {concept}. "
        "In that moment, they realized that {realization}. "
        "And so began the greatest adventure of all.",
    ]
    adj_pool = ["brave", "mysterious", "ancient", "golden", "silent", "powerful", "gentle", "wise"]
    noun_pool = ["warrior", "dragon", "wizard", "forest", "city", "ocean", "mountain", "star"]
    verb_pool = ["wander", "dance", "sing", "dream", "explore", "soar", "dive", "climb"]
    place_pool = ["mountains", "desert", "ocean", "forest", "city", "valley", "sky", "cave"]
    concept_pool = ["love", "time", "courage", "destiny", "knowledge", "peace", "power", "freedom"]

    samples = []
    for i in range(n):
        prompt = random.choice(WRITING_PROMPTS)
        template = random.choice(templates)
        text = template.format(
            adj=random.choice(adj_pool),
            adj2=random.choice(adj_pool),
            noun=random.choice(noun_pool),
            verb=random.choice(verb_pool),
            place=random.choice(place_pool),
            concept=random.choice(concept_pool),
            event="a {adj} {noun} appeared from the shadows".format(
                adj=random.choice(adj_pool), noun=random.choice(noun_pool)
            ),
            realization="everything happens for a reason",
        )
        samples.append({
            "instruction": prompt,
            "input": "",
            "output": text,
            "task": "write",
        })
    return samples


# ============================================================
# 3. Prediction Data (numeric sequences)
# ============================================================
def generate_prediction_samples(n: int = 1000) -> list:
    samples = []
    for i in range(n):
        seq_type = random.choice(["linear", "fibonacci", "arithmetic", "geometric", "pattern"])
        if seq_type == "linear":
            a = random.uniform(0.5, 5.0)
            b = random.uniform(1.0, 10.0)
            length = random.randint(5, 10)
            seq = [a * x + b for x in range(length)]
            target = a * length + b
        elif seq_type == "fibonacci":
            a, b = random.randint(1, 5), random.randint(1, 5)
            seq = [a, b]
            for _ in range(random.randint(3, 8)):
                seq.append(seq[-1] + seq[-2])
            target = seq[-1]
        elif seq_type == "arithmetic":
            start = random.uniform(1, 20)
            step = random.uniform(0.5, 5.0)
            length = random.randint(4, 8)
            seq = [start + step * x for x in range(length)]
            target = seq[-1]
        elif seq_type == "geometric":
            start = random.uniform(1, 10)
            ratio = random.uniform(1.1, 3.0)
            seq = [start * (ratio ** x) for x in range(4)]
            target = seq[-1]
        else:  # pattern
            pattern = random.choice([
                [1, 2, 3, 4, 5],
                [2, 4, 6, 8, 10],
                [1, 1, 2, 3, 5],
                [1, 4, 9, 16, 25],
                [3, 6, 9, 12, 15],
            ])
            target = pattern[-1] + (pattern[-1] - pattern[-2])

        seq_str = " ".join(f"{x:.2f}" if isinstance(x, float) else str(x) for x in seq)
        target_str = f"{target:.2f}" if isinstance(target, float) else str(target)
        samples.append({
            "instruction": "Predict the next value in this sequence.",
            "input": seq_str,
            "output": target_str,
            "task": "predict",
            "seq_type": seq_type,
        })
    return samples


# ============================================================
# 4. PII Data
# ============================================================
PII_TEMPLATES = [
    "My email address is {email} and my phone number is {phone}.",
    "You can reach me at {phone} or send an email to {email}.",
    "My name is {name}, my SSN is {ssn}, and I live at {address}.",
    "Contact: {email}, Phone: {phone}, Address: {address}.",
    "Dear {name}, your account {email} has been locked.",
    "Please call {phone} and reference ticket #{ticket}.",
    "Credit card {cc} was used for the purchase.",
    "My passport number is {passport}.",
    "The patient {name} has ID {id_num}.",
    "Server IP: {ip}, username: {user}.",
]

NAMES = ["John Smith", "Jane Doe", "Alice Johnson", "Bob Williams", "Carol Brown",
         "David Miller", "Eva Garcia", "Frank Wilson", "Grace Lee", "Henry Taylor"]
DOMAINS = ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "company.org"]
SSNS = [f"{random.randint(100,999)}-{random.randint(10,99)}-{random.randint(1000,9999)}" for _ in range(100)]
CCS = [f"{random.randint(4000,4999)}-{random.randint(1000,9999)}-{random.randint(1000,9999)}-{random.randint(1000,9999)}" for _ in range(100)]


def generate_pii_samples(n: int = 500) -> list:
    samples = []
    for i in range(n):
        template = random.choice(PII_TEMPLATES)
        name = random.choice(NAMES)
        email = f"{name.lower().replace(' ', '.')}{random.randint(1,99)}@{random.choice(DOMAINS)}"
        phone = f"+1-{random.randint(200,999)}-{random.randint(100,999)}-{random.randint(1000,9999)}"
        ssn = random.choice(SSNS)
        cc = random.choice(CCS)
        address = f"{random.randint(100,9999)} {random.choice(['Main', 'Oak', 'Elm', 'Park'])} St, {random.choice(['Springfield', 'Riverside', 'Fairview', 'Madison'])}, {random.choice(['CA', 'NY', 'TX', 'FL'])} {random.randint(10000,99999)}"
        ticket = f"TKT-{random.randint(100000,999999)}"
        passport = f"{random.choice(['AB', 'CD', 'EF', 'GH'])}{random.randint(100000,999999)}"
        id_num = f"ID-{random.randint(10000,99999)}"
        ip = f"{random.randint(10,223)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"
        user = name.lower().replace(' ', '_')

        text = template.format(
            name=name, email=email, phone=phone, ssn=ssn,
            cc=cc, address=address, ticket=ticket,
            passport=passport, id_num=id_num, ip=ip, user=user,
        )
        samples.append({
            "instruction": "Detect and filter PII from this text.",
            "input": text,
            "output": "",
            "task": "pii",
        })
    return samples


# ============================================================
# 5. Chess Data (via Stockfish)
# ============================================================
def generate_chess_samples(n: int = 2000, existing: list = None) -> list:
    from src.data.generate_data import DatasetGenerator

    stockfish_path = "stockfish"
    import shutil
    if not shutil.which(stockfish_path):
        print("[WARNING] Stockfish not found. Using synthetic chess data.")
        return generate_synthetic_chess(n)

    generator = DatasetGenerator(stockfish_path=stockfish_path)

    opening_fens = [
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1",
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",
        "rnbqkb1r/pppppppp/5n2/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2",
        "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    ]
    midgame_fens = [
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        "r1bqk2r/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQK2R w KQkq - 0 6",
        "r2qk2r/ppp1bppp/2np1n2/4p3/2BPP1b1/2N2N2/PPP2PPP/R1BQK2R w KQkq - 0 8",
    ]

    all_fens = opening_fens + midgame_fens
    samples = []
    rounds = max(1, n // len(all_fens))
    for r in range(rounds):
        for fen in all_fens:
            try:
                result = generator.generate_from_fens([fen], cot_depth="medium")
                s = result[0]
                for mode_name, mode_output in s.mode_outputs.items():
                    samples.append({
                        "instruction": f"Analyze this chess position and provide the best move in {mode_name} mode.",
                        "input": s.fen,
                        "output": f"[STH] {s.cot} [ETH] {mode_output}",
                        "mode": mode_name,
                        "best_move": s.best_move,
                        "difficulty": s.difficulty,
                        "game_phase": s.game_phase,
                        "task": "chess",
                    })
            except Exception as e:
                print(f"[Chess] Error on {fen}: {e}")
                continue
        print(f"[Chess] Round {r+1}/{rounds}: {len(samples)} samples so far")
        if len(samples) >= n:
            break

    # Deduplicate
    seen = set()
    unique = []
    for s in samples:
        key = (s["input"], s["mode"], s.get("best_move", ""))
        if key not in seen:
            seen.add(key)
            unique.append(s)
    return unique[:n]


def generate_synthetic_chess(n: int) -> list:
    moves = ["e4", "d4", "Nf3", "c4", "e5", "d5", "Nc6", "Nf6", "Bc4", "Bc5",
             "O-O", "O-O-O", "Re1", "Rd1", "Qe2", "Qd2", "f4", "g3", "c5", "b6"]
    phases = ["opening", "midgame", "endgame"]
    samples = []
    for _ in range(n):
        fen_parts = ["rnbqkbnr", "pppppppp", "8", "8", "8", "8", "PPPPPPPP", "RNBQKBNR"]
        fen = f"{random.choice(fen_parts)}/{random.choice(fen_parts)}/8/8/8/8/{random.choice(fen_parts)}/{random.choice(fen_parts)} w KQkq - 0 1"
        move = random.choice(moves)
        phase = random.choice(phases)
        for mode in ["romaji", "cervantes", "yoda", "kansai", "socratic"]:
            samples.append({
                "instruction": f"Analyze this chess position and provide the best move in {mode} mode.",
                "input": fen,
                "output": f"[STH] {phase} position. [GUESSING] Best: {move} [ETH] {move}",
                "mode": mode,
                "best_move": move,
                "difficulty": random.randint(1, 10),
                "game_phase": phase,
                "task": "chess",
            })
    return samples


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Generate SOTA training data for all 5 tasks")
    parser.add_argument("--samples", type=int, default=5000,
                        help="Total samples to generate (distributed across tasks)")
    parser.add_argument("--output-dir", default="data/processed",
                        help="Output directory")
    parser.add_argument("--chess-only", action="store_true",
                        help="Only generate chess data")
    parser.add_argument("--merge", action="store_true",
                        help="Merge all tasks into single training file")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    targets = {
        "chess": 5000,
        "translate": 2000,
        "write": 1500,
        "predict": 1000,
        "pii": 500,
    }

    all_samples = []

    for task, target in targets.items():
        if args.chess_only and task != "chess":
            continue
        print(f"\n{'='*60}")
        print(f"  Generating {task}: {target} samples")
        print(f"{'='*60}")
        if task == "chess":
            existing = []
            chess_path = os.path.join(args.output_dir, "sft_train.json")
            if os.path.exists(chess_path):
                with open(chess_path) as f:
                    existing = json.load(f)
                print(f"[{task}] Loaded {len(existing)} existing samples")
            samples = generate_chess_samples(target, existing=existing)
        elif task == "translate":
            samples = generate_translation_samples(target)
        elif task == "write":
            samples = generate_writing_samples(target)
        elif task == "predict":
            samples = generate_prediction_samples(target)
        elif task == "pii":
            samples = generate_pii_samples(target)
        else:
            continue

        out_path = os.path.join(args.output_dir, f"{task}_train.json")
        with open(out_path, 'w') as f:
            json.dump(samples, f, indent=2)
        print(f"[{task}] Saved {len(samples)} samples to {out_path}")
        all_samples.extend(samples)

    if args.merge and not args.chess_only:
        merged_path = os.path.join(args.output_dir, "all_tasks_train.json")
        with open(merged_path, 'w') as f:
            json.dump(all_samples, f, indent=2)
        print(f"\n[Merge] Saved {len(all_samples)} total samples to {merged_path}")

    print(f"\n{'='*60}")
    print(f"  Generation complete: {len(all_samples)} total samples")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
