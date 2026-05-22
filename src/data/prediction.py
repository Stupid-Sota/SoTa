"""
Sequence Prediction Data Pipeline.
Generates arithmetic, pattern, letter, and chess move sequences.
Implements: #120-124 (Prediction rewards), predict expert training data.
"""

import random
import math
import json
import os
import logging
from typing import Dict, List, Optional, Tuple
from torch.utils.data import Dataset


SEQUENCE_TYPES = ['arithmetic', 'geometric', 'fibonacci', 'pattern', 'letters', 'chess_moves']

PATTERNS = {
    'even': lambda n: [i * 2 for i in range(1, n + 1)],
    'odd': lambda n: [i * 2 - 1 for i in range(1, n + 1)],
    'squares': lambda n: [i ** 2 for i in range(1, n + 1)],
    'cubes': lambda n: [i ** 3 for i in range(1, n + 1)],
    'powers_of_2': lambda n: [2 ** i for i in range(1, n + 1)],
    'triangular': lambda n: [i * (i + 1) // 2 for i in range(1, n + 1)],
    'factorial': lambda n: [math.factorial(i) for i in range(1, min(n + 1, 10))],
}

LETTER_SEQUENCES = [
    ('vowels', ['a', 'e', 'i', 'o', 'u']),
    ('consonants', ['b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm']),
    ('alphabet_step_2', ['a', 'c', 'e', 'g', 'i', 'k', 'm']),
    ('alphabet_step_3', ['a', 'd', 'g', 'j', 'm', 'p', 's']),
    ('reverse', ['z', 'y', 'x', 'w', 'v', 'u', 't']),
]

CHESS_SEQUENCES = [
    ['e4', 'e5', 'Nf3', 'Nc6', 'Bb5', 'a6', 'Ba4', 'Nf6'],
    ['d4', 'd5', 'c4', 'e6', 'Nc3', 'Nf6', 'Bg5', 'Be7'],
    ['e4', 'c5', 'Nf3', 'd6', 'd4', 'cxd4', 'Nxd4', 'Nf6'],
    ['e4', 'e6', 'd4', 'd5', 'Nc3', 'Bb4', 'e5', 'Ne7'],
    ['d4', 'Nf6', 'c4', 'g6', 'Nc3', 'Bg7', 'e4', 'd6'],
    ['e4', 'e5', 'Nf3', 'Nc6', 'Bc4', 'Bc5', 'c3', 'Nf6'],
    ['e4', 'c5', 'Nf3', 'Nc6', 'd4', 'cxd4', 'Nxd4', 'Nf6'],
    ['e4', 'e5', 'Nf3', 'd6', 'd4', 'exd4', 'Nxd4', 'Nf6'],
]


def generate_sequence(seq_type: str, length: int = 10) -> Tuple[List, str]:
    if seq_type == 'arithmetic':
        a = random.randint(1, 5)
        d = random.randint(1, 10)
        seq = [a + i * d for i in range(length)]
        return seq, f"Arithmetic sequence with diff={d}"

    elif seq_type == 'geometric':
        a = random.randint(1, 3)
        r = random.randint(2, 5)
        seq = [a * (r ** i) for i in range(length)]
        return seq, f"Geometric sequence with ratio={r}"

    elif seq_type == 'fibonacci':
        a, b = random.randint(1, 5), random.randint(1, 5)
        seq = [a, b]
        for _ in range(length - 2):
            seq.append(seq[-1] + seq[-2])
        return seq[:length], "Fibonacci-like sequence"

    elif seq_type == 'pattern':
        name = random.choice(list(PATTERNS.keys()))
        try:
            seq = PATTERNS[name](length)
            return seq, f"{name} pattern"
        except:
            seq = [i * 2 for i in range(1, length + 1)]
            return seq, "even pattern"

    elif seq_type == 'letters':
        name, seq = random.choice(LETTER_SEQUENCES)
        seq = seq[:length]
        return seq, f"letter pattern: {name}"

    elif seq_type == 'chess_moves':
        seq = random.choice(CHESS_SEQUENCES)
        seq = seq[:min(length, len(seq))]
        return seq, "chess opening moves"

    seq = [i * 2 for i in range(1, length + 1)]
    return seq, "default pattern"


class PredictionDataset(Dataset):
    """Dataset for sequence completion/prediction tasks."""

    def __init__(self, tokenizer, max_length: int = 128, size: int = 400):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self._generate_samples(size)

    def _generate_samples(self, size: int) -> List[Dict]:
        samples = []
        for _ in range(size):
            seq_type = random.choice(SEQUENCE_TYPES)
            length = random.randint(6, 12)
            seq, desc = generate_sequence(seq_type, length)

            split_point = max(2, int(length * 0.7))
            visible = seq[:split_point]
            to_predict = seq[split_point:]

            visible_str = ' '.join(str(x) for x in visible)
            target_str = ' '.join(str(x) for x in to_predict)

            prompt = f"complete: {visible_str}"
            target = target_str

            samples.append({
                'prompt': prompt,
                'target': target,
                'seq_type': seq_type,
                'visible': visible_str,
                'hidden': target_str,
            })

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        inputs = self.tokenizer(
            sample['prompt'], max_length=self.max_length, truncation=True,
            padding=False, return_tensors='pt',
        )
        targets = self.tokenizer(
            sample['target'], max_length=self.max_length, truncation=True,
            padding=False, return_tensors='pt',
        )
        if inputs['input_ids'].size(1) >= self.max_length:
            logging.getLogger('sota').warning(
                f"Truncated [predict/prompt] from >={self.max_length} to {self.max_length} tokens")
        if targets['input_ids'].size(1) >= self.max_length:
            logging.getLogger('sota').warning(
                f"Truncated [predict/target] from >={self.max_length} to {self.max_length} tokens")
        labels = targets['input_ids'].squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': labels,
            'task': 'predict',
        }


def generate_prediction_data(tokenizer, num_samples: int = 400,
                              output_path: str = None) -> List[Dict]:
    dataset = PredictionDataset(tokenizer, size=num_samples)
    data = []
    for sample in dataset.samples:
        data.append(sample)
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"[Prediction] Saved {len(data)} samples to {output_path}")
    return data
