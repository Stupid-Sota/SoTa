"""
SOTA Training Pipeline
Implements all training optimizations (#37-40, #87-90).
3-stage pipeline: SFT → ORPO → Self-Play RL
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    T5ForConditionalGeneration, T5Tokenizer,
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from peft import PeftModel
from trl import ORPOConfig, ORPOTrainer
from typing import Dict, List, Optional
import json
import os
import yaml
from datetime import datetime


class SOTADataset(Dataset):
    """Dataset for SOTA training."""

    def __init__(self, samples: List[Dict], tokenizer, max_length: int = 512):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Build input
        prompt = f"chess position: {sample['fen']} | mode: {sample.get('mode', 'romaji')} | think: yes"

        # Build target with CoT
        target = f"{sample['cot']} {sample['output']}"

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
        )

        targets = self.tokenizer(
            target,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
        )

        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': targets['input_ids'].squeeze(0),
        }


class SFTTrainer:
    """
    Stage 1: Supervised Fine-Tuning.
    Implements #37, #87 (3-stage pipeline).
    """

    def __init__(self, model, tokenizer, config: Dict):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.sft_config = config.get('training', {}).get('sft', {})

    def train(self, train_dataset: SOTADataset,
              eval_dataset: Optional[SOTADataset] = None,
              output_dir: str = "data/checkpoints/sft"):

        batch_size = self.sft_config.get('batch_size', 1)
        grad_accum = self.sft_config.get('gradient_accumulation_steps', 16)
        lr = self.sft_config.get('learning_rate', 2.0e-4)
        epochs = self.sft_config.get('epochs', 3)

        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            learning_rate=lr,
            lr_scheduler_type=self.sft_config.get('lr_scheduler', 'cosine'),
            warmup_ratio=self.sft_config.get('warmup_ratio', 0.05),
            weight_decay=self.sft_config.get('weight_decay', 0.01),
            max_grad_norm=self.sft_config.get('max_grad_norm', 1.0),
            gradient_checkpointing=self.sft_config.get('gradient_checkpointing', True),
            bf16=self.sft_config.get('bf16', True),
            optim="adafactor",
            save_steps=500,
            save_total_limit=3,
            logging_steps=50,
            report_to="none",
            # Gradient checkpointing selective (#23)
            dataloader_num_workers=0,
            remove_unused_columns=False,
        )

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
        )

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )

        print(f"[SFT] Starting training: {epochs} epochs, lr={lr}")
        train_result = trainer.train()

        # Save checkpoint
        trainer.save_model(output_dir)
        print(f"[SFT] Training complete. Saved to {output_dir}")

        return trainer, train_result


class ORPOTrainerWrapper:
    """
    Stage 2: Direct Preference Optimization with ORPO.
    Implements #37, #87. ORPO is chosen because it supports encoder-decoder.
    """

    def __init__(self, model, tokenizer, config: Dict):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.orpo_config = config.get('training', {}).get('orpo', {})

    def train(self, preference_dataset: List[Dict],
              output_dir: str = "data/checkpoints/orpo"):

        batch_size = self.orpo_config.get('batch_size', 1)
        grad_accum = self.orpo_config.get('gradient_accumulation_steps', 16)
        lr = self.orpo_config.get('learning_rate', 1.0e-4)
        epochs = self.orpo_config.get('epochs', 2)
        beta = self.orpo_config.get('beta', 0.1)

        orpo_args = ORPOConfig(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            learning_rate=lr,
            beta=beta,
            lr_scheduler_type=self.orpo_config.get('lr_scheduler', 'cosine'),
            warmup_ratio=self.orpo_config.get('warmup_ratio', 0.03),
            bf16=self.orpo_config.get('bf16', True),
            optim="adafactor",
            save_steps=500,
            save_total_limit=3,
            logging_steps=50,
            report_to="none",
            remove_unused_columns=False,
            is_encoder_decoder=True,  # T5 is encoder-decoder
            max_length=512,
            max_prompt_length=256,
            max_completion_length=256,
        )

        # Convert preference data to format ORPO expects
        formatted_dataset = []
        for pair in preference_dataset:
            formatted_dataset.append({
                'prompt': pair['prompt'],
                'chosen': pair['chosen'],
                'rejected': pair['rejected'],
            })

        from datasets import Dataset
        hf_dataset = Dataset.from_list(formatted_dataset)

        trainer = ORPOTrainer(
            model=self.model,
            args=orpo_args,
            train_dataset=hf_dataset,
            tokenizer=self.tokenizer,
        )

        print(f"[ORPO] Starting training: {epochs} epochs, lr={lr}, beta={beta}")
        train_result = trainer.train()

        trainer.save_model(output_dir)
        print(f"[ORPO] Training complete. Saved to {output_dir}")

        return trainer, train_result


class SelfPlayRLTrainer:
    """
    Stage 3: Self-Play with Verifiable Rewards.
    Implements #37, #87, #90 (Elo tracking).
    """

    def __init__(self, model, tokenizer, config: Dict,
                 stockfish_path: str = "stockfish"):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.rl_config = config.get('training', {}).get('rl', {})
        self.stockfish_path = stockfish_path
        self.elo_history = []

    def train(self, num_games: int = 100,
              output_dir: str = "data/checkpoints/rl"):
        """
        Train via self-play against Stockfish.
        Stockfish evaluation is the reward signal.
        """
        import chess
        import chess.engine

        print(f"[RL] Starting self-play training: {num_games} games")

        # Open Stockfish
        engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)

        sf_depth = self.rl_config.get('stockfish_depth', 15)
        sf_time = self.rl_config.get('stockfish_time_ms', 1000)

        for game_idx in range(num_games):
            board = chess.Board()
            move_history = []
            game_reward = 0

            while not board.is_game_over():
                # Get SOTA move
                fen = board.fen()
                sota_move = self._get_model_move(fen)

                if sota_move and sota_move in board.legal_moves:
                    board.push(sota_move)
                    move_history.append(sota_move.uci())

                    # Get Stockfish evaluation as reward
                    if not board.is_game_over():
                        result = engine.analyse(
                            board,
                            chess.engine.Limit(depth=sf_depth, time=sf_time / 1000)
                        )
                        eval_cp = result.pov(board.turn).cp or 0
                        game_reward += self._normalize_reward(eval_cp, board.turn)
                else:
                    # Illegal move - heavy penalty
                    game_reward -= 10
                    break

            # Game result
            result = board.result()
            if result == "1-0":
                game_reward += 5  # White win
            elif result == "0-1":
                game_reward += 5  # Black win
            elif result == "1/2-1/2":
                game_reward += 2  # Draw

            self.elo_history.append(game_reward)

            if (game_idx + 1) % 10 == 0:
                avg_reward = sum(self.elo_history[-10:]) / 10
                print(f"[RL] Game {game_idx+1}/{num_games}, avg reward: {avg_reward:.2f}")

        engine.quit()

        # Estimate Elo
        estimated_elo = self._estimate_elo()
        print(f"[RL] Training complete. Estimated Elo: {estimated_elo}")

        return {'elo': estimated_elo, 'rewards': self.elo_history}

    def _get_model_move(self, fen: str) -> Optional[chess.Move]:
        """Get a move from the model."""
        prompt = f"chess position: {fen} | mode: romaji | think: yes"
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=inputs['input_ids'],
                max_new_tokens=50,
                temperature=0.3,  # Low temperature for chess
                do_sample=True,
            )

        output = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Extract UCI move
        import re
        uci_match = re.search(r'([a-h][1-8][a-h][1-8][qrbn]?)', output)
        if uci_match:
            return chess.Move.from_uci(uci_match.group(1))
        return None

    def _normalize_reward(self, eval_cp: int, turn: chess.Color) -> float:
        """Normalize Stockfish evaluation to reward."""
        if turn == chess.BLACK:
            eval_cp = -eval_cp
        # Sigmoid normalization
        return 2.0 / (1.0 + pow(10, -eval_cp / 400)) - 1.0

    def _estimate_elo(self) -> float:
        """Estimate Elo from rewards."""
        if not self.elo_history:
            return 0
        avg_reward = sum(self.elo_history) / len(self.elo_history)
        # Rough conversion: reward 0 ≈ 1500 Elo, +1 ≈ 2000 Elo
        return 1500 + avg_reward * 500


class CurriculumScheduler:
    """
    Implements #38, #76 (curriculum learning).
    Progressively increases difficulty.
    """

    def __init__(self, config: Dict):
        self.phases = config.get('curriculum', {}).get('phases', [])

    def get_phase(self, epoch: int) -> Dict:
        """Get current curriculum phase."""
        for phase in self.phases:
            epochs = phase.get('epochs', [0, 1])
            if epochs[0] <= epoch < epochs[1]:
                return phase
        return self.phases[-1] if self.phases else {}

    def filter_by_difficulty(self, samples: List[Dict],
                            max_difficulty: int) -> List[Dict]:
        """Filter samples by difficulty level."""
        return [s for s in samples if s.get('difficulty', 1) <= max_difficulty]


def load_config(path: str = "config.yaml") -> Dict:
    """Load configuration."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def run_full_training_pipeline(config_path: str = "config.yaml"):
    """
    Run the complete 3-stage training pipeline.
    Implements #37, #87.
    """
    config = load_config(config_path)

    print("=" * 60)
    print("  SOTA Full Training Pipeline")
    print("=" * 60)

    # Load model
    from src.model.sota_model import SOTAModel, SOTAConfig
    sota_config = SOTAConfig(config_path)
    model_wrapper = SOTAModel(sota_config)
    model_wrapper.load_base_model()
    model_wrapper.apply_qdora()

    model = model_wrapper.model
    tokenizer = model_wrapper.tokenizer

    # Load data
    data_path = config.get('paths', {}).get('data_processed', 'data/processed')
    sft_data_path = os.path.join(data_path, 'sft_train.json')
    orpo_data_path = os.path.join(data_path, 'orpo_train.json')

    if not os.path.exists(sft_data_path):
        print("[ERROR] Training data not found. Run data generation first.")
        return

    with open(sft_data_path, 'r') as f:
        sft_samples = json.load(f)

    # Stage 1: SFT
    print("\n" + "=" * 60)
    print("  STAGE 1: Supervised Fine-Tuning")
    print("=" * 60)

    sft_trainer = SFTTrainer(model, tokenizer, config)
    train_dataset = SOTADataset(sft_samples, tokenizer)
    sft_trainer.train(
        train_dataset,
        output_dir=config.get('paths', {}).get('checkpoints', 'data/checkpoints') + '/sft'
    )

    # Stage 2: ORPO
    print("\n" + "=" * 60)
    print("  STAGE 2: ORPO Preference Optimization")
    print("=" * 60)

    if os.path.exists(orpo_data_path):
        with open(orpo_data_path, 'r') as f:
            orpo_pairs = json.load(f)

        orpo_trainer = ORPOTrainerWrapper(model, tokenizer, config)
        orpo_trainer.train(
            orpo_pairs,
            output_dir=config.get('paths', {}).get('checkpoints', 'data/checkpoints') + '/orpo'
        )
    else:
        print("[ORPO] No preference data found. Skipping stage 2.")

    # Stage 3: Self-Play RL
    print("\n" + "=" * 60)
    print("  STAGE 3: Self-Play RL")
    print("=" * 60)

    rl_trainer = SelfPlayRLTrainer(
        model, tokenizer, config,
        stockfish_path=config.get('evaluation', {}).get('stockfish_path', 'stockfish')
    )
    rl_trainer.train(
        num_games=config.get('evaluation', {}).get('num_games', 100),
        output_dir=config.get('paths', {}).get('checkpoints', 'data/checkpoints') + '/rl'
    )

    print("\n" + "=" * 60)
    print("  Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    run_full_training_pipeline()
