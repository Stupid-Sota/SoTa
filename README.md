<p align="center">
  <img src="https://raw.githubusercontent.com/Stupid-Sota/SoTa/main/.github/assets/banner.svg" alt="SOTA Banner" width="680">
</p>

<p align="center">
  <a href="https://huggingface.co/google/flan-t5-base"><img src="https://img.shields.io/badge/Model-flan--t5--base-6366F1?style=flat&logo=huggingface&logoColor=white&labelColor=0A0A0A" alt="Model"></a>
  <a href="#"><img src="https://img.shields.io/badge/Params-250M-d946ef?style=flat&labelColor=0A0A0A" alt="Parameters"></a>
  <a href="#memory-budget"><img src="https://img.shields.io/badge/RAM-400MB-22c55e?style=flat&labelColor=0A0A0A" alt="Memory"></a>
  <a href="#"><img src="https://img.shields.io/badge/Modes-7-a855f7?style=flat&labelColor=0A0A0A" alt="Modes"></a>
  <a href="#"><img src="https://img.shields.io/badge/PEFT-QDoRA-6366F1?style=flat&labelColor=0A0A0A" alt="PEFT"></a>
  <a href="#"><img src="https://img.shields.io/badge/Quantization-4--bit_NF4-22c55e?style=flat&labelColor=0A0A0A" alt="Quantization"></a>
  <a href="#"><img src="https://img.shields.io/badge/License-MIT-555?style=flat&labelColor=0A0A0A" alt="License"></a>
</p>

<p align="center">
  <a href="#overview">Overview</a> &middot;
  <a href="#output-modes">Modes</a> &middot;
  <a href="#optimizations">Optimizations</a> &middot;
  <a href="#quick-start">Quick Start</a> &middot;
  <a href="#architecture">Architecture</a> &middot;
  <a href="#training">Training</a> &middot;
  <a href="#special-tokens">Tokens</a>
</p>

<br>

<p align="center">
  <img src="https://raw.githubusercontent.com/Stupid-Sota/SoTa/main/.github/assets/wave.svg" alt="Wave Divider" width="600">
</p>

# SOTA — Stupid Omega Transformers Agent

<p align="center">
  <em>The Claude Opus of Chess Agents</em>
</p>

<p align="center">
  A specialized chess-playing AI built on <strong>flan-t5-base</strong> (~250M params)<br>
  with <strong>QDoRA adapters</strong>, structured <strong>Chain-of-Thought reasoning</strong>,<br>
  and <strong>7 stylized output modes</strong> -- designed to run on Android/Termux.
</p>

<br>

<p align="center">
  <img src="https://raw.githubusercontent.com/Stupid-Sota/SoTa/main/.github/assets/divider.svg" alt="Divider" width="600">
</p>

## Overview

SOTA is a fine-tuned language model that plays chess by generating move descriptions in 7 different stylistic modes, complete with structured reasoning traces. Built entirely with free tools and designed to run on a phone.

| | |
|---|---|
| **Base Model** | google/flan-t5-base (250M params) |
| **Fine-tuning** | QDoRA -- QLoRA + DoRA (~15M trainable params) |
| **Quantization** | 4-bit NF4 with double quant (~156 MB) |
| **Inference** | ~300-400 MB RAM |
| **Platform** | Android (Termux), Linux, macOS |
| **Output Modes** | 7 stylized formats |
| **Special Tokens** | 15 functional tokens for CoT |
| **Optimizations** | 90 across 6 categories |
| **Training** | 3-stage: SFT → ORPO → Self-Play RL |

<br>

<p align="center">
  <img src="https://raw.githubusercontent.com/Stupid-Sota/SoTa/main/.github/assets/wave.svg" alt="Wave Divider" width="600">
</p>

## Output Modes

SOTA renders every chess move in one of 7 distinct styles:

| Mode | Trigger | Description | Example Output |
|:-----|:--------|:-------------|:---------------|
| **Romaji** | `romaji` | Invented romaji language | `eichi nana eichi go` |
| **Cervantes** | `cervantes` | Archaic literary Spanish | `Mueve el caballo... con gran maestria` |
| **Python** | `python` | Valid executable code | `board.push_san("Nf3")` |
| **Musical** | `musical` | Musical note notation | `E3... ♪ → G3... ♪` |
| **Morse** | `morse` | Morse code | `. / .-.. / -- / --..` |
| **Neural** | `neural` | Fake neural debug | `[LAYER 3] activating neuron #7712` |
| **Patata** | `patata` | Just says "patata" | `patata` |

Mode is selected via prompt or randomly assigned. Each mode has its own specialized output head.

<br>

<p align="center">
  <img src="https://raw.githubusercontent.com/Stupid-Sota/SoTa/main/.github/assets/divider.svg" alt="Divider" width="600">
</p>

## Optimizations

90 optimizations across 6 categories, numbered as documented in `config.yaml`:

| Category | Range | Key Techniques |
|:---------|:-----:|:---------------|
| **Reasoning & CoT** | 1-18 | Functional tokens, ExGRPO, MCTS, NoT, SGR, SaGoT |
| **Vocabulary** | 9-28 | Self-distillation, glitch token replacement, MCL |
| **Architecture** | 17-36 | QDoRA, hybrid ZO+FO, speculative decoding, MQA |
| **Data** | 25-44 | Master distillation, adversarial, curriculum, synthetic |
| **Inference** | 31-50 | QuantSpec, SpecExec, streaming, parallel modes |
| **Training** | 37-54 | 3-stage pipeline, anti-forgetting, adapter ensemble |

Each optimization is documented with rationale and implementation status in `config.yaml`.

<br>

<p align="center">
  <img src="https://raw.githubusercontent.com/Stupid-Sota/SoTa/main/.github/assets/wave.svg" alt="Wave Divider" width="600">
</p>

## Architecture

<p align="center">
  <img src="https://raw.githubusercontent.com/Stupid-Sota/SoTa/main/.github/assets/architecture.svg" alt="SOTA Architecture" width="760">
</p>

### Components

| Component | File | Lines | Responsibility |
|:----------|:-----|:-----:|:---------------|
| `SOTAConfig` | `src/model/sota_model.py` | 50 | YAML config loading, validation |
| `SOTAModel` | `src/model/sota_model.py` | 248 | Model loading, QDoRA, output heads, CoT generation |
| `StockfishEngine` | `src/data/generate_data.py` | 120 | Stockfish UCI wrapper, multi-PV analysis |
| `CoTGenerator` | `src/data/generate_data.py` | 200 | 3-depth CoT traces, confidence, tactical patterns |
| `ModeRenderer` | `src/data/generate_data.py` | 180 | 7 mode output formatters |
| `DatasetGenerator` | `src/data/generate_data.py` | 201 | Training data orchestration, ORPO pairs |
| `SOTADataset` | `src/training/train.py` | 60 | PyTorch Dataset wrapper |
| `SFTTrainer` | `src/training/train.py` | 100 | Stage 1 supervised fine-tuning |
| `ORPOTrainerWrapper` | `src/training/train.py` | 100 | Stage 2 preference optimization |
| `SelfPlayRLTrainer` | `src/training/train.py` | 120 | Stage 3 RL against Stockfish |
| `CurriculumScheduler` | `src/training/train.py` | 57 | 5-phase curriculum learning |
| `SOTAInference` | `src/inference/engine.py` | 264 | Loading, inference, interactive play, batch eval |

<br>

<p align="center">
  <img src="https://raw.githubusercontent.com/Stupid-Sota/SoTa/main/.github/assets/divider.svg" alt="Divider" width="600">
</p>

## Quick Start

### Install

```bash
# Clone and enter
cd ~/Projects/sota

# Install Python deps
pip install -r requirements.txt

# Install Stockfish (for data generation & self-play)
pkg install stockfish  # Termux
# brew install stockfish  # macOS
# apt install stockfish   # Linux
```

### Commands

| Command | Description |
|:--------|:------------|
| `python main.py generate --samples 1000 --depth medium` | Generate training data |
| `python main.py train` | Run 3-stage training pipeline |
| `python main.py play` | Interactive chess (human vs SOTA) |
| `python main.py eval <FEN> --mode romaji` | Evaluate position |
| `python main.py info` | Show model and config info |

### Evaluate a Position

```bash
python main.py eval "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1" --mode romaji
```

### Play Interactive

```bash
python main.py play
# You: e4
# SOTA: eichi go eichi go... eichi nana eichi go
# ...
```

<br>

<p align="center">
  <img src="https://raw.githubusercontent.com/Stupid-Sota/SoTa/main/.github/assets/wave.svg" alt="Wave Divider" width="600">
</p>

## Training Pipeline

SOTA is trained in 3 stages, all designed to run on CPU (Termux/Android):

### Stage 1: Supervised Fine-Tuning (SFT)

| | |
|:---|:---|
| **Goal** | Learn CoT format, move prediction, mode rendering |
| **Data** | FEN + CoT trace + mode-specific output |
| **Loss** | Cross-entropy (next token prediction) |
| **Duration** | 2-4 days on CPU |
| **LR** | 2e-4, cosine scheduler, warmup 10% |

### Stage 2: ORPO Preference Optimization

| | |
|:---|:---|
| **Goal** | Prefer better moves, avoid mode contamination |
| **Data** | Chosen vs rejected move pairs |
| **Method** | ORPO (Odds Ratio Preference Optimization) |
| **Duration** | 2-3 days on CPU |
| **Beta** | 0.1 |

### Stage 3: Self-Play RL

| | |
|:---|:---|
| **Goal** | Auto-improve via playing against oneself + Stockfish |
| **Reward** | Stockfish centipawn evaluation |
| **Method** | PPO with KL penalty |
| **Duration** | 3-7 days on CPU |
| **Self-play games** | 1000+ |

### Curriculum Learning

5 difficulty phases across 8 epochs:

| Phase | Positions | Epochs |
|:------|:---------:|:------:|
| Endgame basic | 5,000 | 2 |
| Endgame complex | 5,000 | 2 |
| Middlegame tactical | 8,000 | 2 |
| Opening theory | 8,000 | 1 |
| Adversarial (vs Stockfish) | 10,000+ | 1 |

<br>

<p align="center">
  <img src="https://raw.githubusercontent.com/Stupid-Sota/SoTa/main/.github/assets/divider.svg" alt="Divider" width="600">
</p>

## Memory Budget

| Component | Training | Inference |
|:----------|:--------:|:---------:|
| Model 4-bit NF4 | ~156 MB | ~156 MB |
| QDoRA adapters | ~15-25 MB | ~15-25 MB |
| 7 output heads | ~14 MB | ~14 MB |
| Optimizer state | ~24 MB | -- |
| Activations & gradients | ~400-600 MB | -- |
| KV cache | ~50-100 MB | ~50-100 MB |
| **Total** | **~700-1000 MB** | **~300-400 MB** |

Fits comfortably in **7.8 GB RAM** (Termux on Android).

<br>

<p align="center">
  <img src="https://raw.githubusercontent.com/Stupid-Sota/SoTa/main/.github/assets/wave.svg" alt="Wave Divider" width="600">
</p>

## Project Structure

```
sota/
├── config.yaml                    # 90 optimizations, full config (184 lines)
├── main.py                        # CLI entry point (5 commands)
├── requirements.txt               # Python dependencies
├── setup.sh                       # Automated setup
├── data/
│   ├── raw/                       # Raw PGN files, openings
│   ├── processed/                 # Generated training data
│   └── checkpoints/               # Model checkpoints
├── src/
│   ├── model/
│   │   └── sota_model.py          # Core model: SOTAConfig + SOTAModel (298 lines)
│   ├── data/
│   │   └── generate_data.py       # Data gen: Stockfish, CoT, Modes (701 lines)
│   ├── training/
│   │   └── train.py               # 3-stage pipeline + curriculum (437 lines)
│   ├── inference/
│   │   └── engine.py              # Inference, interactive, batch (264 lines)
│   └── utils/
│       └── __init__.py            # Utilities
├── tests/
│   └── test_sota.py               # Unit tests (206 lines)
├── configs/                       # Alternative configs
├── scripts/                       # Helper scripts
├── docs/                          # Documentation
└── flan-t5-base/                  # Local model files (cached)
```

**2,200+ lines of Python. Zero external API calls. Runs entirely offline.**

<br>

<p align="center">
  <img src="https://raw.githubusercontent.com/Stupid-Sota/SoTa/main/.github/assets/divider.svg" alt="Divider" width="600">
</p>

## Special Tokens

15 tokens for structured reasoning and mode control:

| Token | Purpose |
|:------|:--------|
| `[STH]` | Start of thought |
| `[ETH]` | End of thought |
| `[ANALYZE]` | Begin analysis |
| `[COMPARE]` | Compare candidates |
| `[VERIFY]` | Verify conclusion |
| `[DECIDE]` | Make final decision |
| `[NXT]` | Next candidate |
| `[MODE]` | Mode indicator |
| `[CERTAIN]` | High confidence |
| `[LIKELY]` | Medium-high confidence |
| `[UNCERTAIN]` | Medium-low confidence |
| `[GUESSING]` | Low confidence |
| `[WAIT]` | Pause and reconsider |
| `[RECHECK]` | Verify previous reasoning |
| `[CORRECT]` | Self-correction |

### Chain-of-Thought Depth

| Depth | Token Budget | Description |
|:------|:------------:|:------------|
| Shallow | ~20 tokens | Fast, minimal reasoning |
| Medium | ~80 tokens | Balanced analysis |
| Deep | ~150 tokens | Full multi-PV evaluation |

<br>

<p align="center">
  <img src="https://raw.githubusercontent.com/Stupid-Sota/SoTa/main/.github/assets/wave.svg" alt="Wave Divider" width="600">
</p>

## License

**MIT** -- Free to use, modify, and distribute.

The bundled flan-t5-base model is licensed under **Apache 2.0** by Google.

### Credits

| Library | Author | Purpose |
|:--------|:-------|:--------|
| [flan-t5-base](https://huggingface.co/google/flan-t5-base) | Google | Base language model |
| [python-chess](https://python-chess.readthedocs.io/) | Niklas Fiekas | Chess logic |
| [Transformers](https://huggingface.co/docs/transformers) | Hugging Face | Model framework |
| [PEFT](https://huggingface.co/docs/peft) | Hugging Face | Parameter-efficient fine-tuning |
| [TRL](https://huggingface.co/docs/trl) | Hugging Face | Reinforcement learning |
| [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) | Tim Dettmers | 4-bit quantization |
| [Stockfish](https://stockfishchess.org/) | Stockfish Team | Chess engine evaluation |

<br>

<p align="center">
  <img src="https://raw.githubusercontent.com/Stupid-Sota/SoTa/main/.github/assets/footer.svg" alt="Footer" width="400">
</p>

<p align="center">
  <a href="https://github.com/Stupid-Sota">
    <img src="https://img.shields.io/badge/Made_by_Stupid_SOTA-2026-000?style=flat" alt="Stupid SOTA">
  </a>
</p>
