# SOTA — Stupid Omega Transformers Agent

A multi-task general reasoning agent built on `flan-t5-base` (250M params).  
MoE architecture with QDoRA adapters, Chain-of-Thought reasoning, and 5 task domains.  
Designed to run entirely offline on a phone (Termux/Android).

`~300 MB RAM · 15M trainable params · 17 special tokens · 300+ optimizations`

---

## Tasks

| Task | Capability |
|------|-----------|
| **Chess** | Position evaluation with 7 stylized output modes + CoT traces |
| **Translate** | Sequence-to-sequence translation between languages |
| **Write** | Creative writing (stories, poems, prompts) |
| **Predict** | Sequence completion and forecasting |
| **PII** | Detect, mask, and filter personally identifiable information |

---

## Architecture

```
flan-t5-base (250M)
  ├── Block Attention (sliding window + global tokens)
  ├── QDoRA adapters (base)
  ├── MoE Router (top-2, noisy gating)
  │   ├── Chess Expert LoRA (r=16, DoRA)
  │   ├── Translate Expert LoRA (r=12, DoRA)
  │   ├── Write Expert LoRA (r=16, DoRA)
  │   ├── Predict Expert LoRA (r=8, DoRA)
  │   └── PII Expert LoRA (r=8, DoRA)
  └── Task Heads
      ├── ChessHeads (7 modes + value head)
      ├── TranslateHead
      ├── WriteHead
      ├── PredictHead (class + regression)
      └── PIIHead (3-class)
```

### Chess Output Modes

| Mode | Trigger | Example |
|:-----|:--------|:--------|
| **Romaji** | `romaji` | `eichi nana eichi go` |
| **Cervantes** | `cervantes` | Archaic literary Spanish |
| **Python** | `python` | `board.push_san("Nf3")` |
| **Musical** | `musical` | `E3... ♪ → G3... ♪` |
| **Morse** | `morse` | `. / .-.. / -- / --..` |
| **Neural** | `neural` | `[LAYER 3] activating neuron #7712` |
| **Patata** | `patata` | `patata` |

### Special Tokens (CoT)

`[STH]` `[ETH]` `[ANALYZE]` `[COMPARE]` `[VERIFY]` `[DECIDE]` `[NXT]`  
`[MODE]` `[CERTAIN]` `[LIKELY]` `[UNCERTAIN]` `[GUESSING]`  
`[WAIT]` `[RECHECK]` `[CORRECT]` `[REDACTED]` `[PAUSE]`

Structured reasoning at 3 depths: **Shallow** (~20 tok), **Medium** (~80 tok), **Deep** (~150 tok).

---

## Quick Start

```bash
# Install
cd ~/Projects/sota
pip install -r requirements.txt
pkg install stockfish  # Termux / brew / apt

# Generate training data
python main.py generate --all --samples 1000

# Run the full 3-stage training pipeline
python main.py train

# Play chess interactively
python main.py chess play

# Evaluate a position
python main.py chess eval "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1" --mode romaji

# Auto-detect task from text
python main.py run "translate en to es: Hello world"
python main.py run "write a haiku about chess"
python main.py run "complete: 1, 1, 2, 3, 5, "
python main.py run "pii filter: email me at test@example.com"

# Show model info
python main.py info
```

---

## Training Pipeline

Three stages, all designed for CPU (Termux/Android):

| Stage | Method | Duration | What it learns |
|:------|:--------|:--------:|:---------------|
| SFT | Cross-entropy next-token pred | 2-4 days | CoT format, move prediction, mode rendering |
| ORPO | Odds Ratio Preference Opt | 2-3 days | Prefer better moves, avoid mode contamination |
| Self-Play RL | PPO + Stockfish eval | 3-7 days | Auto-improvement via self-play |

### Curriculum Learning (5 phases)

| Phase | Positions | Epochs |
|:------|:---------:|:------:|
| Endgame basic | 5,000 | 2 |
| Endgame complex | 5,000 | 2 |
| Middlegame tactical | 8,000 | 2 |
| Opening theory | 8,000 | 1 |
| Adversarial (vs Stockfish) | 10,000+ | 1 |

### Memory Budget

| Component | Training | Inference |
|:----------|:--------:|:---------:|
| Model (4-bit NF4) | 156 MB | 156 MB |
| QDoRA adapters | 15-25 MB | 15-25 MB |
| Task heads | ~14 MB | ~14 MB |
| KV cache | 50-100 MB | 50-100 MB |
| **Total** | **~700-1000 MB** | **~300-400 MB** |

Fits in 7.8 GB RAM (Termux).

---

## Project Structure

```
sota/
├── config.yaml                 # Full config with 300+ optimizations
├── main.py                     # CLI: chess, translate, write, predict, pii, train, run, info
├── requirements.txt
├── setup.sh
│
├── src/
│   ├── model/
│   │   ├── sota_model.py       # Core: SOTAConfig + SOTAModel (603 lines)
│   │   ├── moe.py              # MoE router + 5 expert LoRAs
│   │   ├── multi_task.py       # Task-specific output heads
│   │   └── block_attention.py  # Sliding window attention
│   ├── data/
│   │   ├── generate_data.py    # Chess data: Stockfish, CoT, modes
│   │   ├── translation.py      # Translation data generator
│   │   ├── writing.py          # Creative writing data
│   │   ├── prediction.py       # Sequence prediction data
│   │   └── pii_filter.py       # PII detection/filtering
│   ├── training/
│   │   ├── train.py            # 3-stage pipeline (SFT → ORPO → RL)
│   │   ├── multi_task_train.py # Multi-task training loop
│   │   └── improvements.py     # 300 optimizations implementation
│   ├── inference/
│   │   ├── engine.py           # Inference, interactive, batch eval
│   │   └── optimizations.py    # Tokenizer cache, prefix cache, adaptive temp
│   └── rewards/
│       ├── reward_manager.py
│       ├── translation_rewards.py
│       ├── writing_rewards.py
│       ├── prediction_rewards.py
│       └── pii_rewards.py
│
├── data/
│   ├── raw/                    # PGN files, openings
│   ├── processed/              # Generated training data
│   └── checkpoints/            # Model checkpoints
│
├── tests/
│   └── test_sota.py            # 900+ line test suite
├── configs/                    # Alternative configs
└── scripts/                    # Data download & logging utilities
```

**~2,500+ lines of Python. Zero external API calls. Entirely offline.**

---

## Key Optimizations

| Category | Range | Highlights |
|:---------|:-----:|:-----------|
| Reasoning & CoT | 1-18 | Functional tokens, ExGRPO, MCTS, NoT, SGR, SaGoT |
| Vocabulary | 9-28 | Self-distillation, glitch token replacement, MCL |
| Architecture | 17-36 | QDoRA, hybrid ZO+FO, speculative decoding, MQA |
| Data | 25-44 | Master distillation, adversarial, curriculum, synthetic |
| Inference | 31-50 | QuantSpec, SpecExec, streaming, parallel modes |
| Training | 37-54 | 3-stage pipeline, anti-forgetting, adapter ensemble |

Each optimization documented with rationale in `config.yaml`.

---

## License

**MIT** — Free to use, modify, distribute.

Base model (`flan-t5-base`) by Google — Apache 2.0.

### Built With

| Library | Author |
|:--------|:-------|
| [flan-t5-base](https://huggingface.co/google/flan-t5-base) | Google |
| [python-chess](https://python-chess.readthedocs.io/) | Niklas Fiekas |
| [Transformers](https://huggingface.co/docs/transformers) | Hugging Face |
| [PEFT](https://huggingface.co/docs/peft) | Hugging Face |
| [TRL](https://huggingface.co/docs/trl) | Hugging Face |
| [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) | Tim Dettmers |
| [Stockfish](https://stockfishchess.org/) | Stockfish Team |

---

<p align="center">
  <sub>Built by <a href="https://github.com/Stupid-Sota">Stupid SOTA</a> · 2026</sub>
</p>
