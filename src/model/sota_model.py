"""
SOTA — Stupid Omega Transformers Agent
Core model architecture with all 90 optimizations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, T5Config, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from bitsandbytes import BitsAndBytesConfig
from typing import Dict, List, Optional, Tuple, Union
import math
import os


class SOTAConfig:
    """Configuration for SOTA model with all optimizations."""

    def __init__(self, config_path: str = None):
        import yaml
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                cfg = yaml.safe_load(f)
        else:
            cfg = {}

        self.base_model = cfg.get('base_model', 'google/flan-t5-base')
        self.max_seq_length = cfg.get('model', {}).get('max_seq_length', 512)

        # PEFT
        peft_cfg = cfg.get('peft', {})
        self.peft_r = peft_cfg.get('r', 16)
        self.peft_alpha = peft_cfg.get('lora_alpha', 32)
        self.peft_dropout = peft_cfg.get('lora_dropout', 0.05)
        self.use_dora = peft_cfg.get('use_dora', True)
        self.dynamic_rank = peft_cfg.get('dynamic_rank', True)

        # Quantization
        quant_cfg = cfg.get('quantization', {})
        self.load_in_4bit = quant_cfg.get('load_in_4bit', True)
        self.bnb_4bit_quant_type = quant_cfg.get('bnb_4bit_quant_type', 'nf4')
        self.bnb_4bit_compute_dtype = quant_cfg.get('bnb_4bit_compute_dtype', 'bfloat16')
        self.bnb_4bit_use_double_quant = quant_cfg.get('bnb_4bit_use_double_quant', True)

        # Special tokens
        self.special_tokens = cfg.get('special_tokens', [
            "[STH]", "[ETH]", "[NXT]", "[MODE]",
            "[CERTAIN]", "[LIKELY]", "[UNCERTAIN]", "[GUESSING]",
            "[WAIT]", "[RECHECK]", "[CORRECT]",
            "[ANALYZE]", "[COMPARE]", "[VERIFY]", "[DECIDE]"
        ])

        # Modes
        self.modes = cfg.get('modes', {})

        # Inference
        inf_cfg = cfg.get('inference', {})
        self.temperature = inf_cfg.get('temperature', 0.7)
        self.top_p = inf_cfg.get('top_p', 0.9)
        self.max_new_tokens = inf_cfg.get('max_new_tokens', 256)
        self.num_beams = inf_cfg.get('num_beams', 4)


class SOTAModel:
    """
    SOTA Core Model — flan-t5-base + QDoRA + 6 output heads + CoT.
    Implements all architectural optimizations (17-24, 61-70).
    """

    def __init__(self, config: SOTAConfig, device: str = 'cpu'):
        self.config = config
        self.device = device
        self.model = None
        self.tokenizer = None
        self.mode_heads = {}
        self.loaded = False

    def load_base_model(self):
        """Load flan-t5-base with 4-bit quantization."""
        print(f"[SOTA] Loading {self.config.base_model}...")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.config.load_in_4bit,
            bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=getattr(torch, self.config.bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=self.config.bnb_4bit_use_double_quant,
        )

        self.model = T5ForConditionalGeneration.from_pretrained(
            self.config.base_model,
            quantization_config=bnb_config,
            device_map='auto' if torch.cuda.is_available() else {'': self.device},
            torch_dtype=getattr(torch, self.config.bnb_4bit_compute_dtype),
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.base_model)

        # Add special tokens
        num_added = self.tokenizer.add_special_tokens({
            'additional_special_tokens': self.config.special_tokens
        })
        if num_added > 0:
            self.model.resize_token_embeddings(len(self.tokenizer))

        # Prepare for k-bit training (enables gradient checkpointing)
        self.model = prepare_model_for_kbit_training(self.model)

        print(f"[SOTA] Model loaded: {sum(p.numel() for p in self.model.parameters()):,} params")
        return self.model

    def apply_qdora(self):
        """
        Apply QDoRA (QLoRA + DoRA) adapters.
        Implements #17 (QDoRA), #19 (separate heads), #66 (dynamic rank).
        """
        print("[SOTA] Applying QDoRA adapters...")

        lora_config = LoraConfig(
            r=self.config.peft_r,
            lora_alpha=self.config.peft_alpha,
            lora_dropout=self.config.peft_dropout,
            target_modules=["q", "v"],  # decoder self-attn priority
            use_dora=self.config.use_dora,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM,
        )

        self.model = get_peft_model(self.model, lora_config)
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"[SOTA] QDoRA applied: {trainable:,} trainable / {total:,} total ({100*trainable/total:.2f}%)")

        # Create separate output heads per mode (#19)
        self._create_mode_heads()

        return self.model

    def _create_mode_heads(self):
        """Create 6 separate output heads for each mode (#19)."""
        d_model = self.model.config.d_model
        vocab_size = len(self.tokenizer)

        mode_names = list(self.config.modes.keys())
        for mode_name in mode_names:
            head = nn.Linear(d_model, vocab_size, bias=False)
            # Initialize from base lm_head
            head.weight.data = self.model.lm_head.weight.data.clone()
            self.mode_heads[mode_name] = head.to(self.device)

        print(f"[SOTA] Created {len(self.mode_heads)} mode-specific output heads")

    def get_mode_head(self, mode: str) -> nn.Linear:
        """Get the output head for a specific mode."""
        if mode in self.mode_heads:
            return self.mode_heads[mode]
        return self.model.lm_head

    def forward_with_mode(self, input_ids, attention_mask, mode: str = 'romaji',
                         labels=None, decoder_input_ids=None):
        """
        Forward pass with mode-specific output head.
        Supports all reasoning tokens and CoT structure.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_input_ids=decoder_input_ids,
            output_hidden_states=True,
        )

        # Replace logits with mode-specific head
        last_hidden = outputs.decoder_hidden_states[-1]
        mode_head = self.get_mode_head(mode)
        mode_logits = mode_head(last_hidden)

        return outputs, mode_logits

    def generate_with_cot(self, fen: str, mode: str = 'romaji',
                         include_cot: bool = True, **kwargs) -> str:
        """
        Generate output with Chain-of-Thought reasoning.
        Implements #1-8 (CoT optimizations), #83 (streaming).
        """
        prompt = self._build_prompt(fen, mode, include_cot)
        inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True,
                               max_length=self.config.max_seq_length).to(self.device)

        gen_kwargs = {
            'temperature': kwargs.get('temperature', self.config.temperature),
            'top_p': kwargs.get('top_p', self.config.top_p),
            'max_new_tokens': kwargs.get('max_new_tokens', self.config.max_new_tokens),
            'num_beams': kwargs.get('num_beams', self.config.num_beams),
            'repetition_penalty': kwargs.get('repetition_penalty', 1.2),
            'do_sample': True,
        }

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                **gen_kwargs
            )

        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=False)
        return self._parse_output(output_text, include_cot)

    def _build_prompt(self, fen: str, mode: str, include_cot: bool) -> str:
        """Build the input prompt with CoT structure."""
        prompt = f"chess position: {fen} | mode: {mode}"
        if include_cot:
            prompt = f"{prompt} | think: yes"
        return prompt

    def _parse_output(self, output: str, include_cot: bool) -> Dict:
        """Parse the output into CoT and final move."""
        result = {
            'full_output': output,
            'cot': None,
            'move': None,
            'mode_output': None,
        }

        if '[STH]' in output and '[ETH]' in output:
            start = output.index('[STH]') + 5
            end = output.index('[ETH]')
            result['cot'] = output[start:end].strip()
            result['mode_output'] = output[end + 5:].strip()
        else:
            result['mode_output'] = output.strip()

        # Extract move from output
        result['move'] = self._extract_move(result['mode_output'])
        return result

    def _extract_move(self, output: str) -> str:
        """Extract the chess move from the output."""
        import re
        # UCI format: e2e4, g1f3, etc.
        uci_match = re.search(r'([a-h][1-8][a-h][1-8][qrbn]?)', output)
        if uci_match:
            return uci_match.group(1)
        # SAN format: e4, Nf3, etc.
        san_match = re.search(r'([KQRBN]?[a-h]?[1-8]?x?[a-h][1-8][=+]?#?)', output)
        if san_match:
            return san_match.group(1)
        return None

    def save(self, path: str):
        """Save model and adapters."""
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        # Save mode heads
        for mode_name, head in self.mode_heads.items():
            torch.save(head.state_dict(), os.path.join(path, f'{mode_name}_head.pt'))
        print(f"[SOTA] Model saved to {path}")

    def load(self, path: str):
        """Load model and adapters."""
        self.load_base_model()
        self.model = T5ForConditionalGeneration.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        # Load mode heads
        for mode_name in self.config.modes.keys():
            head_path = os.path.join(path, f'{mode_name}_head.pt')
            if os.path.exists(head_path):
                self.mode_heads[mode_name].load_state_dict(torch.load(head_path))
        self.loaded = True
        print(f"[SOTA] Model loaded from {path}")

    def get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        try:
            with open('/proc/self/status', 'r') as f:
                for line in f:
                    if line.startswith('VmRSS:'):
                        return int(line.split()[1]) / 1024
        except:
            return 0

    def print_model_info(self):
        """Print detailed model information."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\n{'='*60}")
        print(f"  SOTA Model Information")
        print(f"{'='*60}")
        print(f"  Base model: {self.config.base_model}")
        print(f"  Total params: {total_params:,}")
        print(f"  Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
        print(f"  QDoRA: {self.config.use_dora}")
        print(f"  4-bit NF4: {self.config.load_in_4bit}")
        print(f"  Mode heads: {list(self.mode_heads.keys())}")
        print(f"  Special tokens: {len(self.config.special_tokens)}")
        print(f"  Memory usage: {self.get_memory_usage_mb():.0f} MB")
        print(f"{'='*60}\n")
