"""
SOTA — Stupid Omega Transformers Agent
Core model architecture with MoE, Block Attention, Multi-Task.
Integrates all 300 improvements across 5 experts + 5 task heads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, T5Config, AutoTokenizer
from transformers.utils.quantization_config import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from typing import Dict, List, Optional, Tuple, Union
import math
import os
import re
import json
import yaml
import logging

from src.model.block_attention import apply_block_attention
from src.model.moe import MoERouter, ExpertLoRAManager, EXPERT_NAMES, EXPERT_CONFIGS
from src.model.multi_task import TaskHeadsManager, TASK_NAMES


class SOTAConfig:
    """Expanded config for multi-task MoE SOTA."""

    def __init__(self, config_path: str = None):
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                cfg = yaml.safe_load(f)
        else:
            cfg = {}

        self.config = cfg
        self.base_model = cfg.get('base_model', './flan-t5-base')
        self.max_seq_length = cfg.get('model', {}).get('max_seq_length', 512)
        self.window_size = cfg.get('model', {}).get('window_size', 128)
        self.n_global_tokens = cfg.get('model', {}).get('n_global_tokens', 4)
        self.compile_model_flag = cfg.get('model', {}).get('compile', False)
        self.freeze_encoder = cfg.get('training', {}).get('freeze_encoder', True)

        peft_cfg = cfg.get('peft', {})
        self.peft_r = peft_cfg.get('r', 16)
        self.peft_alpha = peft_cfg.get('lora_alpha', 32)
        self.peft_dropout = peft_cfg.get('lora_dropout', 0.05)
        self.use_dora = peft_cfg.get('use_dora', True)

        quant_cfg = cfg.get('quantization', {})
        self.load_in_4bit = quant_cfg.get('load_in_4bit', False)
        self.bnb_4bit_quant_type = quant_cfg.get('bnb_4bit_quant_type', 'nf4')
        self.bnb_4bit_compute_dtype = quant_cfg.get('bnb_4bit_compute_dtype', 'float32')

        self.special_tokens = cfg.get('special_tokens', [
            "[STH]", "[ETH]", "[NXT]", "[MODE]",
            "[CERTAIN]", "[LIKELY]", "[UNCERTAIN]", "[GUESSING]",
            "[WAIT]", "[RECHECK]", "[CORRECT]",
            "[ANALYZE]", "[COMPARE]", "[VERIFY]", "[DECIDE]",
            "[REDACTED]", "[PAUSE]",
        ])

        self.modes = cfg.get('modes', {})

        inf_cfg = cfg.get('inference', {})
        self.temperature = inf_cfg.get('temperature', 0.7)
        self.top_p = inf_cfg.get('top_p', 0.9)
        self.top_k = inf_cfg.get('top_k', 50)
        self.max_new_tokens = inf_cfg.get('max_new_tokens', 256)
        self.num_beams = inf_cfg.get('num_beams', 4)
        self.repetition_penalty = inf_cfg.get('repetition_penalty', 1.2)

    def get(self, key, default=None):
        return getattr(self, key, default)


class SOTAModel:
    """
    SOTA Core — T5-base + Block Attention + MoE Router + 5 Expert LoRAs + 5 Task Heads.
    Supports: chess (7 modes), translation, writing, prediction, PII filtering.
    Implements ALL 300 improvements.
    """

    def __init__(self, config: SOTAConfig, device: str = 'cpu'):
        self.config = config
        self.device = device
        self.model = None
        self.tokenizer = None
        self.router = None
        self.expert_manager = None
        self.task_heads = None
        self.loaded = False
        self.compiled = False
        self.tokenizer_cache = None
        self.prefix_cache = None
        self.ngram_blocker = None
        self.adaptive_temp = None
        self.early_stopper = None
        self.adaptive_temp = None
        self.dialogue_manager = None
        self._setup_inference_optimizations()

    def _setup_inference_optimizations(self):
        """Initialize inference optimizations (#18-#24, #31-#36)."""
        from src.inference.optimizations import (
            TokenizerCache, PrefixCache, NGramBlock,
            AdaptiveTemperature, EarlyStopping,
        )
        self.tokenizer_cache = TokenizerCache(max_size=1024)
        self.prefix_cache = PrefixCache(max_size=256)
        self.ngram_blocker = NGramBlock(n=3, max_repeats=1)
        inf_cfg = self.config.config if hasattr(self.config, 'config') else {}
        temp = inf_cfg.get('inference', {}).get('temperature', 0.7)
        self.adaptive_temp = AdaptiveTemperature(base_temp=self.config.temperature)
        self.early_stopper = EarlyStopping(
            confidence_threshold=0.95, min_tokens=5, window=3
        )
        from src.training.improvements import DialogueManager
        self.dialogue_manager = DialogueManager(max_history=10, max_tokens=2048)

    def load_base_model(self):
        print(f"[SOTA] Loading {self.config.base_model}...")
        torch_dtype = getattr(torch, self.config.bnb_4bit_compute_dtype, torch.float32)

        if torch.cuda.is_available():
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=self.config.load_in_4bit,
                bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
            )
            self.model = T5ForConditionalGeneration.from_pretrained(
                self.config.base_model,
                quantization_config=bnb_config,
                device_map='auto',
                torch_dtype=torch_dtype,
            )
        else:
            self.model = T5ForConditionalGeneration.from_pretrained(
                self.config.base_model,
                torch_dtype=torch_dtype,
            ).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.base_model)

        num_added = self.tokenizer.add_special_tokens({
            'additional_special_tokens': self.config.special_tokens
        })
        if num_added > 0:
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.model = prepare_model_for_kbit_training(self.model)

        print(f"[SOTA] Model loaded: {sum(p.numel() for p in self.model.parameters()):,} params")
        return self.model

    def apply_qdora(self, auto_rank: bool = True):
        """Apply base QDoRA adapters with optional AutoLoRA rank."""
        from src.training.improvements import AutoLoRA
        self.auto_lora = AutoLoRA(min_rank=4, max_rank=32, window=100)

        r = self.config.peft_r
        if auto_rank:
            default_rank = self.auto_lora.update_rank('base', 1.0)
            r = default_rank
            print(f"[SOTA] AutoLoRA initial rank: {r}")

        print(f"[SOTA] Applying QDoRA adapters (r={r})...")
        lora_config = LoraConfig(
            r=r,
            lora_alpha=self.config.peft_alpha,
            lora_dropout=self.config.peft_dropout,
            target_modules=["q", "v"],
            use_dora=self.config.use_dora,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM,
        )
        self.model = get_peft_model(self.model, lora_config)
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"[SOTA] QDoRA applied: {trainable:,} trainable / {total:,} total")
        return self.model

    def apply_block_attention(self):
        """Apply sliding window attention — improvement #1."""
        self.model = apply_block_attention(
            self.model,
            window_size=self.config.window_size,
            n_global=self.config.n_global_tokens,
        )

    def freeze_encoder(self):
        """Freeze encoder — improvement #4, #27."""
        if not hasattr(self.model, 'encoder'):
            return
        for param in self.model.encoder.parameters():
            param.requires_grad = False
        frozen = sum(p.numel() for p in self.model.encoder.parameters())
        print(f"[SOTA] Encoder frozen: {frozen:,} params")

    def freeze_bottom_layers(self, n_layers: int = 4):
        """Freeze bottom N encoder layers — improvement #27."""
        if not hasattr(self.model, 'encoder'):
            return
        for i, block in enumerate(self.model.encoder.block):
            if i < n_layers:
                for param in block.parameters():
                    param.requires_grad = False

    def setup_moe(self):
        """Initialize MoE router + expert manager — improvements #41-60."""
        d_model = self.model.config.d_model
        self.router = MoERouter(
            d_model=d_model,
            n_experts=len(EXPERT_NAMES),
            top_k=2,
            use_noisy_gating=True,
            noise_std=0.1,
            router_dropout=0.05,
        ).to(self.device)

        self.expert_manager = ExpertLoRAManager(self.model)
        self.expert_manager.create_experts()

        print(f"[MoE] Router + {len(EXPERT_NAMES)} experts ready")

    def setup_task_heads(self):
        """Initialize task-specific heads — improvements #81-100."""
        d_model = self.model.config.d_model
        vocab_size = len(self.tokenizer)
        mode_names = list(self.config.modes.keys())

        self.task_heads = TaskHeadsManager(d_model, vocab_size, mode_names).to(self.device)
        print(f"[Tasks] {len(TASK_NAMES)} task heads ready ({', '.join(TASK_NAMES)})")

    def detect_task_from_prompt(self, input_text: str) -> str:
        """Detect task from prompt text."""
        if input_text.startswith('chess') or 'fen' in input_text.lower():
            return 'chess'
        if input_text.startswith('translate') or 'translate' in input_text:
            return 'translate'
        if input_text.startswith('write') or 'story' in input_text.lower() or 'poem' in input_text.lower():
            return 'write'
        if input_text.startswith('predict') or 'complete' in input_text.lower() or 'sequence' in input_text.lower():
            return 'predict'
        if input_text.startswith('pii') or 'filter' in input_text.lower():
            return 'pii'
        return 'chess'

    def detect_task_from_input_ids(self, input_ids: torch.Tensor) -> str:
        """Detect task from tokenized input."""
        text = self.tokenizer.decode(input_ids[0, :20], skip_special_tokens=True).lower()
        return self.detect_task_from_prompt(text)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None,
                labels: torch.Tensor = None, task: str = 'auto', mode: str = 'romaji',
                return_dict: bool = True, output_hidden_states: bool = True,
                **kwargs) -> Dict:
        if task == 'auto' and input_ids is not None:
            task = self.detect_task_from_input_ids(input_ids)

        if self.router is not None and task in EXPERT_NAMES:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=True,
                **kwargs
            )

            if self.task_heads is not None and hasattr(outputs, 'decoder_hidden_states'):
                hidden = outputs.decoder_hidden_states[-1]
                with torch.no_grad():
                    gates, expert_ids, router_logits = self.router(hidden)

                task_logits = self.task_heads.forward(task, hidden, mode=mode)

                if labels is not None:
                    if task == 'chess':
                        shifted = task_logits[:, :-1, :].contiguous()
                        shift_labels = labels[:, 1:].contiguous()
                        task_loss = F.cross_entropy(
                            shifted.view(-1, shifted.size(-1)),
                            shift_labels.view(-1),
                            ignore_index=-100,
                        )
                    elif task in ('translate', 'write'):
                        shifted = task_logits[:, :-1, :].contiguous()
                        shift_labels = labels[:, 1:].contiguous()
                        task_loss = F.cross_entropy(
                            shifted.view(-1, shifted.size(-1)),
                            shift_labels.view(-1),
                            ignore_index=-100,
                        )
                    elif task == 'predict':
                        task_loss = F.mse_loss(task_logits.squeeze(-1), labels.float())
                    else:
                        task_loss = F.cross_entropy(
                            task_logits.view(-1, task_logits.size(-1)),
                            labels.view(-1),
                            ignore_index=-100,
                        )

                    weighted_loss, loss_info = self.task_heads.compute_loss(
                        task, task_logits, labels, task_loss
                    )

                    aux_losses = self.router.compute_aux_loss(router_logits, expert_ids)
                    total_loss = ExpertLoRAManager.compute_total_moe_loss(
                        weighted_loss, aux_losses
                    )

                    return {'loss': total_loss, 'logits': task_logits,
                            'task': task, 'aux_losses': aux_losses,
                            'loss_info': loss_info}

                return {'logits': task_logits, 'task': task}

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            **kwargs
        )
        result = {'logits': outputs.logits if hasattr(outputs, 'logits') else None,
                  'task': task}
        if hasattr(outputs, 'loss'):
            result['loss'] = outputs.loss
        return result

    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None,
                 task: str = 'auto', mode: str = 'romaji', **kwargs) -> torch.Tensor:
        if task == 'auto' and input_ids is not None:
            task = self.detect_task_from_input_ids(input_ids)

        from src.inference.optimizations import (
            top_p_filtering, top_k_filtering, apply_temperature,
            diverse_beam_search,
        )

        gen_kwargs = {
            'max_new_tokens': kwargs.get('max_new_tokens', self.config.max_new_tokens),
            'temperature': kwargs.get('temperature', self.config.temperature),
            'top_p': kwargs.get('top_p', self.config.top_p),
            'top_k': kwargs.get('top_k', self.config.top_k),
            'num_beams': kwargs.get('num_beams', self.config.num_beams),
            'repetition_penalty': kwargs.get('repetition_penalty', self.config.repetition_penalty),
            'do_sample': kwargs.get('do_sample', True),
            'early_stopping': kwargs.get('early_stopping', True),
        }

        use_prefix_cache = kwargs.get('use_prefix_cache', True)
        if use_prefix_cache and task != 'auto' and self.prefix_cache is not None:
            cache_key = self.prefix_cache.make_key(task=task, mode=mode)
            cached = self.prefix_cache.get(cache_key)
            if cached is not None and cached['input_ids'].shape[-1] == input_ids.shape[-1]:
                return cached['output_ids']

        if self.expert_manager is not None and task in EXPERT_NAMES:
            self.expert_manager.activate_expert(task)

        with torch.no_grad():
            if gen_kwargs['num_beams'] > 1:
                output_ids = diverse_beam_search(
                    self.model, input_ids,
                    attention_mask=attention_mask,
                    **gen_kwargs,
                )
            else:
                output_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **gen_kwargs,
                )

        if use_prefix_cache and task != 'auto' and self.prefix_cache is not None:
            cache_key = self.prefix_cache.make_key(task=task, mode=mode)
            self.prefix_cache.set(cache_key, {
                'input_ids': input_ids,
                'output_ids': output_ids,
            })

        return output_ids

    def run(self, input_text: str, task: str = 'auto', mode: str = 'romaji',
            include_cot: bool = True, session_id: str = None, **kwargs) -> Dict:
        if task == 'auto':
            task = self.detect_task_from_prompt(input_text)

        session_id = session_id or 'default'
        if self.dialogue_manager is not None:
            input_text = self.dialogue_manager.add_to_context(
                session_id, input_text, role='user'
            )

        if task == 'chess':
            result = self._run_chess(input_text, mode, include_cot, **kwargs)
        elif task == 'translate':
            result = self._run_translate(input_text, **kwargs)
        elif task == 'write':
            result = self._run_write(input_text, **kwargs)
        elif task == 'predict':
            result = self._run_predict(input_text, **kwargs)
        elif task == 'pii':
            result = self._run_pii(input_text, **kwargs)
        else:
            return {'error': f'Unknown task: {task}'}

        if self.dialogue_manager is not None and 'error' not in result:
            output_text = result.get('text') or result.get('translation') or \
                          result.get('prediction') or result.get('full_output') or ''
            self.dialogue_manager.add_to_context(session_id, output_text, role='assistant')

        return result

    def _run_chess(self, input_text: str, mode: str = 'romaji',
                   include_cot: bool = True, **kwargs) -> Dict:
        fen = self._extract_fen(input_text)
        prompt = f"chess position: {fen} | mode: {mode} | think: {'yes' if include_cot else 'no'}"
        inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True,
                                max_length=self.config.max_seq_length).to(self.device)

        output_ids = self.generate(inputs['input_ids'], inputs['attention_mask'],
                                    task='chess', mode=mode, **kwargs)
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=False)

        result = {
            'full_output': output_text,
            'cot': None,
            'move': None,
            'mode_output': None,
            'task': 'chess',
        }

        if '[STH]' in output_text and '[ETH]' in output_text:
            start = output_text.index('[STH]') + 5
            end = output_text.index('[ETH]')
            result['cot'] = output_text[start:end].strip()
            result['mode_output'] = output_text[end + 5:].strip()
        else:
            result['mode_output'] = output_text.strip()

        result['move'] = self._extract_move(result['mode_output'])
        return result

    def _run_translate(self, input_text: str, **kwargs) -> Dict:
        prompt = f"translate: {input_text}"
        inputs = self.tokenizer(prompt, return_tensors='pt',
                                max_length=self.config.max_seq_length,
                                truncation=True).to(self.device)
        output_ids = self.generate(inputs['input_ids'], inputs['attention_mask'],
                                    task='translate', **kwargs)
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return {'translation': output_text, 'task': 'translate'}

    def _run_write(self, input_text: str, **kwargs) -> Dict:
        prompt = f"write: {input_text}"
        inputs = self.tokenizer(prompt, return_tensors='pt',
                                max_length=self.config.max_seq_length,
                                truncation=True).to(self.device)
        default_new_tokens = self.config.max_new_tokens
        output_ids = self.generate(inputs['input_ids'], inputs['attention_mask'],
                                    task='write',
                                    max_new_tokens=kwargs.get('max_new_tokens', default_new_tokens),
                                    temperature=kwargs.get('temperature', 0.8),
                                    **kwargs)
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return {'text': output_text, 'task': 'write'}

    def _run_predict(self, input_text: str, **kwargs) -> Dict:
        prompt = f"complete: {input_text}"
        inputs = self.tokenizer(prompt, return_tensors='pt',
                                max_length=self.config.max_seq_length,
                                truncation=True).to(self.device)
        default_new_tokens = self.config.max_new_tokens
        output_ids = self.generate(inputs['input_ids'], inputs['attention_mask'],
                                    task='predict',
                                    max_new_tokens=kwargs.get('max_new_tokens', default_new_tokens),
                                    **kwargs)
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return {'prediction': output_text, 'task': 'predict'}

    def _run_pii(self, input_text: str, **kwargs) -> Dict:
        from src.data.pii_filter import PIIProcessor
        processor = PIIProcessor()
        detected = processor.detect(input_text)
        filtered = processor.filter(input_text, mode='mask')

        prompt = f"pii filter: {filtered}"
        inputs = self.tokenizer(prompt, return_tensors='pt',
                                max_length=self.config.max_seq_length,
                                truncation=True).to(self.device)
        default_new_tokens = self.config.max_new_tokens
        output_ids = self.generate(inputs['input_ids'], inputs['attention_mask'],
                                    task='pii',
                                    max_new_tokens=kwargs.get('max_new_tokens', default_new_tokens),
                                    **kwargs)
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return {
            'original': input_text,
            'filtered': filtered,
            'detected_pii': detected,
            'model_output': output_text,
            'task': 'pii',
        }

    def _extract_fen(self, text: str) -> str:
        fen_pattern = r'[rnbqkpRNBQKP1-8]{1,70}(?:/[rnbqkpRNBQKP1-8]{1,70}){7}'
        match = re.search(fen_pattern, text)
        if match:
            return match.group(0)
        return "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    def _extract_move(self, output: str) -> Optional[str]:
        uci_match = re.search(r'([a-h][1-8][a-h][1-8][qrbn]?)', output)
        if uci_match:
            return uci_match.group(1)
        san_match = re.search(r'([KQRBN]?[a-h]?[1-8]?x?[a-h][1-8][=+]?#?)', output)
        if san_match:
            return san_match.group(1)
        return None

    def compile_model(self):
        """Apply torch.compile — improvement #3."""
        if self.compiled:
            return
        try:
            self.model.forward = torch.compile(
                self.model.forward,
                mode="reduce-overhead",
                dynamic=True,
                options={"epilogue_fusion": True, "max_autotune": False},
            )
            self.model.generate = torch.compile(
                self.model.generate,
                mode="reduce-overhead",
                dynamic=True,
            )
            self.compiled = True
            print("[SOTA] Model compiled with torch.compile")
        except Exception as e:
            print(f"[SOTA] torch.compile failed (continuing without): {e}")

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing — improvement #17."""
        try:
            self.model.gradient_checkpointing_enable()
            print("[SOTA] Gradient checkpointing enabled")
        except Exception as e:
            print(f"[SOTA] gradient_checkpointing failed: {e}")

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        if self.router is not None:
            torch.save(self.router.state_dict(), os.path.join(path, 'moe_router.pt'))
        if self.task_heads is not None:
            torch.save(self.task_heads.state_dict(), os.path.join(path, 'task_heads.pt'))
        print(f"[SOTA] Model saved to {path}")

    def load(self, path: str):
        self.load_base_model()
        self.model = T5ForConditionalGeneration.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        router_path = os.path.join(path, 'moe_router.pt')
        if os.path.exists(router_path):
            self.setup_moe()
            self.router.load_state_dict(torch.load(router_path))
        heads_path = os.path.join(path, 'task_heads.pt')
        if os.path.exists(heads_path):
            self.setup_task_heads()
            self.task_heads.load_state_dict(torch.load(heads_path))
        self.loaded = True
        print(f"[SOTA] Model loaded from {path}")

    def get_memory_usage_mb(self) -> float:
        try:
            with open('/proc/self/status', 'r') as f:
                for line in f:
                    if line.startswith('VmRSS:'):
                        return int(line.split()[1]) / 1024
        except:
            return 0

    def print_model_info(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\n{'='*60}")
        print(f"  SOTA Multi-Task MoE")
        print(f"{'='*60}")
        print(f"  Base model: {self.config.base_model}")
        print(f"  Total params: {total_params:,}")
        print(f"  Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
        print(f"  Tasks: {', '.join(TASK_NAMES)}")
        print(f"  Experts: {', '.join(EXPERT_NAMES)}")
        print(f"  Modes: {list(self.config.modes.keys())}")
        print(f"  Window size: {self.config.window_size}")
        print(f"  Compiled: {self.compiled}")
        print(f"  Memory: {self.get_memory_usage_mb():.0f} MB")
        print(f"{'='*60}\n")
