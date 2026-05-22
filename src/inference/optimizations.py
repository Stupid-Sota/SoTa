"""
Inference Optimizations (#18-#24, #31-#36).
Adaptive sampling, tokenizer cache, n-gram blocking, prefix cache.
"""

import torch
import torch.nn.functional as F
import logging
from typing import Callable, Dict, List, Optional, Tuple
from collections import OrderedDict


# ============================================================
# #18 — Dynamic Byte-Pair Encoding Cache (tokenizer caching)
# ============================================================

class TokenizerCache:
    """LRU cache for tokenizer encode/decode operations."""

    def __init__(self, max_size: int = 1024):
        self._encode_cache: OrderedDict = OrderedDict()
        self._decode_cache: OrderedDict = OrderedDict()
        self.max_size = max_size

    def encode(self, text: str, tokenizer, **kwargs) -> dict:
        if text in self._encode_cache:
            self._encode_cache.move_to_end(text)
            return self._encode_cache[text]
        result = tokenizer(text, **kwargs)
        if len(self._encode_cache) >= self.max_size:
            self._encode_cache.popitem(last=False)
        self._encode_cache[text] = result
        return result

    def decode(self, token_ids: tuple, tokenizer, **kwargs) -> str:
        if token_ids in self._decode_cache:
            self._decode_cache.move_to_end(token_ids)
            return self._decode_cache[token_ids]
        result = tokenizer.decode(list(token_ids), **kwargs)
        if len(self._decode_cache) >= self.max_size:
            self._decode_cache.popitem(last=False)
        self._decode_cache[token_ids] = result
        return result

    def clear(self):
        self._encode_cache.clear()
        self._decode_cache.clear()


# ============================================================
# #19 — Adaptive Temperature Sampling
# ============================================================

class AdaptiveTemperature:
    """Adjusts temperature based on confidence/entropy."""

    def __init__(self, base_temp: float = 0.7, min_temp: float = 0.1,
                 max_temp: float = 1.5, entropy_threshold: float = 2.0):
        self.base_temp = base_temp
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.entropy_threshold = entropy_threshold

    def adjust(self, logits: torch.Tensor, confidence: Optional[float] = None) -> float:
        probs = F.softmax(logits[-1:].float(), dim=-1)
        entropy = -torch.sum(probs * torch.log(probs.clamp(min=1e-10)))
        if confidence is not None:
            temp = self.base_temp * (1.0 + (1.0 - confidence))
        elif entropy.item() > self.entropy_threshold:
            temp = self.base_temp * 1.3
        else:
            temp = self.base_temp * 0.7
        return max(self.min_temp, min(self.max_temp, temp))


# ============================================================
# #20 — Min-p Sampling
# ============================================================

def min_p_sampling(logits: torch.Tensor, p: float = 0.1,
                   min_tokens: int = 1) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    max_prob = probs.max(dim=-1, keepdim=True).values
    cutoff = max_prob * p
    mask = probs >= cutoff
    if mask.sum(dim=-1) < min_tokens:
        topk = torch.topk(probs, k=min_tokens, dim=-1)
        mask = torch.zeros_like(probs).scatter_(-1, topk.indices, 1)
    masked_probs = probs * mask.float()
    masked_probs = masked_probs / masked_probs.sum(dim=-1, keepdim=True)
    return masked_probs


# ============================================================
# #21 — Repetition Penalty with N-gram Blocking
# ============================================================

class NGramBlock:
    """Blocks repeating n-grams during generation."""

    def __init__(self, n: int = 3, max_repeats: int = 1):
        self.n = n
        self.max_repeats = max_repeats
        self._history: Dict[Tuple[int, ...], int] = {}

    def update(self, token_ids: List[int]):
        if len(token_ids) < self.n:
            return
        ngram = tuple(token_ids[-self.n:])
        self._history[ngram] = self._history.get(ngram, 0) + 1
        if len(self._history) > 10000:
            cutoff = len(token_ids) - self.n - 500
            self._history = {k: v for k, v in self._history.items()
                            if k[0] > cutoff}

    def should_block(self, token_ids: List[int], next_token: int) -> bool:
        if len(token_ids) < self.n - 1:
            return False
        ngram = tuple(token_ids[-(self.n - 1):] + [next_token])
        count = self._history.get(ngram, 0)
        return count >= self.max_repeats

    def apply_to_logits(self, logits: torch.Tensor, token_ids: List[int],
                        penalty: float = 1.2) -> torch.Tensor:
        if len(token_ids) < 1:
            return logits
        for i in range(logits.size(-1)):
            if self.should_block(token_ids, i):
                logits[-1, i] -= penalty * 10.0
        return logits


# ============================================================
# #22 — KV Cache Compression (4-bit quantized cache placeholder)
# ============================================================

class KVCacheCompressor:
    """Compresses KV cache entries to reduce memory."""

    def __init__(self, bits: int = 4):
        self.bits = bits
        self.scales: Dict[str, torch.Tensor] = {}
        self.zeros: Dict[str, torch.Tensor] = {}

    def quantize(self, key: str, tensor: torch.Tensor) -> torch.Tensor:
        if self.bits >= 16:
            return tensor
        min_val = tensor.min()
        max_val = tensor.max()
        q_range = 2 ** self.bits - 1
        scale = (max_val - min_val) / q_range
        zero = min_val
        self.scales[key] = scale
        self.zeros[key] = zero
        q = ((tensor - zero) / (scale + 1e-10)).round().clamp(0, q_range)
        return q.to(torch.uint8 if self.bits <= 8 else torch.int16)

    def dequantize(self, key: str, q_tensor: torch.Tensor) -> torch.Tensor:
        if key not in self.scales:
            return q_tensor
        return q_tensor.float() * self.scales[key] + self.zeros[key]


# ============================================================
# #23 — Prompt Prefix Caching
# ============================================================

class PrefixCache:
    """Caches encoded prefixes (e.g. FEN prompt prefixes)."""

    def __init__(self, max_size: int = 256):
        self.cache: OrderedDict = OrderedDict()
        self.max_size = max_size

    def get(self, prefix_key: str) -> Optional[dict]:
        if prefix_key in self.cache:
            self.cache.move_to_end(prefix_key)
            return self.cache[prefix_key]
        return None

    def set(self, prefix_key: str, encoded: dict):
        if len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)
        self.cache[prefix_key] = encoded

    def make_key(self, task: str, **kwargs) -> str:
        parts = [task]
        for k, v in sorted(kwargs.items()):
            parts.append(f"{k}={v}")
        return "|".join(parts)


# ============================================================
# #24 — Batch Speculative Decoding
# ============================================================

class SpeculativeDecoder:
    """
    Drafts multiple tokens and verifies them in parallel.
    For CPU: uses same model with reduced beam width.
    """

    def __init__(self, model, draft_tokens: int = 3):
        self.model = model
        self.draft_tokens = draft_tokens

    def draft(self, input_ids: torch.Tensor,
              attention_mask: torch.Tensor, **gen_kwargs) -> List[int]:
        max_new = gen_kwargs.pop('max_new_tokens', self.draft_tokens)
        draft_kwargs = {**gen_kwargs, 'max_new_tokens': max_new,
                        'do_sample': True, 'temperature': 0.6}
        with torch.no_grad():
            draft_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **draft_kwargs,
            )
        return draft_ids[0, input_ids.size(-1):].tolist()

    def verify(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
               draft_tokens: List[int], **gen_kwargs) -> Tuple[List[int], int]:
        if not draft_tokens:
            return [], 0
        extended = torch.cat([
            input_ids,
            torch.tensor([draft_tokens], device=input_ids.device)
        ], dim=-1)
        draft_mask = torch.ones((1, extended.size(-1)), device=input_ids.device)
        with torch.no_grad():
            outputs = self.model(
                input_ids=extended, attention_mask=draft_mask, labels=extended
            )
        accepted = 0
        for i, token in enumerate(draft_tokens):
            logits = outputs.logits[0, input_ids.size(-1) + i]
            accepted_prob = F.softmax(logits, dim=-1)[token].item()
            if accepted_prob > 0.1 * (0.9 ** i):
                accepted += 1
            else:
                break
        return draft_tokens[:accepted], accepted


# ============================================================
# #31 — Top-p Nucleus Sampling
# ============================================================

def top_p_filtering(logits: torch.Tensor, top_p: float = 0.9,
                    filter_value: float = float('-inf')) -> torch.Tensor:
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    indices_to_remove = sorted_indices_to_remove.scatter(
        -1, sorted_indices, sorted_indices_to_remove
    )
    return logits.masked_fill(indices_to_remove, filter_value)


# ============================================================
# #32 — Top-k Sampling
# ============================================================

def top_k_filtering(logits: torch.Tensor, top_k: int = 50,
                    filter_value: float = float('-inf')) -> torch.Tensor:
    values, _ = torch.topk(logits, top_k, dim=-1)
    min_values = values[..., -1, None]
    return logits.masked_fill(logits < min_values, filter_value)


# ============================================================
# #33 — Temperature Scaling
# ============================================================

def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        return logits
    return logits / temperature


# ============================================================
# #34 — Beam Search with Diversity Penalty
# ============================================================

def diverse_beam_search(model, input_ids, num_beams: int = 4,
                        diversity_penalty: float = 0.5,
                        num_beam_groups: int = 2, **kwargs):
    try:
        return model.generate(
            input_ids=input_ids,
            num_beams=num_beams,
            diversity_penalty=diversity_penalty,
            num_beam_groups=num_beam_groups,
            **kwargs,
        )
    except Exception as e:
        logging.getLogger('sota').warning(f"Diverse beam search failed: {e}")
        return model.generate(input_ids=input_ids, num_beams=num_beams, **kwargs)


# ============================================================
# #36 — Early Stopping with Confidence
# ============================================================

class EarlyStopping:
    """Stops generation when confidence exceeds threshold."""

    def __init__(self, confidence_threshold: float = 0.95,
                 min_tokens: int = 5, window: int = 3):
        self.threshold = confidence_threshold
        self.min_tokens = min_tokens
        self.window = window
        self._conf_history = []

    def should_stop(self, step: int, logits: torch.Tensor) -> bool:
        if step < self.min_tokens:
            return False
        probs = F.softmax(logits[-1:].float(), dim=-1)
        conf = probs.max().item()
        self._conf_history.append(conf)
        if len(self._conf_history) > self.window:
            self._conf_history.pop(0)
        avg_conf = sum(self._conf_history) / len(self._conf_history)
        return avg_conf >= self.threshold
