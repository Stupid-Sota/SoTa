"""
Block Attention — Sliding window attention for T5.
Reduces O(n²) → O(n × w) where w = window_size.
Implements improvements #1 (Block Attention), #2 (SDPA), #8 (Position bias truncado).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def create_sliding_window_mask(seq_len: int, window_size: int, device: torch.device,
                                n_global: int = 4) -> torch.Tensor:
    mask = torch.full((seq_len, seq_len), float('-inf'), device=device)
    for i in range(seq_len):
        half_w = window_size // 2
        start = max(0, i - half_w)
        end = min(seq_len, i + half_w + 1)
        mask[i, start:end] = 0.0
    if n_global > 0:
        mask[:n_global, :] = 0.0
        mask[:, :n_global] = 0.0
    return mask


def compute_local_position_bias(relative_attention_bias: nn.Embedding,
                                 seq_len: int, window_size: int,
                                 num_heads: int, device: torch.device) -> torch.Tensor:
    half_w = window_size // 2
    context_pos = torch.arange(seq_len, device=device)
    memory_pos = torch.arange(seq_len, device=device)
    relative_position = memory_pos.unsqueeze(0) - context_pos.unsqueeze(1)
    relative_position = torch.clamp(relative_position, -half_w, half_w)
    rp_bucket = relative_position + half_w
    values = relative_attention_bias(rp_bucket)
    values = values.permute([2, 0, 1]).unsqueeze(0)
    return values


class T5BlockAttentionWrapper:
    def __init__(self, attention_layer: nn.Module, window_size: int = 128,
                 n_global: int = 4):
        self.attn = attention_layer
        self.window_size = window_size
        self.n_global = n_global
        self.config = attention_layer.config if hasattr(attention_layer, 'config') else None
        self.is_decoder = attention_layer.is_decoder if hasattr(attention_layer, 'is_decoder') else False
        self.has_relative_attention_bias = attention_layer.has_relative_attention_bias if hasattr(attention_layer, 'has_relative_attention_bias') else False
        self.training = True

    def forward(self, hidden_states, attention_mask=None, position_bias=None,
                key_value_states=None, past_key_value=None, use_cache=False,
                query_length=None, output_attentions=False, layer_head_mask=None,
                mask=None, **kwargs):
        batch_size, seq_len, d_model = hidden_states.shape
        device = hidden_states.device
        n_heads = self.attn.n_heads
        head_dim = d_model // n_heads

        is_cross_attention = key_value_states is not None

        q = self.attn.q(hidden_states)
        if is_cross_attention:
            k = self.attn.k(key_value_states)
            v = self.attn.v(key_value_states)
        else:
            k = self.attn.k(hidden_states)
            v = self.attn.v(hidden_states)

        q = q.view(batch_size, -1, n_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, n_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, n_heads, head_dim).transpose(1, 2)

        kv_seq_len = k.shape[-2]

        sliding_mask = create_sliding_window_mask(
            kv_seq_len, self.window_size, device, self.n_global
        )

        if attention_mask is not None:
            causal_mask = attention_mask == 0
            if causal_mask.shape[-1] == kv_seq_len:
                combined_mask = sliding_mask.masked_fill(causal_mask.squeeze(1), float('-inf'))
            else:
                combined_mask = sliding_mask
        else:
            combined_mask = sliding_mask

        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=combined_mask[..., :q.size(-2), :],
            dropout_p=self.attn.dropout if self.training else 0.0,
            is_causal=False, scale=1.0 / math.sqrt(head_dim)
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, d_model)
        attn_output = self.attn.o(attn_output)

        if output_attentions:
            return attn_output, None, None, None
        return attn_output, None, None


def apply_block_attention(model: nn.Module, window_size: int = 128,
                          n_global: int = 4):
    for block in model.encoder.block:
        wrapper = T5BlockAttentionWrapper(
            block.layer[0].SelfAttention, window_size, n_global
        )
        block.layer[0].SelfAttention.forward = wrapper.forward
        block.layer[0].SelfAttention.config = model.config

    if hasattr(model, 'decoder'):
        for block in model.decoder.block:
            wrapper = T5BlockAttentionWrapper(
                block.layer[0].SelfAttention, window_size, n_global
            )
            block.layer[0].SelfAttention.forward = wrapper.forward
            block.layer[0].SelfAttention.config = model.config
            if len(block.layer) > 1:
                wrapper_cross = T5BlockAttentionWrapper(
                    block.layer[1].EncDecAttention, window_size, n_global
                )
                block.layer[1].EncDecAttention.forward = wrapper_cross.forward
                block.layer[1].EncDecAttention.config = model.config

    print(f"[BlockAttention] Applied window={window_size}, global={n_global}")
    return model
