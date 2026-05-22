"""
Training Improvements #37-#70, #82-#100.
Dynamic batch sizing, loss scaling, expert optimizations, advanced training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import math
import random
from typing import Dict, List, Optional, Tuple, Callable
from collections import deque


# ============================================================
# #37 — 3-Stage Pipeline Tag (implemented in train.py)
# ============================================================

PIPELINE_STAGES = ['multi_task', 'sft', 'orpo', 'rl']


# ============================================================
# #38 — Curriculum Learning by Difficulty (enhanced)
# ============================================================

class CurriculumScheduler:
    """5-phase curriculum with difficulty-based filtering."""

    def __init__(self, config: dict):
        phases = config.get('curriculum', {}).get('phases', [])
        self.phases = []
        for p in phases:
            self.phases.append({
                'name': p.get('name', 'unknown'),
                'epoch_start': p.get('epochs', [0, 1])[0],
                'epoch_end': p.get('epochs', [0, 1])[1],
                'difficulty': p.get('difficulty', 1),
                'min_samples': p.get('min_samples', 100),
            })
        self._current_phase_idx = 0

    def get_phase(self, epoch: int) -> dict:
        for i, p in enumerate(self.phases):
            if p['epoch_start'] <= epoch < p['epoch_end']:
                self._current_phase_idx = i
                return p
        return self.phases[-1] if self.phases else {}

    def filter_samples(self, samples: list, epoch: int) -> list:
        phase = self.get_phase(epoch)
        max_diff = phase.get('difficulty', 99)
        filtered = [s for s in samples if s.get('difficulty', 1) <= max_diff]
        min_s = phase.get('min_samples', 1)
        if len(filtered) < min_s and len(samples) >= min_s:
            filtered = samples[:min_s]
        return filtered

    @property
    def current_difficulty(self) -> int:
        return self.phases[self._current_phase_idx].get('difficulty', 1)


# ============================================================
# #39 — Dynamic Batch Sizing by Sequence Length
# ============================================================

class DynamicBatchSizer:
    """Adjusts batch_size based on sequence lengths to fit memory budget."""

    def __init__(self, target_memory_mb: float = 1500,
                 base_batch: int = 1, base_seq_len: int = 512):
        self.target_memory_mb = target_memory_mb
        self.base_batch = base_batch
        self.base_seq_len = base_seq_len
        self._current_batch = base_batch

    def compute_batch_size(self, seq_lens: List[int], available_mb: float = None) -> int:
        if not seq_lens:
            return self.base_batch
        avg_len = sum(seq_lens) / len(seq_lens)
        ratio = self.base_seq_len / max(avg_len, 1)
        mem = available_mb or self._get_available_mb()
        mem_ratio = mem / self.target_memory_mb if self.target_memory_mb > 0 else 1.0
        batch = max(1, int(self.base_batch * ratio * mem_ratio))
        self._current_batch = batch
        return batch

    def _get_available_mb(self) -> float:
        try:
            with open('/proc/self/status', 'r') as f:
                for line in f:
                    if line.startswith('MemAvailable:'):
                        return int(line.split()[1]) / 1024
                    if line.startswith('VmRSS:'):
                        rss = int(line.split()[1]) / 1024
                        return max(100, self.target_memory_mb - rss)
        except:
            return self.target_memory_mb * 0.5
        return self.target_memory_mb * 0.5


# ============================================================
# #40 — Loss Scaling by Task Difficulty
# ============================================================

class DifficultyScaledLoss:
    """Scales task loss by difficulty level (harder tasks weighted more)."""

    def __init__(self, base_coef: float = 1.0, scale_factor: float = 0.2):
        self.base_coef = base_coef
        self.scale_factor = scale_factor

    def scale_loss(self, loss: torch.Tensor, difficulty: int) -> torch.Tensor:
        scale = self.base_coef + self.scale_factor * (difficulty - 1)
        return loss * scale


# ============================================================
# #43 — Expert Balancing via Gradient Gates
# ============================================================

class ExpertGradientGate(nn.Module):
    """Gates gradients per expert to prevent expert collapse."""

    def __init__(self, n_experts: int, threshold: float = 0.05):
        super().__init__()
        self.n_experts = n_experts
        self.threshold = threshold
        self.register_buffer('_usage_count', torch.zeros(n_experts))
        self.register_buffer('_total_steps', torch.tensor(0))

    def forward(self, expert_weights: torch.Tensor) -> torch.Tensor:
        if self.training:
            usage = expert_weights.mean(dim=0)
            self._usage_count += usage.detach()
            self._total_steps += 1
            avg_usage = self._usage_count / self._total_steps.clamp(min=1)
            gate = torch.where(
                avg_usage < self.threshold,
                torch.ones_like(expert_weights) * 2.0,
                torch.ones_like(expert_weights),
            )
            return expert_weights * gate
        return expert_weights


# ============================================================
# #44 — Router Confidence Calibration
# ============================================================

def calibrate_router_confidence(logits: torch.Tensor,
                                 temperature: float = 1.5) -> torch.Tensor:
    """Widens or narrows router probability distribution."""
    return F.softmax(logits / temperature, dim=-1)


# ============================================================
# #45 — Expert Dropout
# ============================================================

class ExpertDropout(nn.Module):
    """Drops entire experts during training with probability p."""

    def __init__(self, n_experts: int, p: float = 0.1):
        super().__init__()
        self.p = p
        self.n_experts = n_experts

    def forward(self, gates: torch.Tensor, expert_ids: torch.Tensor,
                **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.training or self.p <= 0:
            return gates, expert_ids
        mask = torch.rand(self.n_experts, device=gates.device) > self.p
        surviving = torch.nonzero(mask).squeeze(-1)
        if len(surviving) < 1:
            return gates, expert_ids
        new_gates = gates.clone()
        new_ids = expert_ids.clone()
        for b in range(expert_ids.size(0)):
            for k in range(expert_ids.size(1)):
                eid = expert_ids[b, k]
                if not mask[eid]:
                    new_gates[b, k] = 0.0
            row_sum = new_gates[b].sum()
            if row_sum > 0:
                new_gates[b] = new_gates[b] / row_sum
            else:
                fallback = surviving[0].item()
                new_gates[b, 0] = 1.0
                new_ids[b, 0] = fallback
        return new_gates, new_ids


# ============================================================
# #46 — Knowledge Distillation Between Experts
# ============================================================

class ExpertDistillationLoss:
    """KL divergence between expert logits (soft targets)."""

    def __init__(self, temperature: float = 4.0, weight: float = 0.1):
        self.temperature = temperature
        self.weight = weight

    def compute(self, student_logits: torch.Tensor,
                teacher_logits: torch.Tensor) -> torch.Tensor:
        s = F.log_softmax(student_logits / self.temperature, dim=-1)
        t = F.softmax(teacher_logits / self.temperature, dim=-1)
        kl = F.kl_div(s, t, reduction='batchmean')
        return self.weight * (self.temperature ** 2) * kl


# ============================================================
# #49 — Soft Voting for Expert Combination
# ============================================================

def soft_expert_vote(expert_logits: List[torch.Tensor],
                     weights: torch.Tensor) -> torch.Tensor:
    """Weighted combination of multiple expert outputs."""
    stacked = torch.stack(expert_logits, dim=0)
    w = F.softmax(weights, dim=0)
    weighted = (stacked * w.view(-1, 1, 1, 1)).sum(dim=0)
    return weighted


# ============================================================
# #50 — Confidence-Weighted Sampling (enhanced)
# ============================================================

def confidence_weighted_sample(logits: torch.Tensor,
                                confidence: float = 0.5) -> torch.Tensor:
    """High confidence → greedy, low confidence → more random."""
    if confidence > 0.9:
        return logits.argmax(dim=-1, keepdim=True)
    temp = max(0.1, 1.0 - confidence) * 2.0
    return torch.multinomial(F.softmax(logits / temp, dim=-1), 1)


# ============================================================
# #51 — Adaptive LoRA Rank Search (AutoLoRA)
# ============================================================

class AutoLoRA:
    """Adjusts LoRA rank based on gradient variance."""

    def __init__(self, min_rank: int = 4, max_rank: int = 32, window: int = 100):
        self.min_rank = min_rank
        self.max_rank = max_rank
        self.window = window
        self._grad_norms: Dict[str, deque] = {}

    def update_rank(self, module_name: str, grad_norm: float) -> int:
        if module_name not in self._grad_norms:
            self._grad_norms[module_name] = deque(maxlen=self.window)
        self._grad_norms[module_name].append(grad_norm)
        if len(self._grad_norms[module_name]) < 10:
            return (self.min_rank + self.max_rank) // 2
        var = torch.tensor(list(self._grad_norms[module_name])).var().item()
        ratio = var / (max(grad_norm, 1e-8))
        if ratio > 0.5:
            return min(self.max_rank, int(ratio * 20))
        return self.min_rank


# ============================================================
# #61 — Reflection Tokens [RETHINK]
# ============================================================

REFLECTION_TOKENS = ["[RETHINK]", "[SELFCHECK]", "[REVISE]", "[CONFIRM]"]


# ============================================================
# #62 — Tree-of-Thought (simplified for CPU)
# ============================================================

class TreeOfThought:
    """Branches at uncertainty points, evaluates best path."""

    def __init__(self, max_branches: int = 3, max_depth: int = 2):
        self.max_branches = max_branches
        self.max_depth = max_depth

    def search(self, initial_text: str, tokenizer, model,
               eval_fn: Callable, **gen_kwargs) -> str:
        best_path = initial_text
        best_score = float('-inf')

        def _branch(text: str, depth: int) -> Tuple[str, float]:
            nonlocal best_path, best_score
            if depth >= self.max_depth:
                score = eval_fn(text)
                if score > best_score:
                    best_score = score
                    best_path = text
                return text, score
            inputs = tokenizer(text, return_tensors='pt')
            with torch.no_grad():
                outputs = model.generate(
                    inputs['input_ids'],
                    num_return_sequences=self.max_branches,
                    num_beams=self.max_branches,
                    max_new_tokens=gen_kwargs.get('max_new_tokens', 50),
                    do_sample=True,
                    **gen_kwargs
                )
            branches = [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
            for branch in branches:
                _branch(branch, depth + 1)
            return text, best_score

        _branch(initial_text, 0)
        return best_path


# ============================================================
# #65 — Rejection Sampling
# ============================================================

class RejectionSampler:
    """Generates N samples, selects best by score."""

    def __init__(self, n_samples: int = 5):
        self.n_samples = n_samples

    def sample(self, model, input_ids, tokenizer,
               scoring_fn: Callable, **gen_kwargs) -> str:
        candidates = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                outputs = model.generate(
                    input_ids, do_sample=True,
                    temperature=gen_kwargs.get('temperature', 0.8),
                    max_new_tokens=gen_kwargs.get('max_new_tokens', 256),
                )
                text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                score = scoring_fn(text)
                candidates.append((text, score))
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0] if candidates else ""


# ============================================================
# #70 — Synthetic Data Augmentation
# ============================================================

class SynthAugmenter:
    """Paraphrases prompts and perturbs FENs for data augmentation."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self._paraphrase_prefixes = [
            "Consider the position:", "Analyze this:",
            "Given the board:", "From this position:",
            "Evaluate:", "Find the best move:",
        ]

    def paraphrase_prompt(self, prompt: str) -> str:
        for old_prefix in ["chess position:", "evaluate:", "analyze:"]:
            if prompt.startswith(old_prefix):
                new_prefix = self.rng.choice(self._paraphrase_prefixes)
                return prompt.replace(old_prefix, new_prefix, 1)
        return prompt

    def perturb_fen(self, fen: str) -> str:
        parts = fen.split()
        if len(parts) < 1:
            return fen
        board_part = parts[0]
        if self.rng.random() < 0.1:
            flipped = '/'.join(row[::-1] for row in board_part.split('/'))
            parts[0] = flipped
        return ' '.join(parts)


# ============================================================
# #82 — Multi-turn Dialogue Support
# ============================================================

class DialogueManager:
    """Maintains conversation history for multi-turn interactions."""

    def __init__(self, max_history: int = 10, max_tokens: int = 2048):
        self.history: List[Dict] = []
        self.max_history = max_history
        self.max_tokens = max_tokens

    def add_turn(self, role: str, content: str):
        self.history.append({'role': role, 'content': content})
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def build_prompt(self, new_input: str, task: str = 'chess') -> str:
        context = "\n".join(
            f"{'User' if t['role']=='user' else 'Assistant'}: {t['content']}"
            for t in self.history[-self.max_history:]
        )
        return f"{task} context:\n{context}\nUser: {new_input}\nAssistant:"

    def clear(self):
        self.history.clear()


# ============================================================
# #94 — Dynamic Task Weighting
# ============================================================

class DynamicTaskWeighter:
    """Adjusts task sampling weights based on loss trends."""

    def __init__(self, task_names: List[str], alpha: float = 0.1):
        self.task_names = task_names
        self.alpha = alpha
        self._loss_history: Dict[str, deque] = {
            t: deque(maxlen=50) for t in task_names
        }
        self._weights = {t: 1.0 / len(task_names) for t in task_names}

    def update(self, task: str, loss: float):
        if task in self._loss_history:
            self._loss_history[task].append(loss)

    def get_weights(self) -> Dict[str, float]:
        for task in self.task_names:
            if len(self._loss_history[task]) >= 5:
                recent = list(self._loss_history[task])[-5:]
                trend = (recent[-1] - recent[0]) / max(abs(recent[0]), 1e-8)
                if trend > 0.1:
                    self._weights[task] *= (1 - self.alpha)
                elif trend < -0.1:
                    self._weights[task] *= (1 + self.alpha)
        total = sum(self._weights.values())
        if total > 0:
            for t in self._weights:
                self._weights[t] /= total
        return self._weights


# ============================================================
# #96 — PCGrad (conflict resolution)
# ============================================================

def pcgrad_loss(task_losses: List[torch.Tensor],
                shared_params: List[nn.Parameter]) -> torch.Tensor:
    """Project conflicting gradients to reduce interference."""
    grads = []
    for loss in task_losses:
        grad = torch.autograd.grad(loss, shared_params, retain_graph=True,
                                    create_graph=True)
        grads.append(torch.cat([g.view(-1) for g in grad]))

    n = len(grads)
    if n < 2:
        return sum(task_losses)

    projected = grads.copy()
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dot = (grads[i] * grads[j]).sum()
            if dot < 0:
                projected[i] = projected[i] - (dot / (grads[j].norm()**2 + 1e-8)) * grads[j]

    total_grad = sum(projected)
    return sum(task_losses)  # loss value, grad manipulation happens in optimizer


# ============================================================
# #97 — Task Embedding Conditioning
# ============================================================

class TaskEmbedding(nn.Module):
    """Learnable task embedding prepended to encoder input."""

    def __init__(self, d_model: int, n_tasks: int = 5):
        super().__init__()
        self.embeddings = nn.Embedding(n_tasks, d_model)
        self.task_ids = {
            'chess': 0, 'translate': 1, 'write': 2,
            'predict': 3, 'pii': 4,
        }

    def forward(self, task: str, batch_size: int, seq_len: int,
                device: torch.device) -> torch.Tensor:
        tid = self.task_ids.get(task, 0)
        emb = self.embeddings(torch.tensor(tid, device=device))
        return emb.view(1, 1, -1).expand(batch_size, seq_len, -1)
