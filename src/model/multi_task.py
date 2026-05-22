"""
Multi-Task Heads — Task-specific output heads for all SOTA capabilities.
Implements: #50 (Policy+Value dual head), #81 (Shared encoder + task adapters),
            #83 (Uncertainty-weighted loss), #35 (Step-level value function).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union


TASK_NAMES = ['chess', 'translate', 'write', 'predict', 'pii']


class ChessHeads(nn.Module):
    """
    7 mode-specific output heads + value head.
    Policy head: vocabulary logits per mode.
    Value head: scalar evaluation (win probability).
    """
    def __init__(self, d_model: int, vocab_size: int, mode_names: List[str]):
        super().__init__()
        self.mode_names = mode_names
        self.mode_heads = nn.ModuleDict({
            name: nn.Linear(d_model, vocab_size, bias=False)
            for name in mode_names
        })
        self.value_head = nn.Linear(d_model, 1)

    def forward(self, hidden_states: torch.Tensor, mode: str = 'romaji'
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        last_hidden = hidden_states[:, -1, :]
        logits = self.mode_heads[mode](last_hidden)
        value = self.value_head(last_hidden)
        return logits, value


class TranslateHead(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.head(hidden_states)


class WriteHead(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.head(hidden_states)


class PredictHead(nn.Module):
    """Prediction head: supports both classification and regression."""
    def __init__(self, d_model: int, n_classes: int = 1000):
        super().__init__()
        self.classifier = nn.Linear(d_model, n_classes)
        self.regressor = nn.Linear(d_model, 1)

    def forward(self, hidden_states: torch.Tensor, mode: str = 'regression'
               ) -> torch.Tensor:
        last_hidden = hidden_states[:, -1, :]
        if mode == 'regression':
            return self.regressor(last_hidden)
        return self.classifier(last_hidden)


class PIIHead(nn.Module):
    """PII classification head: 3 classes [normal, pii, redacted]."""
    def __init__(self, d_model: int):
        super().__init__()
        self.head = nn.Linear(d_model, 3)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.head(hidden_states)


class TaskHeadsManager(nn.Module):
    """
    Manages all task-specific heads with uncertainty-weighted loss.

    Implements: #83 (Uncertainty-weighted loss balancing).
    """
    def __init__(self, d_model: int, vocab_size: int, mode_names: List[str]):
        super().__init__()
        self.chess = ChessHeads(d_model, vocab_size, mode_names)
        self.translate = TranslateHead(d_model, vocab_size)
        self.write = WriteHead(d_model, vocab_size)
        self.predict = PredictHead(d_model)
        self.pii = PIIHead(d_model)

        self.log_sigmas = nn.ParameterDict({
            task: nn.Parameter(torch.zeros(1))
            for task in TASK_NAMES
        })

    def forward(self, task: str, hidden_states: torch.Tensor,
                mode: str = 'romaji', predict_mode: str = 'regression'
                ) -> torch.Tensor:
        if task == 'chess':
            logits, _ = self.chess(hidden_states, mode)
            return logits
        elif task == 'translate':
            return self.translate(hidden_states)
        elif task == 'write':
            return self.write(hidden_states)
        elif task == 'predict':
            return self.predict(hidden_states, predict_mode)
        elif task == 'pii':
            return self.pii(hidden_states)
        raise ValueError(f"Unknown task: {task}")

    def compute_loss(self, task: str, logits: torch.Tensor,
                     labels: torch.Tensor, task_loss: torch.Tensor = None
                     ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        sigma_sq = (2 * self.log_sigmas[task]).exp()
        if task_loss is None:
            if logits.dim() == 3:
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )
            else:
                loss = F.mse_loss(logits, labels.float())
            task_loss = loss

        weighted_loss = task_loss / sigma_sq + 0.5 * self.log_sigmas[task]
        return weighted_loss, {'raw_loss': task_loss.item(), 'sigma': sigma_sq.item()}
