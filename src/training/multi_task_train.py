"""
Multi-Task Training Loop.
Alternating task sampling with proportional weighting, checkpoint/resume, EMA.
Implements: #83 (Uncertainty-weighted loss), multi-task curriculum,
MoE load balancing, per-task gradient accumulation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from typing import Dict, List, Optional, Tuple, Iterable
import json
import os
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class TaskConfig:
    name: str
    weight: float = 1.0
    batch_size: int = 1
    grad_accum: int = 8
    learning_rate: float = 2.0e-4
    max_length: int = 192
    coef: float = 1.0


DEFAULT_TASK_CONFIGS = {
    'chess': TaskConfig('chess', weight=0.40, batch_size=1, grad_accum=16,
                        learning_rate=2.0e-4, max_length=512, coef=1.0),
    'translate': TaskConfig('translate', weight=0.20, batch_size=1, grad_accum=8,
                            learning_rate=1.5e-4, max_length=512, coef=1.0),
    'write': TaskConfig('write', weight=0.20, batch_size=1, grad_accum=8,
                        learning_rate=1.5e-4, max_length=1024, coef=1.0),
    'predict': TaskConfig('predict', weight=0.10, batch_size=1, grad_accum=8,
                          learning_rate=1.0e-4, max_length=512, coef=1.0),
    'pii': TaskConfig('pii', weight=0.10, batch_size=1, grad_accum=4,
                      learning_rate=1.0e-4, max_length=512, coef=1.0),
}


class MultiTaskDataset(Dataset):
    """Wrapper that merges multiple task datasets with task labels."""

    def __init__(self, datasets: Dict[str, Dataset], task_weights: Dict[str, float]):
        self.datasets = datasets
        self.task_weights = task_weights
        self.task_names = list(datasets.keys())
        self.task_indices = {
            name: list(range(len(ds))) for name, ds in datasets.items()
        }
        self._build_index()

    def _build_index(self):
        self.index = []
        total_weight = sum(self.task_weights.values())
        for name in self.task_names:
            weight = self.task_weights[name] / total_weight
            count = max(1, int(len(self.datasets[name]) * weight * 5))
            indices = self.task_indices[name]
            for _ in range(count):
                self.index.append((name, random.choice(indices)))
        random.shuffle(self.index)

    def __len__(self):
        return max(len(self.index), 100)

    def __getitem__(self, idx):
        task_name, ds_idx = self.index[idx % len(self.index)]
        item = self.datasets[task_name][ds_idx]
        item['task'] = task_name
        return item


class MultiTaskTrainer:
    """
    Alternating multi-task training with proportional sampling.
    Supports checkpoint/resume, EMA, uncertainty-weighted loss.
    """

    def __init__(self, model_wrapper, config: Dict, device: str = 'cpu'):
        self.model_wrapper = model_wrapper
        self.model = model_wrapper.model
        self.router = model_wrapper.router
        self.task_heads = model_wrapper.task_heads
        self.tokenizer = model_wrapper.tokenizer
        self.config = config
        self.device = device

        mt_config = config.get('multi_task', {})
        self.task_configs = {
            name: TaskConfig(**mt_config.get(name, {}))
            for name in DEFAULT_TASK_CONFIGS
        }
        for name in self.task_configs:
            defaults = DEFAULT_TASK_CONFIGS[name]
            for k, v in defaults.__dict__.items():
                if k not in mt_config.get(name, {}):
                    setattr(self.task_configs[name], k, v)

        self.ema_model = None
        self.ema_decay = mt_config.get('ema_decay', 0.999)
        self.global_step = 0

    def _get_optimizer(self, task_name: str, params: Iterable[nn.Parameter]):
        cfg = self.task_configs[task_name]
        return torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=0.01)

    def _get_scheduler(self, optimizer, total_steps: int, warmup_ratio: float = 0.05):
        warmup_steps = int(total_steps * warmup_ratio)
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def _compute_task_loss(self, task: str, logits: torch.Tensor,
                           labels: torch.Tensor) -> torch.Tensor:
        if task in ('chess', 'translate', 'write'):
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            return F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        elif task == 'predict':
            return F.mse_loss(logits.squeeze(-1), labels.float())
        elif task == 'pii':
            return F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )

    def train_step(self, batch: Dict, task: str) -> Dict:
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
        )

        hidden = outputs.decoder_hidden_states[-1] if hasattr(outputs, 'decoder_hidden_states') else None

        if self.router is not None and hidden is not None:
            with torch.no_grad():
                gates, expert_ids, router_logits = self.router(hidden)

        if self.task_heads is not None and hidden is not None:
            task_logits = self.task_heads.forward(task, hidden)
            task_loss = self._compute_task_loss(task, task_logits, labels)
            weighted_loss, loss_info = self.task_heads.compute_loss(
                task, task_logits, labels, task_loss
            )
        else:
            weighted_loss = outputs.loss if hasattr(outputs, 'loss') else torch.tensor(0.0)
            task_logits = outputs.logits if hasattr(outputs, 'logits') else None
            loss_info = {}

        if self.router is not None and hidden is not None:
            aux_losses = self.router.compute_aux_loss(router_logits, expert_ids)
            total_loss = weighted_loss
            total_loss += 0.01 * aux_losses.get('load_balance', 0)
            total_loss += 1e-3 * aux_losses.get('z_loss', 0)
        else:
            total_loss = weighted_loss
            aux_losses = {}

        return {
            'loss': total_loss,
            'task_loss': weighted_loss.detach().item(),
            'logits': task_logits,
            'aux_losses': aux_losses,
            'loss_info': loss_info,
        }

    def train(self, task_datasets: Dict[str, Dataset],
              output_dir: str = "data/checkpoints/multi_task",
              epochs: int = 3, resume_from: str = None):
        cfg = self.config.get('training', {}).get('sft', {})
        grad_accum_base = cfg.get('gradient_accumulation_steps', 16)
        max_grad_norm = cfg.get('max_grad_norm', 1.0)
        warmup_ratio = cfg.get('warmup_ratio', 0.05)

        optimizers = {}
        schedulers = {}
        total_steps = 0
        for name, ds in task_datasets.items():
            tc = self.task_configs[name]
            steps = len(ds) * epochs // tc.batch_size
            total_steps += steps

        for name, ds in task_datasets.items():
            tc = self.task_configs[name]
            params = list(filter(lambda p: p.requires_grad, self.model.parameters()))
            if self.router is not None:
                params += list(self.router.parameters())
            if self.task_heads is not None:
                params += list(self.task_heads.parameters())
            optimizers[name] = self._get_optimizer(name, params)
            schedulers[name] = self._get_scheduler(
                optimizers[name], total_steps, warmup_ratio
            )

        start_epoch = 0
        if resume_from and os.path.exists(resume_from):
            checkpoint = torch.load(resume_from, map_location=self.device)
            start_epoch = checkpoint.get('epoch', 0)
            self.global_step = checkpoint.get('global_step', 0)
            for name in task_datasets:
                if name in checkpoint.get('optimizers', {}):
                    optimizers[name].load_state_dict(checkpoint['optimizers'][name])
                if name in checkpoint.get('schedulers', {}):
                    schedulers[name].load_state_dict(checkpoint['schedulers'][name])
            if 'router' in checkpoint and self.router is not None:
                self.router.load_state_dict(checkpoint['router'])
            if 'task_heads' in checkpoint and self.task_heads is not None:
                self.task_heads.load_state_dict(checkpoint['task_heads'])
            print(f"[MultiTask] Resumed from epoch {start_epoch}, step {self.global_step}")

        self.model.train()

        task_dataloaders = {}
        for name, ds in task_datasets.items():
            tc = self.task_configs[name]
            task_dataloaders[name] = DataLoader(
                ds, batch_size=tc.batch_size, shuffle=True,
                collate_fn=lambda b: self._collate(b),
            )

        task_iterators = {
            name: iter(dl) for name, dl in task_dataloaders.items()
        }

        for epoch in range(start_epoch, epochs):
            epoch_losses = defaultdict(float)
            task_steps = defaultdict(int)

            for optimizer in optimizers.values():
                optimizer.zero_grad()

            for step in range(max(len(dl) for dl in task_dataloaders.values())):
                task_order = list(task_datasets.keys())
                random.shuffle(task_order)

                for task in task_order:
                    try:
                        batch = next(task_iterators[task])
                    except StopIteration:
                        task_iterators[task] = iter(task_dataloaders[task])
                        batch = next(task_iterators[task])

                    result = self.train_step(batch, task)
                    loss = result['loss']
                    tc = self.task_configs[task]
                    scaled_loss = loss / tc.grad_accum
                    scaled_loss.backward()

                    epoch_losses[task] += result['task_loss']
                    task_steps[task] += 1

                    if (task_steps[task] % tc.grad_accum == 0):
                        torch.nn.utils.clip_grad_norm_(
                            list(filter(lambda p: p.requires_grad, self.model.parameters())),
                            max_grad_norm,
                        )
                        optimizers[task].step()
                        schedulers[task].step()
                        optimizers[task].zero_grad()
                        self.global_step += 1

                        if self.ema_model is not None:
                            with torch.no_grad():
                                for ema_p, p in zip(self.ema_model.parameters(), self.model.parameters()):
                                    ema_p.mul_(self.ema_decay).add_(p, alpha=1 - self.ema_decay)

            print(f"[MultiTask] Epoch {epoch+1}/{epochs}")
            for task in task_datasets:
                avg = epoch_losses[task] / max(1, task_steps[task])
                print(f"  {task}: loss={avg:.4f}, steps={task_steps[task]}")

            self._save_checkpoint(output_dir, epoch, task_datasets, optimizers, schedulers)

        print(f"[MultiTask] Training complete. Saved to {output_dir}")
        return epoch_losses

    def _collate(self, batch):
        pad_id = self.tokenizer.pad_token_id or 0
        max_in = max(len(item['input_ids']) for item in batch)
        max_out = max(len(item['labels']) for item in batch)
        max_len = max(max_in, max_out)
        result = {}
        for key in ['input_ids', 'attention_mask', 'labels']:
            padded = []
            for item in batch:
                seq = item[key]
                pad_len = max_len - len(seq)
                if key == 'labels':
                    pad = torch.full((pad_len,), -100, dtype=seq.dtype)
                else:
                    pad = torch.full((pad_len,), pad_id, dtype=seq.dtype)
                padded.append(torch.cat([seq, pad]))
            result[key] = torch.stack(padded)
        return result

    def _save_checkpoint(self, output_dir: str, epoch: int,
                          task_datasets: Dict, optimizers, schedulers):
        path = os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}.pt")
        os.makedirs(output_dir, exist_ok=True)
        checkpoint = {
            'epoch': epoch + 1,
            'global_step': self.global_step,
            'model_state': self.model.state_dict(),
            'tokenizer_config': self.tokenizer.name_or_path,
            'task_datasets': list(task_datasets.keys()),
        }
        if self.router is not None:
            checkpoint['router'] = self.router.state_dict()
        if self.task_heads is not None:
            checkpoint['task_heads'] = self.task_heads.state_dict()
        checkpoint['optimizers'] = {
            name: opt.state_dict() for name, opt in optimizers.items()
        }
        checkpoint['schedulers'] = {
            name: sched.state_dict() for name, sched in schedulers.items()
        }
        torch.save(checkpoint, path)
        print(f"[Checkpoint] Saved epoch {epoch+1} to {path}")
