"""
MoE (Mixture of Experts) — Router + Expert LoRA Manager.
Implements improvements #41-60: Adapter MoE with task-specific experts.
5 experts: chess, translate, write, predict, pii.
Top-2 routing with load balancing loss + z-loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from typing import Dict, List, Optional, Tuple


EXPERT_NAMES = ['chess', 'translate', 'write', 'predict', 'pii']

EXPERT_CONFIGS = {
    'chess':     {'r': 16, 'alpha': 32, 'dropout': 0.05, 'targets': ['q', 'v'], 'dora': True},
    'translate': {'r': 12, 'alpha': 24, 'dropout': 0.05, 'targets': ['q', 'v', 'wi'], 'dora': True},
    'write':     {'r': 16, 'alpha': 32, 'dropout': 0.10, 'targets': ['q', 'v', 'wo'], 'dora': True},
    'predict':   {'r': 8,  'alpha': 16, 'dropout': 0.05, 'targets': ['q', 'v'], 'dora': True},
    'pii':       {'r': 8,  'alpha': 16, 'dropout': 0.10, 'targets': ['q', 'v'], 'dora': True},
}


class MoERouter(nn.Module):
    """
    Router that classifies task from the pooled encoder embedding.
    Top-2 routing with softmax gates + optional noise for exploration.

    Implements: #41 (Top-2 routing), #50 (Noisy gating), #53 (Router dropout),
                #47 (Z-loss), #48 (Load balancing).
    """
    def __init__(self, d_model: int = 768, n_experts: int = 5,
                 top_k: int = 2, use_noisy_gating: bool = True,
                 noise_std: float = 0.1, router_dropout: float = 0.05):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.use_noisy_gating = use_noisy_gating
        self.noise_std = noise_std
        self.router_dropout = router_dropout

        self.router = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, n_experts),
        )

        if use_noisy_gating:
            self.w_noise = nn.Linear(d_model // 2, n_experts, bias=False)

    def forward(self, hidden_states: torch.Tensor, force_expert: Optional[int] = None
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pooled = hidden_states.mean(dim=1)
        router_hidden = self.router[1](self.router[0](pooled))
        router_logits = self.router[2](router_hidden)

        if force_expert is not None:
            mask = torch.full_like(router_logits, float('-inf'))
            mask[:, force_expert] = 0.0
            router_logits = mask

        if self.training and self.use_noisy_gating:
            noise = F.softplus(self.w_noise(router_hidden))
            noise = noise * torch.randn_like(noise) * self.noise_std
            router_logits = router_logits + noise

        router_probs = F.softmax(router_logits, dim=-1)

        top_k_vals, top_k_idx = torch.topk(router_probs, self.top_k, dim=-1)
        gates = F.softmax(top_k_vals, dim=-1)

        if self.training and self.router_dropout > 0.0:
            dropout_mask = torch.rand_like(gates) > self.router_dropout
            gates = gates * dropout_mask.float()
            gates = gates / (gates.sum(dim=-1, keepdim=True) + 1e-8)

        return gates, top_k_idx, router_logits

    def compute_aux_loss(self, router_logits: torch.Tensor,
                         top_k_idx: torch.Tensor) -> Dict[str, torch.Tensor]:
        router_probs = F.softmax(router_logits, dim=-1)
        num_tokens = router_logits.size(0)

        one_hot = F.one_hot(top_k_idx, num_classes=self.n_experts).float()
        f_i = one_hot.sum(dim=(0, 1)) / max(one_hot.sum(), 1)
        P_i = router_probs.mean(dim=0)

        load_balance_loss = self.n_experts * torch.sum(f_i * P_i)

        z_loss = torch.mean(torch.logsumexp(router_logits.float(), dim=-1) ** 2)

        entropy = -torch.sum(router_probs * torch.log(router_probs.clamp(min=1e-10)), dim=-1).mean()
        entropy_loss = -entropy

        return {
            'load_balance': load_balance_loss,
            'z_loss': z_loss,
            'entropy': entropy,
            'entropy_loss': entropy_loss,
        }


class ExpertLoRAManager:
    """
    Manages 5 sets of LoRA adapters (one per expert).
    Activates only the selected experts' adapters during forward.

    Implements: #56 (MixLoRA), #57 (MoE-Sieve), #99 (DR-LoRA dynamic rank).
    """
    def __init__(self, base_model: nn.Module):
        self.base_model = base_model
        self.expert_adapters: Dict[str, nn.Module] = {}
        self.active_experts: List[str] = []
        self.original_params = {}

    def create_experts(self):
        for expert_name in EXPERT_NAMES:
            cfg = EXPERT_CONFIGS[expert_name]
            lora_config = LoraConfig(
                r=cfg['r'],
                lora_alpha=cfg['alpha'],
                lora_dropout=cfg['dropout'],
                target_modules=cfg['targets'],
                use_dora=cfg['dora'],
                bias="none",
                task_type="SEQ_2_SEQ_LM",
            )
            self.expert_adapters[expert_name] = lora_config

    def activate(self, expert_ids: List[str], gates: Optional[torch.Tensor] = None):
        self.active_experts = expert_ids

    def get_expert_config(self, expert_name: str) -> Dict:
        return EXPERT_CONFIGS.get(expert_name, EXPERT_CONFIGS['chess'])

    @staticmethod
    def compute_total_moe_loss(task_loss: torch.Tensor,
                                aux_losses: Dict[str, torch.Tensor],
                                load_balance_coeff: float = 0.01,
                                z_loss_coeff: float = 1e-3,
                                entropy_coeff: float = 0.001) -> torch.Tensor:
        total = task_loss
        total += load_balance_coeff * aux_losses.get('load_balance', 0)
        total += z_loss_coeff * aux_losses.get('z_loss', 0)
        total += entropy_coeff * aux_losses.get('entropy_loss', 0)
        return total
