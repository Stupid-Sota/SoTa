"""
ARM Memory Manager (#125-#130, #148-#155).
Dynamic batch sizing, memory-aware gradient accumulation, emergency checkpoint.
"""

import gc
import os
import torch
import math


class ARMMemoryManager:
    """Manages memory on ARM CPU with adaptive batching."""

    def __init__(self, target_memory_mb: int = 1800, safety_margin: float = 0.15):
        self.target_memory_mb = target_memory_mb
        self.safety_margin = safety_margin
        self.current_batch_size = 1

    def get_available_memory_mb(self) -> float:
        try:
            with open('/proc/self/status', 'r') as f:
                for line in f:
                    if line.startswith('VmRSS:'):
                        rss = int(line.split()[1]) / 1024
                        break
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if line.startswith('MemAvailable:'):
                        avail = int(line.split()[1]) / 1024
                        return avail
        except:
            return self.target_memory_mb
        return self.target_memory_mb

    def memory_pressure(self) -> float:
        avail = self.get_available_memory_mb()
        if avail <= 0:
            return 1.0
        return 1.0 - min(1.0, avail / self.target_memory_mb)

    def compute_batch_size(self, seq_len: int, base_batch: int = 4,
                           base_seq_len: int = 512) -> int:
        avail = self.get_available_memory_mb()
        mem_ratio = avail / self.target_memory_mb
        mem_ratio = max(0.1, min(1.0, mem_ratio))
        length_ratio = base_seq_len / max(seq_len, 1)
        batch = int(base_batch * mem_ratio * length_ratio)
        return max(1, batch)

    def compute_grad_accum(self, batch_size: int, target_batch: int = 8) -> int:
        if batch_size <= 0:
            return 8
        return max(1, math.ceil(target_batch / batch_size))

    def safe_call(self, fn, *args, **kwargs):
        try:
            result = fn(*args, **kwargs)
            self._collect_garbage()
            return result
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print("[Memory] OOM detected — collecting garbage and retrying")
                self._collect_garbage()
                return fn(*args, **kwargs)
            raise

    def _collect_garbage(self):
        gc.collect()
        if hasattr(torch, 'mps') and torch.mps.is_available():
            torch.mps.empty_cache()

    def auto_device_map(self, model) -> dict:
        return {'': 'cpu'}

    def emergency_checkpoint(self, model, optimizer, path: str, step: int):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            'step': step,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
        }
        tmp_path = path + '.tmp'
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, path)
        print(f"[Memory] Emergency checkpoint saved at step {step}")


def auto_batch_size(config: dict, seq_len: int) -> int:
    mem = ARMMemoryManager(
        target_memory_mb=config.get('target_memory_mb', 1800)
    )
    return mem.compute_batch_size(
        seq_len,
        base_batch=config.get('base_batch', 4),
        base_seq_len=config.get('base_seq_len', 512),
    )
