"""
SOTA Logging Utilities.
Handles truncation warnings, memory monitoring, experiment tracking.
Implements: FASE 1.5 (truncation warnings), memory logging.
"""

import os
import sys
import logging
import time
from datetime import datetime
from typing import Optional


_LOG_CONFIG = {}


def setup_logging(config: dict = None):
    """Initialize logging with optional config from yaml."""
    log_cfg = config.get('logging', {}) if config else {}
    level = getattr(logging, log_cfg.get('level', 'INFO').upper(), logging.INFO)
    _LOG_CONFIG['log_truncations'] = log_cfg.get('log_truncations', True)
    _LOG_CONFIG['log_memory'] = log_cfg.get('log_memory', True)
    _LOG_CONFIG['csv_log'] = log_cfg.get('csv_log', True)

    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
        stream=sys.stdout,
    )

    logger = logging.getLogger('sota')
    logger.info(f"Logging initialized at level {logging.getLevelName(level)}")
    return logger


def get_logger():
    return logging.getLogger('sota')


def warn_truncation(task: str, original_len: int, max_len: int, source: str):
    """Log a warning when a sample is truncated."""
    if not _LOG_CONFIG.get('log_truncations', True):
        return
    logger = logging.getLogger('sota')
    logger.warning(
        f"Truncated [{task}] from {original_len} to {max_len} tokens "
        f"({source})"
    )


def log_memory(stage: str = ""):
    """Log current memory usage."""
    if not _LOG_CONFIG.get('log_memory', True):
        return
    logger = logging.getLogger('sota')
    try:
        with open('/proc/self/status', 'r') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    mem_mb = int(line.split()[1]) / 1024
                    logger.info(f"[MEM {stage}] {mem_mb:.0f} MB RSS")
                    return
    except Exception:
        pass


class ExperimentTracker:
    """CSV-based experiment tracking with per-step metrics."""

    def __init__(self, log_dir: str = "logs", experiment_name: str = None):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        if experiment_name is None:
            experiment_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.csv_path = os.path.join(log_dir, f"{experiment_name}.csv")
        self.start_time = time.time()
        self._header_written = False

    def log_step(self, metrics: dict):
        """Log a step with arbitrary metrics dict."""
        elapsed = time.time() - self.start_time
        metrics['elapsed_s'] = f"{elapsed:.1f}"
        headers = list(metrics.keys())

        if not self._header_written:
            with open(self.csv_path, 'w') as f:
                f.write(','.join(headers) + '\n')
            self._header_written = True

        with open(self.csv_path, 'a') as f:
            row = ','.join(str(metrics.get(h, '')) for h in headers)
            f.write(row + '\n')

    def close(self):
        elapsed = time.time() - self.start_time
        logger = logging.getLogger('sota')
        logger.info(f"Experiment log saved to {self.csv_path} ({elapsed:.0f}s)")
