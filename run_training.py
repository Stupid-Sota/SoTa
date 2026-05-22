import sys
import gc
gc.collect()

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger('sota')

sys.stdout.write(f"Memory before: {__import__('psutil').Process().memory_info().rss / 1e9:.2f}GB\n")
sys.stdout.flush()

sys.path.insert(0, '.')
from src.training.train import run_full_training_pipeline

try:
    run_full_training_pipeline('config.yaml')
except Exception as e:
    logger.error(f"Training failed: {e}", exc_info=True)
    sys.exit(1)
