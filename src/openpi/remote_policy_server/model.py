"""Model loading and inference logic."""

import logging

from openpi.policies.policy import Policy
from openpi.policies.policy_config import create_trained_policy
from openpi.remote_policy_server.config import ModelConfig
from openpi.shared import download
from openpi.training.config import get_config as get_train_config

logger = logging.getLogger(__name__)


def load_model(config: ModelConfig) -> Policy:
    """Load the π₀ model with the given configuration."""
    logger.info(f"Loading π₀ model from {config.checkpoint_path}...")
    openpi_cfg = get_train_config(config.model_name)
    checkpoint_dir = download.maybe_download(config.checkpoint_path)
    policy = create_trained_policy(openpi_cfg, checkpoint_dir)
    logger.info("Model loaded successfully!")
    return policy
