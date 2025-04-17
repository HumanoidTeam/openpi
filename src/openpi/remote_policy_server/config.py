"""Configuration for the remote policy server."""

from dataclasses import dataclass
from typing import Tuple

@dataclass
class ModelConfig:
    model_name: str = "pi0_fast_rainbow"
    # checkpoint_path: str = "/home/ubuntu/isaacsim/checkpoints/pi0_fast_rainbow/pi0_rainbow_chips/11000"
    checkpoint_path: str = "s3://hm-vla/checkpoints/pi0_fast_rainbow_poc"
    state_shape: Tuple[int, ...] = (16,)
    image_shape: Tuple[int, ...] = (480, 848, 3)
    action_shape: Tuple[int, ...] = (10, 16)
    prompt: str = "Pick up the bowl" 