"""Configuration for the remote policy server."""

from dataclasses import dataclass
from typing import Tuple

@dataclass
class ModelConfig:
    # model_name: str = "pi0_fast_rainbow"
    model_name: str = "pi0_fast_cut_states_v2_full_run"
    # checkpoint_path: str = "/home/ubuntu/isaacsim/checkpoints/pi0_fast_rainbow/pi0_rainbow_chips/11000"
    # checkpoint_path: str = "s3://hm-vla/checkpoints/pi0_fast_rainbow_poc"
    checkpoint_path: str = "/home/ubuntu/downloaded_checkpoint/60000/"
    state_shape: Tuple[int, ...] = (16,)
    # image_shape: Tuple[int, ...] = (480, 848, 3)
    image_shape: Tuple[int, ...] = (224, 224, 3)
    action_shape: Tuple[int, ...] = (30, 16)
    # prompt: str = "Pick up the bowl" 
    prompt: str = "Pick up Quavers"
