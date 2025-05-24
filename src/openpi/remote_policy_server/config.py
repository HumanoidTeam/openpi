"""Configuration for the remote policy server."""

from dataclasses import dataclass
from typing import Tuple

@dataclass
class ModelConfig:
    # model_name: str = "pi0_fast_rainbow"
    # model_name: str = "pi0_fast_cut_states_v2_full_run"
    model_name: str = "denis_pi0_r2quavers_32bs"
    # checkpoint_path: str = "/home/ubuntu/isaacsim/checkpoints/pi0_fast_rainbow/pi0_rainbow_chips/11000"
    # checkpoint_path: str = "s3://hm-vla/checkpoints/pi0_fast_rainbow_poc"
    # checkpoint_path: str = "/home/ubuntu/downloaded_checkpoint/60000/"
    # checkpoint_path: str = "/home/ikot/openpi_mariano/checkpoints/pi0_fast_cut_states_v2_full_run/exp_pi0_fast_cut_states_v2_full_run/99999/"
    checkpoint_path: str = "/home/ikot/denis_checkpoint/"
    # checkpoint_path: str = "/home/ikot/openpi_mariano/checkpoints/pi0_fast_cut_states_v2_full_run/exp_pi0_fast_cut_states_v2_full_run/60000/"
    state_shape: Tuple[int, ...] = (16,)
    # image_shape: Tuple[int, ...] = (480, 848, 3)
    image_shape: Tuple[int, ...] = (224, 224, 3)
    action_shape: Tuple[int, ...] = (30, 16)
    # prompt: str = "Pick up the bowl" 
    # prompt: str = "Pick up Quavers"
    prompt: str = "Pick up yellow crisps"
    # is_8dof: bool = True
    is_8dof: bool = False
