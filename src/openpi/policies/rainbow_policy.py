"""Rainbow robot policy transforms."""

import dataclasses
import logging


import einops
import numpy as np
import torch

from openpi import transforms
from openpi.models import model as _model


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("compute_norm_stats.log")],
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def make_rainbow_example() -> dict:
    """Creates a random input example for the Rainbow policy."""
    return {
        "observation.state": np.random.rand(16),
        "observation.image.head": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation.image.wrist_right": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    """Ensures that the image is a uint8 array with shape (H, W, C).
    Converts from float (assumed to be in [0,1]) or from a (C, H, W) layout.
    """
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    # If the image is in (C, H, W) format, convert it to (H, W, C)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class RainbowInputs(transforms.DataTransformFn):
    """Data transformation for the Rainbow robot.
    Converts raw dataset inputs into the model's expected format.

    Expected keys in the input dictionary:
      - "observation.state": a 16-dimensional array (float64).
      - "observation.image.head": an image array (480x640x3).
      - "observation.image.wrist_right": an image array (480x640x3).
      - "prompt": a string instruction.
      - "action": a 16-dimensional array (float64).
    """

    action_dim: int  # For Rainbow, set this to 16
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        # We only mask padding for pi0 model, not pi0-FAST. Do not change this for your own dataset.
        mask_padding = self.model_type == _model.ModelType.PI0

        # Log available keys for debugging
        logger.debug(f"Available keys in input data: {list(data.keys())}")

        # Process the proprioceptive state

        state = transforms.pad_to_dim(
            self._get_state_short(data["observation.state"]), self.action_dim
        )
        # state = data["observation.state"]

        # Process the images
        base_image = _parse_image(data["observation.image.head"])
        wrist_image = _parse_image(data["observation.image.wrist_right"])

        # Verify image dimensions
        # if base_image.shape != (480, 640, 3) or wrist_image.shape != (480, 640, 3):
        #     raise ValueError(
        #         f"Expected image shapes (480, 640, 3), got {base_image.shape} and {wrist_image.shape}"
        #     )

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                # "wrist_right_0_rgb": wrist_image,
                "left_wrist_0_rgb": np.zeros_like(base_image),
                "right_wrist_0_rgb": wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                # "wrist_right_0_rgb": np.True_,
                # Mask any non-existent images with False (if ``mask_padding`` is True).
                "left_wrist_0_rgb": np.False_ if mask_padding else np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            # We are padding to the model action dim.
            inputs["actions"] = transforms.pad_to_dim(
                self._get_actions_short(data["actions"]), self.action_dim
            )
            # inputs["actions"] = data["actions"]
        elif "action" in data:
            inputs["actions"] = transforms.pad_to_dim(
                self._get_actions_short(data["action"]), self.action_dim
            )
            # inputs["actions"] = data["action"]

        # Add prompt if available
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        logger.debug(f"Transformed input keys: {list(inputs.keys())}")
        return inputs

    @staticmethod
    def _get_state_short(joint_states: torch.Tensor) -> torch.Tensor:
        """Extracts the first 7 elements of the action array - for the right wrist, and one gripper states"""
        if isinstance(joint_states, np.ndarray):
            joint_states = torch.from_numpy(joint_states)
        ret_value = torch.cat([joint_states[:7], joint_states[-2:-1]], dim=0)
        assert len(ret_value) == 8, f"Expected 8 elements, got {len(ret_value)}"
        return ret_value

    @staticmethod
    def _get_actions_short(actions_chunk: torch.Tensor) -> torch.Tensor:
        # actions_chunk expected to be of shape (action_horizon, action_dim)
        if isinstance(actions_chunk, np.ndarray):
            actions_chunk = torch.from_numpy(actions_chunk)
        ret_value = torch.cat([actions_chunk[:, :7], actions_chunk[:, -2:-1]], dim=1)
        assert ret_value.shape[1] == 8, f"Expected 8 elements, got {ret_value.shape[1]}"
        return ret_value


@dataclasses.dataclass(frozen=True)
class RainbowOutputs(transforms.DataTransformFn):
    """Converts model outputs back to the Rainbow dataset format."""

    def __call__(self, data: dict) -> dict:
        # Log available keys for debugging
        logger.debug(f"Available keys in output data: {list(data.keys())}")
        actions = data["actions"]
        assert actions.shape[1] == 8, f"Expected 8 elements, got {actions.shape[1]}"
        return {"actions": actions}


@dataclasses.dataclass(frozen=True)
class RainbowInputs8DOF(RainbowInputs):
    """Data transformation for the Rainbow robot with 8-DOF (single arm).
    Converts raw dataset inputs into the model's expected format.

    Expected keys in the input dictionary:
      - "observation.state": an 8-dimensional array (float64) for single arm.
      - "observation.image.head": an image array (480x640x3).
      - "observation.image.wrist_right": an image array (480x640x3).
      - "prompt": a string instruction.
      - "action": an 8-dimensional array (float64).
    """

    action_dim: int  # For single arm Rainbow, set this to 8

    def __call__(self, data: dict) -> dict:
        # Process the proprioceptive state - only use first 8 dimensions
        state = data["observation.state"][:8]  # Take only first 8 DOFs

        # Process the images
        base_image = _parse_image(data["observation.image.head"])
        wrist_image = _parse_image(data["observation.image.wrist_right"])

        # Verify image dimensions
        if base_image.shape != (480, 640, 3) or wrist_image.shape != (480, 640, 3):
            raise ValueError(
                f"Expected image shapes (480, 640, 3), got {base_image.shape} and {wrist_image.shape}"
            )

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "wrist_right_0_rgb": wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "wrist_right_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            # Only take first 8 dimensions of actions
            inputs["actions"] = data["actions"][:, :8]
        elif "action" in data:
            inputs["actions"] = data["action"][:8]

        # Add prompt if available
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class RainbowOutputs8DOF(RainbowOutputs):
    """Converts model outputs back to the Rainbow dataset format for 8-DOF."""

    def __call__(self, data: dict) -> dict:
        # Only return the first 8 dimensions
        return {"actions": np.asarray(data["actions"][:, :8])}
