"""Rainbow robot policy transforms."""

import dataclasses
import logging


import einops
import enum
import numpy as np
import torch

from openpi import transforms
from openpi.models import model as _model

class ImagePaddingType(enum.Enum):
    """Supported modes for padding missing images in inputs."""
    NONE = "none" # No "fake" image will be added, existing images have mask set to True
    
    PAD_MID = "pad_mid" # Add a fake image as the 2nd of 3 images
    PAD_LAST = "pad_last" # Add a fake image as the last of 3 images
    

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("compute_norm_stats.log")],
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def make_rainbow_example() -> dict:
    # TODO: fixme
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
      - "observation.state": a 8/16-dimensional array.
      - "observation.image.head": an image array (HxWx3).
      - "observation.image.wrist_right": an image array (HxWx3).
      - "prompt": a string instruction.
      - "action": a 8/16-dimensional array.
    """

    action_dim: int
    model_type: _model.ModelType = _model.ModelType.PI0
    cut_to_8_dof: bool = False
    image_padding_type: ImagePaddingType = ImagePaddingType.PAD_MID

    def __call__(self, data: dict) -> dict:
        # We only mask padding for pi0 model, not pi0-FAST.
        mask_padding = self.model_type == _model.ModelType.PI0

        logger.debug(f"Available keys in input data: {list(data.keys())}")

        state_base = data["observation.state"]
        if self.cut_to_8_dof:
            state_base = self._get_state_short(data["observation.state"])
        
        state = transforms.pad_to_dim(state_base, self.action_dim)
        images_dict, image_masks_dict = self._get_images_and_masks(data, mask_padding)
        inputs = {
            "state": state,
            "image": images_dict,
            "image_mask": image_masks_dict,
        }

        assert not ("actions" in data and "action" in data), \
            "`data` must contain *zero* or *one* of {'actions', 'action'}"
        acts = data.get("actions", data.get("action"))
        # None if both are missing -> likely, we're doing inference
        if acts is not None:
            if self.cut_to_8_dof:
                acts = self._get_actions_short(acts)
            inputs["actions"] =  transforms.pad_to_dim(acts, self.action_dim)

        # Prompt should always be present
        assert "prompt" in data, f"Expected 'prompt' in data, got {list(data.keys())}"
        inputs["prompt"] = data["prompt"]

        logger.debug(f"Transformed input keys: {list(inputs.keys())}")
        return inputs

    @staticmethod
    def _get_state_short(joint_states: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        """Extracts the first 7 elements of the action array - for the right wrist, and one gripper state"""
        if isinstance(joint_states, np.ndarray):
            joint_states = torch.from_numpy(joint_states)
        ret_value = torch.cat([joint_states[:7], joint_states[-2:-1]], dim=0)
        assert len(ret_value) == 8, f"Expected 8 elements, got {len(ret_value)}"
        if isinstance(joint_states, np.ndarray):
            ret_value = ret_value.numpy()
        return ret_value

    @staticmethod
    def _get_actions_short(actions_chunk: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        # actions_chunk expected to be of shape (action_horizon, action_dim)
        if isinstance(actions_chunk, np.ndarray):
            actions_chunk = torch.from_numpy(actions_chunk)
        ret_value = torch.cat([actions_chunk[:, :7], actions_chunk[:, -2:-1]], dim=1)
        assert ret_value.shape[1] == 8, f"Expected 8 elements, got {ret_value.shape[1]}"
        if isinstance(actions_chunk, np.ndarray):
            ret_value = ret_value.numpy()
        return ret_value
    
    def _get_images_and_masks(self, data, mask_padding):
        base_image = _parse_image(data["observation.image.head"])
        wrist_image = _parse_image(data["observation.image.wrist_right"])

        images_dict = {}
        image_masks_dict = {}
        # The first image is always the same & mask is True
        images_dict['base_0_rgb'] = base_image
        image_masks_dict['base_0_rgb'] = np.True_
        
        match self.image_padding_type:
            case ImagePaddingType.NONE:
                assert self.model_type != _model.ModelType.PI0, \
                    "Padding type NONE is not supported for PI0 model — it accepts only 3 images."
                images_dict['right_wrist_0_rgb'] = wrist_image
                image_masks_dict['right_wrist_0_rgb'] = np.True_
            case ImagePaddingType.PAD_MID:
                # Add a fake image in the middle (left wrist)
                images_dict['left_wrist_0_rgb'] = np.zeros_like(wrist_image)
                image_masks_dict['left_wrist_0_rgb'] = np.False_ if mask_padding else np.True_
                
                images_dict['right_wrist_0_rgb'] = wrist_image
                image_masks_dict['right_wrist_0_rgb'] = np.True_
            case ImagePaddingType.PAD_LAST:
                # Add a fake image at the end (right wrist)
                images_dict['left_wrist_0_rgb'] = wrist_image
                image_masks_dict['left_wrist_0_rgb'] = np.True_

                images_dict['right_wrist_0_rgb'] = np.zeros_like(wrist_image)
                image_masks_dict['right_wrist_0_rgb'] = np.False_ if mask_padding else np.True_
            case _:
                raise ValueError(f"Unsupported image padding type: {self.image_padding_type}")
        return images_dict, image_masks_dict

@dataclasses.dataclass(frozen=True)
class RainbowOutputs(transforms.DataTransformFn):
    """Converts model outputs back to the Rainbow dataset format."""
    is_8_dof: bool = False

    def __call__(self, data: dict) -> dict:
        # Log available keys for debugging
        logger.debug(f"Available keys in output data: {list(data.keys())}")
        actions = data["actions"]
        idx = 8 if self.is_8_dof else 16
        actions = actions[:, :idx]
        assert actions.shape[1] == idx, \
            f"Expected {idx} elements, got {actions.shape[1]}"
        return {"actions": actions}

@dataclasses.dataclass(frozen=True)
class RainbowInputsWithRotation(RainbowInputs):
    """Rainbow inputs with 180-degree rotation of the head camera image."""

    # set to True ONLY for some legacy models that were trained with the wrong rotation
    do_legacy_rotation: bool = False

    def __call__(self, data: dict) -> dict:
        # Get the head image before standard processing
        if "observation.image.head" in data:
            img = np.asarray(data["observation.image.head"])
            
            if self.do_legacy_rotation:
                # There was a short period where this bug existed
                # To support evaluation of those models, we keep this in the code
                # TODO: remove this in the future
                data["observation.image.head"] = np.flip(np.flip(img, axis=0), axis=1)
            else:
                # This is the proper way to rotate the image and should be used for all runs
                data["observation.image.head"] = np.flip(np.flip(img, axis=1), axis=2)

        return super().__call__(data)

@dataclasses.dataclass(frozen=True)
class RainbowInputs8DOF(transforms.DataTransformFn):
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
class RainbowOutputs8DOF(transforms.DataTransformFn):
    """Converts model outputs back to the Rainbow dataset format for 8-DOF."""

    def __call__(self, data: dict) -> dict:
        # Only return the first 8 dimensions
        return {"actions": np.asarray(data["actions"][:, :8])}
