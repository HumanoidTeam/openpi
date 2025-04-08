import dataclasses
import numpy as np
import einops

from openpi import transforms
from openpi.models import model as _model

def make_rainbow_example() -> dict:
    """
    Creates a random input example for the Rainbow policy.
    """
    return {
        "observation.state": np.random.rand(16),
        "observation.image.head": np.random.randint(256, size=(480, 848, 3), dtype=np.uint8),
        "observation.image.wrist_right": np.random.randint(256, size=(480, 848, 3), dtype=np.uint8),
        "prompt": "do something",
        "action": np.random.rand(16),  # only used during training
    }

def _parse_image(image) -> np.ndarray:
    """
    Ensures that the image is a uint8 array with shape (H, W, C).
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
    """
    Data transformation for the Rainbow robot.
    Converts raw dataset inputs into the model's expected format.
    
    Expected keys in the input dictionary:
      - "observation.state": a 16-dimensional array.
      - "observation.image.head": an image array (480x848x3).
      - "observation.image.wrist_right": an image array (480x848x3).
      - "prompt": a string instruction.
      - "action": (optional) a 16-dimensional array (for training).
    """
    action_dim: int  # For Rainbow, set this to 16.
    model_type: _model.ModelType = _model.ModelType.PI0  # This remains the same.

    def __call__(self, data: dict) -> dict:
        # Process the proprioceptive state and pad to the model's action dimension.
        state = transforms.pad_to_dim(data["observation.state"], self.action_dim)
        
        # Process the images.
        head_image = _parse_image(data["observation.image.head"])
        wrist_image = _parse_image(data["observation.image.wrist_right"])
        
        images = {
            "head_0_rgb": head_image,
            "wrist_right_0_rgb": wrist_image,
        }
        image_masks = {
            "head_0_rgb": np.True_,
            "wrist_right_0_rgb": np.True_,
        }
        
        inputs = {
            "state": state,
            "image": images,
            "image_mask": image_masks,
        }
        
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        
        # Include actions during training.
        if "action" in data:  # Note: dataset uses "action" not "actions"
            actions = transforms.pad_to_dim(data["action"], self.action_dim)
            inputs["actions"] = actions  # Model expects "actions" plural
        
        return inputs

@dataclasses.dataclass(frozen=True)
class RainbowOutputs(transforms.DataTransformFn):
    """
    Converts model outputs back to the Rainbow dataset format.
    This class converts from the model's "actions" format to the dataset's "action" format.
    """
    def __call__(self, data: dict) -> dict:
        # Convert from model's "actions" to dataset's "action"
        return {"action": np.asarray(data["actions"][:, :16])}  # Dataset expects "action" singular