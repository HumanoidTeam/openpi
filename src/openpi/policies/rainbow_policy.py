"""Rainbow robot policy transforms."""

import dataclasses
import logging


import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('compute_norm_stats.log')
    ]
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
        # Log available keys for debugging
        logger.debug(f"Available keys in input data: {list(data.keys())}")
        
        # Process the proprioceptive state
        
        state = data["observation.state"]
        
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
            # We are padding to the model action dim.
            inputs["actions"] = data["actions"]
        elif "action" in data: 
            inputs["actions"] = data["action"]
            
        # Add prompt if available
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        
        logger.debug(f"Transformed input keys: {list(inputs.keys())}")
        return inputs


@dataclasses.dataclass(frozen=True)
class RainbowOutputs(transforms.DataTransformFn):
    """Converts model outputs back to the Rainbow dataset format."""
    def __call__(self, data: dict) -> dict:
        # Log available keys for debugging
        logger.debug(f"Available keys in output data: {list(data.keys())}")
        
        return {"actions": np.asarray(data["actions"][:, :16])}


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


@dataclasses.dataclass(frozen=True)
class RainbowInputs224(RainbowInputs):
    """Data transformation for the Rainbow robot with 224x224 input images.
    Converts raw dataset inputs into the model's expected format.
    
    Expected keys in the input dictionary:
      - "observation.state": a 16-dimensional array (float64).
      - "observation.image.head": an image array (224x224x3).
      - "observation.image.wrist_right": an image array (224x224x3).
      - "prompt": a string instruction.
      - "action": a 16-dimensional array (float64).
    """
    action_dim: int  # For Rainbow, set this to 16
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        # Log available keys for debugging
        logger.debug(f"Available keys in input data: {list(data.keys())}")
        
        # Process the proprioceptive state
        state = data["observation.state"]
        
        # Process the images
        base_image = _parse_image(data["observation.image.head"])
        wrist_image = _parse_image(data["observation.image.wrist_right"])

        # Verify image dimensions for 224x224
        if base_image.shape != (224, 224, 3) or wrist_image.shape != (224, 224, 3):
            raise ValueError(
                f"Expected image shapes (224, 224, 3), got {base_image.shape} and {wrist_image.shape}"
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
            inputs["actions"] = data["actions"]
        elif "action" in data: 
            inputs["actions"] = data["action"]
            
        # Add prompt if available
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        
        logger.debug(f"Transformed input keys: {list(inputs.keys())}")
        return inputs

@dataclasses.dataclass(frozen=True)
class RainbowInputsRotated224(RainbowInputs224):
    """Data transformation for the Rainbow robot with 180-degree rotated head camera and 224x224 input images."""
    
    def __call__(self, data: dict) -> dict:
        # Get the head image before standard processing
        if "observation.image.head" in data:
            # Rotate image using NumPy (more consistent with codebase)
            img = np.asarray(data["observation.image.head"])
            # 180 degree rotation = flip both horizontally and vertically
            data["observation.image.head"] = np.flip(np.flip(img, axis=1), axis=2)
        
        # Call the parent method to do the standard processing
        return super().__call__(data)




"""Rainbow robot policy transforms."""

import dataclasses
import logging


import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('compute_norm_stats.log')
    ]
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def make_rainbow_example() -> dict:
    """Creates a random input example for the Rainbow policy."""
    return {
        "observation.state": np.random.rand(16),
        "observation.image.head": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation.image.wrist_right": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
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
class RainbowInputsPi0(transforms.DataTransformFn):
    """Data transformation for the Rainbow robot.
    Converts raw dataset inputs into the model's expected format.
    
    Expected keys in the input dictionary:
      - "observation.state": a 16-dimensional array (float64).
      - "observation.image.head": an image array (224x224x3).
      - "observation.image.wrist_right": an image array (224x224x3).
      - "prompt": a string instruction.
      - "action": a 16-dimensional array (float64).
    """
    action_dim: int  # For Rainbow, set this to 16 or 8 if one arm.
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        mask_padding = self.model_type == _model.ModelType.PI0

        # Log available keys for debugging
        logger.debug(f"Available keys in input data: {list(data.keys())}")
        
        # Process the proprioceptive state
        
        # state = data["observation.state"]
        state = transforms.pad_to_dim(data["observation.state"], self.action_dim)
        
        # Process the images
        base_image = _parse_image(data["observation.image.head"])
        wrist_image = _parse_image(data["observation.image.wrist_right"])


        # Verify image dimensions
        if base_image.shape != (224, 224, 3) or wrist_image.shape != (224, 224, 3):
            raise ValueError(
                f"Expected image shapes (224, 224, 3), got {base_image.shape} and {wrist_image.shape}"
            )
        
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # Pad any non-existent images with zero-arrays of the appropriate shape.
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # Mask any non-existent images with False (if ``mask_padding`` is True).
                "right_wrist_0_rgb": np.False_ if mask_padding else np.True_,
            },
        }

        if "actions" in data:
            # We are padding to the model action dim.
            inputs["actions"] = transforms.pad_to_dim(data["actions"], self.action_dim)
        elif "action" in data:
            inputs["actions"] = transforms.pad_to_dim(data["action"], self.action_dim)
            
        # Add prompt if available
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        
        logger.debug(f"Transformed input keys: {list(inputs.keys())}")
        return inputs


@dataclasses.dataclass(frozen=True)
class RainbowOutputsPi0(transforms.DataTransformFn):
    """Converts model outputs back to the Rainbow dataset format."""
    def __call__(self, data: dict) -> dict:
        # Log available keys for debugging
        logger.debug(f"Available keys in output data: {list(data.keys())}")
        
        return {"actions": np.asarray(data["actions"][:, :16])}