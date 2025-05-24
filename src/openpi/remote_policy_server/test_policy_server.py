import asyncio
import io
import websockets
import json
import numpy as np
import logging 
from PIL import Image
import base64

logger = logging.getLogger("InferenceTest")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')


# IMAGE_SHAPE = (480, 848, 3)
# IMAGE_SHAPE = (480, 640, 3)
IMAGE_SHAPE = (224, 224, 3)

from typing import Union
# import torch
from numpy.typing import NDArray

# TODO: fix copy-paste from the hmnd monorepo
def encode_image(
    img,
    *,
    output_size: tuple[int, int] = (224, 224),
    img_format: str = "PNG",          # PNG is loss-less;
    quality: int = 90                 # ignored by PNG, used by JPEG
) -> bytes:
    """
    Converts an image tensor/array in the range [0, 1] with shape
    (1, 3, H, W) or (3, H, W) to a compressed byte string.

    Returns
    -------
    bytes
        The raw compressed bytes ready to be sent as a binary WebSocket frame
        (or base-64 encode them if you must send text).
    """
    # --- normalise incoming data -------------------------------------------
    # if isinstance(img, torch.Tensor):
    #     img = img.detach().cpu().numpy()

    if len(img.shape) == 4:           # drop batch dimension
        img = img.squeeze(0)

    if img.shape[0] != 3:
        raise ValueError("Expecting shape (3, H, W) after squeeze")

    # -------------------------------------------------
    # CHW → HWC & uint8
    # -------------------------------------------------
    img_uint8 = (img * 255.0).round().astype(np.uint8)
    img_uint8 = np.transpose(img_uint8, (1, 2, 0))  # HWC

    pil_img = Image.fromarray(img_uint8, mode="RGB")
    #TODO: what's the best method of resizing? Now it's just BILINEAR, as in image_tools in PI repo
    pil_img = pil_img.resize(output_size)

    # --- encode ------------------------------------------------------------
    buff = io.BytesIO()
    if img_format.upper() == "JPEG":
        pil_img.save(buff, format="JPEG", quality=quality, optimize=True)
    else:  # default PNG
        pil_img.save(buff, format="PNG", optimize=True)

    return buff.getvalue()

async def test_policy_server():
    uri = "ws://localhost:9005/ws"
    async with websockets.connect(uri) as websocket:
        logger.info("Connected to policy server")
        
        # Create dummy data with correct shapes
        img_ref = Image.open('_100058428_mediaitem100058424.jpg')
        img = np.array(img_ref, dtype=np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC → CHW
        img = np.expand_dims(img, axis=0)  # CHW → BCHW
        data = {
            "observation/state": np.zeros((1,16), dtype=np.float32).tolist(),
            "observation/image": base64.b64encode(encode_image(img)).decode(),
            "observation/wrist_image": base64.b64encode(encode_image(img)).decode(),
        }

        # Send data
        logger.info("Sending data...")
        await websocket.send(json.dumps(data))
        
        # Receive response
        logger.info("Waiting for response...")
        response = await websocket.recv()
        logger.info(f"Response received: {json.loads(response)}")
        
    logger.info("WebSocket connection closed.")


if __name__ == "__main__":
    try:
        asyncio.run(test_policy_server())
    except Exception as e:
        logger.error(f"An error occurred: {e}")
