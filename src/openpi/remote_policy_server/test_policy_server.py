import asyncio
import websockets
import json
import numpy as np
import logging 

logger = logging.getLogger("InferenceTest")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')


# IMAGE_SHAPE = (480, 848, 3)
IMAGE_SHAPE = (480, 640, 3)

async def test_policy_server():
    uri = "ws://localhost:9005/ws"
    async with websockets.connect(uri) as websocket:
        logger.info("Connected to policy server")
        
        # Create dummy data with correct shapes
        data = {
            "observation/state": np.zeros((1,16), dtype=np.float32).tolist(),
            "observation/image": np.zeros(IMAGE_SHAPE, dtype=np.uint8).tolist(),
            "observation/wrist_image": np.zeros(IMAGE_SHAPE, dtype=np.uint8).tolist()
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
