import asyncio
import websockets
import json
import numpy as np


# IMAGE_SHAPE = (480, 848, 3)
IMAGE_SHAPE = (480, 640, 3)

async def test_policy_server():
    uri = "ws://localhost:8000/ws"
    async with websockets.connect(uri) as websocket:
        print("Connected to policy server")
        
        # Create dummy data with correct shapes
        data = {
            "observation/state": np.zeros(16, dtype=np.float32).tolist(),
            "observation/image": np.zeros(IMAGE_SHAPE, dtype=np.uint8).tolist(),
            "observation/wrist_image": np.zeros(IMAGE_SHAPE, dtype=np.uint8).tolist()
        }

        # Send data
        print("Sending data...")
        await websocket.send(json.dumps(data))
        
        # Receive response
        print("Waiting for response...")
        response = await websocket.recv()
        print("Response received:", json.loads(response))

if __name__ == "__main__":
    asyncio.run(test_policy_server()) 