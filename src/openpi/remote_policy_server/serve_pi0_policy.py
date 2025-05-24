"""Remote policy server for π₀ model inference."""

import io
import json
import logging
import time
import traceback

from fastapi import FastAPI
from fastapi import WebSocket
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from starlette.websockets import WebSocketDisconnect
import uvicorn
import base64

from PIL import Image
from numpy.typing import NDArray

from openpi.remote_policy_server.config import ModelConfig
from openpi.remote_policy_server.model import load_model


def numpy_hook(dct):
    """Convert numpy arrays in JSON to numpy arrays."""
    for key, value in dct.items():
        if isinstance(value, list):
            dct[key] = np.array(value)
    return dct

def decode_image(data: bytes) -> NDArray[np.float32]:
    """
    Decodes the byte string produced by `encode_image` back into a
    float32 numpy array (H, W, 3) in the range [0, 1].
    """
    pil_img = Image.open(io.BytesIO(data))
    arr = np.asarray(pil_img, dtype=np.float32) / 255.0
    return arr


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class PolicyServer:
    def __init__(self, config: ModelConfig | None = None):
        self.config = config or ModelConfig()
        self.app = FastAPI()
        self.policy = load_model(self.config)

        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Register routes
        self.app.websocket("/ws")(self.websocket_endpoint)

    def load_model(self):
        """Load the model if not already loaded."""
        if self.policy is None:
            logger.info("Loading model...")
            self.policy = load_model(self.config)
            logger.info("Model loaded successfully")

    async def websocket_endpoint(self, websocket: WebSocket):
        client_id = id(websocket)
        logger.info(f"New WebSocket connection request from client {client_id}")
        await websocket.accept()
        logger.info(f"WebSocket connection established with client {client_id}")

        while True:
            try:
                start_receive = time.time()
                logger.info(f"Waiting for data from client {client_id}...")
                raw_data = await websocket.receive_text()  # Receive as text instead of json
                data = json.loads(raw_data, object_hook=numpy_hook)  # Use numpy_hook
                receive_time = time.time() - start_receive
                logger.info(f"Received data from client {client_id} in {receive_time:.3f}s")

                try:
                    start_process = time.time()
                    logger.info("Processing received data...")

                    # Convert the received data into numpy arrays
                    state = np.array(data["observation/state"], dtype=np.float32)
                    head_image = decode_image(base64.b64decode(data["observation/image"]))
                    wrist_image = decode_image(base64.b64decode(data["observation/wrist_image"]))
                    process_time = time.time() - start_process

                    logger.info(
                        f"Data shapes - State: {state.shape}, Head Image: {head_image.shape}, Wrist Image: {wrist_image.shape}"
                    )

                    # Validate input shapes
                    if state.shape != self.config.state_shape:
                        raise ValueError(f"Expected state shape {self.config.state_shape}, got {state.shape}")
                    if head_image.shape != self.config.image_shape:
                        raise ValueError(f"Expected head image shape {self.config.image_shape}, got {head_image.shape}")
                    if wrist_image.shape != self.config.image_shape:
                        raise ValueError(
                            f"Expected wrist image shape {self.config.image_shape}, got {wrist_image.shape}"
                        )

                    logger.info("Input validation successful, preparing observation...")

                    # Prepare observation
                    observation = {
                        "observation.state": state.squeeze(),
                        "observation.image.head": head_image,
                        "observation.image.wrist_right": wrist_image,
                        "prompt": self.config.prompt,
                    }

                    logger.info("Running inference...")
                    # Run inference
                    try:
                        start_inference = time.time()
                        result = self.policy.infer(observation)
                        
                        print(f'result shape: {result["actions"].shape}')
                        
                        if self.config.is_8dof:
                            filler_state = state.squeeze().copy() # Shape: (16,)
                            
                            actions_to_send = np.zeros(self.config.action_shape, dtype=np.float32)
                            actions_to_send[:, :7] = result["actions"][:, :7]
                            actions_to_send[:, -2:-1] = result["actions"][:, 7:]
                            # fill the same values for the left arm
                            actions_to_send[:, 7:14] = filler_state[7:14]
                            actions_to_send[:, -1:] = filler_state[-1]
                            
                            action_sequence = actions_to_send
                        else:
                            action_sequence = result["actions"][:, :16]  # Shape: (10, 16)
                        print(f'action_sequence shape: {action_sequence.shape}')
                        
                        log_data = {
                            "observation": {
                                "observation.state": observation["observation.state"].tolist(),
                                "observation.image.head": observation["observation.image.head"].tolist(),
                                "observation.image.wrist_right": observation["observation.image.wrist_right"].tolist(),
                                "prompt": observation["prompt"],
                            },
                            "result": {
                                "actions": action_sequence.tolist()
                            }
                        }
                        timestamp_str = time.strftime("%Y%m%d-%H%M%S")
                        filename = f"dump_logs/inference_{timestamp_str}.json"
                        with open(filename, "w") as f:
                            json.dump(log_data, f)
                        
                        
                        inference_time = time.time() - start_inference
                        logger.info(f"Inference successful, action shape: {action_sequence.shape}")

                        # Print the generated actions
                        logger.info("\nGenerated actions from π₀ policy:")
                        first_action = action_sequence[0]

                        logger.info("Right Arm Actions (7 joints):")
                        for i in range(7):
                            logger.info(f"  right_arm_{i}: {first_action[i]:.4f}")

                        logger.info("\nLeft Arm Actions (7 joints):")
                        for i in range(7):
                            logger.info(f"  left_arm_{i}: {first_action[i + 7]:.4f}")

                        logger.info("\nGripper Actions (2 joints):")
                        logger.info(f"  right_robotiq_85_left_knuckle_joint: {first_action[14]:.4f}")
                        logger.info(f"  left_robotiq_85_left_knuckle_joint: {first_action[15]:.4f}")

                        logger.info("Sending response back to client...")
                        start_send = time.time()
                        await websocket.send_json(
                            {
                                "actions": action_sequence.tolist(),
                                "status": "success",
                                "timing": {
                                    "receive": receive_time,
                                    "process": process_time,
                                    "inference": inference_time,
                                },
                            }
                        )
                        send_time = time.time() - start_send
                        logger.info(f"Response sent in {send_time:.3f}s")
                        logger.info(f"Total request time: {time.time() - start_receive:.3f}s")

                    except WebSocketDisconnect:
                        logger.info(f"Client {client_id} disconnected gracefully.")
                        break
                    except Exception as inference_error:
                        logger.error(f"Inference error: {inference_error!s}")
                        logger.error(f"Inference error traceback: {traceback.format_exc()}")
                        await websocket.send_json(
                            {
                                "status": "error",
                                "error": f"Inference error: {inference_error!s}",
                                "timing": {
                                    "receive": receive_time,
                                    "process": process_time,
                                    "error_time": time.time() - start_inference,
                                },
                            }
                        )
                        # continue

                except Exception as processing_error:
                    logger.error(f"Data processing error: {processing_error!s}")
                    logger.error(f"Processing error traceback: {traceback.format_exc()}")
                    await websocket.send_json(
                        {
                            "status": "error",
                            "error": f"Processing error: {processing_error!s}",
                            "timing": {"receive": receive_time, "error_time": time.time() - start_process},
                        }
                    )
                    continue

            except Exception as websocket_error:
                # logger.error(f"WebSocket error with client {client_id}: {websocket_error!s}")
                # logger.error(f"WebSocket error traceback: {traceback.format_exc()}")
                if "disconnect" in str(websocket_error).lower() or "closed" in str(websocket_error).lower():
                    # logger.info(f"Client {client_id} disconnected")
                    break
                continue


def create_app(config: ModelConfig | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    server = PolicyServer(config)
    return server.app


def main():
    """Run the server."""
    import argparse

    parser = argparse.ArgumentParser(description="Remote policy server for π₀ model inference")
    parser.add_argument("--model-name", default="pi0_fast_rainbow_poc", help="Name of the model to load")
    parser.add_argument("--checkpoint-path", help="Path to the model checkpoint")
    parser.add_argument("--prompt", help="Prompt to use for inference")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    args = parser.parse_args()

    # config = ModelConfig(state_shape=(1, 16), image_shape=(480, 640, 3), action_shape=(10, 16))
    # config = ModelConfig(state_shape=(1, 16), image_shape=(480, 640, 3))
    config = ModelConfig(state_shape=(1, 16), image_shape=(224, 224, 3))

    if args.model_name:
        config.model_name = args.model_name
    if args.checkpoint_path:
        config.checkpoint_path = args.checkpoint_path
    if args.prompt:
        config.prompt = args.prompt

    app = create_app(config)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        timeout_keep_alive=60,
        ws_ping_interval=5,
        ws_ping_timeout=20,
        log_level="info",
        ws_max_size=104857600,  # 100MB in bytes
    )


if __name__ == "__main__":
    main()
