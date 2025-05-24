"""Remote policy server for π₀ model inference with fast image processing."""

import argparse
import base64
from io import BytesIO
import json
import logging
import time
import traceback

from fastapi import FastAPI
from fastapi import WebSocket
from fastapi.middleware.cors import CORSMiddleware
import jax
import numpy as np
from PIL import Image
from starlette.websockets import WebSocketDisconnect
import uvicorn

from openpi.policies import policy_config
from openpi.shared import download
from openpi.training import config


def numpy_hook(dct):
    """Convert numpy arrays in JSON to numpy arrays."""
    for key, value in dct.items():
        if isinstance(value, list):
            dct[key] = np.array(value)
    return dct


def decode_compressed_image(encoded_str):
    """Decode base64 encoded image to numpy array."""
    try:
        # Decode base64 string to bytes
        img_bytes = base64.b64decode(encoded_str)
        # Convert to PIL Image
        pil_image = Image.open(BytesIO(img_bytes))
        # Convert to numpy array
        return np.array(pil_image)
    except Exception as e:
        logger.error("Failed to decode image: %s", e)
        raise ValueError(f"Image decoding failed: {e!s}") from e


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("[Pi0-Server]")

# Log JAX device information
logger.info("=" * 60)
logger.info("JAX CONFIGURATION")
logger.info("=" * 60)
logger.getChild("JAX").info("JAX devices: %s", jax.devices())
logger.getChild("JAX").info("JAX default device: %s", jax.default_backend())
logger.getChild("JAX").info("JAX platform: %s", jax.default_backend())
logger.info("=" * 60)


class FastPolicyServer:
    """FastAPI server for serving π₀ policy model with WebSocket support for real-time inference."""

    def __init__(
        self,
        model_path: str = "s3://hm-vla/checkpoints/44000/",
        config_name: str = "pi0_fast_rainbow_poc_aftereight_qs_deea_250t_128bz_h200",
    ):
        self.app = FastAPI()
        self.model_path = model_path
        self.config_name = config_name
        logger.info("=" * 60)
        logger.info("MODEL INITIALIZATION")
        logger.info("=" * 60)
        logger.info("Model path: %s", model_path)
        logger.info("Config name: %s", config_name)
        logger.info("=" * 60)
        self.policy = None
        self.load_model()

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
        """Load the π₀ model."""
        logger.info("Loading π₀ model...")
        config_obj = config.get_config(self.config_name)
        checkpoint_dir = download.maybe_download(self.model_path)
        self.policy = policy_config.create_trained_policy(config_obj, checkpoint_dir)
        logger.info("Model loaded successfully!")

    async def websocket_endpoint(self, websocket: WebSocket):
        """Handle WebSocket connections for real-time policy inference requests."""
        client_id = id(websocket)
        logger.info("New WebSocket connection request from client %s", client_id)
        await websocket.accept()
        logger.info("WebSocket connection established with client %s", client_id)

        while True:
            try:
                start_receive = time.time()
                logger.info("Waiting for data from client %s...", client_id)
                data = await websocket.receive_json()
                receive_time = time.time() - start_receive
                logger.info("Received data from client %s in %.3fs", client_id, receive_time)

                try:
                    start_process = time.time()
                    logger.info("Processing received data...")

                    # Convert the received data into numpy arrays
                    state = np.array(data["observation/state"], dtype=np.float32)
                    if "observation/image_format" in data:
                        # Handle compressed images
                        head_image = decode_compressed_image(data["observation/image"])
                        wrist_image = decode_compressed_image(data["observation/wrist_image"])
                    else:
                        # Handle legacy format (raw arrays)
                        head_image = np.array(data["observation/image"], dtype=np.uint8)
                        wrist_image = np.array(data["observation/wrist_image"], dtype=np.uint8)
                    process_time = time.time() - start_process

                    logger.info(
                        "Data shapes - State: %s, Head Image: %s, Wrist Image: %s",
                        state.shape,
                        head_image.shape,
                        wrist_image.shape,
                    )

                    # Validate input shapes
                    if state.shape != (1, 16):
                        raise ValueError(f"Expected state shape (1, 16), got {state.shape}")
                    if head_image.shape != (480, 640, 3):
                        raise ValueError(f"Expected head image shape (480, 640, 3), got {head_image.shape}")
                    if wrist_image.shape != (480, 640, 3):
                        raise ValueError(f"Expected wrist image shape (480, 640, 3), got {wrist_image.shape}")

                    logger.info("Input validation successful, preparing observation...")

                    # Prepare observation
                    prompt = data.get("prompt", "Stop.")
                    logger.info('Using prompt: "%s"', prompt)
                    observation = {
                        "observation.state": state.squeeze(),  # Convert (1, 16) to (16,)
                        "observation.image.head": head_image,
                        "observation.image.wrist_right": wrist_image,
                        "prompt": prompt,
                    }

                    logger.info("Running inference...")
                    try:
                        start_inference = time.time()
                        result = self.policy.infer(observation)
                        action_sequence = result["actions"][:, :16]  # Shape: (10, 16)
                        inference_time = time.time() - start_inference
                        logger.info("Inference successful, action shape: %s", action_sequence.shape)

                        # Print the generated actions
                        logger.info("\nGenerated actions from π₀ policy:")
                        first_action = action_sequence[0]

                        logger.info("Right Arm Actions (7 joints):")
                        for i in range(7):
                            logger.info("  right_arm_%d: %.4f", i, first_action[i])

                        logger.info("\nLeft Arm Actions (7 joints):")
                        for i in range(7):
                            logger.info("  left_arm_%d: %.4f", i, first_action[i + 7])

                        logger.info("\nGripper Actions (2 joints):")
                        logger.info("  right_robotiq_85_left_knuckle_joint: %.4f", first_action[14])
                        logger.info("  left_robotiq_85_left_knuckle_joint: %.4f", first_action[15])

                        logger.critical(
                            "!!! TIMING REPORT: Inference took %.3fs, Processing took %.3fs, Receive took %.3fs !!!",
                            inference_time,
                            process_time,
                            receive_time,
                        )

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
                        logger.info("Response sent in %.3fs", send_time)
                        logger.info("Total request time: %.3fs", time.time() - start_receive)

                    except WebSocketDisconnect:
                        logger.info("Client %s disconnected gracefully.", client_id)
                        break
                    except Exception as inference_error:
                        logger.exception("Inference error: ")
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
                        continue

                except Exception as processing_error:
                    logger.exception("Data processing error: ")
                    await websocket.send_json(
                        {
                            "status": "error",
                            "error": f"Processing error: {processing_error!s}",
                            "timing": {"receive": receive_time, "error_time": time.time() - start_process},
                        }
                    )
                    continue

            except Exception as websocket_error:
                logger.exception("WebSocket error with client %s: %s", client_id, websocket_error)
                if "disconnect" in str(websocket_error).lower() or "closed" in str(websocket_error).lower():
                    logger.info("Client %s disconnected", client_id)
                    break
                continue


def create_app(
    model_path: str = "s3://hm-vla/checkpoints/44000/",
    config_name: str = "pi0_fast_rainbow_poc_aftereight_qs_deea_250t_128bz_h200",
) -> FastAPI:
    """Create and configure the FastAPI application."""
    server = FastPolicyServer(model_path, config_name)
    return server.app


def main():
    """Run the server."""
    parser = argparse.ArgumentParser(description="π₀ Policy Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Logging level (default: info)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="s3://hm-vla/checkpoints/44000/",
        help="Path to the model checkpoint (default: s3://hm-vla/checkpoints/44000/)",
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default="pi0_fast_rainbow_poc_aftereight_qs_deea_250t_128bz_h200",
        help="Name of the model configuration to use (default: pi0_fast_rainbow_poc_aftereight_qs_deea_250t_128bz_h200)",
    )

    args = parser.parse_args()

    # Configure logging based on arguments
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Ensure the root logger is also set to the same level
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))

    logger.info("=" * 60)
    logger.info("SERVER CONFIGURATION")
    logger.info("=" * 60)
    logger.info("Starting server on %s:%s...", args.host, args.port)
    logger.info("CORS is enabled for all origins")
    logger.info("WebSocket ping interval: 5s")
    logger.info("WebSocket ping timeout: 20s")
    logger.info("Keep-alive timeout: 60s")
    logger.info("=" * 60)

    app = create_app(args.model_path, args.config_name)
    uvicorn.run(
        app,
        host=args.host,
        port=int(args.port),
        timeout_keep_alive=60,
        ws_ping_interval=5,
        ws_ping_timeout=20,
        log_level=args.log_level,
        proxy_headers=True,
        forwarded_allow_ips="*",
        access_log=True,
    )


if __name__ == "__main__":
    main()
