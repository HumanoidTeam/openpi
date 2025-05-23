from openpi.training import config
from openpi.policies import policy_config
from openpi.shared import download
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import json
import traceback
import time
import logging
import argparse
import cv2
import base64

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load the π₀ model
logger.info("Loading π₀ model from 8...")
MODEL_PATH = "s3://hm-vla/checkpoints/44000/"
config = config.get_config("pi0_fast_rainbow_poc_aftereight_qs_deea_250t_128bz_h200")
checkpoint_dir = download.maybe_download(MODEL_PATH)
policy = policy_config.create_trained_policy(config, checkpoint_dir)
logger.info("Model loaded successfully!")

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def decode_compressed_image(encoded_str):
    # Decode base64 string to bytes
    img_bytes = base64.b64decode(encoded_str)
    # Convert bytes to numpy array
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    # Decode JPEG to image
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    client_id = id(websocket)
    logger.info(f"New WebSocket connection request from client {client_id}")
    await websocket.accept()
    logger.info(f"WebSocket connection established with client {client_id}")

    while True:
        try:
            start_receive = time.time()
            logger.info(f"Waiting for data from client {client_id}...")
            data = await websocket.receive_json()
            receive_time = time.time() - start_receive
            logger.info(f"Received data from client {client_id} in {receive_time:.3f}s")

            try:
                start_process = time.time()
                logger.info("Processing received data...")

                # Convert the received data into numpy arrays
                state = np.array(data["observation/state"], dtype=np.float32)
                if "observation/image_format" in data and data["observation/image_format"] == "jpeg_base64":
                    # Handle compressed images
                    head_image = decode_compressed_image(data["observation/image"])
                    wrist_image = decode_compressed_image(data["observation/wrist_image"])
                else:
                    # Handle legacy format (raw arrays)
                    head_image = np.array(data["observation/image"], dtype=np.uint8)
                    wrist_image = np.array(data["observation/wrist_image"], dtype=np.uint8)
                process_time = time.time() - start_process

                logger.info(
                    f"Data shapes - State: {state.shape}, Head Image: {head_image.shape}, Wrist Image: {wrist_image.shape}"
                )

                # Validate input shapes
                if state.shape != (16,):
                    raise ValueError(f"Expected state shape (16,), got {state.shape}")
                if head_image.shape != (480, 640, 3):
                    raise ValueError(f"Expected head image shape (480, 640, 3), got {head_image.shape}")
                if wrist_image.shape != (480, 640, 3):
                    raise ValueError(f"Expected wrist image shape (480, 640, 3), got {wrist_image.shape}")

                logger.info("Input validation successful, preparing observation...")

                # Prepare observation
                prompt = data.get("prompt", "Stop.")
                logger.info(f'Using prompt: "{prompt}"')
                observation = {
                    "observation.state": state,
                    "observation.image.head": head_image,
                    "observation.image.wrist_right": wrist_image,
                    "prompt": prompt,
                }

                logger.info("Running inference...")
                # Run inference
                try:
                    start_inference = time.time()
                    result = policy.infer(observation)
                    action_sequence = result["actions"][:, :16]  # Shape: (10, 16)
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

                    logger.critical(
                        f"!!! TIMING REPORT: Inference took {inference_time:.3f}s, Processing took {process_time:.3f}s, Receive took {receive_time:.3f}s !!!"
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
                    logger.info(f"Response sent in {send_time:.3f}s")
                    logger.info(f"Total request time: {time.time() - start_receive:.3f}s")

                except Exception as inference_error:
                    logger.error(f"Inference error: {str(inference_error)}")
                    logger.error(f"Inference error traceback: {traceback.format_exc()}")
                    await websocket.send_json(
                        {
                            "status": "error",
                            "error": f"Inference error: {str(inference_error)}",
                            "timing": {
                                "receive": receive_time,
                                "process": process_time,
                                "error_time": time.time() - start_inference,
                            },
                        }
                    )
                    continue

            except Exception as processing_error:
                logger.error(f"Data processing error: {str(processing_error)}")
                logger.error(f"Processing error traceback: {traceback.format_exc()}")
                await websocket.send_json(
                    {
                        "status": "error",
                        "error": f"Processing error: {str(processing_error)}",
                        "timing": {"receive": receive_time, "error_time": time.time() - start_process},
                    }
                )
                continue

        except Exception as websocket_error:
            logger.error(f"WebSocket error with client {client_id}: {str(websocket_error)}")
            logger.error(f"WebSocket error traceback: {traceback.format_exc()}")
            if "disconnect" in str(websocket_error).lower() or "closed" in str(websocket_error).lower():
                logger.info(f"Client {client_id} disconnected")
                break
            continue


if __name__ == "__main__":
    # Parse command line arguments
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

    args = parser.parse_args()

    # Configure logging based on arguments
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()), format="%(asctime)s - %(levelname)s - %(message)s"
    )

    logger.info(f"Starting server on {args.host}:{args.port}...")
    logger.info("CORS is enabled for all origins")
    logger.info("WebSocket ping interval: 5s")
    logger.info("WebSocket ping timeout: 20s")
    logger.info("Keep-alive timeout: 60s")

    print(f"DEBUG: About to start uvicorn with port={args.port}")

    # Run the server
    uvicorn.run(
        app,
        host=args.host,
        port=int(args.port),
        timeout_keep_alive=60,
        ws_ping_interval=5,
        ws_ping_timeout=20,
        log_level=args.log_level,
        proxy_headers=True,  # Trust proxy headers for proper client IP handling
        forwarded_allow_ips="*",  # Trust forwarded IPs from all sources
        access_log=True,  # Enable access logging
    )
