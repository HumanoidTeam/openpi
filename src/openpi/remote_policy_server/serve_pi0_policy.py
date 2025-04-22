"""Remote policy server for π₀ model inference."""

import json
import logging
import time
import traceback
from typing import Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from openpi.remote_policy_server.config import ModelConfig
from openpi.remote_policy_server.model import load_model

def numpy_hook(dct):
    """Convert numpy arrays in JSON to numpy arrays."""
    for key, value in dct.items():
        if isinstance(value, list):
            dct[key] = np.array(value)
    return dct

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PolicyServer:
    def __init__(self, config: Optional[ModelConfig] = None):
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
            self.policy = load_model(self.config)
    
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
                    head_image = np.array(data["observation/image"], dtype=np.uint8)
                    wrist_image = np.array(data["observation/wrist_image"], dtype=np.uint8)
                    process_time = time.time() - start_process
                    
                    logger.info(f"Data shapes - State: {state.shape}, Head Image: {head_image.shape}, Wrist Image: {wrist_image.shape}")
                    
                    # Validate input shapes
                    if state.shape != self.config.state_shape:
                        raise ValueError(f"Expected state shape {self.config.state_shape}, got {state.shape}")
                    if head_image.shape != self.config.image_shape:
                        raise ValueError(f"Expected head image shape {self.config.image_shape}, got {head_image.shape}")
                    if wrist_image.shape != self.config.image_shape:
                        raise ValueError(f"Expected wrist image shape {self.config.image_shape}, got {wrist_image.shape}")
                    
                    logger.info("Input validation successful, preparing observation...")
                    
                    # Prepare observation
                    observation = {
                        "observation.state": state,
                        "observation.image.head": head_image,
                        "observation.image.wrist_right": wrist_image,
                        "prompt": self.config.prompt
                    }
                    
                    logger.info("Running inference...")
                    # Run inference
                    try:
                        start_inference = time.time()
                        result = self.policy.infer(observation)
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
                            logger.info(f"  left_arm_{i}: {first_action[i+7]:.4f}")
                        
                        logger.info("\nGripper Actions (2 joints):")
                        logger.info(f"  right_robotiq_85_left_knuckle_joint: {first_action[14]:.4f}")
                        logger.info(f"  left_robotiq_85_left_knuckle_joint: {first_action[15]:.4f}")
                        
                        logger.info("Sending response back to client...")
                        start_send = time.time()
                        await websocket.send_json({
                            "actions": action_sequence.tolist(),
                            "status": "success",
                            "timing": {
                                "receive": receive_time,
                                "process": process_time,
                                "inference": inference_time,
                            }
                        })
                        send_time = time.time() - start_send
                        logger.info(f"Response sent in {send_time:.3f}s")
                        logger.info(f"Total request time: {time.time() - start_receive:.3f}s")
                        
                    except Exception as inference_error:
                        logger.error(f"Inference error: {str(inference_error)}")
                        logger.error(f"Inference error traceback: {traceback.format_exc()}")
                        await websocket.send_json({
                            "status": "error",
                            "error": f"Inference error: {str(inference_error)}",
                            "timing": {
                                "receive": receive_time,
                                "process": process_time,
                                "error_time": time.time() - start_inference
                            }
                        })
                        continue
                        
                except Exception as processing_error:
                    logger.error(f"Data processing error: {str(processing_error)}")
                    logger.error(f"Processing error traceback: {traceback.format_exc()}")
                    await websocket.send_json({
                        "status": "error",
                        "error": f"Processing error: {str(processing_error)}",
                        "timing": {
                            "receive": receive_time,
                            "error_time": time.time() - start_process
                        }
                    })
                    continue
                    
            except Exception as websocket_error:
                logger.error(f"WebSocket error with client {client_id}: {str(websocket_error)}")
                logger.error(f"WebSocket error traceback: {traceback.format_exc()}")
                if "disconnect" in str(websocket_error).lower() or "closed" in str(websocket_error).lower():
                    logger.info(f"Client {client_id} disconnected")
                    break
                continue

def create_app(config: Optional[ModelConfig] = None) -> FastAPI:
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
    
    config = ModelConfig()
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
        log_level="info"
    )

if __name__ == "__main__":
    main() 