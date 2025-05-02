#!/usr/bin/env python
"""
Debug tool for Pi0_FAST checkpoints, with Rainbow robot defaults.

Author: Mariano

This script allows debugging and investigating Pi0_FAST policy checkpoints:
- Load a trained checkpoint
- Analyze token distributions
- Visualize gripper control
- Debug token mappings
- Test on real episodes from HuggingFace datasets

Usage:
  python scripts/debug_fast_checkpoint.py --checkpoint_path [CHECKPOINT_PATH] --action_dim [DIM] --action_horizon [HORIZON]

Example for Rainbow robot:
  python scripts/debug_fast_checkpoint.py --checkpoint_path ./checkpoints/pi0_fast_rainbow/my_exp/10000/params --gripper_indices 7,15 --analyze_tokens --plot_actions

Example with HuggingFace dataset:
  python scripts/debug_fast_checkpoint.py --checkpoint_path ./checkpoints/pi0_fast_rainbow_poc_crumpets_deea_250t_384bz/pi0_crumpets_deea_250t_384bz_h200_8x/33000/params --hf_dataset HumanoidTeam/AfterEightDeea23041956 --num_episodes 5
"""

import argparse
import logging
import numpy as np
import jax
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json
import time
import os
from collections import defaultdict
from datasets import load_dataset
import tensorflow as tf

from openpi.models import model as _model
from openpi.models import pi0_fast
from openpi.models.tokenizer import FASTTokenizer
from openpi.shared import download
from openpi.transforms import TokenizeFASTInputs, ExtractFASTActions

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Rainbow robot configuration
RAINBOW_CONFIG = {
    "action_dim": 16,
    "action_horizon": 50,
    "max_token_len": 250,
    "gripper_indices": [7, 15],  # Default gripper indices for Rainbow robot
    "camera_keys": ["base_0_rgb", "wrist_0_rgb"],  # Default camera keys for Rainbow
    "camera_shape": (480, 640, 3),  # Original camera capture dimensions
    "model_image_shape": (224, 224, 3),  # Resized shape for model input
}

def parse_args():
    parser = argparse.ArgumentParser(description="Debug Pi0_FAST checkpoint (Rainbow defaults)")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                       help="Path to checkpoint (e.g., './checkpoints/pi0_fast_rainbow/exp/10000/params')")
    parser.add_argument("--action_dim", type=int, default=RAINBOW_CONFIG["action_dim"],
                       help=f"Action dimensions (default: {RAINBOW_CONFIG['action_dim']} for Rainbow)")
    parser.add_argument("--action_horizon", type=int, default=RAINBOW_CONFIG["action_horizon"],
                       help=f"Action horizon (default: {RAINBOW_CONFIG['action_horizon']})")
    parser.add_argument("--max_token_len", type=int, default=RAINBOW_CONFIG["max_token_len"],
                       help=f"Maximum token length (default: {RAINBOW_CONFIG['max_token_len']})")
    parser.add_argument("--sample_state", action="store_true",
                       help="Generate a sample random state instead of using zeros")
    parser.add_argument("--prompt", type=str, default="Pick up the green object and place it in the empty box.",
                       help="Sample prompt to use for tokenization")
    parser.add_argument("--gripper_indices", type=str, default="7,15",
                       help="Comma-separated indices of gripper dimensions (default: '7,15' for Rainbow robot)")
    parser.add_argument("--analyze_tokens", action="store_true",
                       help="Analyze token distributions")
    parser.add_argument("--plot_actions", action="store_true",
                       help="Plot sample action sequences")
    parser.add_argument("--output_dir", type=str, default="./debug_output",
                       help="Directory to save output files")
    parser.add_argument("--measure_time", action="store_true",
                       help="Measure inference time")
    parser.add_argument("--robot", type=str, choices=["rainbow"], default="rainbow",
                       help="Robot type to use (currently only rainbow is supported)")
    
    # HuggingFace dataset options
    parser.add_argument("--hf_dataset", type=str, 
                       help="HuggingFace dataset to load (e.g., 'HumanoidTeam/AfterEightDeea23041956')")
    parser.add_argument("--num_episodes", type=int, default=1,
                       help="Number of episodes to test from the dataset")
    parser.add_argument("--episode_offset", type=int, default=0,
                       help="Offset to start from in the dataset")
    parser.add_argument("--task_prompt_key", type=str, default="prompt",
                       help="Key for prompt in the dataset's tasks.json (default: 'prompt')")
    
    return parser.parse_args()

def load_checkpoint(checkpoint_path: str, action_dim: int, action_horizon: int, max_token_len: int):
    """Load the Pi0_FAST model from checkpoint."""
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    
    # Create model config
    config = pi0_fast.Pi0FASTConfig(
        action_dim=action_dim,
        action_horizon=action_horizon,
        max_token_len=max_token_len
    )
    
    # Initialize model
    key = jax.random.key(0)
    model = config.create(key)
    
    # Load checkpoint params
    try:
        params = _model.restore_params(
            download.maybe_download(checkpoint_path),
            restore_type=np.ndarray
        )
        
        # Create model with loaded params
        try:
            # For newer JAX versions
            model = model.replace_vars(params)
        except AttributeError:
            # Fallback for older JAX versions
            model_def, state = jax.tree_util.tree_flatten(model)
            state_dict = jax.tree_util.tree_leaves(params)
            model = jax.tree_util.tree_unflatten(model_def, state_dict)
        
        logger.info("Successfully loaded checkpoint")
        return model, config
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        raise

def prepare_sample_input(
    config: pi0_fast.Pi0FASTConfig,
    prompt: str,
    sample_state: bool = False,
    camera_keys: List[str] = None
) -> Tuple[Dict, FASTTokenizer]:
    """Prepare a sample input for the model, using Rainbow camera naming conventions."""
    # Create tokenizer
    tokenizer = FASTTokenizer(max_len=config.max_token_len)
    
    # Create sample state (either zeros or random)
    if sample_state:
        state = np.random.uniform(-0.5, 0.5, (config.action_dim)).astype(np.float32)
    else:
        state = np.zeros((config.action_dim), dtype=np.float32)
        
    # Tokenize input
    tokens, token_mask, ar_mask, loss_mask = tokenizer.tokenize(prompt, state, actions=None)
    
    # Set default camera keys if not provided
    if camera_keys is None:
        camera_keys = RAINBOW_CONFIG["camera_keys"]
    
    # Create observation dict with Rainbow-specific image keys
    # Note: We use model_image_shape here as this is what the model expects after the transform pipeline
    observation = {
        "state": state,
        "images": {
            key: np.zeros(RAINBOW_CONFIG["model_image_shape"], dtype=np.uint8) for key in camera_keys
        },
        "image_masks": {
            key: np.array(True) for key in camera_keys
        },
        "tokenized_prompt": tokens[None, :],  # Add batch dimension
        "tokenized_prompt_mask": token_mask[None, :],
        "token_ar_mask": ar_mask[None, :],
        "token_loss_mask": loss_mask[None, :],
    }
    
    return observation, tokenizer

def extract_actions(
    tokenizer: FASTTokenizer, 
    tokens: np.ndarray, 
    action_horizon: int, 
    action_dim: int
) -> np.ndarray:
    """Extract actions from generated tokens."""
    return tokenizer.extract_actions(tokens, action_horizon, action_dim)

def analyze_tokens(
    tokenizer: FASTTokenizer, 
    prompt: str, 
    state: np.ndarray, 
    actions: Optional[np.ndarray] = None
) -> Dict:
    """Analyze token distributions and provide insights."""
    tokens, token_mask, ar_mask, loss_mask = tokenizer.tokenize(prompt, state, actions)
    
    # Get prefix and action tokens
    prefix_end = np.argmax(ar_mask) if np.any(ar_mask) else len(tokens)
    prefix_tokens = tokens[:prefix_end]
    action_tokens = tokens[prefix_end:] if prefix_end < len(tokens) else []
    
    # Count valid tokens (non-padding)
    valid_tokens = sum(token_mask)
    prefix_len = len(prefix_tokens)
    action_len = sum(token_mask[prefix_len:])
    
    # Prepare analysis results
    analysis = {
        "total_tokens": len(tokens),
        "valid_tokens": valid_tokens,
        "prefix_length": prefix_len,
        "action_length": action_len,
        "prefix_tokens": prefix_tokens.tolist(),
        "action_tokens": [t for t, m in zip(tokens[prefix_len:], token_mask[prefix_len:]) if m],
    }
    
    # Decode the tokenized prompt for verification
    try:
        decoded_prefix = tokenizer._paligemma_tokenizer.decode(prefix_tokens.tolist())
        analysis["decoded_prefix"] = decoded_prefix
    except Exception as e:
        analysis["decoded_prefix_error"] = str(e)
    
    return analysis

def plot_action_sequence(
    actions: np.ndarray, 
    gripper_indices: Optional[List[int]] = None,
    save_path: Optional[str] = None,
    title: Optional[str] = None
):
    """Plot the predicted action sequence with gripper highlights."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Set title if provided
    if title:
        fig.suptitle(title, fontsize=14)
    
    # Plot action sequence
    time_steps = np.arange(actions.shape[0])
    for i in range(actions.shape[1]):
        if gripper_indices and i in gripper_indices:
            # Highlight gripper dimensions
            axes[0].plot(time_steps, actions[:, i], 'o-', linewidth=3, label=f"Gripper {i}")
        else:
            axes[0].plot(time_steps, actions[:, i], '--', alpha=0.7, label=f"Dim {i}")
    
    axes[0].set_title("Action Sequence")
    axes[0].set_xlabel("Time Step")
    axes[0].set_ylabel("Action Value")
    axes[0].grid(True, alpha=0.3)
    
    # Only show gripper dimensions in second plot if specified
    if gripper_indices:
        for i in gripper_indices:
            axes[1].plot(time_steps, actions[:, i], 'o-', linewidth=3, label=f"Gripper {i}")
        axes[1].set_title("Gripper Actions Only")
        axes[1].set_xlabel("Time Step")
        axes[1].set_ylabel("Gripper Value")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
    else:
        axes[1].axis('off')
        axes[1].text(0.5, 0.5, "No gripper indices specified", 
                   ha='center', va='center', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Saved action plot to {save_path}")
        plt.close(fig)  # Close the figure to avoid showing it
    else:
        plt.show()

def analyze_gripper_behavior(actions: np.ndarray, gripper_indices: List[int]) -> str:
    """Analyze gripper behavior in the action sequence."""
    result = []
    
    # For each gripper index
    for idx in gripper_indices:
        gripper_values = actions[:, idx]
        
        # Check if gripper opens (decreasing values)
        opens = np.any(np.diff(gripper_values) < -0.1)
        
        # Check if gripper closes (increasing values)
        closes = np.any(np.diff(gripper_values) > 0.1)
        
        # Determine pattern
        if opens and closes:
            pattern = "opens and closes"
        elif opens:
            pattern = "opens but doesn't close"
        elif closes:
            pattern = "closes but doesn't open"
        else:
            pattern = "doesn't show significant movement"
        
        result.append(f"Gripper {idx}: {pattern}")
        
        # Add range info
        min_val = np.min(gripper_values)
        max_val = np.max(gripper_values)
        result.append(f"  Value range: {min_val:.4f} to {max_val:.4f}")
        
        # Add recommendation based on patterns
        if min_val > 0.8 and max_val > 0.8:
            result.append("  Recommendation: Gripper may be stuck in closed position")
        elif min_val < 0.2 and max_val < 0.2: 
            result.append("  Recommendation: Gripper may be stuck in open position")
    
    return "\n".join(result)

def load_hf_dataset(dataset_name: str, num_episodes: int = 1, offset: int = 0, task_prompt_key: str = "prompt"):
    """Load episodes from a HuggingFace dataset."""
    logger.info(f"Loading dataset {dataset_name}")
    
    try:
        # Load dataset
        dataset = load_dataset(dataset_name)
        
        # Check if dataset has the expected structure
        if "train" not in dataset:
            raise ValueError(f"Dataset {dataset_name} does not have a 'train' split")
        
        # Get the tasks file (to get prompts)
        tasks_file_path = os.path.join(dataset._data_dir, "tasks.jsonl")
        tasks = []
        if os.path.exists(tasks_file_path):
            with open(tasks_file_path, 'r') as f:
                for line in f:
                    tasks.append(json.loads(line))
            logger.info(f"Loaded {len(tasks)} tasks from {tasks_file_path}")
        else:
            logger.warning(f"No tasks.jsonl found at {tasks_file_path}")
        
        # Load episodes
        episodes = []
        for i in range(offset, min(offset + num_episodes, len(dataset["train"]))):
            episode = dataset["train"][i]
            
            # Try to find the corresponding task/prompt
            prompt = None
            if tasks and "task_index" in episode:
                task_index = episode["task_index"]
                if 0 <= task_index < len(tasks) and task_prompt_key in tasks[task_index]:
                    prompt = tasks[task_index][task_prompt_key]
            
            # If prompt not found, use a default
            if not prompt:
                prompt = "Perform the task"
                logger.warning(f"No prompt found for episode {i}, using default")
                
            # Create episode dict
            episode_data = {
                "index": i,
                "prompt": prompt,
                "observations": episode["observations"],
                "actions": episode["actions"],
                "task_index": episode.get("task_index", None)
            }
            episodes.append(episode_data)
            
        logger.info(f"Loaded {len(episodes)} episodes from {dataset_name}")
        return episodes
        
    except Exception as e:
        logger.error(f"Failed to load dataset {dataset_name}: {e}")
        raise

def test_on_hf_episode(
    model, 
    tokenizer, 
    episode, 
    action_dim: int, 
    action_horizon: int, 
    gripper_indices: List[int] = None,
    output_dir: Optional[Path] = None,
    plot_actions: bool = False
):
    """Test the model on a single episode from a HuggingFace dataset."""
    # Extract prompt and first state
    prompt = episode["prompt"]
    first_obs = episode["observations"][0]
    
    # Create tokenizer input
    if "robot_state" in first_obs:
        state = np.array(first_obs["robot_state"], dtype=np.float32)
    else:
        state = np.zeros(action_dim, dtype=np.float32)
        logger.warning("No robot_state in observation, using zeros")
    
    # Tokenize input
    tokens, token_mask, ar_mask, loss_mask = tokenizer.tokenize(prompt, state, actions=None)
    
    # Ensure state has correct dimensions
    if len(state) != action_dim:
        if len(state) < action_dim:
            state = np.pad(state, (0, action_dim - len(state)), 'constant')
        else:
            state = state[:action_dim]
    
    # Create observation dict
    observation = {
        "state": state,
        "images": {
            # Use empty images for now - in a real setup you'd use the actual images
            # Note: We use model_image_shape here as this is what the model expects after the transform pipeline
            key: np.zeros(RAINBOW_CONFIG["model_image_shape"], dtype=np.uint8) 
            for key in RAINBOW_CONFIG["camera_keys"]
        },
        "image_masks": {
            key: np.array(True) for key in RAINBOW_CONFIG["camera_keys"]
        },
        "tokenized_prompt": tokens[None, :],  # Add batch dimension
        "tokenized_prompt_mask": token_mask[None, :],
        "token_ar_mask": ar_mask[None, :],
        "token_loss_mask": loss_mask[None, :],
    }
    
    # Sample actions from the model
    key = jax.random.key(0)
    output_tokens = model.sample_actions(key, observation)
    
    # Extract actions from tokens
    predicted_actions = extract_actions(tokenizer, output_tokens[0], action_horizon, action_dim)
    
    # Get ground truth actions
    true_actions = np.array(episode["actions"][:action_horizon])
    if len(true_actions) < action_horizon:
        # Pad if episode is shorter than action horizon
        pad_length = action_horizon - len(true_actions)
        true_actions = np.pad(true_actions, ((0, pad_length), (0, 0)), mode='edge')
    
    # Calculate MSE between predicted and true actions
    mse = np.mean((predicted_actions - true_actions)**2)
    
    # Prepare results dict
    results = {
        "episode_index": episode["index"],
        "prompt": prompt,
        "predicted_actions_shape": list(predicted_actions.shape),
        "true_actions_shape": list(true_actions.shape),
        "mse": float(mse),
    }
    
    # Log results
    logger.info(f"Episode {episode['index']}: MSE = {mse:.6f}")
    
    # Plot actions comparison if requested
    if plot_actions and output_dir:
        plot_dir = output_dir / "episode_plots"
        plot_dir.mkdir(exist_ok=True, parents=True)
        
        # Plot predicted actions
        pred_plot_path = plot_dir / f"episode_{episode['index']}_predicted.png"
        plot_action_sequence(
            predicted_actions, 
            gripper_indices, 
            str(pred_plot_path),
            title=f"Episode {episode['index']} - Predicted Actions"
        )
        
        # Plot true actions
        true_plot_path = plot_dir / f"episode_{episode['index']}_true.png"
        plot_action_sequence(
            true_actions, 
            gripper_indices, 
            str(true_plot_path),
            title=f"Episode {episode['index']} - True Actions"
        )
        
        results["predicted_actions_plot"] = str(pred_plot_path)
        results["true_actions_plot"] = str(true_plot_path)
    
    return results

def main():
    args = parse_args()
    
    # Create output directory if needed
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model from checkpoint
    model, config = load_checkpoint(
        args.checkpoint_path, 
        args.action_dim, 
        args.action_horizon, 
        args.max_token_len
    )
    
    # Parse gripper indices if provided
    gripper_indices = None
    if args.gripper_indices:
        gripper_indices = [int(idx) for idx in args.gripper_indices.split(',')]
        logger.info(f"Using gripper indices: {gripper_indices}")
    
    # Create debug info dictionary to save at the end
    debug_info = {
        "checkpoint_path": args.checkpoint_path,
        "model_type": "pi0_fast",
        "action_dim": args.action_dim,
        "action_horizon": args.action_horizon,
        "robot": args.robot,
    }
    
    # Test on HuggingFace dataset if provided
    if args.hf_dataset:
        logger.info(f"Testing on HuggingFace dataset: {args.hf_dataset}")
        debug_info["hf_dataset"] = args.hf_dataset
        debug_info["num_episodes"] = args.num_episodes
        debug_info["episode_offset"] = args.episode_offset
        
        # Create tokenizer for dataset testing
        tokenizer = FASTTokenizer(max_len=config.max_token_len)
        
        try:
            # Load episodes from dataset
            episodes = load_hf_dataset(
                args.hf_dataset, 
                args.num_episodes, 
                args.episode_offset,
                args.task_prompt_key
            )
            
            # Test on each episode
            episode_results = []
            for episode in episodes:
                result = test_on_hf_episode(
                    model,
                    tokenizer,
                    episode,
                    args.action_dim,
                    args.action_horizon,
                    gripper_indices,
                    output_dir,
                    args.plot_actions
                )
                episode_results.append(result)
            
            # Save episode results
            debug_info["episode_results"] = episode_results
            
            # Calculate average MSE
            avg_mse = np.mean([r["mse"] for r in episode_results])
            debug_info["average_mse"] = float(avg_mse)
            
            logger.info(f"Tested on {len(episode_results)} episodes, average MSE: {avg_mse:.6f}")
            
        except Exception as e:
            logger.error(f"Error testing on dataset: {e}")
            debug_info["dataset_error"] = str(e)
    
    # Standard debug test using synthetic prompt (if not using HF dataset or if it failed)
    else:
        # Prepare sample input
        observation, tokenizer = prepare_sample_input(
            config, 
            args.prompt, 
            args.sample_state,
            RAINBOW_CONFIG["camera_keys"]
        )
        
        debug_info["prompt"] = args.prompt
        
        # Sample actions from the model
        logger.info("Sampling actions from the model...")
        key = jax.random.key(0)
        
        # Warm-up run
        _ = model.sample_actions(key, observation)
        
        # Actual run with timing if requested
        if args.measure_time:
            start_time = time.time()
            output_tokens = model.sample_actions(key, observation)
            end_time = time.time()
            
            inference_time_ms = (end_time - start_time) * 1000
            inference_time_per_step_ms = inference_time_ms / args.action_horizon
            
            debug_info["inference_time_ms"] = inference_time_ms
            debug_info["inference_time_per_step_ms"] = inference_time_per_step_ms
            
            print(f"\nInference Timing:")
            print(f"  Total inference time: {inference_time_ms:.2f} ms")
            print(f"  Per-step inference time: {inference_time_per_step_ms:.2f} ms/step")
        else:
            output_tokens = model.sample_actions(key, observation)
        
        # Extract actions from tokens
        actions = extract_actions(tokenizer, output_tokens[0], args.action_horizon, args.action_dim)
        logger.info(f"Generated action sequence shape: {actions.shape}")
        debug_info["action_sequence_shape"] = list(actions.shape)
        
        # Print sample of the action sequence
        print("\nSample action sequence:")
        for i, action in enumerate(actions[:min(5, len(actions))]):
            gripper_info = ""
            if gripper_indices:
                gripper_info = " | " + " ".join([f"Gripper {j}: {action[j]:.4f}" for j in gripper_indices])
            print(f"Step {i}: {action[:5]}...{gripper_info}")
        
        # Analyze gripper behavior if indices are provided
        if gripper_indices:
            print("\nGripper behavior analysis:")
            gripper_analysis = analyze_gripper_behavior(actions, gripper_indices)
            print(gripper_analysis)
            debug_info["gripper_analysis"] = gripper_analysis.split("\n")
            
            # Save gripper analysis
            analysis_path = output_dir / "gripper_analysis.txt"
            with open(analysis_path, "w") as f:
                f.write("Gripper Behavior Analysis\n")
                f.write("------------------------\n")
                f.write(f"Prompt: '{args.prompt}'\n\n")
                f.write(gripper_analysis)
            
            logger.info(f"Gripper analysis saved to {analysis_path}")
        
        # Analyze tokens if requested
        if args.analyze_tokens:
            logger.info("Analyzing token distributions...")
            token_analysis = analyze_tokens(
                tokenizer, 
                args.prompt, 
                observation["state"], 
                actions
            )
            debug_info["token_analysis"] = token_analysis
            
            # Save analysis results
            analysis_path = output_dir / "token_analysis.txt"
            with open(analysis_path, "w") as f:
                f.write(f"Token Analysis for Prompt: '{args.prompt}'\n")
                f.write(f"-------------------------------------------\n")
                f.write(f"Total tokens: {token_analysis['total_tokens']}\n")
                f.write(f"Valid tokens: {token_analysis['valid_tokens']}\n")
                f.write(f"Prefix length: {token_analysis['prefix_length']}\n")
                f.write(f"Action length: {token_analysis['action_length']}\n\n")
                
                f.write(f"Decoded prefix: {token_analysis.get('decoded_prefix', 'N/A')}\n\n")
                
                f.write(f"Prefix tokens: {token_analysis['prefix_tokens']}\n\n")
                f.write(f"Action tokens: {token_analysis['action_tokens']}\n")
            
            logger.info(f"Token analysis saved to {analysis_path}")
        
        # Plot actions if requested
        if args.plot_actions:
            logger.info("Plotting action sequence...")
            plot_path = output_dir / "action_sequence.png"
            plot_action_sequence(actions, gripper_indices, str(plot_path))
    
    # Save debug info
    debug_info_path = output_dir / "debug_info.json"
    with open(debug_info_path, "w") as f:
        json.dump(debug_info, f, indent=2)
    
    logger.info(f"Debug complete! Output saved to {output_dir}")
    print(f"\nRun this command to check results: open {output_dir}")

if __name__ == "__main__":
    main() 