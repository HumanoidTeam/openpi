#!/usr/bin/env python
"""
Debug tool for Pi0_FAST checkpoints, with Rainbow robot defaults.

This script allows debugging and investigating Pi0_FAST policy checkpoints:
- Load a trained checkpoint
- Analyze token distributions
- Visualize gripper control
- Debug token mappings
- Count parameters by component

Usage:
  python scripts/debug_fast_checkpoint.py --checkpoint_path [CHECKPOINT_PATH] --action_dim [DIM] --action_horizon [HORIZON]

Example for Rainbow robot:
  python scripts/debug_fast_checkpoint.py --checkpoint_path ./checkpoints/pi0_fast_rainbow/my_exp/10000/params --gripper_indices 7,15 --analyze_tokens --plot_actions
"""

import argparse
import logging
import numpy as np
import jax
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json
from collections import defaultdict

from openpi.models import model as _model
from openpi.models import pi0_fast
from openpi.models.tokenizer import FASTTokenizer
from openpi.shared import download
from openpi.transforms import TokenizeFASTInputs, ExtractFASTActions

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Debug Pi0_FAST checkpoint (Rainbow defaults)")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                       help="Path to checkpoint (e.g., './checkpoints/pi0_fast_rainbow/exp/10000/params')")
    parser.add_argument("--action_dim", type=int, default=16,
                       help="Action dimensions (default: 16 for Rainbow)")
    parser.add_argument("--action_horizon", type=int, default=50,
                       help="Action horizon (default: 50)")
    parser.add_argument("--max_token_len", type=int, default=250,
                       help="Maximum token length (default: 250)")
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
    parser.add_argument("--count_params", action="store_true",
                       help="Count and categorize parameters in the model")
    parser.add_argument("--output_dir", type=str, default="./debug_output",
                       help="Directory to save output files")
    
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
        model_def, state = jax.tree_util.tree_flatten(model)
        state.replace_by_pure_dict(params)
        model = jax.tree_util.tree_unflatten(model_def, state)
        
        logger.info("Successfully loaded checkpoint")
        return model, config, params
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        raise

def count_parameters(params: Any) -> Tuple[Dict[str, int], int]:
    """Count the number of parameters in each component of the model."""
    # Flatten the parameter tree to get all leaves (actual parameters)
    flat_params = jax.tree_util.tree_flatten(params)[0]
    
    # Count total parameters
    total_params = sum(p.size for p in flat_params)
    
    # Categorize parameters by component
    param_tree = jax.tree_util.tree_map(lambda x: x.size, params)
    
    # Extract component counts
    component_counts = {}
    
    # Extract PaliGemma components if present
    if "PaliGemma" in params:
        # Vision encoder (SigLIP)
        if "img" in params["PaliGemma"]:
            img_params = sum(jax.tree_util.tree_flatten(params["PaliGemma"]["img"])[0][i].size 
                          for i in range(len(jax.tree_util.tree_flatten(params["PaliGemma"]["img"])[0])))
            component_counts["vision_encoder"] = img_params
        
        # Language model (Gemma)
        if "llm" in params["PaliGemma"]:
            llm_params = sum(jax.tree_util.tree_flatten(params["PaliGemma"]["llm"])[0][i].size 
                          for i in range(len(jax.tree_util.tree_flatten(params["PaliGemma"]["llm"])[0])))
            component_counts["language_model"] = llm_params
    
    return component_counts, total_params

def calculate_param_size_mb(params: Any) -> float:
    """Calculate the approximate size of parameters in MB."""
    # Flatten the parameter tree to get all leaves (actual parameters)
    flat_params = jax.tree_util.tree_flatten(params)[0]
    
    # Calculate total bytes
    total_bytes = sum(p.nbytes for p in flat_params)
    
    # Convert to MB
    total_mb = total_bytes / (1024 * 1024)
    
    return total_mb

def prepare_sample_input(
    config: pi0_fast.Pi0FASTConfig,
    prompt: str,
    sample_state: bool = False
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
    
    # Create observation dict with Rainbow-specific image keys
    observation = {
        "state": state,
        "images": {
            # Rainbow uses head and wrist_right camera naming
            "base_0_rgb": np.zeros((224, 224, 3), dtype=np.uint8),  # head camera
            "wrist_0_rgb": np.zeros((224, 224, 3), dtype=np.uint8),  # right wrist camera
        },
        "image_masks": {
            "base_0_rgb": np.array(True),
            "wrist_0_rgb": np.array(True),
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
    save_path: Optional[str] = None
):
    """Plot the predicted action sequence with gripper highlights."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
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

def main():
    args = parse_args()
    
    # Create output directory if needed
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model from checkpoint
    model, config, params = load_checkpoint(
        args.checkpoint_path, 
        args.action_dim, 
        args.action_horizon, 
        args.max_token_len
    )
    
    # Count parameters if requested
    if args.count_params:
        logger.info("Analyzing parameter counts...")
        component_counts, total_params = count_parameters(params)
        param_size_mb = calculate_param_size_mb(params)
        
        print("\nParameter Analysis:")
        print(f"Total parameters: {total_params:,}")
        print(f"Model size: {param_size_mb:.2f} MB")
        print("\nComponent breakdown:")
        for component, count in component_counts.items():
            print(f"  {component}: {count:,} parameters ({count/total_params*100:.2f}%)")
        
        # Save parameter analysis
        param_path = output_dir / "parameter_analysis.json"
        with open(param_path, "w") as f:
            param_analysis = {
                "total_parameters": total_params,
                "model_size_mb": param_size_mb,
                "components": component_counts,
                "components_percent": {k: v/total_params*100 for k, v in component_counts.items()}
            }
            json.dump(param_analysis, f, indent=2)
        
        logger.info(f"Parameter analysis saved to {param_path}")
    
    # Parse gripper indices if provided
    gripper_indices = None
    if args.gripper_indices:
        gripper_indices = [int(idx) for idx in args.gripper_indices.split(',')]
        logger.info(f"Using gripper indices: {gripper_indices}")
    
    # Prepare sample input
    observation, tokenizer = prepare_sample_input(config, args.prompt, args.sample_state)
    
    # Sample actions from the model
    logger.info("Sampling actions from the model...")
    key = jax.random.key(0)
    output_tokens = model.sample_actions(key, observation)
    
    # Extract actions from tokens
    actions = extract_actions(tokenizer, output_tokens[0], args.action_horizon, args.action_dim)
    logger.info(f"Generated action sequence shape: {actions.shape}")
    
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
    
    logger.info(f"Debug complete! Output saved to {output_dir}")
    print(f"\nRun this command to check results: open {output_dir}")

if __name__ == "__main__":
    main() 