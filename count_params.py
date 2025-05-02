#!/usr/bin/env python
"""
Simple script to count parameters in a Pi0_FAST checkpoint.

Author: Mariano
"""

import argparse
import jax
import numpy as np
import json
import os
import subprocess
import time
from pathlib import Path
from openpi.shared import download
from openpi.models import model as _model
from openpi.models import pi0_fast

def parse_args():
    parser = argparse.ArgumentParser(description="Count parameters in a Pi0_FAST checkpoint")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to checkpoint params directory")
    parser.add_argument("--output_file", type=str, default="parameter_counts.json",
                        help="Output file to save parameter counts")
    parser.add_argument("--action_dim", type=int, default=16,
                        help="Action dimensions (default: 16 for Rainbow)")
    parser.add_argument("--action_horizon", type=int, default=50,
                        help="Action horizon (default: 50)")
    parser.add_argument("--measure_gpu_memory", action="store_true",
                        help="Measure GPU memory during model loading and inference")
    return parser.parse_args()

def get_gpu_memory_usage():
    """Get GPU memory usage in MB."""
    try:
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
            encoding='utf-8'
        )
        # Result contains memory usage values for all GPUs, line by line
        return [int(x) for x in result.strip().split('\n')]
    except (subprocess.SubprocessError, FileNotFoundError):
        print("Warning: Could not get GPU memory usage. Is nvidia-smi available?")
        return None

def count_parameters(params):
    """Count the number of parameters in each component of the model."""
    # Flatten the parameter tree to get all leaves (actual parameters)
    flat_params = jax.tree_util.tree_leaves(params)
    
    # Count total parameters
    total_params = sum(p.size for p in flat_params)
    
    # Calculate size in MB and GB
    total_bytes = sum(p.nbytes for p in flat_params)
    total_mb = total_bytes / (1024 * 1024)
    total_gb = total_mb / 1024
    
    # Calculate billions of parameters
    total_billions = total_params / 1_000_000_000
    
    # Try to categorize parameters by component
    component_counts = {}
    component_billions = {}
    
    # Extract component counts for PaliGemma if present
    if isinstance(params, dict) and "PaliGemma" in params:
        # Vision encoder (SigLIP)
        if "img" in params["PaliGemma"]:
            img_params = sum(p.size for p in jax.tree_util.tree_leaves(params["PaliGemma"]["img"]))
            component_counts["vision_encoder"] = img_params
            component_billions["vision_encoder"] = img_params / 1_000_000_000
        
        # Language model (Gemma)
        if "llm" in params["PaliGemma"]:
            llm_params = sum(p.size for p in jax.tree_util.tree_leaves(params["PaliGemma"]["llm"]))
            component_counts["language_model"] = llm_params
            component_billions["language_model"] = llm_params / 1_000_000_000
    
    return {
        "total_parameters": total_params,
        "total_parameters_billions": total_billions,
        "model_size_mb": total_mb,
        "model_size_gb": total_gb,
        "components": component_counts,
        "components_billions": component_billions,
        "components_percent": {k: v/total_params*100 for k, v in component_counts.items()} if total_params > 0 else {}
    }

def measure_inference_memory(checkpoint_path, action_dim, action_horizon):
    """Measure GPU memory during model initialization and inference."""
    memory_metrics = {
        "before_load": None,
        "after_load": None,
        "after_inference": None,
        "peak_usage": None,
        "inference_usage": None,
        "inference_time_ms": None,
        "inference_time_per_step_ms": None
    }
    
    try:
        print("\nMeasuring GPU memory utilization...")
        
        # Get initial memory usage
        initial_memory = get_gpu_memory_usage()
        if initial_memory:
            memory_metrics["before_load"] = initial_memory[0]  # Use first GPU
            print(f"Initial GPU memory usage: {initial_memory[0]} MB")
        
        # Create model config
        config = pi0_fast.Pi0FASTConfig(
            action_dim=action_dim,
            action_horizon=action_horizon,
            max_token_len=250  # Typical value for Rainbow
        )
        
        # Initialize model
        key = jax.random.key(0)
        print("Initializing model...")
        model = config.create(key)
        
        # Load checkpoint params
        print(f"Loading parameters from {checkpoint_path}...")
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
        
        # Check memory after loading
        after_load_memory = get_gpu_memory_usage()
        if after_load_memory:
            memory_metrics["after_load"] = after_load_memory[0]
            print(f"GPU memory after loading model: {after_load_memory[0]} MB")
        
        # Create a dummy input for inference
        print("Running inference on dummy input...")
        dummy_input = {
            "state": np.zeros((action_dim,), dtype=np.float32),
            "images": {
                "base_0_rgb": np.zeros((224, 224, 3), dtype=np.uint8),
                "wrist_0_rgb": np.zeros((224, 224, 3), dtype=np.uint8),
            },
            "image_masks": {
                "base_0_rgb": np.array(True),
                "wrist_0_rgb": np.array(True),
            },
            "tokenized_prompt": np.zeros((1, 100), dtype=np.int32),
            "tokenized_prompt_mask": np.ones((1, 100), dtype=bool),
            "token_ar_mask": np.ones((1, 100), dtype=bool),
            "token_loss_mask": np.zeros((1, 100), dtype=bool),
        }
        
        # Warm up run
        _ = model.sample_actions(key, dummy_input)
        
        # Run inference with timing
        start_time = time.time()
        _ = model.sample_actions(key, dummy_input)
        end_time = time.time()
        
        # Calculate inference time in milliseconds
        inference_time_ms = (end_time - start_time) * 1000
        inference_time_per_step_ms = inference_time_ms / action_horizon
        
        memory_metrics["inference_time_ms"] = inference_time_ms
        memory_metrics["inference_time_per_step_ms"] = inference_time_per_step_ms
        
        print(f"Inference time: {inference_time_ms:.2f} ms")
        print(f"Inference time per step: {inference_time_per_step_ms:.2f} ms/step")
        
        # Check memory after inference
        after_inference_memory = get_gpu_memory_usage()
        if after_inference_memory and after_load_memory:
            memory_metrics["after_inference"] = after_inference_memory[0]
            memory_metrics["inference_usage"] = after_inference_memory[0] - initial_memory[0]
            memory_metrics["peak_usage"] = max(after_load_memory[0], after_inference_memory[0]) - initial_memory[0]
            
            print(f"GPU memory after inference: {after_inference_memory[0]} MB")
            print(f"Inference memory usage: {memory_metrics['inference_usage']} MB")
            print(f"Peak memory usage: {memory_metrics['peak_usage']} MB")
        
        # Delete model to free memory
        del model
        
    except Exception as e:
        print(f"Error measuring inference memory: {str(e)}")
    
    return memory_metrics

def extract_checkpoint_specs(checkpoint_path):
    """Extract specifications from the checkpoint path."""
    path = Path(checkpoint_path)
    full_path = str(path)
    
    # Get checkpoint directory path
    if path.name == "params":
        # If we're pointing to the params directory, go up one level
        checkpoint_dir = path.parent
    else:
        checkpoint_dir = path
    
    # Extract information from path parts
    parts = str(checkpoint_dir).split(os.sep)
    specs = {
        "checkpoint_path": full_path,
        "checkpoint_name": os.path.basename(str(checkpoint_dir)),
        "model_type": "pi0_fast"  # Default to pi0_fast since that's what we're analyzing
    }
    
    # Look for common indicators in the path
    for part in parts:
        # Extract checkpoint step if it's a number
        if part.isdigit():
            specs["checkpoint_step"] = int(part)
        
        # Look for batch size indicators
        if "bz" in part.lower() or "bs" in part.lower():
            batch_size = None
            for segment in part.split("_"):
                # Look for patterns like "384bz" or "bs256"
                if segment.lower().endswith("bz") or segment.lower().endswith("bs"):
                    try:
                        batch_size = int(segment[:-2])
                        break
                    except ValueError:
                        pass
                elif segment.lower().startswith("bz") or segment.lower().startswith("bs"):
                    try:
                        batch_size = int(segment[2:])
                        break
                    except ValueError:
                        pass
            if batch_size:
                specs["batch_size"] = batch_size
        
        # Look for hardware indicators
        if "h100" in part.lower():
            specs["hardware"] = "H100"
        elif "h200" in part.lower():
            specs["hardware"] = "H200"
        elif "a100" in part.lower():
            specs["hardware"] = "A100"
    
    return specs

def main():
    args = parse_args()
    
    print(f"Loading checkpoint from {args.checkpoint_path}")
    try:
        # Load checkpoint params
        params = _model.restore_params(
            download.maybe_download(args.checkpoint_path),
            restore_type=np.ndarray
        )
        
        # Count parameters
        param_counts = count_parameters(params)
        
        # Extract checkpoint specs
        checkpoint_specs = extract_checkpoint_specs(args.checkpoint_path)
        param_counts["checkpoint_specs"] = checkpoint_specs
        
        # Display results
        print("\nParameter Analysis:")
        print(f"Total parameters: {param_counts['total_parameters']:,} ({param_counts['total_parameters_billions']:.2f}B)")
        print(f"Model size: {param_counts['model_size_mb']:.2f} MB ({param_counts['model_size_gb']:.2f} GB)")
        
        if param_counts['components']:
            print("\nComponent breakdown:")
            for component, count in param_counts['components'].items():
                percent = param_counts['components_percent'][component]
                billions = param_counts['components_billions'][component]
                print(f"  {component}: {count:,} parameters ({billions:.2f}B, {percent:.2f}%)")
        
        # Display checkpoint specs
        if checkpoint_specs:
            print("\nCheckpoint Specifications:")
            for key, value in checkpoint_specs.items():
                print(f"  {key}: {value}")
        
        # Measure GPU memory during inference if requested
        if args.measure_gpu_memory:
            memory_metrics = measure_inference_memory(
                args.checkpoint_path, 
                args.action_dim, 
                args.action_horizon
            )
            param_counts["memory_metrics"] = memory_metrics
            
            # Display inference metrics
            if "inference_time_ms" in memory_metrics and memory_metrics["inference_time_ms"] is not None:
                print("\nInference Metrics:")
                print(f"  Total inference time: {memory_metrics['inference_time_ms']:.2f} ms")
                print(f"  Per-step inference time: {memory_metrics['inference_time_per_step_ms']:.2f} ms/step")
        
        # Save to file
        output_file = Path(args.output_file)
        with open(output_file, "w") as f:
            json.dump(param_counts, f, indent=2)
        
        print(f"\nParameter analysis saved to {output_file}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 