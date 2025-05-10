"""Compute normalization statistics for a config using JAX for GPU acceleration.

This script is a JAX-accelerated version of compute_norm_stats.py that computes the
normalization statistics for a given config. It computes the mean, standard deviation,
and quantiles of the data in the dataset and saves it to the config assets directory.
"""

import os
import numpy as np
import tqdm
import tyro
import jax
import jax.numpy as jnp
from functools import partial
from typing import Dict, List, Tuple, Any, Optional

import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def create_dataset(config: _config.TrainConfig) -> tuple[_config.DataConfig, _data_loader.Dataset]:
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_dataset(data_config, config.model)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    return data_config, dataset


class JaxRunningStats:
    """JAX implementation of RunningStats for GPU acceleration."""

    def __init__(self):
        self._count = 0
        self._mean = None
        self._mean_of_squares = None
        self._min = None
        self._max = None
        self._histograms = None
        self._bin_edges = None
        self._num_quantile_bins = 5000  # Keep same as original for identical results
        self._initialized = False

    def _initialize(self, batch: np.ndarray) -> None:
        """Initialize statistics with the first batch."""
        if batch.ndim == 1:
            batch = batch.reshape(-1, 1)
        
        num_elements, vector_length = batch.shape
        self._count = num_elements
        
        # Convert to JAX arrays
        batch_jax = jnp.array(batch)
        self._mean = jnp.mean(batch_jax, axis=0)
        self._mean_of_squares = jnp.mean(batch_jax**2, axis=0)
        self._min = jnp.min(batch_jax, axis=0)
        self._max = jnp.max(batch_jax, axis=0)
        
        # Initialize histograms with numpy initially (for compatibility)
        self._histograms = [np.zeros(self._num_quantile_bins) for _ in range(vector_length)]
        self._bin_edges = [
            np.linspace(float(self._min[i]) - 1e-10, float(self._max[i]) + 1e-10, self._num_quantile_bins + 1)
            for i in range(vector_length)
        ]
        
        # Update histograms with initial batch (using numpy for exact compatibility)
        for i in range(vector_length):
            hist, _ = np.histogram(batch[:, i], bins=self._bin_edges[i])
            self._histograms[i] += hist
            
        self._initialized = True

    # JIT-compiled function to update mean and mean of squares
    @partial(jax.jit, static_argnums=(0,))
    def _update_moments_jit(self, current_count, current_mean, current_mean_sq, batch, batch_count):
        """Update mean and mean of squares using JAX."""
        new_count = current_count + batch_count
        batch_mean = jnp.mean(batch, axis=0)
        batch_mean_sq = jnp.mean(batch**2, axis=0)
        
        # Update running statistics using the same formula as original
        new_mean = current_mean + (batch_mean - current_mean) * (batch_count / new_count)
        new_mean_sq = current_mean_sq + (batch_mean_sq - current_mean_sq) * (batch_count / new_count)
        
        return new_count, new_mean, new_mean_sq

    def _adjust_histograms(self):
        """Adjust histograms when min or max changes - keep original logic."""
        for i in range(len(self._histograms)):
            old_edges = self._bin_edges[i]
            new_edges = np.linspace(float(self._min[i]), float(self._max[i]), self._num_quantile_bins + 1)
            
            # Redistribute existing histogram counts to new bins (keep original logic)
            new_hist, _ = np.histogram(old_edges[:-1], bins=new_edges, weights=self._histograms[i])
            
            self._histograms[i] = new_hist
            self._bin_edges[i] = new_edges

    def update(self, batch: np.ndarray) -> None:
        """Update running statistics with a batch of vectors."""
        # Convert input to numpy for consistency
        if isinstance(batch, jnp.ndarray):
            batch = np.array(batch)
            
        if batch.ndim == 1:
            batch = batch.reshape(-1, 1)
            
        num_elements, vector_length = batch.shape
        
        # Initialize if first batch
        if not self._initialized:
            self._initialize(batch)
            return
            
        # Check dimension consistency
        if vector_length != len(self._mean):
            raise ValueError(f"Expected {len(self._mean)} features, but got {vector_length}")
        
        # Update min and max (on CPU to match original logic)
        new_max = np.maximum(np.array(self._max), np.max(batch, axis=0))
        new_min = np.minimum(np.array(self._min), np.min(batch, axis=0))
        
        max_changed = np.any(new_max > self._max)
        min_changed = np.any(new_min < self._min)
        
        # Store as JAX arrays
        self._max = jnp.array(new_max)
        self._min = jnp.array(new_min)
        
        if max_changed or min_changed:
            self._adjust_histograms()
        
        # Update count, mean and mean_of_squares using JIT-compiled function
        batch_jax = jnp.array(batch)
        self._count, self._mean, self._mean_of_squares = self._update_moments_jit(
            self._count, self._mean, self._mean_of_squares, batch_jax, num_elements
        )
        
        # Update histograms (using numpy for exact compatibility)
        for i in range(vector_length):
            hist, _ = np.histogram(batch[:, i], bins=self._bin_edges[i])
            self._histograms[i] += hist

    def _compute_quantiles(self, quantiles):
        """Compute quantiles based on histograms - match original implementation."""
        results = []
        for q in quantiles:
            target_count = q * self._count
            q_values = []
            
            for hist, edges in zip(self._histograms, self._bin_edges):
                cumsum = np.cumsum(hist)
                idx = np.searchsorted(cumsum, target_count)
                q_values.append(edges[idx])
                
            results.append(np.array(q_values))
        return results

    def get_statistics(self) -> normalize.NormStats:
        """Compute and return the statistics (matches original implementation)."""
        if self._count < 2:
            raise ValueError("Cannot compute statistics for less than 2 vectors.")
        
        # Convert JAX arrays to numpy for consistency with original
        mean_np = np.array(self._mean)
        mean_sq_np = np.array(self._mean_of_squares)
        
        # Compute variance and standard deviation
        variance = mean_sq_np - mean_np**2
        stddev = np.sqrt(np.maximum(0, variance))
        
        # Compute quantiles using the same method as original
        q01, q99 = self._compute_quantiles([0.01, 0.99])
        
        return normalize.NormStats(mean=mean_np, std=stddev, q01=q01, q99=q99)


def main(config_name: str, max_frames: int | None = None):
    """Main function that computes and saves normalization statistics."""
    # Print info about JAX devices
    print(f"JAX is using {jax.device_count()} devices: {jax.devices()}")
    
    # Use same setup as original
    config = _config.get_config(config_name)
    data_config, dataset = create_dataset(config)

    num_frames = len(dataset)
    shuffle = False

    if max_frames is not None and max_frames < num_frames:
        num_frames = max_frames
        shuffle = True

    # Calculate appropriate batch size for GPUs - larger than original
    # For H100s with 80GB, we can use very large batch sizes
    devices = jax.devices()
    num_devices = len(devices)
    per_device_batch = 128  # Adjust based on your data size
    batch_size = per_device_batch * num_devices
    
    # Limit batch size based on dataset size
    batch_size = min(batch_size, num_frames // 10) 
    batch_size = max(batch_size, 16)  # Ensure minimum batch size
    
    print(f"Using batch size: {batch_size} across {num_devices} devices")
    
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=16,  # More workers for faster loading
        shuffle=shuffle,
        num_batches=num_frames,
        prefetch_factor=4,
        pin_memory=True,
    )

    # Same keys as original
    keys = ["state", "actions"]
    stats = {key: JaxRunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=(num_frames + batch_size - 1) // batch_size, desc="Computing stats"):
        for key in keys:
            values = np.asarray(batch[key])
            # Reshape to (total_elements, feature_dim) like in original
            if values.ndim > 2:
                values = values.reshape(-1, values.shape[-1])
            stats[key].update(values)

    # Convert statistics to the same format as original
    norm_stats = {key: stats[key].get_statistics() for key in stats.items()}

    # Save to the same location with same format
    output_path = config.assets_dirs / data_config.repo_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main) 