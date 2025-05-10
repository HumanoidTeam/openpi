import jax
import gc

def clear_jax_gpu_memory():
    # Clear JAX's compilation and device caches explicitly
    jax.clear_backends()
    jax.clear_caches()
    gc.collect()

# Usage Example:
clear_jax_gpu_memory()