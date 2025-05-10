# src/openpi/training/clean_mem.py

import jax
import gc

def clear_jax_gpu_memory():
    """
    Frees JAX’s compilation caches and triggers Python GC.
    Note: JAX will only free device memory for arrays
    that no longer have any Python references.
    """
    # 1) Clear JAX’s compilation & staging caches
    jax.clear_caches()  

    # 2) Force a Python garbage-collection pass
    gc.collect()        

if __name__ == "__main__":
    clear_jax_gpu_memory()
    print("JAX caches cleared and garbage collection run.")# src/openpi/training/clean_mem.py

import jax
import gc

def clear_jax_gpu_memory():
    """
    Frees JAX’s compilation caches and triggers Python GC.
    Note: JAX will only free device memory for arrays
    that no longer have any Python references.
    """
    # 1) Clear JAX’s compilation & staging caches
    jax.clear_caches()  

    # 2) Force a Python garbage-collection pass
    gc.collect()        

if __name__ == "__main__":
    clear_jax_gpu_memory()
    print("JAX caches cleared and garbage collection run.")# src/openpi/training/clean_mem.py

import jax
import gc

def clear_jax_gpu_memory():
    """
    Frees JAX’s compilation caches and triggers Python GC.
    Note: JAX will only free device memory for arrays
    that no longer have any Python references.
    """
    # 1) Clear JAX’s compilation & staging caches
    jax.clear_caches()  

    # 2) Force a Python garbage-collection pass
    gc.collect()        

if __name__ == "__main__":
    clear_jax_gpu_memory()
    print("JAX caches cleared and garbage collection run.")