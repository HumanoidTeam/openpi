import jax
import jaxlib
import flax
import optax

print("=== Version Info ===")
print("JAX version:", jax.__version__)
print("jaxlib version:", jaxlib.__version__)
print("Flax version:", flax.__version__)
print("Optax version:", optax.__version__)
print()

print("=== JAX Devices ===")
for device in jax.devices():
    print(f"  {device}")

print()

print("=== Backend Info ===")
print("Backend platform:", jax.default_backend())
print("Backend version:", jaxlib.version.__version__)
print()

print("=== Device Memory Info ===")
for device in jax.devices():
    mem_info = device.memory_stats()
    total_memory = mem_info.get('bytes_limit', 0)
    print(f"Device {device.id}: {device.device_kind}, Total Memory: {total_memory / (1024 ** 3):.2f} GB")

