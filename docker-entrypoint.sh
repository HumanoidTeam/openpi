#!/bin/bash
set -e

# Set default values if not provided
MODEL_NAME=${MODEL_NAME:-"pi0_fast_rainbow_poc"}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"s3://hm-vla/checkpoints/pi0_fast_rainbow_poc/pi0_rainbow_poc_chips/20000"}

# Set JAX environment variables
# export JAX_PLATFORMS=${JAX_PLATFORMS:-"cpu"}
# export XLA_FLAGS=${XLA_FLAGS:-"--xla_force_host_platform_device_count=1"}

# Run the policy server
exec uv run serve-pi0-policy \
    --model-name "${MODEL_NAME}" \
    --checkpoint-path "${CHECKPOINT_PATH}" \
    --host 0.0.0.0 \
    --port 8000 