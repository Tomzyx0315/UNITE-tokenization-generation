#!/usr/bin/env bash
set -euo pipefail

## Launches first-pass AWM fine-tuning on 8 GPUs.
## Set CKPT_PATH to a pretrained UNITE checkpoint.
: "${CKPT_PATH:?Set CKPT_PATH=/path/to/pretrained_unite.pt before running this script.}"

TORCH_DISTRIBUTED_DEBUG=INFO torchrun --nproc_per_node=8 \
  --master_port=12343 main_train_awm.py \
  --config configs/imagenet_awm.yaml \
  --ckpt "$CKPT_PATH" \
  --results-dir outputs_awm \
  --experiment-name unite-awm-dinov2
