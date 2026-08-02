#!/usr/bin/env bash
set -euo pipefail

## Launches AWM + UNITE reconstruction fine-tuning on 8 GPUs.
## Set CKPT_PATH to a pretrained UNITE checkpoint and DATA_PATH to ImageNet train.
: "${CKPT_PATH:?Set CKPT_PATH=/path/to/pretrained_unite.pt before running this script.}"
: "${DATA_PATH:?Set DATA_PATH=/path/to/imagenet/train before running this script.}"

TORCH_DISTRIBUTED_DEBUG=INFO torchrun --nproc_per_node=8 \
  --master_port=12343 main_train_awm.py \
  --config configs/imagenet_awm_recon.yaml \
  --ckpt "$CKPT_PATH" \
  --data-path "$DATA_PATH" \
  --results-dir outputs_awm \
  --experiment-name unite-awm-recon-dinov2
