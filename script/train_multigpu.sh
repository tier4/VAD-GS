#!/bin/bash
# Multi-GPU training with torchrun.
#
# Usage:
#   bash script/train_multigpu.sh [config] [nproc]
#
# Examples:
#   bash script/train_multigpu.sh                                        # default: 8 GPUs
#   bash script/train_multigpu.sh configs/example/t4_train_h100x8.yaml 4 # 4 GPUs
#
# Single-GPU training (existing method):
#   python train.py --config configs/example/t4_train_example.yaml

set -euo pipefail

CONFIG="${1:-configs/example/t4_train_h100x8.yaml}"
NPROC="${2:-8}"

echo "=== Multi-GPU Training ==="
echo "Config : ${CONFIG}"
echo "GPUs   : ${NPROC}"
echo "=========================="

# Step 1: single-process preprocessing (COLMAP + pointcloud).
# Skipped automatically when the cache already exists.
echo "--- Preprocess ---"
python script/t4/preprocess.py --config "${CONFIG}"

# Step 2: distributed training.
echo "--- Train (torchrun) ---"
torchrun \
    --nproc_per_node="${NPROC}" \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=29500 \
    train.py --config "${CONFIG}"
