#!/bin/bash
set -e

source /home/masaya/workspace/VAD-GS/.venv/bin/activate
cd /home/masaya/workspace/VAD-GS

CONFIG=configs/example/t4_train_example.yaml

echo "=== Phase 1: Background mask generation (SAM vit_h) ==="
echo "Started at: $(date)"
cd script/t4
python3 generate_bkgd_masks.py --config /home/masaya/workspace/VAD-GS/$CONFIG --model-type vit_h
echo "Mask generation completed at: $(date)"

echo "=== Phase 2: Training ==="
echo "Started at: $(date)"
cd /home/masaya/workspace/VAD-GS
python3 train.py --config $CONFIG
echo "Training completed at: $(date)"
