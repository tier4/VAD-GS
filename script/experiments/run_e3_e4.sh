#!/bin/bash
# Sequentially run the E-3 / E-4 / combined verification experiments.
#
# Each run uses all NPROC GPUs via torchrun (same recipe as
# script/train_multigpu.sh). Output dirs are disambiguated by the
# exp_name in each YAML, so logs and ckpts don't collide.
#
# Prereqs:
#   * Stop the sweep first — it occupies all 8 GPUs:
#       pkill -f "wandb agent.*nu6sy3bm"
#   * Preprocess cache should exist (the sweep has been running, so it does).
#
# Usage:
#   bash script/experiments/run_e3_e4.sh [nproc]
#   bash script/experiments/run_e3_e4.sh 8    # default
#
# Total wall-clock: ~3 × (6250 iter / nproc step rate) ≈ 3-4 hours at nproc=8.

set -euo pipefail

NPROC="${1:-8}"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}"

# Group all three runs together in the wandb UI so they're easy to compare.
# WANDB_ENTITY / WANDB_PROJECT are loaded from <repo>/.env inside train.py.
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-e3e4_pdense_verify}"

CONFIGS=(
  "configs/experiments/e4_pdense_bkgd.yaml"
  "configs/experiments/e3_pdense_obj.yaml"
  "configs/experiments/e34_combined.yaml"
)

for cfg in "${CONFIGS[@]}"; do
  echo
  echo "================================================================"
  echo "Running: ${cfg}  (nproc=${NPROC})"
  echo "Started: $(date)"
  echo "================================================================"

  # Single-process preprocess (no-op if cache hits).
  python script/t4/preprocess.py --config "${cfg}"

  torchrun \
      --nproc_per_node="${NPROC}" \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=localhost \
      --master_port=29500 \
      train.py --config "${cfg}"

  echo "Finished ${cfg} at: $(date)"
done

echo
echo "All experiments done. Compare test/test_view PSNR (psnr_obj,"
echo "psnr_bkgd_near) at iter 5000/6250 across:"
echo "  output/t4_exp/tokyo_teleport_h100x8/      # baseline (existing)"
echo "  output/t4_exp/e34_e4_pdense015/           # E-4 only"
echo "  output/t4_exp/e34_e3_pdense_obj07/        # E-3 only"
echo "  output/t4_exp/e34_combined_pd015_obj07/   # E-3 + E-4"
