#!/usr/bin/env bash
# Run train.py briefly under torch.profiler to get an operator-level
# breakdown of the inner loop. After warmup+active iters complete the
# script writes a Chrome/Perfetto trace, prints a top-N op table, and
# exits.
#
# View the trace by uploading <out_dir>/*.json.gz to:
#   - https://ui.perfetto.dev   (recommended)
#   - chrome://tracing
#
# Usage:
#   bash script/perf/run_torch_profile.sh
#
# Env knobs:
#   GPU                       default 0
#   CONFIG                    default configs/sweep/profile_quick.yaml
#   THREADS                   intra-op thread cap (default 28)
#   CACHE_FROM                preprocess cache (default output/t4_exp/tokyo_teleport_h100x8)
#   PROFILE_WARMUP            iters skipped before recording (default 5)
#   PROFILE_ACTIVE            iters actually profiled (default 10)
#                             — short on purpose; trace files balloon fast.

set -euo pipefail

cd "$(dirname "$0")/../.."

GPU="${GPU:-0}"
CONFIG="${CONFIG:-configs/sweep/profile_quick.yaml}"
THREADS="${THREADS:-28}"
CACHE_FROM="${CACHE_FROM:-output/t4_exp/tokyo_teleport_h100x8}"
PROFILE_WARMUP="${PROFILE_WARMUP:-5}"
PROFILE_ACTIVE="${PROFILE_ACTIVE:-10}"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="output/perf/torch_profile_${STAMP}"
LOG_FILE="output/perf/torch_profile_${STAMP}.log"
mkdir -p "$OUT_DIR"

# Symlink preprocess cache (input_ply, colmap) into the run's model_path so
# COLMAP is not rebuilt every time we kick off a profile run. (Shared with
# run_profile.sh — keep these two scripts in sync if you tweak this block.)
read -r EXP_NAME TASK < <(uv run python - "$CONFIG" <<'PY'
import sys, yaml
def load(path):
    with open(path) as f:
        return yaml.safe_load(f) or {}
cfg = load(sys.argv[1])
parent = cfg.get('parent_cfg')
if parent:
    base = load(parent)
    base.update(cfg)
    cfg = base
print(cfg.get('exp_name', 'profile_quick'), cfg.get('task', 't4_exp'))
PY
)
MODEL_DIR="output/$TASK/$EXP_NAME"
mkdir -p "$MODEL_DIR"
for d in input_ply colmap; do
    src="$(pwd)/$CACHE_FROM/$d"
    dst="$MODEL_DIR/$d"
    if [[ -e "$dst" || -L "$dst" ]]; then
        rm -rf "$dst"
    fi
    if [[ -d "$src" ]]; then
        ln -s "$src" "$dst"
    fi
done

echo "[torch.profile] GPU=$GPU CONFIG=$CONFIG warmup=$PROFILE_WARMUP active=$PROFILE_ACTIVE"
echo "[torch.profile] out -> $OUT_DIR"
echo "[torch.profile] log -> $LOG_FILE"

CUDA_VISIBLE_DEVICES="$GPU" \
OMP_NUM_THREADS="$THREADS" \
MKL_NUM_THREADS="$THREADS" \
OPENBLAS_NUM_THREADS="$THREADS" \
NUMEXPR_NUM_THREADS="$THREADS" \
VAD_GS_NUM_THREADS="$THREADS" \
VAD_GS_TORCH_PROFILE_ITERS="$PROFILE_ACTIVE" \
VAD_GS_TORCH_PROFILE_WARMUP="$PROFILE_WARMUP" \
VAD_GS_TORCH_PROFILE_OUT="$OUT_DIR" \
    uv run python train.py --config "$CONFIG" 2>&1 | tee "$LOG_FILE"

echo
echo "[torch.profile] done."
ls -lh "$OUT_DIR" || true
echo "[torch.profile] upload the *.json.gz to https://ui.perfetto.dev"
