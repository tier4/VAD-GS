#!/usr/bin/env bash
# One-shot profile run for VAD-GS training.
#
# Runs train.py for ~500 iters on a single GPU with the perf_timer enabled.
# Sweep / wandb are completely out of the picture. Per-section averages are
# printed every VAD_GS_PERF_EVERY iters; a CSV is also written to
# output/perf/perf.csv for later plotting.
#
# Usage:
#   bash script/perf/run_profile.sh                # GPU 0, default config
#   GPU=3 bash script/perf/run_profile.sh
#   CONFIG=configs/sweep/profile_quick.yaml \
#     bash script/perf/run_profile.sh
#
# Env knobs (with sane defaults):
#   GPU                 default 0
#   CONFIG              default configs/sweep/profile_quick.yaml
#   THREADS             intra-op thread cap (default: 28, same as 8-agent sweep)
#   CACHE_FROM          model dir to symlink input_ply / colmap from
#                       (default: output/t4_exp/tokyo_teleport_h100x8)
#   PROFILE_EVERY       print every N iters (default 50)
#   PROFILE_SYNC        1 (default) inserts cuda.synchronize around sections
#                       so the print reports true wall time; 0 to measure
#                       Python-launch time only.

set -euo pipefail

cd "$(dirname "$0")/../.."

GPU="${GPU:-0}"
CONFIG="${CONFIG:-configs/sweep/profile_quick.yaml}"
THREADS="${THREADS:-28}"
CACHE_FROM="${CACHE_FROM:-output/t4_exp/tokyo_teleport_h100x8}"
PROFILE_EVERY="${PROFILE_EVERY:-50}"
PROFILE_SYNC="${PROFILE_SYNC:-1}"

PERF_DIR="output/perf"
mkdir -p "$PERF_DIR"
CSV_FILE="$PERF_DIR/perf_$(date +%Y%m%d_%H%M%S).csv"
LOG_FILE="$PERF_DIR/perf_$(date +%Y%m%d_%H%M%S).log"

# Symlink preprocess cache (input_ply, colmap) into the run's model_path so
# we do not rebuild COLMAP every time we kick off a profile run.
read -r EXP_NAME TASK < <(uv run python - "$CONFIG" <<'PY'
import sys, yaml
def load(path):
    with open(path) as f:
        return yaml.safe_load(f) or {}
cfg = load(sys.argv[1])
parent = cfg.get('parent_cfg')
if parent:
    # parent_cfg in yaml is interpreted relative to cwd by yacs make_cfg,
    # and run_profile.sh cd's to the repo root before this call.
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

echo "[profile] GPU=$GPU CONFIG=$CONFIG THREADS=$THREADS"
echo "[profile] log -> $LOG_FILE"
echo "[profile] csv -> $CSV_FILE"

CUDA_VISIBLE_DEVICES="$GPU" \
OMP_NUM_THREADS="$THREADS" \
MKL_NUM_THREADS="$THREADS" \
OPENBLAS_NUM_THREADS="$THREADS" \
NUMEXPR_NUM_THREADS="$THREADS" \
VAD_GS_NUM_THREADS="$THREADS" \
VAD_GS_PERF=1 \
VAD_GS_PERF_SYNC="$PROFILE_SYNC" \
VAD_GS_PERF_EVERY="$PROFILE_EVERY" \
VAD_GS_PERF_FILE="$CSV_FILE" \
VAD_GS_TORCH_COMPILE="${VAD_GS_TORCH_COMPILE:-}" \
    uv run python train.py --config "$CONFIG" 2>&1 | tee "$LOG_FILE"
