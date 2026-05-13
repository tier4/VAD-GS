#!/usr/bin/env bash
# Launch N parallel wandb agents, one per GPU.
#
# Usage:
#   # Create a sweep and immediately launch 8 agents:
#   bash script/sweep/sweep_launch.sh configs/sweep/t4_sweep.yaml
#
#   # Attach more agents to an existing sweep:
#   SWEEP_ID=entity/project/abc123 bash script/sweep/sweep_launch.sh
#
# Env vars:
#   NUM_AGENTS              number of agents to spawn (default: 8)
#   GPU_OFFSET              first GPU id to use (default: 0; agents run on
#                           offset..offset+NUM_AGENTS-1)
#   LOG_DIR                 where agent logs go (default: output/sweep_logs)
#   COUNT                   per-agent --count arg (number of runs each agent
#                           does before exiting; default: unset = unlimited)
#   VAD_GS_SWEEP_CACHE_FROM relative or absolute path of a pre-built model
#                           dir whose input_ply/ + colmap/ should be
#                           symlinked into each run's model_path. Exported
#                           to every agent so they share preprocessing.
#                           Default: output/t4_exp/tokyo_teleport_h100x8
#   THREADS_PER_AGENT       OpenMP/MKL/OpenBLAS thread cap per agent.
#                           Default: nproc / NUM_AGENTS. Without this each
#                           PyTorch process tries to grab every CPU core,
#                           and 8 of them oversubscribe the machine into
#                           uselessness (load avg > 300, GPU util ~0%).

set -euo pipefail

cd "$(dirname "$0")/../.."  # repo root

SWEEP_YAML="${1:-}"
NUM_AGENTS="${NUM_AGENTS:-8}"
GPU_OFFSET="${GPU_OFFSET:-0}"
LOG_DIR="${LOG_DIR:-output/sweep_logs}"
COUNT="${COUNT:-}"
VAD_GS_SWEEP_CACHE_FROM="${VAD_GS_SWEEP_CACHE_FROM:-output/t4_exp/tokyo_teleport_h100x8}"
export VAD_GS_SWEEP_CACHE_FROM

# Cap per-agent thread count so 8 PyTorch processes do not oversubscribe
# the box. nproc returns logical cores; default to floor(nproc / NUM_AGENTS).
TOTAL_CPUS="$(nproc 2>/dev/null || echo 8)"
THREADS_PER_AGENT="${THREADS_PER_AGENT:-$((TOTAL_CPUS / NUM_AGENTS))}"
if [[ "$THREADS_PER_AGENT" -lt 1 ]]; then
    THREADS_PER_AGENT=1
fi
echo "[launch] $TOTAL_CPUS logical CPUs / $NUM_AGENTS agents = $THREADS_PER_AGENT threads/agent"

mkdir -p "$LOG_DIR"

# Sanity-check the preprocessing cache. If it's missing, every agent will
# rebuild COLMAP + bkgd PLY (~10 min each) — warn loudly but don't abort,
# since a user may genuinely want a from-scratch run.
if [[ -n "$VAD_GS_SWEEP_CACHE_FROM" ]]; then
    cache_abs="$VAD_GS_SWEEP_CACHE_FROM"
    [[ "$cache_abs" = /* ]] || cache_abs="$(pwd)/$cache_abs"
    if [[ ! -f "$cache_abs/input_ply/points3D_bkgd.ply" ]]; then
        echo "[launch] WARNING: $cache_abs/input_ply/points3D_bkgd.ply missing." >&2
        echo "[launch]          Each agent will rebuild COLMAP + bkgd PLY from scratch." >&2
        echo "[launch]          To prebuild once, run:" >&2
        echo "[launch]            python script/sweep/sweep_prebuild_cache.py --config configs/sweep/base.yaml" >&2
    else
        echo "[launch] preprocess cache: $cache_abs"
    fi
fi

if [[ -z "${SWEEP_ID:-}" ]]; then
    if [[ -z "$SWEEP_YAML" ]]; then
        echo "Usage: $0 <sweep_yaml>   or   SWEEP_ID=<id> $0" >&2
        exit 1
    fi
    echo "[launch] creating sweep from $SWEEP_YAML"
    SWEEP_ID=$(python script/sweep/sweep_init.py "$SWEEP_YAML" | tail -n1)
    echo "[launch] SWEEP_ID=$SWEEP_ID"
fi

declare -a AGENT_PIDS=()

cleanup() {
    echo "[launch] terminating agents..."
    for pid in "${AGENT_PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
}
trap cleanup INT TERM

for i in $(seq 0 $((NUM_AGENTS - 1))); do
    gpu=$((GPU_OFFSET + i))
    log_file="$LOG_DIR/agent_gpu${gpu}.log"
    echo "[launch] agent $i -> GPU $gpu, log=$log_file"

    if [[ -n "$COUNT" ]]; then
        CUDA_VISIBLE_DEVICES="$gpu" \
        OMP_NUM_THREADS="$THREADS_PER_AGENT" \
        MKL_NUM_THREADS="$THREADS_PER_AGENT" \
        OPENBLAS_NUM_THREADS="$THREADS_PER_AGENT" \
        NUMEXPR_NUM_THREADS="$THREADS_PER_AGENT" \
        VAD_GS_NUM_THREADS="$THREADS_PER_AGENT" \
            wandb agent --count "$COUNT" "$SWEEP_ID" \
            > "$log_file" 2>&1 &
    else
        CUDA_VISIBLE_DEVICES="$gpu" \
        OMP_NUM_THREADS="$THREADS_PER_AGENT" \
        MKL_NUM_THREADS="$THREADS_PER_AGENT" \
        OPENBLAS_NUM_THREADS="$THREADS_PER_AGENT" \
        NUMEXPR_NUM_THREADS="$THREADS_PER_AGENT" \
        VAD_GS_NUM_THREADS="$THREADS_PER_AGENT" \
            wandb agent "$SWEEP_ID" \
            > "$log_file" 2>&1 &
    fi
    AGENT_PIDS+=("$!")
done

echo "[launch] ${#AGENT_PIDS[@]} agents running; PIDs: ${AGENT_PIDS[*]}"
echo "[launch] tail -f $LOG_DIR/agent_gpu*.log"
wait
