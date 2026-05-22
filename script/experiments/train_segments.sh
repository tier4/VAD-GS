#!/usr/bin/env bash
# Direct per-segment training for the segmented-BG-merge pipeline.
#
# No sweep agent — just runs `python train.py --config seg_XX.yaml` for
# each of the 13 segments, throttled to NUM_GPUS in parallel. Each
# training is single-GPU (the seg_base.yaml has dist.enabled=false), so
# 8 segments run in parallel per wave on an 8-GPU box. With 13 segments
# / 8 GPUs this is 2 waves → ~100 min wall-clock.
#
# All 13 trainings use the same hyperparameters (seg_base.yaml inherits
# from nuscenes_train_000.yaml). No HP tuning. If you want HP search per
# segment, use a sweep (see configs/sweep/t4_segmented.yaml).
#
# Each run logs to wandb under WANDB_RUN_GROUP so the 13 runs cluster
# together in the UI. Run names use the yaml's `exp_name`
# (e.g. `t4_seg_03_f120_180`).
#
# After this script exits, the next steps are:
#
#   1. Pick best per segment (since there's only 1 run per segment, this
#      is essentially "pick the only run"):
#        python script/experiments/select_best_per_segment.py \\
#            --sweep-id <ignored — see note> --emit-merge-cmd
#      Or skip this and assemble the merge command by hand from the
#      output/t4_exp/t4_seg_*_f*/trained_model/iteration_30000.pth paths.
#
#   2. Merge BG checkpoints:
#        python script/experiments/merge_bg_checkpoints.py \\
#            --ckpts output/t4_exp/t4_seg_00_f0_60/trained_model/iteration_30000.pth ... \\
#            --out  output/t4_exp/merged_bg/iteration_0.pth \\
#            --dedup-voxel-size 0.15
#
#   3. Fine-tune on the full sequence (8-GPU distributed):
#        bash script/train_multigpu.sh \\
#            configs/experiments/segmented/finetune_merged.yaml 8
#
# See configs/experiments/segmented/README.md for the full pipeline.
#
# Usage:
#   bash script/experiments/train_segments.sh                       # 8 GPUs from 0
#   NUM_GPUS=4 bash script/experiments/train_segments.sh            # 4 GPUs
#   GPU_OFFSET=2 NUM_GPUS=6 bash script/experiments/train_segments.sh # GPUs 2..7
#
# Env vars:
#   NUM_GPUS          number of GPUs to use in parallel (default 8)
#   GPU_OFFSET        first GPU id to use (default 0)
#   LOG_DIR           per-segment stdout/stderr destination (default output/segmented_logs)
#   WANDB_RUN_GROUP   wandb group name (default t4_segmented_<timestamp>)
#   ONLY_SEGMENTS     space-separated list of segment indices to run
#                     (e.g. "0 5 12"). Default: all 13.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}"

NUM_GPUS="${NUM_GPUS:-8}"
GPU_OFFSET="${GPU_OFFSET:-0}"
LOG_DIR="${LOG_DIR:-output/segmented_logs}"
mkdir -p "${LOG_DIR}"

# Group label so the 13 wandb runs are easy to find / compare in the UI.
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-t4_segmented_${TIMESTAMP}}"

# Default: all 13 segments. ONLY_SEGMENTS=... overrides.
if [[ -n "${ONLY_SEGMENTS:-}" ]]; then
    # shellcheck disable=SC2206
    SEG_INDICES=( ${ONLY_SEGMENTS} )
else
    SEG_INDICES=( 0 1 2 3 4 5 6 7 8 9 10 11 12 )
fi

# Cap per-job thread count so N PyTorch processes do not oversubscribe
# the box. nproc returns logical cores; divide evenly.
TOTAL_CPUS="$(nproc 2>/dev/null || echo 8)"
THREADS_PER_JOB="$(( TOTAL_CPUS / NUM_GPUS ))"
(( THREADS_PER_JOB < 1 )) && THREADS_PER_JOB=1

# --- Pre-flight checks --------------------------------------------------

# 1) Refuse to launch if other GPU work is going on (sweep agents etc.).
existing=$(pgrep -f "train\.py|sweep_run\.py|wandb agent" 2>/dev/null || true)
if [[ -n "${existing}" ]]; then
    echo "[train_segments] ERROR: existing train/sweep processes detected (PIDs: ${existing})." >&2
    echo "[train_segments]        Stop them first:  pkill -9 -f train.py; pkill -9 -f sweep_run.py" >&2
    exit 1
fi

# 2) All segment yamls must exist.
for i in "${SEG_INDICES[@]}"; do
    seg=$(printf "seg_%02d" "$i")
    yaml="configs/experiments/segmented/${seg}.yaml"
    if [[ ! -f "${yaml}" ]]; then
        echo "[train_segments] ERROR: missing ${yaml}" >&2
        exit 1
    fi
done

# 3) Tools available.
if [[ ! -x ".venv/bin/python" ]]; then
    echo "[train_segments] WARNING: .venv/bin/python not found — using PATH python." >&2
    PYTHON="${PYTHON:-python}"
else
    PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
fi

echo "============================================================"
echo "[train_segments] Segmented BG-merge pipeline — phase 1 (direct train)"
echo "[train_segments] segments     : ${SEG_INDICES[*]}  (count=${#SEG_INDICES[@]})"
echo "[train_segments] NUM_GPUS     : ${NUM_GPUS} (offset=${GPU_OFFSET})"
echo "[train_segments] threads/job  : ${THREADS_PER_JOB}"
echo "[train_segments] LOG_DIR      : ${LOG_DIR}"
echo "[train_segments] WANDB group  : ${WANDB_RUN_GROUP}"
echo "[train_segments] Started at   : $(date)"
echo "============================================================"

# --- Wave scheduler -----------------------------------------------------
# Spawn jobs in waves of NUM_GPUS, wait for each wave to finish before
# starting the next. Each job pins to one GPU via CUDA_VISIBLE_DEVICES.
# Simpler than a dynamic scheduler; the runtime variance across
# similarly-configured segments is small enough that wave slack is OK.

PIDS=()
PID_GPUS=()
PID_SEGS=()

dispatch() {
    local seg_idx="$1"
    local gpu_slot="$2"
    local gpu_id=$(( GPU_OFFSET + gpu_slot ))
    local seg=$(printf "seg_%02d" "${seg_idx}")
    local log_file="${LOG_DIR}/${seg}_gpu${gpu_id}.log"
    local yaml="configs/experiments/segmented/${seg}.yaml"

    echo "[dispatch] ${seg} -> GPU ${gpu_id}, log=${log_file}"

    (
        CUDA_VISIBLE_DEVICES="${gpu_id}" \
        OMP_NUM_THREADS="${THREADS_PER_JOB}" \
        MKL_NUM_THREADS="${THREADS_PER_JOB}" \
        OPENBLAS_NUM_THREADS="${THREADS_PER_JOB}" \
        NUMEXPR_NUM_THREADS="${THREADS_PER_JOB}" \
        VAD_GS_NUM_THREADS="${THREADS_PER_JOB}" \
        WANDB_RUN_GROUP="${WANDB_RUN_GROUP}" \
            "${PYTHON}" train.py --config "${yaml}" \
            > "${log_file}" 2>&1
    ) &
    local pid=$!
    PIDS+=( "${pid}" )
    PID_GPUS+=( "${gpu_id}" )
    PID_SEGS+=( "${seg}" )
}

cleanup() {
    echo "[train_segments] received interrupt — killing in-flight jobs..."
    for pid in "${PIDS[@]}"; do
        kill "${pid}" 2>/dev/null || true
    done
    wait 2>/dev/null || true
    exit 130
}
trap cleanup INT TERM

wave_idx=0
i=0
while (( i < ${#SEG_INDICES[@]} )); do
    # Fill the wave with up to NUM_GPUS jobs.
    PIDS=(); PID_GPUS=(); PID_SEGS=()
    slot=0
    wave_start_idx="${i}"
    while (( slot < NUM_GPUS && i < ${#SEG_INDICES[@]} )); do
        dispatch "${SEG_INDICES[$i]}" "${slot}"
        slot=$(( slot + 1 ))
        i=$(( i + 1 ))
    done

    wave_idx=$(( wave_idx + 1 ))
    echo "[wave ${wave_idx}] ${#PIDS[@]} jobs running; waiting..."

    # Wait for every job in this wave and record exit codes individually.
    wave_failed=0
    for k in "${!PIDS[@]}"; do
        pid="${PIDS[$k]}"
        seg="${PID_SEGS[$k]}"
        gpu="${PID_GPUS[$k]}"
        if wait "${pid}"; then
            echo "[wave ${wave_idx}] ✓ ${seg} (GPU ${gpu}) finished OK"
        else
            ec=$?
            echo "[wave ${wave_idx}] ✗ ${seg} (GPU ${gpu}) FAILED (exit ${ec}) — see ${LOG_DIR}/${seg}_gpu${gpu}.log" >&2
            wave_failed=$(( wave_failed + 1 ))
        fi
    done

    if (( wave_failed > 0 )); then
        echo "[train_segments] ${wave_failed} job(s) failed in wave ${wave_idx}." >&2
        echo "[train_segments] Continuing with remaining waves; rerun failed segments via ONLY_SEGMENTS=..." >&2
    fi
done

echo
echo "============================================================"
echo "[train_segments] All requested segments processed at $(date)"
echo "============================================================"

# Print the ckpt paths for the next phase.
echo
echo "Per-segment checkpoints (input to merge_bg_checkpoints.py):"
ckpts=()
for i in "${SEG_INDICES[@]}"; do
    seg=$(printf "seg_%02d" "$i")
    # Resolve the exp_name from the yaml so the path is correct.
    exp_name=$(grep -E '^exp_name:' "configs/experiments/segmented/${seg}.yaml" | awk '{print $2}')
    ckpt="output/t4_exp/${exp_name}/trained_model/iteration_30000.pth"
    if [[ -f "${ckpt}" ]]; then
        echo "  ✓ ${ckpt}"
        ckpts+=( "${ckpt}" )
    else
        echo "  ✗ ${ckpt}   (missing — segment may have failed)"
    fi
done

echo
echo "Next steps:"
echo
echo "  # Merge BG Gaussians from all completed segments:"
echo "  python script/experiments/merge_bg_checkpoints.py \\"
echo "      --ckpts ${ckpts[*]} \\"
echo "      --out output/t4_exp/merged_bg/iteration_0.pth \\"
echo "      --dedup-voxel-size 0.15"
echo
echo "  # Then fine-tune on the full sequence:"
echo "  bash script/train_multigpu.sh \\"
echo "      configs/experiments/segmented/finetune_merged.yaml 8"
