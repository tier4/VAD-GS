#!/usr/bin/env bash
# End-to-end driver for the segmented-BG-merge pipeline.
#
#   Phase 1  per-segment training, 13 segments, 8-parallel waves
#            (single-GPU each; seg_base.yaml has dist.enabled=false)
#   Phase 2  merge BG + merge OBJ checkpoints (CPU)
#   Phase 3  fine-tune on the full 561-frame sequence
#            (8-GPU DDP via torchrun, bg_lidar_prune enabled)
#
# Output goes to ${WORKSPACE_DIR}/output/... — matches the workspace
# override added to configs/experiments/segmented/seg_base.yaml so
# both the python side (cfg.workspace) and the bash side agree on
# the same root.
#
# Usage:
#   bash script/experiments/train_full_pipeline.sh
#
# Resume after partial completion:
#   SKIP_SEGMENTS=1 bash script/experiments/train_full_pipeline.sh
#   SKIP_SEGMENTS=1 SKIP_MERGE=1 bash script/experiments/train_full_pipeline.sh
#
# Env vars:
#   WORKSPACE_DIR   output root (default /mnt/nvme/kataoka/VAD-GS)
#   NUM_GPUS        parallel slots for phase 1 (default 8)
#   GPU_OFFSET      first GPU id (default 0)
#   FINETUNE_NPROC  DDP world size for phase 3 (default 8)
#   WANDB_RUN_GROUP wandb group name (default t4_segpipe_<timestamp>)
#   SKIP_SEGMENTS   set to 1 to skip phase 1
#   SKIP_MERGE      set to 1 to skip phase 2
#   SKIP_FINETUNE   set to 1 to skip phase 3
#   ONLY_SEGMENTS   space-separated indices to limit phase 1 (e.g. "3 7 12")

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}"

WORKSPACE_DIR="${WORKSPACE_DIR:-/mnt/nvme/kataoka/VAD-GS}"
NUM_GPUS="${NUM_GPUS:-8}"
GPU_OFFSET="${GPU_OFFSET:-0}"
FINETUNE_NPROC="${FINETUNE_NPROC:-8}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-t4_segpipe_${TIMESTAMP}}"

OUT_ROOT="${WORKSPACE_DIR}/output/t4_exp"
LOG_DIR="${WORKSPACE_DIR}/output/segmented_logs"
mkdir -p "${LOG_DIR}" "${OUT_ROOT}"

if [[ -n "${ONLY_SEGMENTS:-}" ]]; then
    # shellcheck disable=SC2206
    SEG_INDICES=( ${ONLY_SEGMENTS} )
else
    SEG_INDICES=( 0 1 2 3 4 5 6 7 8 9 10 11 12 )
fi

# Spread CPU threads so N parallel pytorch processes don't oversubscribe.
TOTAL_CPUS="$(nproc 2>/dev/null || echo 8)"
THREADS_PER_JOB="$(( TOTAL_CPUS / NUM_GPUS ))"
(( THREADS_PER_JOB < 1 )) && THREADS_PER_JOB=1

UV_RUN=( uv run )

# --- Safety: refuse to launch if other train.py / torchrun is alive ---------
existing=$(pgrep -af 'train\.py|torchrun.*train\.py|sweep_run\.py|wandb agent' 2>/dev/null || true)
if [[ -n "${existing}" ]]; then
    echo "[pipeline] ERROR: existing train/torchrun/sweep processes detected:" >&2
    echo "${existing}" >&2
    echo "[pipeline]        Stop them first:  pkill -9 -f 'train\.py|torchrun.*train\.py|sweep_run\.py'" >&2
    exit 1
fi

# Cleanup handler for in-flight phase 1 jobs.
PIDS=()
cleanup() {
    echo "[pipeline] interrupted — killing in-flight jobs..."
    for pid in "${PIDS[@]:-}"; do kill "${pid}" 2>/dev/null || true; done
    wait 2>/dev/null || true
    exit 130
}
trap cleanup INT TERM

# ============================================================================
# Phase 1 — per-segment training
# ============================================================================
if [[ "${SKIP_SEGMENTS:-0}" != "1" ]]; then
    echo "============================================================"
    echo "[pipeline] Phase 1: per-segment training"
    echo "[pipeline]   workspace : ${WORKSPACE_DIR}"
    echo "[pipeline]   segments  : ${SEG_INDICES[*]}  (count=${#SEG_INDICES[@]})"
    echo "[pipeline]   num GPUs  : ${NUM_GPUS} (offset=${GPU_OFFSET})"
    echo "[pipeline]   threads/j : ${THREADS_PER_JOB}"
    echo "[pipeline]   log dir   : ${LOG_DIR}"
    echo "[pipeline]   wandb grp : ${WANDB_RUN_GROUP}"
    echo "[pipeline]   started   : $(date)"
    echo "============================================================"

    # Validate all segment yamls up front.
    for i in "${SEG_INDICES[@]}"; do
        seg_short=$(printf "seg_%02d" "$i")
        yaml="configs/experiments/segmented/${seg_short}.yaml"
        [[ -f "${yaml}" ]] || { echo "[pipeline] ERROR: missing ${yaml}" >&2; exit 1; }
    done

    PID_SEGS=(); PID_GPUS=()
    dispatch() {
        local seg_idx="$1" slot="$2"
        local gpu_id=$(( GPU_OFFSET + slot ))
        local seg_short=$(printf "seg_%02d" "${seg_idx}")
        local yaml="configs/experiments/segmented/${seg_short}.yaml"
        local log_file="${LOG_DIR}/${seg_short}_gpu${gpu_id}.log"
        echo "[dispatch] ${seg_short} -> GPU ${gpu_id}, log=${log_file}"
        (
            CUDA_VISIBLE_DEVICES="${gpu_id}" \
            OMP_NUM_THREADS="${THREADS_PER_JOB}" \
            MKL_NUM_THREADS="${THREADS_PER_JOB}" \
            OPENBLAS_NUM_THREADS="${THREADS_PER_JOB}" \
            NUMEXPR_NUM_THREADS="${THREADS_PER_JOB}" \
            VAD_GS_NUM_THREADS="${THREADS_PER_JOB}" \
            WANDB_RUN_GROUP="${WANDB_RUN_GROUP}" \
                "${UV_RUN[@]}" python train.py --config "${yaml}" \
                > "${log_file}" 2>&1
        ) &
        PIDS+=( "$!" )
        PID_SEGS+=( "${seg_short}" )
        PID_GPUS+=( "${gpu_id}" )
    }

    wave=0; i=0
    while (( i < ${#SEG_INDICES[@]} )); do
        PIDS=(); PID_SEGS=(); PID_GPUS=()
        slot=0
        while (( slot < NUM_GPUS && i < ${#SEG_INDICES[@]} )); do
            dispatch "${SEG_INDICES[$i]}" "${slot}"
            slot=$(( slot + 1 )); i=$(( i + 1 ))
        done
        wave=$(( wave + 1 ))
        echo "[wave ${wave}] ${#PIDS[@]} jobs running; waiting..."

        wave_failed=0
        for k in "${!PIDS[@]}"; do
            if wait "${PIDS[$k]}"; then
                echo "[wave ${wave}] OK  ${PID_SEGS[$k]} (GPU ${PID_GPUS[$k]})"
            else
                ec=$?
                echo "[wave ${wave}] FAIL ${PID_SEGS[$k]} (GPU ${PID_GPUS[$k]}) exit=${ec}  — see ${LOG_DIR}/${PID_SEGS[$k]}_gpu${PID_GPUS[$k]}.log" >&2
                wave_failed=$(( wave_failed + 1 ))
            fi
        done
        if (( wave_failed > 0 )); then
            echo "[pipeline] WARNING: ${wave_failed} job(s) failed in wave ${wave} — continuing; rerun via ONLY_SEGMENTS=..." >&2
        fi
    done

    echo "[pipeline] Phase 1 done at $(date)"
else
    echo "[pipeline] SKIP_SEGMENTS=1 — skipping segment training"
fi

# ============================================================================
# Phase 2 — merge BG + OBJ
# ============================================================================
if [[ "${SKIP_MERGE:-0}" != "1" ]]; then
    echo
    echo "============================================================"
    echo "[pipeline] Phase 2: merge BG + OBJ"
    echo "============================================================"

    ckpts=()
    missing=()
    for i in "${SEG_INDICES[@]}"; do
        seg_short=$(printf "seg_%02d" "$i")
        exp_name=$(grep -E '^exp_name:' "configs/experiments/segmented/${seg_short}.yaml" | awk '{print $2}')
        ckpt="${OUT_ROOT}/${exp_name}/trained_model/iteration_30000.pth"
        if [[ -f "${ckpt}" ]]; then
            ckpts+=( "${ckpt}" )
        else
            missing+=( "${seg_short}" )
        fi
    done

    if (( ${#missing[@]} > 0 )); then
        echo "[pipeline] WARNING: ${#missing[@]} segments missing ckpt — proceeding without: ${missing[*]}" >&2
    fi
    if (( ${#ckpts[@]} == 0 )); then
        echo "[pipeline] ERROR: no segment ckpts found under ${OUT_ROOT} — abort" >&2
        exit 1
    fi

    mkdir -p "${OUT_ROOT}/merged_bg" "${OUT_ROOT}/merged_obj"

    echo "[pipeline] merging BG (n=${#ckpts[@]})..."
    "${UV_RUN[@]}" python script/experiments/merge_bg_checkpoints.py \
        --ckpts "${ckpts[@]}" \
        --out "${OUT_ROOT}/merged_bg/iteration_0.pth" \
        --dedup-voxel-size 0.15

    echo "[pipeline] merging OBJ..."
    "${UV_RUN[@]}" python script/experiments/merge_obj_checkpoints.py \
        --seg-dir-glob "${OUT_ROOT}/t4_seg_*" \
        --out "${OUT_ROOT}/merged_obj/iteration_0.pth" \
        --log-dir "${LOG_DIR}"

    echo "[pipeline] Phase 2 done at $(date)"
else
    echo "[pipeline] SKIP_MERGE=1 — skipping merge"
fi

# ============================================================================
# Phase 3 — finetune on full sequence (8-GPU DDP via torchrun)
# ============================================================================
if [[ "${SKIP_FINETUNE:-0}" != "1" ]]; then
    echo
    echo "============================================================"
    echo "[pipeline] Phase 3: finetune (DDP, ${FINETUNE_NPROC} GPUs)"
    echo "============================================================"

    CONFIG="configs/experiments/segmented/finetune_merged.yaml"

    # Single-process preprocessing (idempotent; skipped if cache exists).
    "${UV_RUN[@]}" python script/t4/preprocess.py --config "${CONFIG}"

    "${UV_RUN[@]}" torchrun \
        --nproc_per_node="${FINETUNE_NPROC}" \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=localhost \
        --master_port=29500 \
        train.py --config "${CONFIG}" \
        train.bg_lidar_prune_enable True

    echo "[pipeline] Phase 3 done at $(date)"
else
    echo "[pipeline] SKIP_FINETUNE=1 — stopping after merge"
fi

echo
echo "[pipeline] ALL DONE at $(date)"
