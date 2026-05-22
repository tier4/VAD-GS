# Segmented BG Merge Pipeline

Train one Gaussian set per time-window segment (HP-tuned via sweep), merge
the BG Gaussians across segments, and fine-tune on the full sequence at
low LR so multiple Gaussians integrate into one consistent scene.

Designed for T4 driving sequences (561 frames ≈ 56 sec at 10Hz) where a
single full-sequence run hits HP ceilings (see analysis in
`wandb/sweep-nu6sy3bm`). Each segment matches the nuscenes reference
image count (5 cam × 61 frame = **305 imgs**), which the existing
nuscenes HP recipe is known-good for.

## Files

| Path | Role |
|---|---|
| `seg_base.yaml` | Common HP shared by all 13 segments (nuscenes_train_000.yaml-derived, T4-specific). |
| `seg_00.yaml` … `seg_12.yaml` | Segment-specific overrides — only `data.selected_frames` and `exp_name`. |
| `../../sweep/t4_segmented.yaml` | wandb sweep config. Treats `data.selected_frames` as a categorical axis (13 segments) plus HP axes. |
| `finetune_merged.yaml` | Final low-LR fine-tune. Loads merged BG checkpoint via `train.bg_init_from`. |
| `../../../script/experiments/merge_bg_checkpoints.py` | Concatenates BG Gaussians from N segment checkpoints. Optional voxel-dedup. |
| `../../../script/experiments/select_best_per_segment.py` | wandb API → picks best run per segment via summary metric. |

## Segmentation

13 overlapping 61-frame windows covering all 561 frames at stride 40
(~33% overlap). The last window is shifted to end at frame 560 so the
tail isn't dropped.

```
Seg 00 : [  0,  60]   Seg 07 : [280, 340]
Seg 01 : [ 40, 100]   Seg 08 : [320, 380]
Seg 02 : [ 80, 140]   Seg 09 : [360, 420]
Seg 03 : [120, 180]   Seg 10 : [400, 460]
Seg 04 : [160, 220]   Seg 11 : [440, 500]
Seg 05 : [200, 260]   Seg 12 : [500, 560]   ← shifted, covers tail
Seg 06 : [240, 300]
```

Overlap helps BG points near segment boundaries get supervision from
both neighbours → smoother stitching after merge. Seg 11 ↔ Seg 12
overlap is only 1 frame (the shift trade-off), but ego motion already
gives spatial separation in that gap.

## Workflow

### 0. Stop any other GPU work

The sweep needs all 8 GPUs. If `nu6sy3bm` is still running:
```bash
pkill -9 -f "sweep_run.py"; pkill -9 -f "wandb agent.*nu6sy3bm"
```

### 1. Train all 13 segments (direct, no sweep)

`train_segments.sh` runs `python train.py` for each segment, throttled
to NUM_GPUS in parallel. All segments use the same nuscenes-derived HP
from `seg_base.yaml` — no HP tuning per segment.

```bash
bash script/experiments/train_segments.sh
```

With 8 GPUs and 13 segments, this is 2 waves of 8 + 5 jobs at ~50 min
each → **~100 min wall-clock**. Each run logs to wandb under a shared
`WANDB_RUN_GROUP` (default `t4_segmented_<timestamp>`) so the 13 runs
are easy to compare in the UI.

```bash
# Watch progress:
tail -f output/segmented_logs/*.log

# Rerun a failed segment:
ONLY_SEGMENTS="5 11" bash script/experiments/train_segments.sh

# Use a subset of GPUs:
GPU_OFFSET=4 NUM_GPUS=4 bash script/experiments/train_segments.sh
```

> **Alternative: HP sweep instead of direct train.** If you'd rather
> sweep HP per segment, use `configs/sweep/t4_segmented.yaml` with
> `python script/sweep/sweep_init.py`+`bash script/sweep/sweep_launch.sh`
> (with `VAD_GS_SWEEP_CACHE_FROM=""`). Note that wandb bayes does NOT
> guarantee uniform per-segment coverage — switch to `method: grid` if
> that matters. For the default "just train each segment with the
> nuscenes recipe" goal, the direct launcher above is simpler.

### 2a. Backfill the track_id_map.json for segments trained before the patch

`street_gaussian_model.setup_functions` now saves `track_id_map.json`
alongside each `model_path` automatically — but segments trained
before that patch landed don't have it. Run the backfill once:

```bash
python script/experiments/build_track_id_maps.py
```

This re-runs the T4 actor parsing (cheap, ~1 min total for 13 segments)
and writes `output/t4_exp/t4_seg_*/track_id_map.json`. Idempotent.

### 2b. Merge BG checkpoints

Concatenates BG Gaussians from all completed segments, optional voxel
dedup to collapse near-duplicates from the overlapping windows:

```bash
python script/experiments/merge_bg_checkpoints.py \
    --ckpts output/t4_exp/t4_seg_00_f0_60/trained_model/iteration_30000.pth \
            output/t4_exp/t4_seg_01_f40_100/trained_model/iteration_30000.pth \
            output/t4_exp/t4_seg_02_f80_140/trained_model/iteration_30000.pth \
            output/t4_exp/t4_seg_03_f120_180/trained_model/iteration_30000.pth \
            output/t4_exp/t4_seg_04_f160_220/trained_model/iteration_30000.pth \
            output/t4_exp/t4_seg_05_f200_260/trained_model/iteration_30000.pth \
            output/t4_exp/t4_seg_06_f240_300/trained_model/iteration_30000.pth \
            output/t4_exp/t4_seg_07_f280_340/trained_model/iteration_30000.pth \
            output/t4_exp/t4_seg_08_f320_380/trained_model/iteration_30000.pth \
            output/t4_exp/t4_seg_09_f360_420/trained_model/iteration_30000.pth \
            output/t4_exp/t4_seg_10_f400_460/trained_model/iteration_30000.pth \
            output/t4_exp/t4_seg_11_f440_500/trained_model/iteration_30000.pth \
            output/t4_exp/t4_seg_12_f500_560/trained_model/iteration_30000.pth \
    --out output/t4_exp/merged_bg/iteration_0.pth
# defaults: --dedup-voxel-size 0.025  --prune-opacity 0.01  --prune-max-scale 2.0  --prune-min-scale 0.001
# (the old 0.15m voxel was too aggressive after densify — collapsed ~80%
# of Gaussians and destroyed dense detail. See merge_bg_checkpoints.py
# docstring for the trade-off table.)
```

### 2c. Merge OBJ checkpoints (best segment per actor)

For each T4 track_id present in any segment, picks the segment whose
test/test_view psnr_obj at the final iter is highest, and saves that
segment's actor state keyed by T4 original ID. Sky is intentionally
*not* carried forward (it trains from scratch in the fine-tune phase).

```bash
python script/experiments/merge_obj_checkpoints.py \
    --seg-dir-glob 'output/t4_exp/t4_seg_*' \
    --iteration 30000 \
    --metric psnr_obj \
    --out output/t4_exp/merged_obj/iteration_0.pth
```

`--dedup-voxel-size 0.15` collapses near-duplicate Gaussians from the
overlapping segments down to one per voxel-cell (keeps the most-opaque
representative). Drop the flag if you want full concat (more Gaussians,
more VRAM, possibly more detail). Without dedup, expect ~13× the
single-segment Gaussian count.

### 3. Fine-tune on the full sequence

```bash
bash script/train_multigpu.sh \
    configs/experiments/segmented/finetune_merged.yaml 8
```

Runs 16k iter (~30 min on 8×H100) on the full 561-frame sequence with:
- **BG**: replaced by merged checkpoint via `train.bg_init_from`, polished at 1/10 LR (no densify, no opacity reset)
- **Obj**: per-actor best-segment state loaded via `train.obj_init_from`, then trained further at full LR (per-tag overrides `position_lr_init_obj` etc.)
- **Sky**: rebuilt fresh from the full-sequence PLY, trained at the base 1/10 LR (no per-tag override exists for sky today)
- All Adam moments reset by merge — first iters warm up

The final checkpoint at `output/t4_exp/t4_finetune_merged/trained_model/iteration_8000.pth`
is the deliverable. Eval with `render.py --config configs/experiments/segmented/finetune_merged.yaml`.

## Knobs to consider per phase

| Phase | Knob | Default | When to change |
|---|---|---|---|
| Segment training | `iterations` | 30000 | Lower to 16000 if you want a faster cycle |
| Segment training | `densify_until_iter` | 24000 | Lower if obj count explodes |
| Segment training | parallelism | NUM_GPUS=8 | Lower (NUM_GPUS=4) to share box with other work |
| Merge | `--dedup-voxel-size` | 0.15 | Raise to 0.30 for smaller models; drop for max quality |
| Fine-tune | `train.iterations` | 8000 | Raise to 16000 if obj convergence is incomplete |
| Fine-tune | LR scale | 1/10 of seg_base | Lower (1/100) if BG looks brittle, higher (1/3) if still under-fit |

## Notes

- **Sky model** is rebuilt fresh in the fine-tune phase (its PLY differs per segment but the sphere projection is sequence-wide). The merge step only carries BG forward.
- **Actor models** rebuild from full-sequence PLYs, picking up tracks present anywhere in [0, 560]. Per-segment obj training is essentially throwaway — only the BG knowledge is reused.
- **`spatial_lr_scale`** is preserved from segment 0's checkpoint (all segments use the same `data.extent: 10` so they match).
- **Adam optimiser moments** are dropped by merge — the fine-tune's `training_setup` re-initialises them to zeros. With LR/10 this is harmless.
- The pipeline assumes `train.py` carries the `bg_init_from` and `_init_wandb_for_direct_run` patches added on the `feature/segmented-bg-merge` branch.
