"""Merge background Gaussians from N segment checkpoints into one.

Concatenates `xyz`, `feature_dc`, `feature_rest`, `scaling`, `rotation`,
`opacity` along dim 0 (the Gaussian count axis). The merged state can
then be loaded as the BG init for a low-LR fine-tune on the full
sequence (see configs/experiments/segmented/finetune_merged.yaml +
the `train.bg_init_from` flow in train.py).

The output checkpoint is in the same nested form as the source
checkpoints — `{'background': {...}}` plus a synthetic `iter: 0` — so
the receiving train.py can drop into it via gaussians.background.
load_state_dict(state_dict['background']).

Per-Gaussian training state (`max_radii2D`, `denom`,
`xyz_gradient_accum`) is concatenated alongside the params so densify
counters stay coherent if fine-tune still triggers densify
(typically it shouldn't — the recommended fine-tune config disables
densify_from_iter > iterations).

Adam optimiser moments are *not* preserved — they are reset to zero by
the receiving training_setup() before the fine-tune's first
optimizer.step(). Empirically this is fine: position/scaling/rotation
lr in the fine-tune config is already ~1/10 of the segment training,
so missing first-step momentum doesn't matter.

Usage:
    python script/experiments/merge_bg_checkpoints.py \\
        --ckpts path/to/seg0_best.pth path/to/seg1_best.pth path/to/seg2_best.pth \\
        --out  output/t4_exp/merged_bg/iteration_0.pth

Optional dedup (voxel-grid based, keeps the most-opaque Gaussian per
voxel):
    python script/experiments/merge_bg_checkpoints.py \\
        --ckpts ... --out ... --dedup-voxel-size 0.15
"""

from __future__ import annotations

import argparse
import os
from typing import Any

import torch


# Per-Gaussian tensors stored under state_dict['background'] that must be
# concatenated along dim 0 when merging. Order matters for keeping
# (xyz[i], feature_dc[i], ...) coherent within each Gaussian.
_PER_GAUSSIAN_KEYS = (
    "xyz",
    "feature_dc",
    "feature_rest",
    "scaling",
    "rotation",
    "opacity",
)

# Auxiliary per-Gaussian training-state tensors. May be missing on `is_final`
# checkpoints — concatenated only if present in every source ckpt.
_PER_GAUSSIAN_AUX_KEYS = (
    "max_radii2D",
    "denom",
    "xyz_gradient_accum",
)


def _bg_from_ckpt(ckpt_path: str) -> dict[str, Any]:
    state = torch.load(ckpt_path, map_location="cpu")
    if "background" not in state:
        raise KeyError(
            f"{ckpt_path} has no 'background' key — is this a StreetGaussianModel checkpoint?"
        )
    return state["background"]


def _detach_param(t: Any) -> torch.Tensor:
    """Tensor that nn.Parameter or plain tensor — return a detached plain tensor."""
    if isinstance(t, torch.nn.Parameter):
        return t.data.detach().clone()
    if isinstance(t, torch.Tensor):
        return t.detach().clone()
    raise TypeError(f"unexpected param type {type(t)}")


def _prune_by_opacity(merged: dict[str, Any], threshold: float) -> dict[str, Any]:
    """Drop Gaussians whose sigmoid(opacity) < threshold."""
    op_raw = merged["opacity"].squeeze(-1)
    keep = torch.sigmoid(op_raw) >= threshold
    return _apply_mask(merged, keep)


def _prune_by_scale(merged: dict[str, Any], max_scale: float, min_scale: float) -> dict[str, Any]:
    """Drop Gaussians whose max axis scale (= exp(_scaling).max(-1)) lies
    outside [min_scale, max_scale]. Catches the two outlier failure modes:
      * scale > max_scale → over-large Gaussians (would have been pruned
        by big-ws during normal training, but densify is OFF in fine-tune)
      * scale < min_scale → degenerate Gaussians (negligible visual
        contribution but still cost render compute)
    """
    sc = torch.exp(merged["scaling"]).max(dim=1).values
    keep = (sc <= max_scale) & (sc >= min_scale)
    return _apply_mask(merged, keep)


def _apply_mask(merged: dict[str, Any], keep: torch.Tensor) -> dict[str, Any]:
    n = merged["xyz"].shape[0]
    out: dict[str, Any] = {}
    for k, v in merged.items():
        if isinstance(v, torch.Tensor) and v.shape[0] == n:
            out[k] = v[keep]
        else:
            out[k] = v
    return out


def merge(
    ckpt_paths: list[str],
    dedup_voxel_size: float = 0.0,
    prune_opacity: float = 0.0,
    prune_max_scale: float = 0.0,
    prune_min_scale: float = 0.0,
) -> dict[str, Any]:
    """Concatenate BG Gaussians from each ckpt and (optionally) prune.

    Pruning order (each step independent, applied if its arg > 0):
      1. spatial voxel dedup (per-cell top-opacity)
      2. opacity prune (drop sigmoid(opacity) < prune_opacity)
      3. scale prune (drop max(scale) > prune_max_scale OR < prune_min_scale)
    """
    sources = [_bg_from_ckpt(p) for p in ckpt_paths]

    merged: dict[str, Any] = {}
    for k in _PER_GAUSSIAN_KEYS:
        for i, s in enumerate(sources):
            if k not in s:
                raise KeyError(f"ckpt {ckpt_paths[i]} missing 'background.{k}'")
        merged[k] = torch.cat([_detach_param(s[k]) for s in sources], dim=0)

    # Aux tensors: include only if all sources have them. Otherwise leave out
    # and let training_setup re-create from scratch (zeros for radii/denom).
    for k in _PER_GAUSSIAN_AUX_KEYS:
        if all(k in s for s in sources):
            merged[k] = torch.cat([_detach_param(s[k]) for s in sources], dim=0)

    # spatial_lr_scale should be the same across segments (data.extent in
    # seg_base.yaml is fixed at 10m). Use the first one.
    if "spatial_lr_scale" in sources[0]:
        merged["spatial_lr_scale"] = sources[0]["spatial_lr_scale"]

    # active_sh_degree: max across sources (so the SH degrees actually
    # learned in any segment carry through).
    sh_degs = [s.get("active_sh_degree", 0) for s in sources]
    merged["active_sh_degree"] = max(sh_degs)

    n_before = sum(_detach_param(s["xyz"]).shape[0] for s in sources)
    n_concat = merged["xyz"].shape[0]
    print(f"[merge] concatenated {len(sources)} segments: total {n_before} Gaussians")

    if dedup_voxel_size > 0:
        merged = _dedup_by_voxel(merged, voxel_size=dedup_voxel_size)
        n = merged["xyz"].shape[0]
        print(
            f"[merge] voxel dedup ({dedup_voxel_size:.3f}m): "
            f"{n_concat} → {n} ({100.0 * n / n_concat:.1f}% kept)"
        )

    if prune_opacity > 0:
        n_before_op = merged["xyz"].shape[0]
        merged = _prune_by_opacity(merged, threshold=prune_opacity)
        n = merged["xyz"].shape[0]
        print(
            f"[merge] opacity prune (< {prune_opacity}): "
            f"{n_before_op} → {n} ({100.0 * n / n_before_op:.1f}% kept)"
        )

    if prune_max_scale > 0 or prune_min_scale > 0:
        max_s = prune_max_scale if prune_max_scale > 0 else float("inf")
        min_s = prune_min_scale if prune_min_scale > 0 else 0.0
        n_before_sc = merged["xyz"].shape[0]
        merged = _prune_by_scale(merged, max_scale=max_s, min_scale=min_s)
        n = merged["xyz"].shape[0]
        print(
            f"[merge] scale prune ([{min_s:.4f}, {max_s:.4f}]): "
            f"{n_before_sc} → {n} ({100.0 * n / n_before_sc:.1f}% kept)"
        )

    return merged


def _dedup_by_voxel(merged: dict[str, torch.Tensor], voxel_size: float) -> dict[str, torch.Tensor]:
    """Keep at most one Gaussian per voxel cell. Picks the one with highest
    sigmoid-opacity (= visibility-weighted importance) within each cell.

    Implementation: quantise xyz → uint64 voxel id, sort by (voxel_id,
    -opacity), then keep the first occurrence per voxel_id.
    """
    xyz = merged["xyz"]
    opacity_raw = merged["opacity"].squeeze(-1)  # pre-sigmoid logits
    # Quantise positions to voxel ids
    vid = torch.floor(xyz / voxel_size).to(torch.long)
    # Pack 3D voxel id into one int (handles ~+/-2M cells per axis safely)
    OFFSET = 2_097_152  # 2^21
    flat_vid = (
        (vid[:, 0] + OFFSET) * (OFFSET * 4)
        + (vid[:, 1] + OFFSET) * 2
        + (vid[:, 2] + OFFSET)
    )
    # Sort by (voxel_id asc, opacity desc) so the first occurrence per
    # voxel_id is the most-opaque Gaussian in that cell.
    n = xyz.shape[0]
    order = torch.argsort(flat_vid * (10**9) - opacity_raw.float() * 1000)
    sorted_vid = flat_vid[order]
    # Keep first occurrence of each voxel_id
    keep_mask = torch.ones(n, dtype=torch.bool)
    keep_mask[1:] = sorted_vid[1:] != sorted_vid[:-1]
    keep_idx = order[keep_mask]

    out: dict[str, Any] = {}
    for k, v in merged.items():
        if not isinstance(v, torch.Tensor):
            out[k] = v
            continue
        if v.shape[0] == n:
            out[k] = v[keep_idx]
        else:
            out[k] = v
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--ckpts",
        nargs="+",
        required=True,
        help="N segment checkpoint paths (StreetGaussianModel .pth files).",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output checkpoint path. The receiving fine-tune config "
             "(train.bg_init_from) should point at this file.",
    )
    parser.add_argument(
        "--dedup-voxel-size",
        type=float,
        default=0.025,
        help="If > 0, dedup merged Gaussians by quantising positions to a "
             "uniform grid and keeping the most-opaque per cell. Default "
             "0.025m matches typical converged Gaussian min(scale) and "
             "preserves dense detail. The pre-training voxel_size (0.15m) "
             "is too aggressive after densify — it collapses ~80%% of "
             "Gaussians and destroys fine detail in 60k+ dense cells. "
             "Set 0 to disable spatial dedup entirely.",
    )
    parser.add_argument(
        "--prune-opacity",
        type=float,
        default=0.01,
        help="Drop Gaussians with sigmoid(opacity) below this threshold "
             "(default 0.01 = same as seg_base.optim.min_opacity). Catches "
             "low-confidence Gaussians that fine-tune cannot prune (densify "
             "OFF in finetune yaml). Set 0 to disable.",
    )
    parser.add_argument(
        "--prune-max-scale",
        type=float,
        default=2.0,
        help="Drop Gaussians whose max(scale) exceeds this (m). Default 2.0 "
             "matches seg_base.optim.percent_big_ws=0.1 × extent=10. Set 0 "
             "to disable.",
    )
    parser.add_argument(
        "--prune-min-scale",
        type=float,
        default=0.001,
        help="Drop Gaussians whose max(scale) is below this (m). Default "
             "0.001 catches degenerate Gaussians. Set 0 to disable.",
    )
    args = parser.parse_args()

    if len(args.ckpts) < 2:
        print("[merge] WARNING: fewer than 2 ckpts — this is just a pass-through.")

    merged_bg = merge(
        args.ckpts,
        dedup_voxel_size=args.dedup_voxel_size,
        prune_opacity=args.prune_opacity,
        prune_max_scale=args.prune_max_scale,
        prune_min_scale=args.prune_min_scale,
    )
    out_state = {"background": merged_bg, "iter": 0}

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(out_state, args.out)
    print(f"[merge] saved merged checkpoint: {args.out}")
    print(f"[merge] final BG Gaussian count: {merged_bg['xyz'].shape[0]}")


if __name__ == "__main__":
    main()
