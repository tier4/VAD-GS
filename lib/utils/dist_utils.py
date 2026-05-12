"""Distributed training utilities for VAD-GS.

Controlled by ``cfg.dist.enabled`` (default: False).  When disabled, all
helper functions degrade to no-ops so that existing single-GPU code paths
remain unchanged.

Usage
-----
Multi-GPU::

    # In YAML config:
    #   dist:
    #     enabled: true
    #
    # Launch:
    #   torchrun --nproc_per_node=8 train.py --config <config>.yaml

Single-GPU (existing behaviour, no config change needed)::

    python train.py --config <config>.yaml
"""

from __future__ import annotations

import os
from typing import Generator

import torch
import torch.distributed as dist

from lib.config import cfg


# ---------------------------------------------------------------------------
# Setup / teardown
# ---------------------------------------------------------------------------

def setup_distributed() -> tuple[int, int, int]:
    """Initialise the distributed process group when ``cfg.dist.enabled`` is True.

    Returns ``(rank, world_size, local_rank)``.  When distributed training is
    disabled, returns ``(0, 1, 0)`` without touching the process group.
    """
    if not cfg.dist.enabled:
        return 0, 1, 0

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=cfg.dist.backend)

    return rank, world_size, local_rank


def cleanup_distributed() -> None:
    """Destroy the process group if it was initialised."""
    if dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def is_distributed() -> bool:
    """Return True when running in a multi-GPU distributed setting."""
    return dist.is_initialized()


def is_main_process() -> bool:
    """Return True on rank 0 (or always True in single-GPU mode)."""
    return not is_distributed() or dist.get_rank() == 0


# ---------------------------------------------------------------------------
# Internal: iterate all optimizer params
# ---------------------------------------------------------------------------

def _auxiliary_modules(gaussians) -> Generator:
    """Yield non-None auxiliary modules from *gaussians*."""
    if gaussians.actor_pose is not None:
        yield gaussians.actor_pose
    if gaussians.include_sky and gaussians.sky_cubemap is not None:
        yield gaussians.sky_cubemap
    if gaussians.color_correction is not None:
        yield gaussians.color_correction
    if gaussians.pose_correction is not None:
        yield gaussians.pose_correction


def _iter_all_optimizer_params(gaussians) -> Generator[torch.nn.Parameter, None, None]:
    """Yield every trainable parameter across all sub-models and auxiliary modules."""
    for model_name in gaussians.model_name_id.keys():
        sub_model = getattr(gaussians, model_name)
        for group in sub_model.optimizer.param_groups:
            yield from group["params"]
    for module in _auxiliary_modules(gaussians):
        for group in module.optimizer.param_groups:
            yield from group["params"]


# ---------------------------------------------------------------------------
# Gradient synchronisation
# ---------------------------------------------------------------------------

def all_reduce_gradients(gaussians) -> None:
    """All-reduce gradients across ranks and average them.

    Iterates over every sub-model (background, obj_*, sky) held by
    *gaussians* (a :class:`StreetGaussianModel`), plus the auxiliary
    modules (actor_pose, sky_cubemap, color_correction, pose_correction).

    Different ranks see different viewpoints, so different obj_* models
    end up with grad=None on different ranks. NCCL requires every rank
    to call all_reduce in the same order with the same shapes — skipping
    None grads would desync the collective and silently hang.

    Fix: materialize a zero grad for any param missing one, so every rank
    issues the same sequence of all_reduces. After the SUM all_reduce we
    divide by world_size, giving the same average gradient on every rank
    (with the ranks that didn't see this param contributing 0 to the sum).
    """
    # First pass: align grad-mask across ranks by filling zeros so every
    # rank issues the same sequence of collectives in the same order.
    for p in _iter_all_optimizer_params(gaussians):
        if p.grad is None:
            p.grad = torch.zeros_like(p.data)

    world_size = dist.get_world_size()
    for p in _iter_all_optimizer_params(gaussians):
        # NCCL requires contiguous tensors; Gaussian params can carry
        # non-contiguous grad views after densify/prune slicing.
        if not p.grad.is_contiguous():
            p.grad = p.grad.contiguous()
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        p.grad.div_(world_size)


# ---------------------------------------------------------------------------
# Densification stat synchronisation
# ---------------------------------------------------------------------------

def sync_densification_stats(gaussians) -> None:
    """All-reduce densification accumulators so every rank makes identical
    densify/prune decisions.

    Must be called *before* ``densify_and_prune()``.
    """
    for model_name in gaussians.model_name_id.keys():
        sub_model = getattr(gaussians, model_name)
        for attr in ("xyz_gradient_accum", "denom", "max_radii2D"):
            t = getattr(sub_model, attr)
            if not t.is_contiguous():
                t = t.contiguous()
                setattr(sub_model, attr, t)
        dist.all_reduce(sub_model.xyz_gradient_accum, op=dist.ReduceOp.SUM)
        dist.all_reduce(sub_model.denom, op=dist.ReduceOp.SUM)
        dist.all_reduce(sub_model.max_radii2D, op=dist.ReduceOp.MAX)


# ---------------------------------------------------------------------------
# Model-state broadcast (after depth-propagation densification)
# ---------------------------------------------------------------------------

def broadcast_model_params(gaussians, src: int = 0) -> None:
    """Broadcast all trainable parameters from *src* to every other rank.

    Called after rank 0 runs depth-propagation-based densification which
    may change parameter tensor shapes.
    """
    for p in _iter_all_optimizer_params(gaussians):
        if not p.data.is_contiguous():
            p.data = p.data.contiguous()
        dist.broadcast(p.data, src=src)


# ---------------------------------------------------------------------------
# GradScaler synchronisation
# ---------------------------------------------------------------------------

def sync_grad_scaler(scaler) -> None:
    """Broadcast GradScaler's internal scale factor from rank 0.

    Ensures all ranks agree on the AMP loss scale for the next iteration.
    Accesses ``scaler._scale`` (private) because GradScaler exposes no
    public tensor accessor.
    """
    dist.broadcast(scaler._scale, src=0)
