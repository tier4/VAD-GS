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
# Full-state broadcast (after rank-0-only densify / prune / propagation)
# ---------------------------------------------------------------------------
#
# Densify / prune mutate parameter shapes and the optimizer state attached
# to each parameter (Adam's exp_avg / exp_avg_sq are param-shaped). When we
# do them on rank 0 only, every rank then has to receive all of:
#   - the new .data on each nn.Parameter
#   - the matching Adam state (exp_avg, exp_avg_sq, step)
#   - the resized densification accumulators (xyz_gradient_accum, denom,
#     max_radii2D)
#
# NCCL broadcast requires sender/receiver tensors of matching shape, so we
# first send a fixed-size metadata tensor describing (ndim, shape, dtype)
# and let receivers reallocate before the data broadcast.

_MAX_NDIM = 8
_DTYPE_TO_CODE = {
    torch.float32: 0, torch.float16: 1, torch.bfloat16: 2, torch.float64: 3,
    torch.int8: 4, torch.int16: 5, torch.int32: 6, torch.int64: 7,
    torch.uint8: 8, torch.bool: 9,
}
_CODE_TO_DTYPE = {v: k for k, v in _DTYPE_TO_CODE.items()}


def _broadcast_meta(t: torch.Tensor | None, src: int) -> tuple[tuple[int, ...], torch.dtype]:
    """Broadcast (ndim, shape..., dtype) from *src*. Returns the receiver's view."""
    rank = dist.get_rank()
    meta = torch.zeros(_MAX_NDIM + 2, dtype=torch.long, device="cuda")
    if rank == src:
        assert t is not None, "sender must provide tensor"
        assert t.ndim <= _MAX_NDIM, f"tensor ndim {t.ndim} exceeds _MAX_NDIM={_MAX_NDIM}"
        meta[0] = t.ndim
        for i, dim in enumerate(t.shape):
            meta[1 + i] = dim
        meta[-1] = _DTYPE_TO_CODE[t.dtype]
    dist.broadcast(meta, src=src)
    ndim = int(meta[0].item())
    shape = tuple(int(meta[1 + i].item()) for i in range(ndim))
    dtype = _CODE_TO_DTYPE[int(meta[-1].item())]
    return shape, dtype


def _broadcast_param_data(p: torch.nn.Parameter, src: int) -> None:
    """Broadcast p.data from src; replace it on receivers if shape/dtype changed."""
    rank = dist.get_rank()
    shape, dtype = _broadcast_meta(p.data if rank == src else None, src)
    if rank != src and (p.data.shape != shape or p.data.dtype != dtype):
        p.data = torch.empty(shape, dtype=dtype, device=p.data.device)
    if not p.data.is_contiguous():
        p.data = p.data.contiguous()
    dist.broadcast(p.data, src=src)


def _broadcast_attr_tensor(obj, attr: str, src: int) -> None:
    """Broadcast obj.<attr> from src; reassign on receivers if shape/dtype changed."""
    rank = dist.get_rank()
    cur = getattr(obj, attr)
    shape, dtype = _broadcast_meta(cur if rank == src else None, src)
    if rank != src and (cur.shape != shape or cur.dtype != dtype):
        cur = torch.empty(shape, dtype=dtype, device=cur.device)
        setattr(obj, attr, cur)
    if not cur.is_contiguous():
        cur = cur.contiguous()
        setattr(obj, attr, cur)
    dist.broadcast(cur, src=src)


def _broadcast_opt_state(opt: torch.optim.Optimizer, p: torch.nn.Parameter, src: int) -> None:
    """Broadcast Adam's exp_avg / exp_avg_sq for parameter *p*.

    'step' is intentionally not broadcast: every rank invokes
    optimizer.step() once per training iter so the step counter increments
    identically on all ranks, and densify/prune carries the same
    ``stored_state`` (step included) onto the rebuilt parameter without
    resetting it. Broadcasting step would also force us to round-trip
    Adam's CPU 0-d step tensor through CUDA — NCCL can't operate on CPU
    tensors.
    """
    rank = dist.get_rank()
    state = opt.state.setdefault(p, {})

    # Flags: bit 0 = exp_avg, bit 1 = exp_avg_sq
    if rank == src:
        flags = (
            (1 if "exp_avg" in state else 0)
            | (2 if "exp_avg_sq" in state else 0)
        )
    else:
        flags = 0
    flags_t = torch.tensor([flags], dtype=torch.long, device="cuda")
    dist.broadcast(flags_t, src=src)
    flags = int(flags_t.item())

    for key, bit in (("exp_avg", 1), ("exp_avg_sq", 2)):
        if not (flags & bit):
            continue
        if rank == src:
            shape, dtype = _broadcast_meta(state[key], src)
        else:
            shape, dtype = _broadcast_meta(None, src)
            cur = state.get(key)
            if not isinstance(cur, torch.Tensor) or cur.shape != shape or cur.dtype != dtype:
                state[key] = torch.empty(shape, dtype=dtype, device=p.data.device)
        if not state[key].is_contiguous():
            state[key] = state[key].contiguous()
        dist.broadcast(state[key], src=src)


def broadcast_model_state(gaussians, src: int = 0) -> None:
    """Broadcast every piece of state that densify / prune can mutate.

    Includes for every sub-model and auxiliary module:
      * Each nn.Parameter's .data (resizing receivers if shape changed)
      * Each parameter's Adam state (step, exp_avg, exp_avg_sq)
      * Densification accumulators (xyz_gradient_accum, denom, max_radii2D)
        on the Gaussian sub-models.
    """
    if not dist.is_initialized():
        return

    for model_name in gaussians.model_name_id.keys():
        sub_model = getattr(gaussians, model_name)
        for group in sub_model.optimizer.param_groups:
            for p in group["params"]:
                _broadcast_param_data(p, src)
                _broadcast_opt_state(sub_model.optimizer, p, src)
        for attr in ("xyz_gradient_accum", "denom", "max_radii2D"):
            if hasattr(sub_model, attr):
                _broadcast_attr_tensor(sub_model, attr, src)

    for module in _auxiliary_modules(gaussians):
        for group in module.optimizer.param_groups:
            for p in group["params"]:
                _broadcast_param_data(p, src)
                _broadcast_opt_state(module.optimizer, p, src)


# Back-compat alias: train.py used to call broadcast_model_params.
broadcast_model_params = broadcast_model_state


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
