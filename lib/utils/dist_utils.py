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

import hashlib
import os
import sys
import time
from typing import Generator

import torch
import torch.distributed as dist

from lib.config import cfg


# ---------------------------------------------------------------------------
# Debug instrumentation (enabled with VAD_GS_DEBUG_DIST=1)
# ---------------------------------------------------------------------------
# Set VAD_GS_DEBUG_DIST=1 to:
#   - Log each collective entry on every rank with a per-call counter
#   - Before each grad-touching collective, all_gather a fingerprint of
#     "which optimizer params will participate" and abort with a clear
#     message if ranks disagree (which is what silently desyncs NCCL).
_DEBUG_DIST = os.environ.get("VAD_GS_DEBUG_DIST", "0") == "1"
_call_counter: dict[str, int] = {}


def _dbg(label: str, msg: str) -> None:
    if not _DEBUG_DIST:
        return
    rank = dist.get_rank() if dist.is_initialized() else 0
    t = time.strftime("%H:%M:%S")
    print(f"[DBG_DIST {t} rank={rank}] {label}: {msg}", flush=True)
    sys.stdout.flush()


def _next_seq(label: str) -> int:
    n = _call_counter.get(label, 0) + 1
    _call_counter[label] = n
    return n


def _iter_all_optimizer_params_named(gaussians) -> Generator[tuple[str, torch.nn.Parameter], None, None]:
    """Same iteration order as _iter_all_optimizer_params, but with stable names."""
    for model_name in gaussians.model_name_id.keys():
        sub_model = getattr(gaussians, model_name)
        for gi, group in enumerate(sub_model.optimizer.param_groups):
            gname = group.get("name", f"g{gi}")
            for pi, p in enumerate(group["params"]):
                yield (f"{model_name}/{gname}/{pi}", p)
    for module in _auxiliary_modules(gaussians):
        mod_name = type(module).__name__
        for gi, group in enumerate(module.optimizer.param_groups):
            gname = group.get("name", f"g{gi}")
            for pi, p in enumerate(group["params"]):
                yield (f"{mod_name}/{gname}/{pi}", p)


def _check_grad_mask_consensus(label: str, gaussians) -> None:
    """Pre-collective consensus check: every rank reports which params have
    grad != None and which have grad == None. If ranks disagree, abort with
    a readable diff instead of letting NCCL hang minutes later.

    This itself enqueues NCCL byte collectives via all_gather_object, so it
    *also* desyncs if upstream code does, but at least it fails *here* with
    a clear log instead of at a distant 1-elem broadcast.
    """
    if not _DEBUG_DIST or not dist.is_initialized():
        return

    rank = dist.get_rank()
    world = dist.get_world_size()
    seq = _call_counter.get(label, 0)

    names_with_grad: list[str] = []
    names_without_grad: list[str] = []
    for name, p in _iter_all_optimizer_params_named(gaussians):
        (names_with_grad if p.grad is not None else names_without_grad).append(name)

    fp_input = "|".join(names_with_grad)
    fp = hashlib.sha1(fp_input.encode()).hexdigest()[:10]
    payload = (rank, len(names_with_grad), len(names_without_grad), fp)

    gathered: list = [None] * world
    dist.all_gather_object(gathered, payload)

    fps = {entry[3] for entry in gathered}  # type: ignore[index]
    if len(fps) == 1:
        if rank == 0:
            _dbg(label, f"seq={seq} consensus OK n_with_grad={payload[1]} fp={fp}")
        return

    # Mismatch path: gather full name lists so rank 0 can diff.
    full: list = [None] * world
    dist.all_gather_object(full, names_with_grad)

    if rank == 0:
        print(f"[DBG_DIST] !!! {label} seq={seq} GRAD-MASK MISMATCH ACROSS RANKS !!!", flush=True)
        for entry in gathered:
            r, nw, nn, f = entry  # type: ignore[misc]
            print(f"  rank {r}: {nw} with grad / {nn} without grad / fp={f}", flush=True)
        ref = set(full[0])  # type: ignore[arg-type]
        for r in range(1, world):
            cur = set(full[r])  # type: ignore[arg-type]
            only_ref = sorted(ref - cur)
            only_cur = sorted(cur - ref)
            print(
                f"  diff rank0 vs rank{r}: only_in_0={only_ref[:15]}"
                f"{'...' if len(only_ref) > 15 else ''} "
                f"only_in_{r}={only_cur[:15]}"
                f"{'...' if len(only_cur) > 15 else ''}",
                flush=True,
            )
        sys.stdout.flush()
    # Synchronize before aborting so all ranks emit logs.
    dist.barrier()
    raise RuntimeError(
        f"[rank={rank}] grad-mask mismatch detected at {label} seq={seq}; "
        f"see rank-0 log for diff. This is the root cause of the NCCL hang."
    )


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
    """
    label = "all_reduce_gradients"
    _next_seq(label)
    _check_grad_mask_consensus(label, gaussians)

    world_size = dist.get_world_size()
    n_reduced = 0
    n_skipped = 0
    for p in _iter_all_optimizer_params(gaussians):
        if p.grad is not None:
            # NCCL requires contiguous tensors; Gaussian params can carry
            # non-contiguous grad views after densify/prune slicing.
            if not p.grad.is_contiguous():
                p.grad = p.grad.contiguous()
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad.div_(world_size)
            n_reduced += 1
        else:
            n_skipped += 1
    _dbg(label, f"seq={_call_counter[label]} reduced={n_reduced} skipped={n_skipped}")


# ---------------------------------------------------------------------------
# Densification stat synchronisation
# ---------------------------------------------------------------------------

def sync_densification_stats(gaussians) -> None:
    """All-reduce densification accumulators so every rank makes identical
    densify/prune decisions.

    Must be called *before* ``densify_and_prune()``.
    """
    label = "sync_densification_stats"
    seq = _next_seq(label)
    if _DEBUG_DIST:
        names = list(gaussians.model_name_id.keys())
        _dbg(label, f"seq={seq} starting; submodels={len(names)}")
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
    _dbg(label, f"seq={seq} done")


# ---------------------------------------------------------------------------
# Model-state broadcast (after depth-propagation densification)
# ---------------------------------------------------------------------------

def broadcast_model_params(gaussians, src: int = 0) -> None:
    """Broadcast all trainable parameters from *src* to every other rank.

    Called after rank 0 runs depth-propagation-based densification which
    may change parameter tensor shapes.
    """
    label = "broadcast_model_params"
    seq = _next_seq(label)
    if _DEBUG_DIST and dist.is_initialized():
        rank = dist.get_rank()
        # Each rank also sees its own tensor shape - a shape mismatch with
        # src will hang NCCL silently; log shapes so we can spot it.
        first_few = []
        for i, (name, p) in enumerate(_iter_all_optimizer_params_named(gaussians)):
            if i >= 5:
                break
            first_few.append((name, tuple(p.data.shape)))
        _dbg(label, f"seq={seq} src={src} sample_shapes={first_few}")
    for p in _iter_all_optimizer_params(gaussians):
        if not p.data.is_contiguous():
            p.data = p.data.contiguous()
        dist.broadcast(p.data, src=src)
    _dbg(label, f"seq={seq} done")


# ---------------------------------------------------------------------------
# GradScaler synchronisation
# ---------------------------------------------------------------------------

def sync_grad_scaler(scaler) -> None:
    """Broadcast GradScaler's internal scale factor from rank 0.

    Ensures all ranks agree on the AMP loss scale for the next iteration.
    Accesses ``scaler._scale`` (private) because GradScaler exposes no
    public tensor accessor.
    """
    label = "sync_grad_scaler"
    seq = _next_seq(label)
    if _DEBUG_DIST:
        scale_val = float(scaler._scale.item()) if scaler._scale is not None else None
        _dbg(label, f"seq={seq} pre scale={scale_val}")
    dist.broadcast(scaler._scale, src=0)
    _dbg(label, f"seq={seq} done")
