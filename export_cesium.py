#!/usr/bin/env python3
"""Export VAD-GS trained checkpoint (.pth) to Cesium-compatible 3D Tiles.

Uses the 3dgs_io library to produce GLB files with the KHR_gaussian_splatting
glTF extension, wrapped in a 3D Tiles tileset.json.

Requirements:
    pip install 3dgs-io torch numpy
    (Recommended: use .venv_export with Python >=3.13)

Usage:
    python export_cesium.py <checkpoint.pth> [options]

Example:
    python export_cesium.py output/nuscenes_exp/nuscenes_val_000_3cam/trained_model/iteration_30000.pth \
        --output tiles_output \
        --lat 35.6812 --lon 139.7671 --height 0 \
        --background-only
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np
import torch

# 3dgs_io: KHR_gaussian_splatting compliant glTF I/O
from importlib import import_module

_3dgs_io = import_module("3dgs_io")
save_gltf = _3dgs_io.save_gltf
GltfSaveOptions = _3dgs_io.GltfSaveOptions

from spz import GaussianCloud


# ---------------------------------------------------------------------------
# Coordinate conversion helpers (geodetic → ECEF)
# ---------------------------------------------------------------------------

# WGS84 ellipsoid constants
_WGS84_A = 6378137.0  # semi-major axis [m]
_WGS84_B = 6356752.314245  # semi-minor axis [m]
_WGS84_E2 = 1 - (_WGS84_B / _WGS84_A) ** 2  # first eccentricity squared


def _geodetic_to_ecef(lat_deg: float, lon_deg: float, height: float) -> np.ndarray:
    """Convert geodetic (WGS84) coordinates to ECEF."""
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    sin_lon, cos_lon = math.sin(lon), math.cos(lon)
    N = _WGS84_A / math.sqrt(1 - _WGS84_E2 * sin_lat**2)
    x = (N + height) * cos_lat * cos_lon
    y = (N + height) * cos_lat * sin_lon
    z = (N * (1 - _WGS84_E2) + height) * sin_lat
    return np.array([x, y, z], dtype=np.float64)


def _enu_to_ecef_matrix(lat_deg: float, lon_deg: float) -> np.ndarray:
    """Build a 3x3 rotation matrix from ENU to ECEF at the given location."""
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    sin_lon, cos_lon = math.sin(lon), math.cos(lon)
    # Columns: East, North, Up in ECEF
    return np.array(
        [
            [-sin_lon, -sin_lat * cos_lon, cos_lat * cos_lon],
            [cos_lon, -sin_lat * sin_lon, cos_lat * sin_lon],
            [0.0, cos_lat, sin_lat],
        ],
        dtype=np.float64,
    )


def build_tileset_transform(
    lat_deg: float, lon_deg: float, height: float
) -> list[float]:
    """Build a 4x4 column-major transform for 3D Tiles root tile.

    Places the model origin at the given geodetic location,
    with axes mapped from the model's local ENU frame to ECEF.
    """
    origin = _geodetic_to_ecef(lat_deg, lon_deg, height)
    rot = _enu_to_ecef_matrix(lat_deg, lon_deg)
    # 4x4 column-major (as required by 3D Tiles spec)
    m = np.eye(4, dtype=np.float64)
    m[:3, :3] = rot
    m[:3, 3] = origin
    # 3D Tiles stores column-major: flatten column by column
    return m.T.flatten().tolist()


# ---------------------------------------------------------------------------
# Checkpoint → GaussianCloud conversion
# ---------------------------------------------------------------------------


def load_checkpoint(path: Path) -> dict:
    """Load a VAD-GS checkpoint file."""
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    return ckpt


def state_dict_to_gaussian_cloud(
    state_dict: dict, max_sh_degree: int | None = None
) -> GaussianCloud:
    """Convert a VAD-GS model state_dict to a spz.GaussianCloud.

    Parameters
    ----------
    state_dict:
        One model entry from the checkpoint (e.g. ckpt['background']).
    max_sh_degree:
        Limit the exported SH degree. None = export all available.
    """
    def _to_np(t: torch.Tensor) -> np.ndarray:
        return t.detach().cpu().numpy().astype(np.float32)

    xyz = _to_np(state_dict["xyz"])  # (N, 3)
    feature_dc = _to_np(state_dict["feature_dc"])  # (N, K, 3)
    feature_rest = _to_np(state_dict["feature_rest"])  # (N, M, 3)
    scaling = _to_np(state_dict["scaling"])  # (N, 3) log-space
    rotation = _to_np(state_dict["rotation"])  # (N, 4) wxyz
    opacity = _to_np(state_dict["opacity"])  # (N, 1) logit

    n = xyz.shape[0]

    # -- SH DC coefficients --
    # For background: feature_dc is (N, 1, 3)
    # For objects with Fourier features: feature_dc is (N, K, 3) where K > 1
    #   → only the first coefficient is the actual SH DC
    sh_dc = feature_dc[:, 0, :]  # (N, 3)

    # -- Higher-order SH coefficients --
    # feature_rest: (N, num_coef, 3)
    num_sh_rest = feature_rest.shape[1]

    # Determine SH degree from coefficient count
    # degree L has (L+1)^2 total coeffs; rest = (L+1)^2 - 1
    # degree 1: rest=3, degree 2: rest=8, degree 3: rest=15
    if max_sh_degree is not None:
        max_rest = (max_sh_degree + 1) ** 2 - 1
        num_sh_rest = min(num_sh_rest, max_rest)

    # Compute SH degree from rest coefficient count
    # degree L: (L+1)^2 - 1 rest coefficients → degree 1: 3, degree 2: 8, degree 3: 15
    sh_degree = 0
    if num_sh_rest >= 15:
        sh_degree = 3
    elif num_sh_rest >= 8:
        sh_degree = 2
    elif num_sh_rest >= 3:
        sh_degree = 1

    if max_sh_degree is not None:
        sh_degree = min(sh_degree, max_sh_degree)

    actual_rest = (sh_degree + 1) ** 2 - 1 if sh_degree > 0 else 0

    if actual_rest > 0:
        sh_rest = feature_rest[:, :actual_rest, :]  # (N, num_coef, 3)
        sh_flat = sh_rest.reshape(-1).astype(np.float32)
    else:
        sh_flat = np.zeros(0, dtype=np.float32)
        sh_degree = 0

    # -- Quaternion conversion: wxyz (VAD-GS) → xyzw (glTF/spz) --
    rot_xyzw = np.empty_like(rotation)
    rot_xyzw[:, 0] = rotation[:, 1]  # x
    rot_xyzw[:, 1] = rotation[:, 2]  # y
    rot_xyzw[:, 2] = rotation[:, 3]  # z
    rot_xyzw[:, 3] = rotation[:, 0]  # w

    # -- Build GaussianCloud --
    gc = GaussianCloud()
    gc.sh_degree = sh_degree
    gc.positions = xyz.reshape(-1).astype(np.float32)
    gc.colors = sh_dc.reshape(-1).astype(np.float32)
    gc.alphas = opacity.reshape(-1).astype(np.float32)
    gc.rotations = rot_xyzw.reshape(-1).astype(np.float32)
    gc.scales = scaling.reshape(-1).astype(np.float32)
    gc.sh = sh_flat

    print(f"    Gaussians: {n:,}")
    print(f"    SH rest coefficients: {num_sh_rest}")
    print(f"    Position range: [{xyz.min(axis=0)} .. {xyz.max(axis=0)}]")

    return gc


def merge_gaussian_clouds(clouds: list[GaussianCloud]) -> GaussianCloud:
    """Merge multiple GaussianClouds into one."""
    if len(clouds) == 1:
        return clouds[0]

    positions = []
    colors = []
    alphas = []
    rotations = []
    scales = []
    sh_parts = []

    # Find the minimum SH degree across all clouds
    min_sh_degree = None
    for gc in clouds:
        d = gc.sh_degree
        if min_sh_degree is None:
            min_sh_degree = d
        else:
            min_sh_degree = min(min_sh_degree, d)
    if min_sh_degree is None:
        min_sh_degree = 0

    sh_per_point = (min_sh_degree + 1) ** 2 - 1 if min_sh_degree > 0 else 0

    for gc in clouds:
        n = gc.num_points
        if n == 0:
            continue
        positions.append(np.array(gc.positions, dtype=np.float32))
        colors.append(np.array(gc.colors, dtype=np.float32))
        alphas.append(np.array(gc.alphas, dtype=np.float32))
        rotations.append(np.array(gc.rotations, dtype=np.float32))
        scales.append(np.array(gc.scales, dtype=np.float32))

        sh_arr = np.array(gc.sh, dtype=np.float32)
        if sh_per_point > 0 and sh_arr.size > 0:
            sh_reshaped = sh_arr.reshape(n, -1, 3)[:, :sh_per_point, :]
            sh_parts.append(sh_reshaped.reshape(-1))
        elif sh_per_point > 0:
            sh_parts.append(np.zeros(n * sh_per_point * 3, dtype=np.float32))

    merged = GaussianCloud()
    merged.sh_degree = min_sh_degree
    merged.positions = np.concatenate(positions)
    merged.colors = np.concatenate(colors)
    merged.alphas = np.concatenate(alphas)
    merged.rotations = np.concatenate(rotations)
    merged.scales = np.concatenate(scales)
    if sh_parts:
        merged.sh = np.concatenate(sh_parts)
    else:
        merged.sh = np.zeros(0, dtype=np.float32)

    return merged


# ---------------------------------------------------------------------------
# Bounding box computation for tileset.json
# ---------------------------------------------------------------------------


def compute_bounding_box(gc: GaussianCloud) -> dict:
    """Compute an oriented bounding box for 3D Tiles (12-element array).

    Returns the 'box' array: [cx, cy, cz, xx, xy, xz, yx, yy, yz, zx, zy, zz]
    where (cx,cy,cz) is center and the 3 column vectors are half-axes.
    """
    n = gc.num_points
    pos = np.array(gc.positions, dtype=np.float64).reshape(n, 3)
    pmin = pos.min(axis=0)
    pmax = pos.max(axis=0)
    center = ((pmin + pmax) / 2).tolist()
    half = ((pmax - pmin) / 2).tolist()
    return {
        "box": [
            center[0], center[1], center[2],
            half[0], 0, 0,
            0, half[1], 0,
            0, 0, half[2],
        ]
    }


# ---------------------------------------------------------------------------
# Tileset.json generation
# ---------------------------------------------------------------------------


def create_tileset_json(
    glb_filename: str,
    gc: GaussianCloud,
    lat: float,
    lon: float,
    height: float,
    geometric_error: float = 500.0,
) -> dict:
    """Create a 3D Tiles 1.1 tileset.json for a single GLB tile."""
    bv = compute_bounding_box(gc)
    transform = build_tileset_transform(lat, lon, height)

    return {
        "asset": {"version": "1.1", "generator": "VAD-GS export_cesium.py"},
        "geometricError": geometric_error,
        "root": {
            "boundingVolume": bv,
            "geometricError": 0,
            "refine": "ADD",
            "transform": transform,
            "content": {"uri": glb_filename},
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Export VAD-GS checkpoint to Cesium 3D Tiles"
    )
    parser.add_argument(
        "checkpoint", type=Path, help="Path to .pth checkpoint file"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output directory (default: <checkpoint_dir>/../cesium_tiles/iteration_N)",
    )
    parser.add_argument(
        "--background-only",
        action="store_true",
        help="Export only the background model (no dynamic objects)",
    )
    parser.add_argument(
        "--objects",
        nargs="*",
        default=None,
        help="Specific object keys to include (e.g. obj_001 obj_003). "
        "Default: all objects",
    )
    parser.add_argument(
        "--max-sh-degree",
        type=int,
        default=None,
        help="Limit SH degree for export (reduces file size)",
    )
    parser.add_argument(
        "--spz-compression",
        action="store_true",
        help="Use SPZ compression (smaller files, requires compatible viewer)",
    )
    parser.add_argument(
        "--lat",
        type=float,
        default=35.6812,
        help="Latitude for tileset placement (default: Tokyo 35.6812)",
    )
    parser.add_argument(
        "--lon",
        type=float,
        default=139.7671,
        help="Longitude for tileset placement (default: Tokyo 139.7671)",
    )
    parser.add_argument(
        "--height",
        type=float,
        default=0.0,
        help="Height above WGS84 ellipsoid in meters (default: 0)",
    )
    parser.add_argument(
        "--geometric-error",
        type=float,
        default=500.0,
        help="Geometric error for tileset.json (default: 500)",
    )
    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"Error: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        sys.exit(1)

    # Determine output directory
    if args.output is None:
        ckpt_dir = args.checkpoint.parent
        iteration = args.checkpoint.stem  # e.g. "iteration_30000"
        output_dir = ckpt_dir.parent / "cesium_tiles" / iteration
    else:
        output_dir = args.output

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = load_checkpoint(args.checkpoint)
    iteration = ckpt.get("iter", "?")
    print(f"  Training iteration: {iteration}")

    # Convert models
    clouds: list[GaussianCloud] = []

    # Background
    if "background" in ckpt:
        print("Converting background...")
        clouds.append(
            state_dict_to_gaussian_cloud(
                ckpt["background"], max_sh_degree=args.max_sh_degree
            )
        )

    # Objects
    if not args.background_only:
        obj_keys = [k for k in ckpt.keys() if k.startswith("obj_")]
        if args.objects is not None:
            obj_keys = [k for k in obj_keys if k in args.objects]

        for obj_key in sorted(obj_keys):
            print(f"Converting {obj_key}...")
            clouds.append(
                state_dict_to_gaussian_cloud(
                    ckpt[obj_key], max_sh_degree=args.max_sh_degree
                )
            )

    if not clouds:
        print("Error: no models found in checkpoint", file=sys.stderr)
        sys.exit(1)

    # Merge all clouds
    print("Merging gaussian clouds...")
    merged = merge_gaussian_clouds(clouds)
    total_points = merged.num_points
    print(f"  Total gaussians: {total_points:,}")

    # Save GLB
    glb_name = "model.glb"
    glb_path = output_dir / glb_name
    options = GltfSaveOptions(spz_compression=args.spz_compression)
    print(f"Saving GLB: {glb_path}")
    save_gltf(merged, glb_path, options)
    glb_size = glb_path.stat().st_size
    print(f"  GLB size: {glb_size / 1024 / 1024:.1f} MB")

    # Save tileset.json
    tileset = create_tileset_json(
        glb_name,
        merged,
        lat=args.lat,
        lon=args.lon,
        height=args.height,
        geometric_error=args.geometric_error,
    )
    tileset_path = output_dir / "tileset.json"
    with open(tileset_path, "w", encoding="utf-8") as f:
        json.dump(tileset, f, indent=2)
    print(f"Saved tileset: {tileset_path}")

    print(f"\nExport complete!")
    print(f"  Output directory: {output_dir}")
    print(f"  Placement: lat={args.lat}, lon={args.lon}, height={args.height}")
    print(f"  To view in Cesium, serve {output_dir} and load tileset.json")


if __name__ == "__main__":
    main()
