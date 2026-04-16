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

    # Auto-detect coordinates from T4 dataset:
    python export_cesium.py <checkpoint.pth> \
        --t4-dataset 835afe23-ff50-4883-a0b2-421e101a124b \
        --background-only
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
import xml.etree.ElementTree as ET
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
# T4 dataset → geodetic coordinate extraction
# ---------------------------------------------------------------------------

_ANNOTATION_DATASET_BASE = os.path.expanduser("~/.webauto/data/data/annotation_dataset")


def resolve_t4_dataset_path(dataset_id_or_path: str, revision: int = 0) -> Path:
    """Resolve a T4 dataset UUID or path to a filesystem path."""
    candidate = os.path.expanduser(dataset_id_or_path)
    if os.path.isdir(candidate) and os.path.isdir(os.path.join(candidate, "annotation")):
        return Path(candidate)
    id_path = os.path.join(_ANNOTATION_DATASET_BASE, dataset_id_or_path)
    if os.path.isdir(id_path):
        rev_path = os.path.join(id_path, str(revision))
        if os.path.isdir(rev_path):
            return Path(rev_path)
        return Path(id_path)
    return Path(candidate)


def _read_lanelet2_reference_points(
    osm_path: Path,
) -> np.ndarray:
    """Read (lat, lon, local_x, local_y) reference points from lanelet2_map.osm.

    Returns an (N, 4) array: columns are [lat, lon, local_x, local_y].
    """
    tree = ET.parse(osm_path)
    root = tree.getroot()
    rows: list[tuple[float, float, float, float]] = []
    for node in root.iter("node"):
        lat = node.get("lat")
        lon = node.get("lon")
        if lat is None or lon is None:
            continue
        tags = {t.get("k"): t.get("v") for t in node.iter("tag")}
        lx = tags.get("local_x")
        ly = tags.get("local_y")
        if lx is None or ly is None:
            continue
        rows.append((float(lat), float(lon), float(lx), float(ly)))
    if not rows:
        raise RuntimeError(f"No reference points with (lat,lon,local_x,local_y) found in {osm_path}")
    return np.array(rows, dtype=np.float64)


def _fit_local_to_geodetic(
    ref: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit affine transforms from local (x,y) to (lat, lon).

    Returns (lat_coef, lon_coef) where each is [a, b, c] such that
        lat = a*local_x + b*local_y + c
        lon = a*local_x + b*local_y + c
    """
    A = np.column_stack([ref[:, 2], ref[:, 3], np.ones(len(ref))])
    lat_coef, _, _, _ = np.linalg.lstsq(A, ref[:, 0], rcond=None)
    lon_coef, _, _, _ = np.linalg.lstsq(A, ref[:, 1], rcond=None)
    return lat_coef, lon_coef


def geocoord_from_t4_dataset(
    dataset_path: Path,
) -> tuple[float, float, float]:
    """Extract (lat, lon, height) from a T4 dataset.

    Uses lanelet2_map.osm reference points to build a local→geodetic affine
    transform, then applies it to the ego trajectory centroid from ego_pose.json.
    """
    # Read ego pose centroid
    ego_pose_path = dataset_path / "annotation" / "ego_pose.json"
    if not ego_pose_path.exists():
        raise FileNotFoundError(f"ego_pose.json not found: {ego_pose_path}")
    with open(ego_pose_path) as f:
        ego_poses = json.load(f)
    if not ego_poses:
        raise RuntimeError("ego_pose.json is empty")

    xs = [e["translation"][0] for e in ego_poses]
    ys = [e["translation"][1] for e in ego_poses]
    zs = [e["translation"][2] for e in ego_poses]
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)
    cz = sum(zs) / len(zs)

    # Find lanelet2 map
    osm_path = dataset_path / "map" / "lanelet2_map.osm"
    if not osm_path.exists():
        raise FileNotFoundError(
            f"lanelet2_map.osm not found: {osm_path}\n"
            "Cannot auto-detect coordinates. Use --lat/--lon/--height instead."
        )

    ref = _read_lanelet2_reference_points(osm_path)
    lat_coef, lon_coef = _fit_local_to_geodetic(ref)

    lat = float(lat_coef[0] * cx + lat_coef[1] * cy + lat_coef[2])
    lon = float(lon_coef[0] * cx + lon_coef[1] * cy + lon_coef[2])
    height = float(cz)

    # Sanity check: compute residual on reference points
    pred_lat = ref[:, 2] * lat_coef[0] + ref[:, 3] * lat_coef[1] + lat_coef[2]
    pred_lon = ref[:, 2] * lon_coef[0] + ref[:, 3] * lon_coef[1] + lon_coef[2]
    lat_err_m = np.abs(pred_lat - ref[:, 0]).mean() * 111_000
    lon_err_m = np.abs(pred_lon - ref[:, 1]).mean() * 90_000
    print(f"  Affine fit residual: lat {lat_err_m:.3f}m, lon {lon_err_m:.3f}m "
          f"({len(ref):,} reference points)")

    return lat, lon, height


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
    spz_compression: bool = True,
) -> dict:
    """Create a 3D Tiles 1.1 tileset.json for a single GLB tile."""
    bv = compute_bounding_box(gc)
    transform = build_tileset_transform(lat, lon, height)

    tileset: dict = {
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

    # CesiumJS requires 3DTILES_content_gltf extension to detect 3DGS content.
    # Both KHR_gaussian_splatting and the SPZ compression sub-extension must be
    # declared as extensionsRequired for CesiumJS to route to its GS renderer.
    gltf_extensions = ["KHR_gaussian_splatting"]
    if spz_compression:
        gltf_extensions.append("KHR_gaussian_splatting_compression_spz_2")

    tileset["extensionsUsed"] = ["3DTILES_content_gltf"]
    tileset["extensions"] = {
        "3DTILES_content_gltf": {
            "extensionsUsed": gltf_extensions,
            "extensionsRequired": gltf_extensions,
        }
    }

    return tileset


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
        "--no-spz-compression",
        action="store_true",
        help="Disable SPZ compression (larger files, wider viewer compatibility)",
    )
    # Coordinate source: either --t4-dataset or manual --lat/--lon/--height
    coord_group = parser.add_argument_group("coordinate source")
    coord_group.add_argument(
        "--t4-dataset",
        type=str,
        default=None,
        help="T4 dataset path or UUID. Auto-detects lat/lon/height from "
        "lanelet2_map.osm + ego_pose.json. Overrides --lat/--lon/--height.",
    )
    coord_group.add_argument(
        "--t4-revision",
        type=int,
        default=0,
        help="T4 dataset revision subdirectory (default: 0)",
    )
    coord_group.add_argument(
        "--lat",
        type=float,
        default=None,
        help="Latitude for tileset placement",
    )
    coord_group.add_argument(
        "--lon",
        type=float,
        default=None,
        help="Longitude for tileset placement",
    )
    coord_group.add_argument(
        "--height",
        type=float,
        default=None,
        help="Height above WGS84 ellipsoid in meters",
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

    # Resolve coordinates
    if args.t4_dataset is not None:
        print(f"Resolving coordinates from T4 dataset: {args.t4_dataset}")
        ds_path = resolve_t4_dataset_path(args.t4_dataset, args.t4_revision)
        if not ds_path.is_dir():
            print(f"Error: T4 dataset not found: {ds_path}", file=sys.stderr)
            sys.exit(1)
        print(f"  Dataset path: {ds_path}")
        lat, lon, height = geocoord_from_t4_dataset(ds_path)
        # Allow manual overrides for individual components
        args.lat = args.lat if args.lat is not None else lat
        args.lon = args.lon if args.lon is not None else lon
        args.height = args.height if args.height is not None else height
        print(f"  Coordinates: lat={args.lat:.8f}, lon={args.lon:.8f}, height={args.height:.2f}")
    else:
        # Fallback defaults (Tokyo)
        if args.lat is None:
            args.lat = 35.6812
        if args.lon is None:
            args.lon = 139.7671
        if args.height is None:
            args.height = 0.0

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
    options = GltfSaveOptions(spz_compression=not args.no_spz_compression)
    print(f"Saving GLB: {glb_path}")
    save_gltf(merged, glb_path, options)
    glb_size = glb_path.stat().st_size
    print(f"  GLB size: {glb_size / 1024 / 1024:.1f} MB")

    # Save tileset.json
    use_spz = not args.no_spz_compression
    tileset = create_tileset_json(
        glb_name,
        merged,
        lat=args.lat,
        lon=args.lon,
        height=args.height,
        geometric_error=args.geometric_error,
        spz_compression=use_spz,
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
