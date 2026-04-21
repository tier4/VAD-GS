#!/usr/bin/env python3
"""Export VAD-GS trained checkpoint (.pth) to Cesium-compatible 3D Tiles.

Uses the 3dgs_io library to produce GLB files with the KHR_gaussian_splatting
glTF extension, wrapped in a 3D Tiles tileset.json.

Requirements:
    pip install 3dgs-io torch numpy pyyaml
    (Recommended: use .venv_export with Python >=3.13)

Usage:
    python export_cesium.py [checkpoint.pth] [options]

Example:
    # Config only — checkpoint auto-detected from output/<task>/<exp_name>/trained_model/:
    python export_cesium.py --config configs/example/t4_train_example.yaml --background-only

    # Config + explicit checkpoint:
    python export_cesium.py output/t4_exp/t4_scene_000/trained_model/iteration_30000.pth \
        --config configs/example/t4_train_example.yaml --background-only
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

import yaml

import numpy as np
import torch

# 3dgs_io: KHR_gaussian_splatting compliant glTF I/O
from importlib import import_module

_3dgs_io = import_module("3dgs_io")
save_gltf = _3dgs_io.save_gltf
GltfSaveOptions = _3dgs_io.GltfSaveOptions
DatasetType = _3dgs_io.DatasetType

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
    lat_deg: float,
    lon_deg: float,
    height: float,
    model_to_enu_rotation: np.ndarray | None = None,
) -> list[float]:
    """Build a 4x4 column-major transform for 3D Tiles root tile.

    Places the model origin at the given geodetic location,
    with axes mapped from the model's local frame to ECEF.

    Parameters
    ----------
    lat_deg, lon_deg, height:
        Geodetic position of the model origin.
    model_to_enu_rotation:
        Optional 3x3 rotation matrix that maps model-local axes to ENU axes.
        When None, the model is assumed to already be in ENU frame.
    """
    origin = _geodetic_to_ecef(lat_deg, lon_deg, height)
    enu_to_ecef = _enu_to_ecef_matrix(lat_deg, lon_deg)
    if model_to_enu_rotation is not None:
        rot = enu_to_ecef @ model_to_enu_rotation
    else:
        rot = enu_to_ecef
    # 4x4 column-major (as required by 3D Tiles spec)
    m = np.eye(4, dtype=np.float64)
    m[:3, :3] = rot
    m[:3, 3] = origin
    # 3D Tiles stores column-major: flatten column by column
    return m.T.flatten().tolist()


# ---------------------------------------------------------------------------
# Training config (YAML) loading
# ---------------------------------------------------------------------------


def load_training_config(config_path: Path) -> dict:
    """Load a VAD-GS training config YAML and return relevant settings.

    Returns a dict with keys:
        task, exp_name, source_path, revision, scene_index, lidar_channel, data_type
    """
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    data = raw.get("data", {})
    return {
        "task": raw.get("task", ""),
        "exp_name": raw.get("exp_name", ""),
        "source_path": raw.get("source_path", ""),
        "data_type": data.get("type", ""),
        "revision": data.get("revision", 0),
        "scene_index": data.get("scene_index", 0),
        "lidar_channel": data.get("lidar_channel", "LIDAR_CONCAT"),
    }


def _find_latest_checkpoint(trained_model_dir: Path) -> Path | None:
    """Find the checkpoint with the highest iteration number in a directory."""
    candidates = sorted(trained_model_dir.glob("iteration_*.pth"))
    if not candidates:
        return None
    # Sort by iteration number (extract integer from filename)
    def _iter_num(p: Path) -> int:
        try:
            return int(p.stem.split("_", 1)[1])
        except (IndexError, ValueError):
            return -1
    return max(candidates, key=_iter_num)


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


def _quat_wxyz_to_rotmat(quat) -> np.ndarray:
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
    w, x, y, z = quat
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float64,
    )


def _make_transform(translation, rotation_wxyz) -> np.ndarray:
    """Create 4x4 transform matrix from translation and quaternion [w,x,y,z]."""
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _quat_wxyz_to_rotmat(rotation_wxyz)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64)
    return transform


def _compute_lidar_to_world_start(
    dataset_path: Path,
    scene_index: int = 0,
    lidar_channel: str = "LIDAR_CONCAT",
) -> np.ndarray:
    """Compute the lidar_to_world_start transform from a T4 dataset.

    This replicates the normalization used during training (t4_utils.py
    ``load_camera_info_t4``): the first sample's LiDAR ego_pose × sensor_to_ego.

    Returns a 4×4 matrix mapping from the model's local frame to the map frame.
    """
    annotation_dir = dataset_path / "annotation"

    def _load_table(name):
        p = annotation_dir / f"{name}.json"
        if not p.exists():
            return []
        with open(p) as f:
            return json.load(f)

    scenes = _load_table("scene")
    samples_list = _load_table("sample")
    sample_data_list = _load_table("sample_data")
    ego_poses = _load_table("ego_pose")
    calibrated_sensors = _load_table("calibrated_sensor")
    sensors = _load_table("sensor")

    samples_by_token = {s["token"]: s for s in samples_list}
    ego_pose_by_token = {e["token"]: e for e in ego_poses}
    cs_by_token = {cs["token"]: cs for cs in calibrated_sensors}
    sensor_by_token = {s["token"]: s for s in sensors}

    # Follow sample chain to get first sample
    scene = scenes[min(scene_index, len(scenes) - 1)]
    first_sample_token = scene["first_sample_token"]

    # Try specified lidar channel + common fallbacks
    lidar_channels = {lidar_channel, "LIDAR_CONCAT", "LIDAR_TOP", "lidar_top"}

    # Find lidar sample_data for first sample
    first_lidar_sd = None
    first_lidar_cs = None
    for sd in sample_data_list:
        if sd.get("sample_token") != first_sample_token:
            continue
        if not sd.get("is_key_frame", False):
            continue
        cs = cs_by_token.get(sd.get("calibrated_sensor_token", ""))
        if cs is None:
            continue
        sensor = sensor_by_token.get(cs.get("sensor_token", ""))
        if sensor is None:
            continue
        if sensor["channel"] in lidar_channels:
            first_lidar_sd = sd
            first_lidar_cs = cs
            break

    if first_lidar_sd is not None:
        ego = ego_pose_by_token[first_lidar_sd["ego_pose_token"]]
        ego_to_world = _make_transform(ego["translation"], ego["rotation"])
        sensor_to_ego = _make_transform(first_lidar_cs["translation"], first_lidar_cs["rotation"])
        lidar_to_world_start = ego_to_world @ sensor_to_ego
        print(f"  lidar_to_world_start from {sensor_by_token[first_lidar_cs['sensor_token']]['channel']}")
    else:
        # Fallback: use first sample's first available ego_pose
        print("  Warning: LiDAR channel not found, using ego_pose directly")
        for sd in sample_data_list:
            if sd.get("sample_token") == first_sample_token:
                ego = ego_pose_by_token[sd["ego_pose_token"]]
                lidar_to_world_start = _make_transform(ego["translation"], ego["rotation"])
                break
        else:
            raise RuntimeError("Cannot compute lidar_to_world_start: no sample_data for first sample")

    return lidar_to_world_start


def _map_to_enu_rotation(
    lat_coef: np.ndarray,
    lon_coef: np.ndarray,
    lat_deg: float,
) -> np.ndarray:
    """Extract the 3x3 rotation from map coordinates to ENU.

    Uses the affine fit coefficients (local_x, local_y → lat, lon) to
    determine how the map X/Y axes relate to East/North.

    Returns a 3x3 rotation matrix R such that ENU = R @ map_vector.
    """
    cos_lat = math.cos(math.radians(lat_deg))
    # Map X-axis direction in ENU (East, North) — in metric-proportional units
    map_x_east = lon_coef[0] * cos_lat
    map_x_north = lat_coef[0]
    # Map Y-axis direction in ENU
    map_y_east = lon_coef[1] * cos_lat
    map_y_north = lat_coef[1]

    # Build 2D rotation: map (x,y) → ENU (east, north)
    # Columns = map basis vectors expressed in ENU
    R2d = np.array([[map_x_east, map_y_east],
                     [map_x_north, map_y_north]], dtype=np.float64)
    # Normalize to pure rotation (remove scale via SVD)
    U, _, Vt = np.linalg.svd(R2d)
    R2d_pure = U @ Vt
    # Ensure proper rotation (det = +1)
    if np.linalg.det(R2d_pure) < 0:
        U[:, -1] *= -1
        R2d_pure = U @ Vt

    # Extend to 3D (Z = Up stays the same)
    R3d = np.eye(3, dtype=np.float64)
    R3d[:2, :2] = R2d_pure
    return R3d


def geocoord_from_t4_dataset(
    dataset_path: Path,
    scene_index: int = 0,
    lidar_channel: str = "LIDAR_CONCAT",
) -> tuple[float, float, float, np.ndarray]:
    """Extract geodetic coordinates and model-to-ENU rotation from a T4 dataset.

    Uses lanelet2_map.osm reference points to build a local→geodetic affine
    transform, and computes lidar_to_world_start for the model orientation.

    Returns:
        (lat, lon, height, model_to_enu_rotation)
    where model_to_enu_rotation is a 3x3 matrix mapping model-local axes to ENU.
    """
    # Compute lidar_to_world_start (model-local to map transform)
    lidar_to_world_start = _compute_lidar_to_world_start(
        dataset_path, scene_index=scene_index, lidar_channel=lidar_channel,
    )

    # The model origin (0,0,0) corresponds to this position in map coordinates
    origin_map = lidar_to_world_start[:3, 3]
    origin_x, origin_y, origin_z = origin_map

    # Find lanelet2 map
    osm_path = dataset_path / "map" / "lanelet2_map.osm"
    if not osm_path.exists():
        raise FileNotFoundError(
            f"lanelet2_map.osm not found: {osm_path}\n"
            "Cannot auto-detect coordinates. Use --lat/--lon/--height instead."
        )

    ref = _read_lanelet2_reference_points(osm_path)
    lat_coef, lon_coef = _fit_local_to_geodetic(ref)

    # Convert the model origin (first lidar pos) to geodetic
    lat = float(lat_coef[0] * origin_x + lat_coef[1] * origin_y + lat_coef[2])
    lon = float(lon_coef[0] * origin_x + lon_coef[1] * origin_y + lon_coef[2])
    height = float(origin_z)

    # Sanity check: compute residual on reference points
    pred_lat = ref[:, 2] * lat_coef[0] + ref[:, 3] * lat_coef[1] + lat_coef[2]
    pred_lon = ref[:, 2] * lon_coef[0] + ref[:, 3] * lon_coef[1] + lon_coef[2]
    lat_err_m = np.abs(pred_lat - ref[:, 0]).mean() * 111_000
    lon_err_m = np.abs(pred_lon - ref[:, 1]).mean() * 90_000
    print(f"  Affine fit residual: lat {lat_err_m:.3f}m, lon {lon_err_m:.3f}m "
          f"({len(ref):,} reference points)")

    # Build model-to-ENU rotation:
    #   model → map (lidar_to_world_start rotation)
    #   map → ENU  (from affine fit coefficients)
    R_model_to_map = lidar_to_world_start[:3, :3]
    R_map_to_enu = _map_to_enu_rotation(lat_coef, lon_coef, lat)
    model_to_enu = R_map_to_enu @ R_model_to_map

    print(f"  Model origin in map: ({origin_x:.2f}, {origin_y:.2f}, {origin_z:.2f})")

    return lat, lon, height, model_to_enu


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
# Metadata builder
# ---------------------------------------------------------------------------


def build_export_metadata(
    train_cfg: dict | None,
    checkpoint_path: Path,
    iteration: int | str,
    total_points: int,
    background_only: bool,
    object_keys: list[str],
    spz_compression: bool,
    max_sh_degree: int | None,
    lat: float | None = None,
    lon: float | None = None,
    height: float | None = None,
) -> dict:
    """Build metadata dict to embed in glTF asset.extras.

    Records the training data source, export parameters, and model statistics
    so that downstream consumers can trace provenance.
    """
    metadata: dict = {}

    # Training data source
    if train_cfg is not None:
        metadata["dataset_type"] = DatasetType.T4_DATASET.value
        source: dict = {}
        if train_cfg.get("source_path"):
            source["source_path"] = train_cfg["source_path"]
        if train_cfg.get("data_type"):
            source["data_type"] = train_cfg["data_type"]
        if train_cfg.get("revision") is not None:
            source["revision"] = train_cfg["revision"]
        if train_cfg.get("scene_index") is not None:
            source["scene_index"] = train_cfg["scene_index"]
        if train_cfg.get("lidar_channel"):
            source["lidar_channel"] = train_cfg["lidar_channel"]
        if train_cfg.get("task"):
            source["task"] = train_cfg["task"]
        if train_cfg.get("exp_name"):
            source["exp_name"] = train_cfg["exp_name"]
        metadata["training_data"] = source

    # Checkpoint info
    metadata["checkpoint"] = {
        "path": str(checkpoint_path),
        "iteration": iteration if isinstance(iteration, int) else str(iteration),
    }

    # Export parameters
    export_params: dict = {
        "background_only": background_only,
        "spz_compression": spz_compression,
    }
    if max_sh_degree is not None:
        export_params["max_sh_degree"] = max_sh_degree
    if object_keys:
        export_params["object_keys"] = object_keys
    metadata["export"] = export_params

    # Model statistics
    metadata["model"] = {
        "total_gaussians": total_points,
    }

    # Geodetic placement
    if lat is not None and lon is not None:
        placement: dict = {"lat": lat, "lon": lon}
        if height is not None:
            placement["height"] = height
        metadata["placement"] = placement

    metadata["generator"] = "VAD-GS export_cesium.py"

    return metadata


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
    model_to_enu_rotation: np.ndarray | None = None,
) -> dict:
    """Create a 3D Tiles 1.1 tileset.json for a single GLB tile."""
    bv = compute_bounding_box(gc)
    transform = build_tileset_transform(lat, lon, height, model_to_enu_rotation)

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
        "checkpoint",
        type=Path,
        nargs="?",
        default=None,
        help="Path to .pth checkpoint file. "
        "If omitted, the latest checkpoint is found from --config "
        "(output/<task>/<exp_name>/trained_model/).",
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
    # Coordinate source: --config (recommended) or --t4-dataset
    coord_group = parser.add_argument_group("coordinate source")
    coord_group.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Training config YAML (e.g. configs/example/t4_train_example.yaml). "
        "Reads source_path, data.revision, data.scene_index, data.lidar_channel "
        "to auto-detect coordinates and orientation.",
    )
    coord_group.add_argument(
        "--t4-dataset",
        type=str,
        default=None,
        help="T4 dataset path or UUID. Auto-detects coordinates from "
        "lanelet2_map.osm + ego_pose.json.",
    )
    coord_group.add_argument(
        "--t4-revision",
        type=int,
        default=None,
        help="T4 dataset revision subdirectory (default: from config or 0)",
    )
    parser.add_argument(
        "--geometric-error",
        type=float,
        default=500.0,
        help="Geometric error for tileset.json (default: 500)",
    )
    args = parser.parse_args()

    # --- Resolve T4 dataset settings from --config if provided ---
    scene_index = 0
    lidar_channel = "LIDAR_CONCAT"
    train_cfg = None

    if args.config is not None:
        if not args.config.exists():
            print(f"Error: config not found: {args.config}", file=sys.stderr)
            sys.exit(1)
        print(f"Loading training config: {args.config}")
        train_cfg = load_training_config(args.config)

        if train_cfg["data_type"] != "T4":
            print(f"Warning: config data.type={train_cfg['data_type']!r}, expected 'T4'")

        # --config sets t4-dataset unless explicitly overridden
        if args.t4_dataset is None and train_cfg["source_path"]:
            args.t4_dataset = train_cfg["source_path"]
        if args.t4_revision is None:
            args.t4_revision = train_cfg["revision"]
        scene_index = train_cfg["scene_index"]
        lidar_channel = train_cfg["lidar_channel"]

        print(f"  source_path: {args.t4_dataset}")
        print(f"  revision: {args.t4_revision}, scene_index: {scene_index}, "
              f"lidar_channel: {lidar_channel}")

    # --- Resolve checkpoint path ---
    if args.checkpoint is None:
        if train_cfg is None:
            print("Error: checkpoint not specified and no --config given", file=sys.stderr)
            sys.exit(1)
        task = train_cfg["task"]
        exp_name = train_cfg["exp_name"]
        if not task or not exp_name:
            print("Error: config missing 'task' or 'exp_name', cannot locate checkpoint",
                  file=sys.stderr)
            sys.exit(1)
        trained_model_dir = Path("output") / task / exp_name / "trained_model"
        args.checkpoint = _find_latest_checkpoint(trained_model_dir)
        if args.checkpoint is None:
            print(f"Error: no checkpoint found in {trained_model_dir}", file=sys.stderr)
            sys.exit(1)
        print(f"Auto-detected checkpoint: {args.checkpoint}")

    if not args.checkpoint.exists():
        print(f"Error: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        sys.exit(1)

    # Defaults for values not set by config or CLI
    if args.t4_revision is None:
        args.t4_revision = 0

    # --- Resolve coordinates and orientation ---
    model_to_enu_rotation = None  # None = assume model is already ENU-aligned
    if args.t4_dataset is not None:
        print(f"Resolving coordinates from T4 dataset: {args.t4_dataset}")
        ds_path = resolve_t4_dataset_path(args.t4_dataset, args.t4_revision)
        if not ds_path.is_dir():
            print(f"Error: T4 dataset not found: {ds_path}", file=sys.stderr)
            sys.exit(1)
        print(f"  Dataset path: {ds_path}")
        lat, lon, height, model_to_enu_rotation = geocoord_from_t4_dataset(
            ds_path, scene_index=scene_index, lidar_channel=lidar_channel,
        )
        print(f"  Coordinates: lat={lat:.8f}, lon={lon:.8f}, height={height:.2f}")
    else:
        print("Error: no coordinate source specified. Use --config or --t4-dataset.",
              file=sys.stderr)
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
    exported_obj_keys: list[str] = []
    if not args.background_only:
        obj_keys = [k for k in ckpt.keys() if k.startswith("obj_")]
        if args.objects is not None:
            obj_keys = [k for k in obj_keys if k in args.objects]
        exported_obj_keys = sorted(obj_keys)

        for obj_key in exported_obj_keys:
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
    use_spz = not args.no_spz_compression
    metadata = build_export_metadata(
        train_cfg=train_cfg,
        checkpoint_path=args.checkpoint,
        iteration=iteration,
        total_points=total_points,
        background_only=args.background_only,
        object_keys=exported_obj_keys,
        spz_compression=use_spz,
        max_sh_degree=args.max_sh_degree,
        lat=lat,
        lon=lon,
        height=height,
    )
    options = GltfSaveOptions(spz_compression=use_spz, metadata=metadata)
    print(f"Saving GLB: {glb_path}")
    save_gltf(merged, glb_path, options)
    glb_size = glb_path.stat().st_size
    print(f"  GLB size: {glb_size / 1024 / 1024:.1f} MB")

    # Save tileset.json
    tileset = create_tileset_json(
        glb_name,
        merged,
        lat=lat,
        lon=lon,
        height=height,
        geometric_error=args.geometric_error,
        spz_compression=use_spz,
        model_to_enu_rotation=model_to_enu_rotation,
    )
    tileset_path = output_dir / "tileset.json"
    with open(tileset_path, "w", encoding="utf-8") as f:
        json.dump(tileset, f, indent=2)
    print(f"Saved tileset: {tileset_path}")

    print(f"\nExport complete!")
    print(f"  Output directory: {output_dir}")
    print(f"  Placement: lat={lat}, lon={lon}, height={height}")
    print(f"  To view in Cesium, serve {output_dir} and load tileset.json")


if __name__ == "__main__":
    main()
