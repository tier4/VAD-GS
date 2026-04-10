"""Generate LiDAR depth maps from T4 dataset for VAD-GS training.

Reads T4 annotation JSON files, loads LiDAR point clouds and camera
calibration, projects LiDAR points onto camera images, and saves
sparse depth maps in the format expected by VAD-GS.

Usage:
    python script/t4/generate_lidar_depth.py --config configs/example/t4_train_example.yaml
    python script/t4/generate_lidar_depth.py \
        --dataroot /path/to/t4_dataset \
        --output-dir /path/to/t4_dataset/lidar_depth \
        --scene-index 0 \
        --camera-channels CAM_FRONT CAM_FRONT_LEFT CAM_FRONT_RIGHT \
        --lidar-channel LIDAR_CONCAT
"""

import argparse
import os
import json
from pathlib import Path

from config_utils import add_config_arg, apply_config_defaults, resolve_dataroot

import numpy as np
from tqdm import tqdm


def quat_wxyz_to_rotmat(quat):
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


def make_transform(translation, rotation_wxyz):
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = quat_wxyz_to_rotmat(rotation_wxyz)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64)
    return transform


def load_t4_tables(annotation_dir):
    tables = {}
    for name in [
        "scene", "sample", "sample_data", "sample_annotation",
        "calibrated_sensor", "ego_pose", "sensor", "instance", "category",
    ]:
        path = os.path.join(annotation_dir, f"{name}.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                tables[name] = json.load(f)
        else:
            tables[name] = []
    return tables


def index_by_token(items):
    return {item["token"]: item for item in items}


def sample_chain(scene, sample_by_token):
    ordered = []
    token = scene["first_sample_token"]
    while token:
        sample = sample_by_token[token]
        ordered.append(sample)
        token = sample.get("next", "")
    return ordered


def read_lidar(path):
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % 5 == 0:
        raw = raw.reshape(-1, 5)
    elif raw.size % 4 == 0:
        raw = raw.reshape(-1, 4)
    else:
        raw = raw.reshape(-1, 5)
    return raw[:, :3]


def main():
    parser = argparse.ArgumentParser(
        description="Generate LiDAR depth maps from T4 dataset"
    )
    add_config_arg(parser)
    parser.add_argument("--dataroot", type=str, default=None)
    parser.add_argument("--revision", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--scene-index", type=int, default=None)
    parser.add_argument("--camera-channels", nargs="+", default=None)
    parser.add_argument("--lidar-channel", default=None)
    parser.add_argument("--min-depth", type=float, default=1.0)
    parser.add_argument("--max-depth", type=float, default=80.0)
    args = parser.parse_args()

    # Apply config defaults, then hard defaults
    apply_config_defaults(args)
    if args.dataroot is None:
        parser.error("--dataroot is required (provide via --config or CLI)")
    if args.revision is None:
        args.revision = 0
    if args.scene_index is None:
        args.scene_index = 0
    if args.camera_channels is None:
        args.camera_channels = [
            "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
            "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
        ]
    if args.lidar_channel is None:
        args.lidar_channel = "LIDAR_CONCAT"

    dataroot = resolve_dataroot(args.dataroot, revision=args.revision)
    output_dir = args.output_dir or (dataroot / "preprocessed" / "lidar_depth")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find annotation directory
    annotation_dir = dataroot / "annotation"
    if not annotation_dir.exists():
        if (dataroot / "sample.json").exists():
            annotation_dir = dataroot
        else:
            raise FileNotFoundError(f"No annotation directory found at {annotation_dir}")

    print(f"Loading T4 tables from {annotation_dir}")
    tables = load_t4_tables(str(annotation_dir))

    sample_by_token = index_by_token(tables["sample"])
    calibrated_sensor_by_token = index_by_token(tables["calibrated_sensor"])
    sensor_by_token = index_by_token(tables["sensor"])
    ego_pose_by_token = index_by_token(tables["ego_pose"])

    scene = tables["scene"][args.scene_index]
    samples = sample_chain(scene, sample_by_token)
    print(f"Scene: {scene.get('name', 'unknown')}, {len(samples)} samples")

    # Build sample_data -> (channel, sd, cs) map (keyframes only)
    sd_by_sample = {}
    for sd in tables["sample_data"]:
        if not sd.get("is_key_frame", False):
            continue
        cs = calibrated_sensor_by_token.get(sd["calibrated_sensor_token"])
        if cs is None:
            continue
        sensor = sensor_by_token.get(cs["sensor_token"])
        if sensor is None:
            continue
        sd_by_sample.setdefault(sd["sample_token"], {})[sensor["channel"]] = (sd, cs)

    # Map camera channels to indices
    camera_channels = args.camera_channels

    for frame_idx, sample in enumerate(tqdm(samples, desc="Generating depth maps")):
        frame_data = sd_by_sample.get(sample["token"], {})

        # Load LiDAR
        if args.lidar_channel not in frame_data:
            continue
        lidar_sd, lidar_cs = frame_data[args.lidar_channel]
        lidar_path = dataroot / lidar_sd["filename"]
        if not lidar_path.exists():
            continue

        lidar_points = read_lidar(str(lidar_path))

        # LiDAR to world transform
        lidar_ego = ego_pose_by_token[lidar_sd["ego_pose_token"]]
        lidar_sensor_to_ego = make_transform(lidar_cs["translation"], lidar_cs["rotation"])
        lidar_to_world = make_transform(lidar_ego["translation"], lidar_ego["rotation"]) @ lidar_sensor_to_ego

        # Transform to world
        pts_h = np.concatenate([lidar_points, np.ones((lidar_points.shape[0], 1))], axis=1)
        pts_world = (pts_h @ lidar_to_world.T)[:, :3]

        # Project to each camera
        for cam_idx, ch in enumerate(camera_channels):
            if ch not in frame_data:
                continue
            cam_sd, cam_cs = frame_data[ch]
            cam_ego = ego_pose_by_token[cam_sd["ego_pose_token"]]
            cam_sensor_to_ego = make_transform(cam_cs["translation"], cam_cs["rotation"])
            cam_to_world = make_transform(cam_ego["translation"], cam_ego["rotation"]) @ cam_sensor_to_ego
            world_to_cam = np.linalg.inv(cam_to_world)

            K = np.array(cam_cs["camera_intrinsic"], dtype=np.float64)
            H, W = cam_sd["height"], cam_sd["width"]

            # Project
            pts_cam = (np.concatenate([pts_world, np.ones((pts_world.shape[0], 1))], axis=1) @ world_to_cam.T)[:, :3]
            depth = pts_cam[:, 2]
            valid = (depth > args.min_depth) & (depth < args.max_depth)

            uvw = pts_cam @ K.T
            u = uvw[:, 0] / np.maximum(depth, 1e-6)
            v = uvw[:, 1] / np.maximum(depth, 1e-6)
            valid = valid & (u >= 0) & (u < W) & (v >= 0) & (v < H)

            u_int = np.round(u[valid]).astype(np.int32)
            v_int = np.round(v[valid]).astype(np.int32)
            d_valid = depth[valid].astype(np.float32)

            # Clamp to valid pixel range (round can push boundary pixels out)
            in_bounds = (u_int >= 0) & (u_int < W) & (v_int >= 0) & (v_int < H)
            u_int = u_int[in_bounds]
            v_int = v_int[in_bounds]
            d_valid = d_valid[in_bounds]

            # Create depth map
            mask = np.zeros((H, W), dtype=bool)
            value = np.zeros(d_valid.shape[0], dtype=np.float32)

            # Handle duplicate projections (keep closest)
            for j in range(len(u_int)):
                y, x = v_int[j], u_int[j]
                if not mask[y, x] or d_valid[j] < value[j]:
                    mask[y, x] = True

            # Rebuild sparse arrays
            depth_mask = np.zeros((H, W), dtype=bool)
            depth_value_arr = []
            for j in range(len(u_int)):
                y, x = v_int[j], u_int[j]
                depth_mask[y, x] = True

            # Use a simpler approach: save mask and depth values
            depth_map_mask = np.zeros((H, W), dtype=bool)
            depth_map_values = np.zeros((H, W), dtype=np.float32)
            for j in range(len(u_int)):
                y, x = v_int[j], u_int[j]
                if not depth_map_mask[y, x] or d_valid[j] < depth_map_values[y, x]:
                    depth_map_mask[y, x] = True
                    depth_map_values[y, x] = d_valid[j]

            # Save in VAD-GS format: dict with 'mask' and 'value'
            final_mask = depth_map_mask
            final_value = depth_map_values[final_mask]

            depth_data = {"mask": final_mask, "value": final_value}

            # Build image name matching T4 reader convention
            image_name = os.path.splitext(os.path.basename(cam_sd["filename"]))[0]
            cam_out_dir = output_dir / ch
            cam_out_dir.mkdir(parents=True, exist_ok=True)
            save_path = cam_out_dir / f"{image_name}.npy"
            np.save(str(save_path), depth_data)

    print(f"Done. Depth maps saved to {output_dir}")


if __name__ == "__main__":
    main()
