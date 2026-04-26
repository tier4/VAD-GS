"""Generate LiDAR depth maps from T4 dataset for VAD-GS training.

Uses t4-devkit for dataset I/O, coordinate transforms and projection.

Usage:
    python script/t4/generate_lidar_depth.py --config configs/example/t4_train_example.yaml
    python script/t4/generate_lidar_depth.py \
        --dataroot /path/to/t4_dataset \
        --scene-index 0 \
        --camera-channels CAM_FRONT CAM_FRONT_LEFT CAM_FRONT_RIGHT \
        --lidar-channel LIDAR_CONCAT
"""

import argparse
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

from config_utils import add_config_arg, apply_config_defaults
from t4_dataset import T4Dataset


def main():
    parser = argparse.ArgumentParser(
        description="Generate LiDAR depth maps from T4 dataset"
    )
    add_config_arg(parser)
    parser.add_argument("--dataroot", type=str, default=None)
    parser.add_argument("--version", type=int, default=None)
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
    if args.version is None:
        args.version = 0
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

    ds = T4Dataset.from_args(args)
    dataroot = ds.dataroot
    output_dir = args.output_dir or (dataroot / "preprocessed" / "lidar_depth")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Resolved dataroot: {dataroot}")
    print(f"Scene: {ds.scene.name}, {ds.num_samples} samples")

    camera_channels = args.camera_channels

    for sample in tqdm(ds.samples, desc="Generating depth maps"):
        # Get LiDAR sample_data token
        lidar_sd_token = ds.get_lidar_sample_data_token(
            sample.token, args.lidar_channel
        )
        if lidar_sd_token is None:
            continue

        # Load LiDAR points in sensor frame (N, 3)
        lidar_points = ds.load_lidar_points(lidar_sd_token)

        # Project to each camera
        for ch in camera_channels:
            cam_sd_token = sample.data.get(ch)
            if cam_sd_token is None:
                continue

            cam_sd = ds.t4.get("sample_data", cam_sd_token)
            H, W = cam_sd.height, cam_sd.width
            K = ds.get_camera_intrinsic(cam_sd.calibrated_sensor_token)

            # Transform LiDAR points to camera frame
            pts_cam = ds.transform_lidar_to_camera(
                lidar_points, lidar_sd_token, cam_sd_token
            )

            depth = pts_cam[:, 2]
            valid = (depth > args.min_depth) & (depth < args.max_depth)

            # Project to 2D
            uv, in_image = T4Dataset.project_points_to_image(
                pts_cam, K, image_size=(W, H)
            )
            valid = valid & in_image

            u_int = np.round(uv[valid, 0]).astype(np.int32)
            v_int = np.round(uv[valid, 1]).astype(np.int32)
            d_valid = depth[valid].astype(np.float32)

            # Clamp to valid pixel range
            in_bounds = (u_int >= 0) & (u_int < W) & (v_int >= 0) & (v_int < H)
            u_int = u_int[in_bounds]
            v_int = v_int[in_bounds]
            d_valid = d_valid[in_bounds]

            # Build depth map (keep closest point per pixel)
            depth_map_mask = np.zeros((H, W), dtype=bool)
            depth_map_values = np.zeros((H, W), dtype=np.float32)
            for j in range(len(u_int)):
                y, x = v_int[j], u_int[j]
                if not depth_map_mask[y, x] or d_valid[j] < depth_map_values[y, x]:
                    depth_map_mask[y, x] = True
                    depth_map_values[y, x] = d_valid[j]

            # Save in VAD-GS format: dict with 'mask' and 'value'
            depth_data = {"mask": depth_map_mask, "value": depth_map_values}

            image_name = Path(ds.t4.get_sample_data_path(cam_sd_token)).stem
            cam_out_dir = output_dir / ch
            cam_out_dir.mkdir(parents=True, exist_ok=True)
            np.save(str(cam_out_dir / f"{image_name}.npy"), depth_data)

    print(f"Done. Depth maps saved to {output_dir}")


if __name__ == "__main__":
    main()
