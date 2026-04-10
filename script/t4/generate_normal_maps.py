"""Generate monocular surface normal maps for T4 datasets.

Uses t4-devkit for dataset I/O and frame iteration.

Usage:
    python script/t4/generate_normal_maps.py --config configs/example/t4_train_example.yaml
    python script/t4/generate_normal_maps.py --dataroot caf37e66-... --scene-index 0 --batch-size 4
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from config_utils import add_config_arg, apply_config_defaults
from t4_dataset import T4Dataset


def depth_to_normal(depth, K=None):
    """Compute surface normals from depth map using finite differences."""
    h, w = depth.shape
    if K is not None:
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
    else:
        fx = fy = max(h, w)
        cx, cy = w / 2.0, h / 2.0

    u = np.arange(w, dtype=np.float32)
    v = np.arange(h, dtype=np.float32)
    u, v = np.meshgrid(u, v)

    z = depth.astype(np.float64)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    dzdx = np.gradient(z, axis=1)
    dzdy = np.gradient(z, axis=0)
    dxdx = np.gradient(x, axis=1)
    dxdy = np.gradient(x, axis=0)
    dydy = np.gradient(y, axis=0)
    dydx = np.gradient(y, axis=1)

    nx = dydx * dzdy - dzdx * dydy
    ny = dzdx * dxdy - dxdx * dzdy
    nz = dxdx * dydy - dydx * dxdy

    norm = np.sqrt(nx**2 + ny**2 + nz**2) + 1e-10
    nx /= norm
    ny /= norm
    nz /= norm

    flip = nz > 0
    nx[flip] = -nx[flip]
    ny[flip] = -ny[flip]
    nz[flip] = -nz[flip]

    return np.stack([nx, ny, nz], axis=-1).astype(np.float32)


def normal_to_png_bgr(normal):
    """Convert camera-space normals to BGR PNG format matching VAD-GS loading convention."""
    nx, ny, nz = normal[:, :, 0], normal[:, :, 1], normal[:, :, 2]
    png_r = ((-nx + 1) * 0.5 * 255).clip(0, 255).astype(np.uint8)
    png_g = ((-ny + 1) * 0.5 * 255).clip(0, 255).astype(np.uint8)
    png_b = ((-nz + 1) * 0.5 * 255).clip(0, 255).astype(np.uint8)
    return np.stack([png_b, png_g, png_r], axis=-1)


def main():
    parser = argparse.ArgumentParser(description="Generate normal maps for T4 dataset")
    add_config_arg(parser)
    parser.add_argument("--dataroot", type=str, default=None, help="Dataset UUID or path")
    parser.add_argument("--revision", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--scene-index", type=int, default=None)
    parser.add_argument("--camera-channels", nargs="+", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.add_argument("--depth-dir", type=Path, default=None,
                        help="Pre-computed depth dir (default: <dataroot>/preprocessed/depth)")
    args = parser.parse_args()

    apply_config_defaults(args)
    if args.dataroot is None:
        parser.error("--dataroot is required (provide via --config or CLI)")
    if args.revision is None:
        args.revision = 0
    if args.scene_index is None:
        args.scene_index = 0

    ds = T4Dataset.from_args(args)
    dataroot = ds.dataroot
    print(f"Resolved dataroot: {dataroot}")

    output_dir = args.output_dir or (dataroot / "preprocessed" / "normal_img")
    output_dir.mkdir(parents=True, exist_ok=True)
    depth_dir = args.depth_dir or (dataroot / "preprocessed" / "depth")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Scene: {ds.scene.name}, {ds.num_samples} samples")

    # Collect frames with camera intrinsics, skip existing
    entries = []
    for frame in ds.iter_frames(args.camera_channels):
        cam_out_dir = output_dir / frame.camera_channel
        cam_out_dir.mkdir(parents=True, exist_ok=True)
        save_path = cam_out_dir / f"{frame.image_name}.png"
        if args.skip_existing and save_path.exists():
            continue
        K = ds.get_camera_intrinsic(frame.calibrated_sensor_token)
        entries.append((frame, K))

    print(f"Images to process: {len(entries)}")
    if not entries:
        print("Nothing to do.")
        return

    use_precomputed_depth = depth_dir.exists()
    if use_precomputed_depth:
        print(f"Using pre-computed depth from {depth_dir}")
    else:
        print(f"No pre-computed depth at {depth_dir}, running depth estimation...")

    # Load depth model if needed
    depth_model = None
    depth_processor = None
    if not use_precomputed_depth:
        model_id = "depth-anything/Depth-Anything-V2-Small-hf"
        print(f"Loading model: {model_id}")
        depth_processor = AutoImageProcessor.from_pretrained(model_id)
        depth_model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device)
        depth_model.eval()

    for frame, K in tqdm(entries, desc="Normal estimation"):
        ch = frame.camera_channel
        # Get depth
        depth_npz_cam = depth_dir / ch / f"{frame.image_name}.npz"
        depth_npz_flat = depth_dir / f"{frame.image_name}.npz"
        if use_precomputed_depth and depth_npz_cam.exists():
            depth = np.load(str(depth_npz_cam))["depth"]
        elif use_precomputed_depth and depth_npz_flat.exists():
            depth = np.load(str(depth_npz_flat))["depth"]
        elif depth_model is not None:
            image = Image.open(frame.image_path).convert("RGB")
            inputs = depth_processor(images=[image], return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = depth_model(**inputs)
            post = depth_processor.post_process_depth_estimation(
                outputs, target_sizes=[(image.height, image.width)]
            )
            depth = post[0]["predicted_depth"].cpu().numpy().astype(np.float32)
        else:
            print(f"Skipping {frame.image_name}: no depth available")
            continue

        depth_smooth = cv2.GaussianBlur(depth, (5, 5), 1.0)
        normal = depth_to_normal(depth_smooth, K)

        cam_out_dir = output_dir / ch
        bgr = normal_to_png_bgr(normal)
        cv2.imwrite(str(cam_out_dir / f"{frame.image_name}.png"), bgr)

        vis = ((normal + 1) * 0.5 * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(vis).save(str(cam_out_dir / f"{frame.image_name}.jpg"), quality=90)

    print(f"Done. Normal maps saved to {output_dir}")


if __name__ == "__main__":
    main()
