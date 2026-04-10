"""Generate monocular surface normal maps for T4 datasets.

Uses Depth Anything V2 depth + gradient-based normal estimation to produce
surface normal maps in the format expected by VAD-GS.

The normals are computed from depth gradients rather than a dedicated normal
estimation model, avoiding additional model dependencies.

Usage:
    python script/t4/generate_normal_maps.py --config configs/example/t4_train_example.yaml
    python script/t4/generate_normal_maps.py --dataroot caf37e66-... --scene-index 0 --batch-size 4
"""

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from config_utils import add_config_arg, apply_config_defaults, resolve_dataroot


def load_t4_tables(annotation_dir):
    tables = {}
    for name in ["scene", "sample", "sample_data", "calibrated_sensor", "sensor"]:
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


def depth_to_normal(depth, K=None):
    """Compute surface normals from depth map using finite differences.

    Args:
        depth: (H, W) numpy array of depth values
        K: (3, 3) camera intrinsic matrix (optional, uses pixel coords if None)

    Returns:
        normal: (H, W, 3) numpy array of unit normals in camera space [nx, ny, nz]
    """
    h, w = depth.shape

    if K is not None:
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
    else:
        fx = fy = max(h, w)
        cx, cy = w / 2.0, h / 2.0

    # Create pixel coordinate grids
    u = np.arange(w, dtype=np.float32)
    v = np.arange(h, dtype=np.float32)
    u, v = np.meshgrid(u, v)

    # Back-project to 3D
    z = depth.astype(np.float64)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    # Compute gradients
    dzdx = np.gradient(z, axis=1)
    dzdy = np.gradient(z, axis=0)
    dxdx = np.gradient(x, axis=1)
    dxdy = np.gradient(x, axis=0)
    dydy = np.gradient(y, axis=0)
    dydx = np.gradient(y, axis=1)

    # Cross product of partial derivatives
    # tangent_u = (dxdx, dydx, dzdx), tangent_v = (dxdy, dydy, dzdy)
    nx = dydx * dzdy - dzdx * dydy
    ny = dzdx * dxdy - dxdx * dzdy
    nz = dxdx * dydy - dydx * dxdy

    norm = np.sqrt(nx**2 + ny**2 + nz**2) + 1e-10
    nx /= norm
    ny /= norm
    nz /= norm

    # Ensure normals point toward camera (nz should be negative in camera coords)
    flip = nz > 0
    nx[flip] = -nx[flip]
    ny[flip] = -ny[flip]
    nz[flip] = -nz[flip]

    normal = np.stack([nx, ny, nz], axis=-1).astype(np.float32)
    return normal


def normal_to_png_bgr(normal):
    """Convert camera-space normals to BGR PNG format matching VAD-GS loading convention.

    The reader does:
        tmp = cv2.imread(...) / 255 * 2 - 1  (BGR, [-1,1])
        ref_norm[:,:,0] = -tmp[:,:,2]  (R channel, negated)
        ref_norm[:,:,1] = -tmp[:,:,1]  (G channel, negated)
        ref_norm[:,:,2] = -tmp[:,:,0]  (B channel, negated)

    So ref_norm = [-R, -G, -B] where RGB = tmp in RGB order.
    Given normal [nx, ny, nz]:
        nx = -tmp_R => tmp_R = -nx => PNG_R = (-nx + 1) / 2 * 255
        ny = -tmp_G => tmp_G = -ny => PNG_G = (-ny + 1) / 2 * 255
        nz = -tmp_B => tmp_B = -nz => PNG_B = (-nz + 1) / 2 * 255
    """
    nx, ny, nz = normal[:, :, 0], normal[:, :, 1], normal[:, :, 2]
    png_r = ((-nx + 1) * 0.5 * 255).clip(0, 255).astype(np.uint8)
    png_g = ((-ny + 1) * 0.5 * 255).clip(0, 255).astype(np.uint8)
    png_b = ((-nz + 1) * 0.5 * 255).clip(0, 255).astype(np.uint8)
    bgr = np.stack([png_b, png_g, png_r], axis=-1)
    return bgr


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
                        help="Pre-computed depth dir (default: <dataroot>/depth)")
    args = parser.parse_args()

    # Apply config defaults, then hard defaults
    apply_config_defaults(args)
    if args.dataroot is None:
        parser.error("--dataroot is required (provide via --config or CLI)")
    if args.revision is None:
        args.revision = 0
    if args.scene_index is None:
        args.scene_index = 0

    dataroot = resolve_dataroot(args.dataroot, revision=args.revision)
    print(f"Resolved dataroot: {dataroot}")
    output_dir = args.output_dir or (dataroot / "preprocessed" / "normal_img")
    output_dir.mkdir(parents=True, exist_ok=True)
    depth_dir = args.depth_dir or (dataroot / "preprocessed" / "depth")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Load T4 annotations
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

    scene = tables["scene"][args.scene_index]
    samples = sample_chain(scene, sample_by_token)

    # Keyframes only to ensure correct sample-annotation alignment
    sd_by_sample = {}
    for sd in tables["sample_data"]:
        if not sd.get("is_key_frame", False):
            continue
        cs = calibrated_sensor_by_token.get(sd["calibrated_sensor_token"])
        if cs is None:
            continue
        sensor = sensor_by_token.get(cs["sensor_token"])
        if sensor is None or sensor.get("modality") != "camera":
            continue
        sd_by_sample.setdefault(sd["sample_token"], {})[sensor["channel"]] = (sd, cs)

    camera_channels = args.camera_channels
    if camera_channels is None:
        all_channels = set()
        for sensor in tables["sensor"]:
            if sensor.get("modality") == "camera":
                all_channels.add(sensor["channel"])
        camera_channels = sorted(all_channels)

    # Collect entries
    entries = []
    for sample in samples:
        frame_data = sd_by_sample.get(sample["token"], {})
        for ch in camera_channels:
            if ch not in frame_data:
                continue
            sd, cs = frame_data[ch]
            image_path = dataroot / sd["filename"]
            if not image_path.exists():
                continue
            image_name = image_path.stem
            cam_out_dir = output_dir / ch
            cam_out_dir.mkdir(parents=True, exist_ok=True)
            save_path = cam_out_dir / f"{image_name}.png"
            if args.skip_existing and save_path.exists():
                continue
            K = np.array(cs["camera_intrinsic"], dtype=np.float64)
            entries.append((str(image_path), image_name, K, ch))

    print(f"Images to process: {len(entries)}")
    if not entries:
        print("Nothing to do.")
        return

    # Check if pre-computed depth maps exist
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

    batch_size = args.batch_size
    for batch_start in tqdm(range(0, len(entries), batch_size), desc="Normal estimation"):
        batch = entries[batch_start:batch_start + batch_size]

        for image_path, image_name, K, ch in batch:
            # Get depth (try camera subdirectory first, then flat)
            depth_npz_cam = depth_dir / ch / f"{image_name}.npz"
            depth_npz_flat = depth_dir / f"{image_name}.npz"
            if use_precomputed_depth and depth_npz_cam.exists():
                depth = np.load(str(depth_npz_cam))["depth"]
            elif use_precomputed_depth and depth_npz_flat.exists():
                depth = np.load(str(depth_npz_flat))["depth"]
            elif depth_model is not None:
                image = Image.open(image_path).convert("RGB")
                inputs = depth_processor(images=[image], return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = depth_model(**inputs)
                post = depth_processor.post_process_depth_estimation(
                    outputs, target_sizes=[(image.height, image.width)]
                )
                depth = post[0]["predicted_depth"].cpu().numpy().astype(np.float32)
            else:
                print(f"Skipping {image_name}: no depth available")
                continue

            # Smooth depth slightly to reduce noise in normals
            depth_smooth = cv2.GaussianBlur(depth, (5, 5), 1.0)

            # Compute normals
            normal = depth_to_normal(depth_smooth, K)

            cam_out_dir = output_dir / ch
            # Save as BGR PNG
            bgr = normal_to_png_bgr(normal)
            cv2.imwrite(str(cam_out_dir / f"{image_name}.png"), bgr)

            # Save visualization JPEG
            vis = ((normal + 1) * 0.5 * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(vis).save(str(cam_out_dir / f"{image_name}.jpg"), quality=90)

    print(f"Done. Normal maps saved to {output_dir}")


if __name__ == "__main__":
    main()
