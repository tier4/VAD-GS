"""Visualize preprocessed T4 data as per-item per-camera MP4 videos.

Generates videos for each preprocessing item (lidar_depth, mono_depth,
sky_masks, sam_masks, sam_bkgd_masks, normal_img) and each camera channel,
plus an additional "original" video of raw images for comparison.

Videos are saved under <output_dir>/preprocess_vis/<item>/<camera>.mp4.

Usage:
    python script/t4/visualize_preprocess.py --config configs/example/t4_train_example.yaml
    python script/t4/visualize_preprocess.py --dataroot 835afe23-... --scene-index 0
    python script/t4/visualize_preprocess.py --config configs/example/t4_train_example.yaml --fps 10
"""

import argparse
import json
import os
from pathlib import Path

import cv2
import imageio
import numpy as np
from tqdm import tqdm

from config_utils import add_config_arg, apply_config_defaults, resolve_dataroot


# ---------------------------------------------------------------------------
# T4 helpers
# ---------------------------------------------------------------------------

def load_t4_tables(annotation_dir):
    tables = {}
    for name in ["scene", "sample", "sample_data", "calibrated_sensor", "sensor"]:
        path = os.path.join(str(annotation_dir), f"{name}.json")
        if os.path.exists(path):
            with open(path) as f:
                tables[name] = json.load(f)
        else:
            tables[name] = []
    return tables


def _index(records):
    return {r["token"]: r for r in records}


def collect_frame_paths(dataroot, scene_index, camera_channels, tables):
    """Collect per-camera ordered lists of (image_path, image_name) for a scene.

    Returns dict: camera_channel -> [(image_path, image_name), ...]
    """
    sample_by_token = _index(tables["sample"])
    cs_by_token = _index(tables["calibrated_sensor"])
    sensor_by_token = _index(tables["sensor"])

    scene = tables["scene"][scene_index]

    # Build ordered sample chain
    samples = []
    token = scene["first_sample_token"]
    while token:
        samples.append(sample_by_token[token])
        token = sample_by_token[token].get("next", "")

    # Map sample_token -> {channel: sample_data} (keyframes only)
    sd_by_sample = {}
    for sd in tables["sample_data"]:
        if not sd.get("is_key_frame", False):
            continue
        cs = cs_by_token.get(sd["calibrated_sensor_token"])
        if cs is None:
            continue
        sensor = sensor_by_token.get(cs["sensor_token"])
        if sensor is None or sensor.get("modality") != "camera":
            continue
        sd_by_sample.setdefault(sd["sample_token"], {})[sensor["channel"]] = sd

    # Auto-detect cameras if not specified
    if camera_channels is None:
        all_ch = set()
        for sensor in tables["sensor"]:
            if sensor.get("modality") == "camera":
                all_ch.add(sensor["channel"])
        camera_channels = sorted(all_ch)

    result = {ch: [] for ch in camera_channels}
    for sample in samples:
        frame_data = sd_by_sample.get(sample["token"], {})
        for ch in camera_channels:
            if ch in frame_data:
                rel_path = frame_data[ch]["filename"]
                abs_path = dataroot / rel_path
                image_name = Path(rel_path).stem
                result[ch].append((str(abs_path), image_name))

    return result


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def colorize_depth(depth, cmap=cv2.COLORMAP_TURBO):
    """Convert a float32 depth map to a colored uint8 RGB image."""
    x = np.nan_to_num(depth)
    mask = x > 0
    if mask.any():
        mi = np.min(x[mask])
        ma = np.max(x)
    else:
        mi, ma = 0.0, 1.0
    x = (x - mi) / (ma - mi + 1e-8)
    x = (255 * x).clip(0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(x, cmap)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def load_frame_as_rgb(item_name, file_path):
    """Load a preprocessed file and return an RGB uint8 image.

    Handles .npy (lidar_depth), .npz (mono_depth), and .png (masks/normals).
    """
    ext = Path(file_path).suffix.lower()

    if ext == ".npy":
        # lidar_depth: dict-like np array with 'value' and 'mask' keys
        data = np.load(file_path, allow_pickle=True).item()
        depth = data["value"].astype(np.float32)
        mask = data["mask"].astype(bool)
        depth[~mask] = 0
        return colorize_depth(depth)

    elif ext == ".npz":
        # mono_depth: compressed float32 depth
        data = np.load(file_path)
        key = list(data.keys())[0]  # usually 'arr_0' or 'depth'
        depth = data[key].astype(np.float32)
        return colorize_depth(depth)

    elif ext == ".png":
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None

        if item_name == "sky_masks":
            # Binary mask (single channel, 0 or 255) -> green overlay
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            rgb = np.zeros((*img.shape[:2], 3), dtype=np.uint8)
            rgb[..., 0] = 100  # dim background
            rgb[..., 1] = np.where(img > 127, 255, 100)
            rgb[..., 2] = 100
            return rgb

        elif item_name == "sam_masks":
            # Dynamic object masks: uint8 IDs (255=background)
            if img.ndim == 3:
                img = img[:, :, 0]
            rgb = np.full((*img.shape[:2], 3), 40, dtype=np.uint8)
            unique_ids = np.unique(img)
            colors = _generate_colors(len(unique_ids))
            for idx, obj_id in enumerate(unique_ids):
                if obj_id == 255:
                    continue
                mask = img == obj_id
                rgb[mask] = colors[idx]
            return rgb

        elif item_name == "sam_bkgd_masks":
            # BGR segmentation -> RGB
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        elif item_name == "normal_img":
            # Normal map stored as BGR PNG -> RGB
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        else:
            # Generic PNG
            if img.ndim == 2:
                return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return None


def _generate_colors(n):
    """Generate n visually distinct colors."""
    colors = []
    for i in range(n):
        hue = int(180 * i / max(n, 1))
        hsv = np.uint8([[[hue, 200, 230]]])
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0]
        colors.append(rgb)
    return colors


def add_label(img, text, font_scale=0.7, thickness=2):
    """Add a text label to the top-left of an image (in-place)."""
    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    return img


# ---------------------------------------------------------------------------
# Item definitions
# ---------------------------------------------------------------------------

ITEMS = [
    # (item_name, preprocessed_subdir, extension)
    ("original",       None,              None),
    ("lidar_depth",    "lidar_depth",     ".npy"),
    ("mono_depth",     "depth",           ".npz"),
    ("sky_masks",      "sky_masks",       ".png"),
    ("sam_masks",      "sam_masks",       ".png"),
    ("sam_bkgd_masks", "sam_bkgd_masks",  ".png"),
    ("normal_img",     "normal_img",      ".png"),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualize preprocessed T4 data as MP4 videos"
    )
    add_config_arg(parser)
    parser.add_argument("--dataroot", type=str, default=None, help="Dataset UUID or path")
    parser.add_argument("--revision", type=int, default=None)
    parser.add_argument("--scene-index", type=int, default=None)
    parser.add_argument("--camera-channels", nargs="+", default=None)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (default: <dataroot>/preprocess_vis)")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--items", nargs="+", default=None,
                        help="Items to visualize (default: all). "
                             "Choices: original, lidar_depth, mono_depth, sky_masks, "
                             "sam_masks, sam_bkgd_masks, normal_img")
    parser.add_argument("--max-width", type=int, default=960,
                        help="Max frame width for videos (resize if larger)")
    args = parser.parse_args()

    # Apply config defaults
    apply_config_defaults(args)
    if args.dataroot is None:
        parser.error("--dataroot is required (provide via --config or CLI)")
    if args.revision is None:
        args.revision = 0
    if args.scene_index is None:
        args.scene_index = 0

    dataroot = resolve_dataroot(args.dataroot, revision=args.revision)
    print(f"Resolved dataroot: {dataroot}")

    output_dir = args.output_dir or (dataroot / "preprocess_vis")

    # Load T4 tables
    annotation_dir = dataroot / "annotation"
    if not annotation_dir.exists():
        raise FileNotFoundError(f"No annotation directory found at {annotation_dir}")

    tables = load_t4_tables(annotation_dir)
    frame_paths = collect_frame_paths(
        dataroot, args.scene_index, args.camera_channels, tables
    )

    cameras = list(frame_paths.keys())
    if not cameras:
        print("No camera data found.")
        return

    print(f"Cameras: {cameras}")
    for cam in cameras:
        print(f"  {cam}: {len(frame_paths[cam])} frames")

    prep = dataroot / "preprocessed"

    # Filter items
    selected_items = args.items
    items_to_process = []
    for item_name, subdir, ext in ITEMS:
        if selected_items and item_name not in selected_items:
            continue
        if subdir is not None and not (prep / subdir).exists():
            print(f"[SKIP] {item_name}: directory not found at {prep / subdir}")
            continue
        items_to_process.append((item_name, subdir, ext))

    if not items_to_process:
        print("No items to visualize.")
        return

    # Generate videos
    total_videos = 0
    for item_name, subdir, ext in items_to_process:
        for cam in cameras:
            frames_info = frame_paths[cam]
            if not frames_info:
                continue

            frames = []
            for image_path, image_name in tqdm(
                frames_info,
                desc=f"{item_name}/{cam}",
                leave=False,
            ):
                if item_name == "original":
                    # Load the raw image
                    if not os.path.exists(image_path):
                        continue
                    img = cv2.imread(image_path)
                    if img is None:
                        continue
                    frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                else:
                    # Find the preprocessed file
                    file_path = prep / subdir / cam / f"{image_name}{ext}"
                    if not file_path.exists():
                        # Try .png fallback for mono_depth
                        if ext == ".npz":
                            file_path = prep / subdir / cam / f"{image_name}.png"
                        if not file_path.exists():
                            continue
                    frame = load_frame_as_rgb(item_name, str(file_path))
                    if frame is None:
                        continue

                # Resize if needed
                h, w = frame.shape[:2]
                if w > args.max_width:
                    scale = args.max_width / w
                    new_w = args.max_width
                    new_h = int(h * scale)
                    # Ensure even dimensions for video codec
                    new_h = new_h - (new_h % 2)
                    new_w = new_w - (new_w % 2)
                    frame = cv2.resize(frame, (new_w, new_h))
                else:
                    # Ensure even dimensions
                    new_h = h - (h % 2)
                    new_w = w - (w % 2)
                    if new_h != h or new_w != w:
                        frame = frame[:new_h, :new_w]

                # Add label
                add_label(frame, f"{item_name} / {cam}")
                frames.append(frame)

            if not frames:
                continue

            out_dir = output_dir / item_name
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{cam}.mp4"
            imageio.mimwrite(str(out_path), frames, fps=args.fps)
            total_videos += 1
            print(f"  Saved {out_path} ({len(frames)} frames)")

    print(f"\nDone: {total_videos} videos saved to {output_dir}")


if __name__ == "__main__":
    main()
