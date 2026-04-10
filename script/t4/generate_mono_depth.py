"""Generate monocular depth maps using Depth Anything V2 Small for T4 datasets.

Loads images from T4 annotation files, runs inference with
depth-anything/Depth-Anything-V2-Small-hf, and caches the predicted
depth as .npz files (float32 numpy arrays).

Usage:
    python script/t4/generate_mono_depth.py \
        --dataroot /path/to/t4_dataset \
        --scene-index 0 \
        --camera-channels CAM_FRONT CAM_FRONT_LEFT CAM_FRONT_RIGHT \
                          CAM_BACK_LEFT CAM_BACK_RIGHT \
        --batch-size 4
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

_ANNOTATION_DATASET_BASE = os.path.expanduser("~/.webauto/data/data/annotation_dataset")


def resolve_dataroot(dataset_id_or_path, revision=0):
    """Resolve a dataset UUID or path to an absolute directory (same logic as cfg_utils)."""
    candidate = os.path.expanduser(str(dataset_id_or_path))
    if os.path.isdir(candidate) and os.path.isdir(os.path.join(candidate, "annotation")):
        return Path(candidate)
    id_path = os.path.join(_ANNOTATION_DATASET_BASE, str(dataset_id_or_path))
    if os.path.isdir(id_path):
        rev_path = os.path.join(id_path, str(revision))
        if os.path.isdir(rev_path):
            return Path(rev_path)
        return Path(id_path)
    return Path(candidate)


# ---------------------------------------------------------------------------
# T4 helpers (duplicated from generate_lidar_depth.py to keep standalone)
# ---------------------------------------------------------------------------

def load_t4_tables(annotation_dir):
    tables = {}
    for name in [
        "scene", "sample", "sample_data",
        "calibrated_sensor", "sensor",
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate monocular depth maps (Depth Anything V2 Small) for T4 dataset"
    )
    parser.add_argument("--dataroot", type=str, required=True,
                        help="Dataset UUID or path")
    parser.add_argument("--revision", type=int, default=0,
                        help="Sub-directory index under the dataset ID (default: 0)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (default: <dataroot>/depth)")
    parser.add_argument("--scene-index", type=int, default=0)
    parser.add_argument(
        "--camera-channels", nargs="+", default=None,
        help="Camera channel names (auto-detected if omitted)",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", type=str, default=None,
                        help="Device (default: cuda if available)")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="Skip images that already have depth (default: True)")
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    args = parser.parse_args()

    dataroot = resolve_dataroot(args.dataroot, revision=args.revision)
    print(f"Resolved dataroot: {dataroot}")
    output_dir = args.output_dir or (dataroot / "preprocessed" / "depth")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # -- Load T4 annotations --------------------------------------------------
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
    print(f"Scene: {scene.get('name', 'unknown')}, {len(samples)} samples")

    # Build sample_data -> channel map (keyframes only)
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
        if sensor.get("modality") != "camera":
            continue
        sd_by_sample.setdefault(sd["sample_token"], {})[sensor["channel"]] = sd

    # Auto-detect camera channels if not specified
    camera_channels = args.camera_channels
    if camera_channels is None:
        all_channels = set()
        for sensor in tables["sensor"]:
            if sensor.get("modality") == "camera":
                all_channels.add(sensor["channel"])
        camera_channels = sorted(all_channels)
        print(f"Auto-detected camera channels: {camera_channels}")

    # -- Collect image paths ---------------------------------------------------
    image_entries = []  # (image_path, image_name, camera_channel)
    for sample in samples:
        frame_data = sd_by_sample.get(sample["token"], {})
        for ch in camera_channels:
            if ch not in frame_data:
                continue
            sd = frame_data[ch]
            image_path = dataroot / sd["filename"]
            if not image_path.exists():
                continue
            image_name = image_path.stem
            cam_out_dir = output_dir / ch
            cam_out_dir.mkdir(parents=True, exist_ok=True)
            save_path = cam_out_dir / f"{image_name}.npz"
            if args.skip_existing and save_path.exists():
                continue
            image_entries.append((str(image_path), image_name, ch))

    print(f"Images to process: {len(image_entries)}")
    if len(image_entries) == 0:
        print("Nothing to do.")
        return

    # -- Load model ------------------------------------------------------------
    model_id = "depth-anything/Depth-Anything-V2-Small-hf"
    print(f"Loading model: {model_id}")
    image_processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device)
    model.eval()

    # -- Run inference ---------------------------------------------------------
    batch_size = args.batch_size
    for batch_start in tqdm(range(0, len(image_entries), batch_size), desc="Depth inference"):
        batch = image_entries[batch_start:batch_start + batch_size]
        images = [Image.open(p).convert("RGB") for p, _, _ in batch]
        original_sizes = [(img.height, img.width) for img in images]

        inputs = image_processor(images=images, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        post_processed = image_processor.post_process_depth_estimation(
            outputs,
            target_sizes=original_sizes,
        )

        for i, (_, image_name, ch) in enumerate(batch):
            predicted_depth = post_processed[i]["predicted_depth"]  # (H, W) tensor
            depth_np = predicted_depth.cpu().numpy().astype(np.float32)
            cam_out_dir = output_dir / ch
            save_path = cam_out_dir / f"{image_name}.npz"
            np.savez_compressed(str(save_path), depth=depth_np)
            # Save visualization as JPEG
            d_min, d_max = depth_np.min(), depth_np.max()
            if d_max - d_min > 1e-6:
                depth_vis = ((depth_np - d_min) / (d_max - d_min) * 255).astype(np.uint8)
            else:
                depth_vis = np.zeros_like(depth_np, dtype=np.uint8)
            Image.fromarray(depth_vis).save(str(cam_out_dir / f"{image_name}.jpg"), quality=90)

    print(f"Done. Depth maps saved to {output_dir}")


if __name__ == "__main__":
    main()
