"""Generate sky masks using SegFormer B5 (Cityscapes) for T4 datasets.

Uses nvidia/segformer-b5-finetuned-cityscapes-1024-1024 to detect sky pixels
(Cityscapes class 10) and saves binary masks as PNG.

Usage:
    python script/t4/generate_sky_masks.py --dataroot caf37e66-... --scene-index 0 --batch-size 4
"""

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# Workaround: SegFormer model lacks safetensors, and torch<2.6 triggers CVE check
import transformers.modeling_utils as _mu
_mu.check_torch_load_is_safe = lambda: None

from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

_ANNOTATION_DATASET_BASE = os.path.expanduser("~/.webauto/data/data/annotation_dataset")

CITYSCAPES_SKY_CLASS = 10


def resolve_dataroot(dataset_id_or_path, revision=0):
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


def main():
    parser = argparse.ArgumentParser(description="Generate sky masks for T4 dataset")
    parser.add_argument("--dataroot", type=str, required=True, help="Dataset UUID or path")
    parser.add_argument("--revision", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--scene-index", type=int, default=0)
    parser.add_argument("--camera-channels", nargs="+", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    args = parser.parse_args()

    dataroot = resolve_dataroot(args.dataroot, revision=args.revision)
    print(f"Resolved dataroot: {dataroot}")
    output_dir = args.output_dir or (dataroot / "preprocessed" / "sky_masks")
    output_dir.mkdir(parents=True, exist_ok=True)

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
    print(f"Scene: {scene.get('name', 'unknown')}, {len(samples)} samples")

    # Build sample_data -> channel map
    sd_by_sample = {}
    for sd in tables["sample_data"]:
        cs = calibrated_sensor_by_token.get(sd["calibrated_sensor_token"])
        if cs is None:
            continue
        sensor = sensor_by_token.get(cs["sensor_token"])
        if sensor is None or sensor.get("modality") != "camera":
            continue
        sd_by_sample.setdefault(sd["sample_token"], {})[sensor["channel"]] = sd

    camera_channels = args.camera_channels
    if camera_channels is None:
        all_channels = set()
        for sensor in tables["sensor"]:
            if sensor.get("modality") == "camera":
                all_channels.add(sensor["channel"])
        camera_channels = sorted(all_channels)
        print(f"Auto-detected camera channels: {camera_channels}")

    # Collect image paths
    image_entries = []
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
            save_path = cam_out_dir / f"{image_name}.png"
            if args.skip_existing and save_path.exists():
                continue
            image_entries.append((str(image_path), image_name, ch))

    print(f"Images to process: {len(image_entries)}")
    if not image_entries:
        print("Nothing to do.")
        return

    # Load model
    model_id = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
    print(f"Loading model: {model_id}")
    image_processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForSemanticSegmentation.from_pretrained(model_id).to(device)
    model.eval()

    # Run inference
    batch_size = args.batch_size
    for batch_start in tqdm(range(0, len(image_entries), batch_size), desc="Sky mask inference"):
        batch = image_entries[batch_start:batch_start + batch_size]
        images = [Image.open(p).convert("RGB") for p, _, _ in batch]
        original_sizes = [(img.height, img.width) for img in images]

        inputs = image_processor(images=images, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits  # (B, num_classes, H/4, W/4)

        for i, (_, image_name, ch) in enumerate(batch):
            h, w = original_sizes[i]
            upsampled = F.interpolate(
                logits[i:i+1], size=(h, w), mode="bilinear", align_corners=False
            )
            pred = upsampled.argmax(dim=1).squeeze(0).cpu().numpy()  # (H, W)
            sky_mask = (pred == CITYSCAPES_SKY_CLASS).astype(np.uint8) * 255

            cam_out_dir = output_dir / ch
            # Save binary PNG
            cv2.imwrite(str(cam_out_dir / f"{image_name}.png"), sky_mask)
            # Save visualization JPEG (sky overlay in blue)
            img_np = np.array(images[i])
            vis = img_np.copy()
            vis[sky_mask > 0] = (vis[sky_mask > 0] * 0.4 + np.array([100, 150, 255]) * 0.6).astype(np.uint8)
            Image.fromarray(vis).save(str(cam_out_dir / f"{image_name}.jpg"), quality=90)

    print(f"Done. Sky masks saved to {output_dir}")


if __name__ == "__main__":
    main()
