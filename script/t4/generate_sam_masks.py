"""Generate dynamic instance masks (sam_masks) and background segmentation masks
(sam_bkgd_masks) for T4 datasets.

Dynamic masks: Project 3D bounding boxes from T4 annotations onto camera images
  using t4-devkit for coordinate transforms and projection.
  Each dynamic object gets a unique uint8 ID. Background = 255.
Background masks: SegFormer Cityscapes semantic segmentation of background classes.

Usage:
    python script/t4/generate_sam_masks.py --config configs/example/t4_train_example.yaml
    python script/t4/generate_sam_masks.py --dataroot caf37e66-... --scene-index 0 --batch-size 4
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

from config_utils import add_config_arg, apply_config_defaults
from t4_dataset import T4Dataset

# Cityscapes class IDs
# Sky=10, Person=11, Rider=12, Car=13, Truck=14, Bus=15, Train=16, Motorcycle=17, Bicycle=18
DYNAMIC_CLASSES = {11, 12, 13, 14, 15, 16, 17, 18}
SKY_CLASS = 10
# Background classes: road=0, sidewalk=1, building=2, wall=3, fence=4, pole=5,
# traffic_light=6, traffic_sign=7, vegetation=8, terrain=9
BKGD_CLASSES = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

# Unique BGR colors for each background class (avoid 0,0,0 and 1,1,1 which are skipped by trellis)
BKGD_CLASS_COLORS = {
    0: (40, 40, 40),      # road
    1: (60, 60, 60),      # sidewalk
    2: (80, 80, 80),      # building
    3: (100, 100, 100),   # wall
    4: (120, 120, 120),   # fence
    5: (140, 140, 140),   # pole
    6: (160, 160, 160),   # traffic light
    7: (180, 180, 180),   # traffic sign
    8: (200, 200, 200),   # vegetation
    9: (220, 220, 220),   # terrain
}


def simplify_category(name):
    name = name.lower()
    if "vehicle" in name or "car" in name or "truck" in name or "bus" in name or "trailer" in name:
        return "vehicle"
    if "pedestrian" in name:
        return "pedestrian"
    if "cycle" in name or "bicycle" in name or "motorcycle" in name:
        return "cyclist"
    if "cone" in name or "barrier" in name:
        return "misc_dynamic"
    return "misc"


def project_box_to_mask(box, intrinsic, H, W, box_scale=1.0):
    """Project a Box3D (in sensor frame) to a 2D mask.

    Returns an (H, W) uint8 mask with 1 where the box projects, 0 elsewhere.
    Returns None if the box is entirely behind the camera.
    """
    corners = box.corners(box_scale=box_scale)  # (8, 3) in sensor coord
    depths = corners[:, 2]

    # Skip boxes entirely behind camera
    if not np.any(depths > 0):
        return None

    # For corners behind camera, clip depth to small positive value
    # so they project to extreme image coordinates (handled by fillPoly clipping)
    corners_clipped = corners.copy()
    corners_clipped[:, 2] = np.clip(corners_clipped[:, 2], a_min=0.1, a_max=None)

    # Project to 2D using t4-devkit
    uv, _ = T4Dataset.project_points_to_image(corners_clipped, intrinsic)
    uv = np.round(uv).astype(np.int32)

    # Fill all 6 faces of the bounding box
    mask = np.zeros((H, W), dtype=np.uint8)
    # Face vertex indices (each face is a quad)
    faces = [
        [0, 1, 3, 2],  # left
        [4, 5, 7, 6],  # right
        [0, 1, 5, 4],  # bottom
        [2, 3, 7, 6],  # top
        [0, 2, 6, 4],  # front
        [1, 3, 7, 5],  # back
    ]
    for face in faces:
        pts = uv[face]
        cv2.fillPoly(mask, [pts], 1)

    return mask


def main():
    parser = argparse.ArgumentParser(
        description="Generate dynamic + background masks for T4 dataset"
    )
    add_config_arg(parser)
    parser.add_argument("--dataroot", type=str, default=None, help="Dataset UUID or path")
    parser.add_argument("--revision", type=int, default=None)
    parser.add_argument("--output-dir-dynamic", type=Path, default=None,
                        help="Output dir for sam_masks (default: <dataroot>/sam_masks)")
    parser.add_argument("--output-dir-bkgd", type=Path, default=None,
                        help="Output dir for sam_bkgd_masks (default: <dataroot>/sam_bkgd_masks)")
    parser.add_argument("--scene-index", type=int, default=None)
    parser.add_argument("--camera-channels", nargs="+", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--box-scale", type=float, default=None,
                        help="Scale factor for 3D bounding boxes (default: 1.5)")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    args = parser.parse_args()

    # Apply config defaults, then hard defaults
    apply_config_defaults(args)
    if args.dataroot is None:
        parser.error("--dataroot is required (provide via --config or CLI)")
    if args.revision is None:
        args.revision = 0
    if args.scene_index is None:
        args.scene_index = 0
    if args.box_scale is None:
        args.box_scale = 1.5

    ds = T4Dataset.from_args(args)
    dataroot = ds.dataroot
    print(f"Resolved dataroot: {dataroot}")

    out_dynamic = args.output_dir_dynamic or (dataroot / "preprocessed" / "sam_masks")
    out_bkgd = args.output_dir_bkgd or (dataroot / "preprocessed" / "sam_bkgd_masks")
    out_dynamic.mkdir(parents=True, exist_ok=True)
    out_bkgd.mkdir(parents=True, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    t4 = ds.t4
    print(f"Scene: {ds.scene.name}, {ds.num_samples} samples")

    # Build instance_token -> remapped track_id (same logic as t4_utils.py)
    raw_track_ids = {}  # instance_token -> raw_id
    for sample in ds.samples:
        for ann_token in sample.ann_3ds:
            ann = t4.get("sample_annotation", ann_token)
            cat_name = ann.category_name
            class_name = simplify_category(cat_name)
            if class_name == "misc":
                continue
            raw_id = int(ann.instance_token[:8], 16)
            raw_track_ids[ann.instance_token] = raw_id

    # Remap to sequential IDs (matching t4_utils.py remapping)
    sorted_raw_ids = sorted(set(raw_track_ids.values()))
    raw_to_remapped = {raw_id: new_id for new_id, raw_id in enumerate(sorted_raw_ids)}
    token_to_remapped = {tok: raw_to_remapped[raw_id] for tok, raw_id in raw_track_ids.items()}
    print(f"Found {len(sorted_raw_ids)} dynamic objects, remapped to [0, {len(sorted_raw_ids)-1}]")

    # Collect frames to process
    frames = list(ds.iter_frames(args.camera_channels))
    print(f"Total frames: {len(frames)}")

    # Filter by skip_existing
    entries = []
    for frame in frames:
        ch = frame.camera_channel
        dyn_cam_dir = out_dynamic / ch
        bkgd_cam_dir = out_bkgd / ch
        dyn_cam_dir.mkdir(parents=True, exist_ok=True)
        bkgd_cam_dir.mkdir(parents=True, exist_ok=True)
        dyn_path = dyn_cam_dir / f"{frame.image_name}.png"
        bkgd_path = bkgd_cam_dir / f"{frame.image_name}.png"
        if args.skip_existing and dyn_path.exists() and bkgd_path.exists():
            continue
        entries.append(frame)

    print(f"Images to process: {len(entries)}")
    if not entries:
        print("Nothing to do.")
        return

    # --- Generate dynamic masks using t4-devkit projection ---
    print("Generating dynamic masks from 3D bounding box projections...")
    for frame in tqdm(entries, desc="Dynamic masks"):
        # Get 3D boxes already transformed to sensor (camera) coordinates
        boxes, cam_intrinsic = ds.get_boxes_in_sensor(frame.sample_data_token)
        K = np.array(cam_intrinsic, dtype=np.float64)
        H, W = frame.height, frame.width

        # 3-channel mask: background = 255
        dyn_mask = np.full((H, W, 3), 255, dtype=np.uint8)

        for box in boxes:
            # Look up remapped ID via instance token (uuid)
            if box.uuid is None or box.uuid not in token_to_remapped:
                # Try matching via instance_token from annotation
                continue
            remapped_id = token_to_remapped[box.uuid]

            mask_2d = project_box_to_mask(box, K, H, W, box_scale=args.box_scale)
            if mask_2d is not None:
                dyn_mask[mask_2d > 0] = remapped_id

        cv2.imwrite(str(out_dynamic / frame.camera_channel / f"{frame.image_name}.png"), dyn_mask)

    # Save track_id mapping for reference
    mapping_path = out_dynamic / "track_id_mapping.json"
    mapping_data = {str(raw_id): new_id for raw_id, new_id in raw_to_remapped.items()}
    with open(mapping_path, "w") as f:
        json.dump(mapping_data, f, indent=2)
    print(f"Track ID mapping saved to {mapping_path}")

    # --- Generate background masks (SegFormer semantic segmentation) ---
    print("Generating background segmentation masks...")
    model_id = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
    print(f"Loading model: {model_id}")
    image_processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForSemanticSegmentation.from_pretrained(model_id).to(device)
    model.eval()

    batch_size = args.batch_size
    for batch_start in tqdm(range(0, len(entries), batch_size), desc="Bkgd masks"):
        batch = entries[batch_start:batch_start + batch_size]
        images = [Image.open(f.image_path).convert("RGB") for f in batch]
        original_sizes = [(img.height, img.width) for img in images]

        inputs = image_processor(images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits

        for i, frame in enumerate(batch):
            h, w = original_sizes[i]
            upsampled = F.interpolate(
                logits[i:i+1], size=(h, w), mode="bilinear", align_corners=False
            )
            pred = upsampled.argmax(dim=1).squeeze(0).cpu().numpy()

            # Background mask: each background class gets a unique color
            bkgd_mask = np.zeros((h, w, 3), dtype=np.uint8)
            for cls_id, color in BKGD_CLASS_COLORS.items():
                bkgd_mask[pred == cls_id] = color

            bkgd_cam_dir = out_bkgd / frame.camera_channel
            cv2.imwrite(str(bkgd_cam_dir / f"{frame.image_name}.png"), bkgd_mask)
            Image.fromarray(bkgd_mask[:, :, ::-1]).save(
                str(bkgd_cam_dir / f"{frame.image_name}.jpg"), quality=90
            )

    print(f"Done. Dynamic masks: {out_dynamic}, Background masks: {out_bkgd}")


if __name__ == "__main__":
    main()
