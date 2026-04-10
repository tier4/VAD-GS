"""Generate dynamic instance masks (sam_masks) and background segmentation masks
(sam_bkgd_masks) for T4 datasets using SAM3 (Segment Anything Model 3).

Dynamic masks: SAM3 with bounding box prompts from projected 3D annotations.
Each projected 3D bounding box is used as a SAM3 box prompt to obtain a
pixel-accurate instance mask.

Background masks: SAM3 with text prompts for each background class
(road, sidewalk, building, etc.).

Vision embeddings are computed once per image and shared across all prompts.

Usage:
    python script/t4/generate_sam_masks.py --config configs/example/t4_train_example.yaml
    python script/t4/generate_sam_masks.py --dataroot caf37e66-... --scene-index 0 --batch-size 4
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from transformers import Sam3Processor, Sam3Model

from config_utils import add_config_arg, apply_config_defaults
from t4_dataset import T4Dataset

# Background class text prompts and BGR colors
# (avoid 0,0,0 and 1,1,1 which are skipped by trellis)
BKGD_CLASS_PROMPTS = {
    0: "road",
    1: "sidewalk",
    2: "building",
    3: "wall",
    4: "fence",
    5: "pole",
    6: "traffic light",
    7: "traffic sign",
    8: "vegetation",
    9: "terrain",
}

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


def project_box_to_xyxy(box, intrinsic, H, W, box_scale=1.0):
    """Project a Box3D (in sensor frame) to a 2D bounding box [x1, y1, x2, y2].

    Returns [x1, y1, x2, y2] in pixel coordinates, or None if behind camera.
    """
    corners = box.corners(box_scale=box_scale)  # (8, 3) in sensor coord
    depths = corners[:, 2]

    if not np.any(depths > 0):
        return None

    corners_clipped = corners.copy()
    corners_clipped[:, 2] = np.clip(corners_clipped[:, 2], a_min=0.1, a_max=None)

    uv, _ = T4Dataset.project_points_to_image(corners_clipped, intrinsic)

    x1 = max(0, int(np.floor(uv[:, 0].min())))
    y1 = max(0, int(np.floor(uv[:, 1].min())))
    x2 = min(W, int(np.ceil(uv[:, 0].max())))
    y2 = min(H, int(np.ceil(uv[:, 1].max())))

    if x2 <= x1 or y2 <= y1:
        return None

    return [x1, y1, x2, y2]


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
    raw_track_ids = {}
    for sample in ds.samples:
        for ann_token in sample.ann_3ds:
            ann = t4.get("sample_annotation", ann_token)
            class_name = simplify_category(ann.category_name)
            if class_name == "misc":
                continue
            raw_id = int(ann.instance_token[:8], 16)
            raw_track_ids[ann.instance_token] = raw_id

    sorted_raw_ids = sorted(set(raw_track_ids.values()))
    raw_to_remapped = {raw_id: new_id for new_id, raw_id in enumerate(sorted_raw_ids)}
    token_to_remapped = {tok: raw_to_remapped[raw_id] for tok, raw_id in raw_track_ids.items()}
    print(f"Found {len(sorted_raw_ids)} dynamic objects, remapped to [0, {len(sorted_raw_ids)-1}]")

    # Collect frames to process
    entries = []
    for frame in ds.iter_frames(args.camera_channels):
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

    # --- Pre-compute 3D bbox projections per frame ---
    print("Projecting 3D bounding boxes...")
    bbox_projections: dict[int, list[tuple[int, list[int]]]] = {}
    for idx, frame in enumerate(tqdm(entries, desc="BBox projection")):
        boxes, cam_intrinsic = ds.get_boxes_in_sensor(frame.sample_data_token)
        K = np.array(cam_intrinsic, dtype=np.float64)
        H, W = frame.height, frame.width

        projections = []
        for box in boxes:
            if box.uuid is None or box.uuid not in token_to_remapped:
                continue
            remapped_id = token_to_remapped[box.uuid]
            bbox_xyxy = project_box_to_xyxy(box, K, H, W, box_scale=args.box_scale)
            if bbox_xyxy is not None:
                projections.append((remapped_id, bbox_xyxy))
        bbox_projections[idx] = projections

    # --- Load SAM3 (shared for dynamic + background masks) ---
    model_id = "facebook/sam3"
    print(f"Loading model: {model_id}")
    processor = Sam3Processor.from_pretrained(model_id)
    model = Sam3Model.from_pretrained(model_id).to(device)
    model.eval()

    # Pre-compute box-prompt text tokens (identical for all single-box prompts)
    dummy_img = Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8))
    dummy_box_inputs = processor(
        images=dummy_img,
        input_boxes=[[[0, 0, 1, 1]]],
        input_boxes_labels=[[1]],
        return_tensors="pt",
    )
    box_input_ids = dummy_box_inputs["input_ids"].to(device)
    box_attention_mask = dummy_box_inputs["attention_mask"].to(device)

    # Pre-compute text embeddings for background classes (reused across all images)
    print("Pre-computing background class text embeddings...")
    bkgd_text_cache = {}
    for cls_id, prompt in BKGD_CLASS_PROMPTS.items():
        text_inputs = processor(text=prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            text_embeds = model.get_text_features(**text_inputs).pooler_output
        bkgd_text_cache[cls_id] = {
            "text_embeds": text_embeds,
            "attention_mask": text_inputs.attention_mask,
        }

    # --- Single pass: produce both dynamic and background masks ---
    print("Running SAM3 inference (dynamic + background masks)...")
    for idx, frame in enumerate(tqdm(entries, desc="SAM3")):
        image = Image.open(frame.image_path).convert("RGB")
        h, w = frame.height, frame.width

        # Compute vision embeddings once per image (shared for all prompts)
        img_inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            vision_embeds = model.get_vision_features(
                pixel_values=img_inputs.pixel_values
            )
        original_sizes = img_inputs.get("original_sizes").tolist()

        # --- Dynamic mask: SAM3 box prompt per instance ---
        projections = bbox_projections[idx]
        dyn_mask = np.full((h, w, 3), 255, dtype=np.uint8)

        for remapped_id, bbox_xyxy in projections:
            x1, y1, x2, y2 = bbox_xyxy
            # SAM3 processor normalizes boxes to [cx, cy, w, h] / (W, H)
            cx = (x1 + x2) / 2.0 / w
            cy = (y1 + y2) / 2.0 / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            input_boxes = torch.tensor(
                [[[cx, cy, bw, bh]]], dtype=torch.float32, device=device
            )
            input_boxes_labels = torch.tensor(
                [[1]], dtype=torch.int64, device=device
            )

            with torch.no_grad():
                outputs = model(
                    vision_embeds=vision_embeds,
                    input_ids=box_input_ids,
                    attention_mask=box_attention_mask,
                    input_boxes=input_boxes,
                    input_boxes_labels=input_boxes_labels,
                )

            results = processor.post_process_instance_segmentation(
                outputs, threshold=0.5, mask_threshold=0.5,
                target_sizes=original_sizes,
            )[0]

            if len(results["masks"]) > 0:
                # Find the mask that best overlaps with our projected bbox
                bbox_region = np.zeros((h, w), dtype=bool)
                bbox_region[y1:y2, x1:x2] = True

                best_mask_np = None
                best_iou = 0.0
                for mask in results["masks"]:
                    mask_np = mask.cpu().numpy().astype(bool)
                    intersection = (mask_np & bbox_region).sum()
                    union = (mask_np | bbox_region).sum()
                    iou = intersection / union if union > 0 else 0
                    if iou > best_iou:
                        best_iou = iou
                        best_mask_np = mask_np

                if best_mask_np is not None and best_iou > 0.01:
                    dyn_mask[best_mask_np] = remapped_id

        cv2.imwrite(
            str(out_dynamic / frame.camera_channel / f"{frame.image_name}.png"),
            dyn_mask,
        )

        # --- Background mask: SAM3 text prompt per class ---
        bkgd_mask = np.zeros((h, w, 3), dtype=np.uint8)

        for cls_id, cache in bkgd_text_cache.items():
            with torch.no_grad():
                outputs = model(
                    vision_embeds=vision_embeds,
                    text_embeds=cache["text_embeds"],
                    attention_mask=cache["attention_mask"],
                )

            results = processor.post_process_instance_segmentation(
                outputs, threshold=0.5, mask_threshold=0.5,
                target_sizes=original_sizes,
            )[0]

            color = BKGD_CLASS_COLORS[cls_id]
            for mask in results["masks"]:
                bkgd_mask[mask.cpu().numpy().astype(bool)] = color

        cv2.imwrite(
            str(out_bkgd / frame.camera_channel / f"{frame.image_name}.png"),
            bkgd_mask,
        )

    # Save track_id mapping
    mapping_path = out_dynamic / "track_id_mapping.json"
    mapping_data = {str(raw_id): new_id for raw_id, new_id in raw_to_remapped.items()}
    with open(mapping_path, "w") as f:
        json.dump(mapping_data, f, indent=2)
    print(f"Track ID mapping saved to {mapping_path}")

    print(f"Done. Dynamic masks: {out_dynamic}, Background masks: {out_bkgd}")


if __name__ == "__main__":
    main()
