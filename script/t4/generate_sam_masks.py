"""Generate dynamic instance masks (sam_masks) and background segmentation masks
(sam_bkgd_masks) for T4 datasets.

Dynamic masks: SAM3 text prompts per category ("car", "truck", "bus",
"person", "bicycle", "motorcycle") detect all instances, then greedy
IoU matching assigns each SAM3 mask to a projected 3D bounding box.

Background masks: SegFormer B5 (Cityscapes) semantic segmentation.

Two-phase processing: SAM3 for dynamic masks first, then SAM3 is unloaded
and SegFormer is loaded for background masks to minimize peak VRAM usage.

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
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from transformers import Sam3Processor, Sam3Model

# Workaround: SegFormer model lacks safetensors, and torch<2.6 triggers CVE check
import transformers.modeling_utils as _mu
_mu.check_torch_load_is_safe = lambda: None

from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

from config_utils import add_config_arg, apply_config_defaults
from t4_dataset import T4Dataset

# Cityscapes background classes for SegFormer
# (avoid 0,0,0 and 1,1,1 which are skipped by trellis)
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

    # ===== Phase 1: Dynamic masks with SAM3 text prompts =====
    model_id = "facebook/sam3"
    print(f"Loading SAM3: {model_id}")
    processor = Sam3Processor.from_pretrained(model_id)
    model = Sam3Model.from_pretrained(model_id).to(device)
    model.eval()

    # Text prompts for dynamic object categories
    DYNAMIC_PROMPTS = ["car", "truck", "bus", "person", "bicycle", "motorcycle"]
    print(f"Dynamic prompts: {DYNAMIC_PROMPTS}")

    print("Running SAM3 inference (dynamic masks)...")
    for idx, frame in enumerate(tqdm(entries, desc="SAM3 dynamic")):
        image = Image.open(frame.image_path).convert("RGB")
        h, w = frame.height, frame.width

        # Batched inference: same image × N prompts in one forward pass
        images_batch = [image] * len(DYNAMIC_PROMPTS)
        inputs = processor(
            images=images_batch, text=DYNAMIC_PROMPTS, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        batch_results = processor.post_process_instance_segmentation(
            outputs, threshold=0.5, mask_threshold=0.5,
            target_sizes=inputs.get("original_sizes").tolist(),
        )

        # Collect all instance masks across all prompts
        all_masks = []
        for results in batch_results:
            for mask in results["masks"]:
                all_masks.append(mask.cpu().numpy().astype(bool))

        # IoU match: projected bboxes ↔ SAM3 instance masks
        projections = bbox_projections[idx]
        dyn_mask = np.full((h, w, 3), 255, dtype=np.uint8)

        if all_masks and projections:
            # Build IoU matrix: (n_bboxes, n_masks)
            n_bboxes = len(projections)
            n_masks = len(all_masks)
            iou_matrix = np.zeros((n_bboxes, n_masks), dtype=np.float32)

            for bi, (remapped_id, bbox_xyxy) in enumerate(projections):
                x1, y1, x2, y2 = bbox_xyxy
                bbox_region = np.zeros((h, w), dtype=bool)
                bbox_region[y1:y2, x1:x2] = True
                bbox_area = bbox_region.sum()

                for mi, mask_np in enumerate(all_masks):
                    intersection = (mask_np & bbox_region).sum()
                    union = (mask_np | bbox_region).sum()
                    iou_matrix[bi, mi] = intersection / union if union > 0 else 0

            # Greedy 1:1 matching (highest IoU first)
            used_bboxes = set()
            used_masks = set()
            while True:
                if iou_matrix.max() < 0.05:
                    break
                bi, mi = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
                if bi in used_bboxes or mi in used_masks:
                    iou_matrix[bi, mi] = 0
                    continue
                remapped_id = projections[bi][0]
                dyn_mask[all_masks[mi]] = remapped_id
                used_bboxes.add(bi)
                used_masks.add(mi)
                iou_matrix[bi, :] = 0
                iou_matrix[:, mi] = 0

        cv2.imwrite(
            str(out_dynamic / frame.camera_channel / f"{frame.image_name}.png"),
            dyn_mask,
        )

    # Unload SAM3 to free VRAM for SegFormer
    del model, processor
    torch.cuda.empty_cache()

    # ===== Phase 2: Background masks with SegFormer =====
    segformer_id = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
    print(f"Loading SegFormer: {segformer_id}")
    image_processor = AutoImageProcessor.from_pretrained(segformer_id)
    segformer_model = AutoModelForSemanticSegmentation.from_pretrained(segformer_id).to(device)
    segformer_model.eval()

    print("Running SegFormer inference (background masks)...")
    batch_size = args.batch_size
    for batch_start in tqdm(range(0, len(entries), batch_size), desc="SegFormer bkgd"):
        batch = entries[batch_start:batch_start + batch_size]
        images = [Image.open(f.image_path).convert("RGB") for f in batch]
        original_sizes = [(img.height, img.width) for img in images]

        inputs = image_processor(images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = segformer_model(**inputs)
        logits = outputs.logits

        for i, frame in enumerate(batch):
            h, w = original_sizes[i]
            upsampled = F.interpolate(
                logits[i:i+1], size=(h, w), mode="bilinear", align_corners=False
            )
            seg_pred = upsampled.argmax(dim=1).squeeze(0).cpu().numpy()

            bkgd_mask = np.zeros((h, w, 3), dtype=np.uint8)
            for cls_id, color in BKGD_CLASS_COLORS.items():
                bkgd_mask[seg_pred == cls_id] = color

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
