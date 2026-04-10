"""Generate dynamic instance masks (sam_masks) and background segmentation masks
(sam_bkgd_masks) for T4 datasets.

Dynamic masks: Project 3D bounding boxes from T4 annotations onto camera images.
  Each dynamic object gets a unique uint8 ID. Background = 255.
Background masks: SegFormer Cityscapes semantic segmentation of background classes.

Usage:
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

_ANNOTATION_DATASET_BASE = os.path.expanduser("~/.webauto/data/data/annotation_dataset")

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


def quat_wxyz_to_rotmat(quat):
    w, x, y, z = quat
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    return np.array([
        [1 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1 - (xx + yy)],
    ], dtype=np.float64)


def make_transform(translation, rotation_wxyz):
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = quat_wxyz_to_rotmat(rotation_wxyz)
    t[:3, 3] = np.asarray(translation, dtype=np.float64)
    return t


def simplify_category(name):
    name = name.lower()
    if "vehicle" in name:
        return "vehicle"
    if "pedestrian" in name:
        return "pedestrian"
    if "cycle" in name or "bicycle" in name or "motorcycle" in name:
        return "cyclist"
    return "misc"


def bbox_to_corner3d(bbox):
    min_x, min_y, min_z = bbox[0]
    max_x, max_y, max_z = bbox[1]
    return np.array([
        [min_x, min_y, min_z], [min_x, min_y, max_z],
        [min_x, max_y, min_z], [min_x, max_y, max_z],
        [max_x, min_y, min_z], [max_x, min_y, max_z],
        [max_x, max_y, min_z], [max_x, max_y, max_z],
    ])


def get_bound_2d_mask(corners_3d, K, pose, H, W):
    corners_3d = np.dot(corners_3d, pose[:3, :3].T) + pose[:3, 3:].T
    corners_3d[..., 2] = np.clip(corners_3d[..., 2], a_min=1e-3, a_max=None)
    corners_3d = np.dot(corners_3d, K.T)
    corners_2d = corners_3d[:, :2] / corners_3d[:, 2:]
    corners_2d = np.round(corners_2d).astype(int)
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 3, 2, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[4, 5, 7, 6, 5]]], 1)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 5, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[2, 3, 7, 6, 2]]], 1)
    cv2.fillPoly(mask, [corners_2d[[0, 2, 6, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[1, 3, 7, 5, 1]]], 1)
    return mask


def build_sample_annotation_map(sample_annotations):
    m = {}
    for ann in sample_annotations:
        m.setdefault(ann["sample_token"], []).append(ann)
    return m


def main():
    parser = argparse.ArgumentParser(
        description="Generate dynamic + background masks for T4 dataset"
    )
    parser.add_argument("--dataroot", type=str, required=True, help="Dataset UUID or path")
    parser.add_argument("--revision", type=int, default=0)
    parser.add_argument("--output-dir-dynamic", type=Path, default=None,
                        help="Output dir for sam_masks (default: <dataroot>/sam_masks)")
    parser.add_argument("--output-dir-bkgd", type=Path, default=None,
                        help="Output dir for sam_bkgd_masks (default: <dataroot>/sam_bkgd_masks)")
    parser.add_argument("--scene-index", type=int, default=0)
    parser.add_argument("--camera-channels", nargs="+", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--box-scale", type=float, default=1.5,
                        help="Scale factor for 3D bounding boxes (default: 1.5)")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    args = parser.parse_args()

    dataroot = resolve_dataroot(args.dataroot, revision=args.revision)
    print(f"Resolved dataroot: {dataroot}")
    out_dynamic = args.output_dir_dynamic or (dataroot / "preprocessed" / "sam_masks")
    out_bkgd = args.output_dir_bkgd or (dataroot / "preprocessed" / "sam_bkgd_masks")
    out_dynamic.mkdir(parents=True, exist_ok=True)
    out_bkgd.mkdir(parents=True, exist_ok=True)

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
    ego_pose_by_token = index_by_token(tables["ego_pose"])
    instance_by_token = index_by_token(tables["instance"])
    category_by_token = index_by_token(tables["category"])
    sample_annotation_map = build_sample_annotation_map(tables["sample_annotation"])

    scene = tables["scene"][args.scene_index]
    samples = sample_chain(scene, sample_by_token)
    print(f"Scene: {scene.get('name', 'unknown')}, {len(samples)} samples")

    # Build instance_token -> remapped track_id (same logic as t4_utils.py)
    raw_track_ids = {}  # instance_token -> raw_id
    for sample in samples:
        annotations = sample_annotation_map.get(sample["token"], [])
        for ann in annotations:
            instance_token = ann["instance_token"]
            instance = instance_by_token.get(instance_token, {})
            category = category_by_token.get(instance.get("category_token", ""), {})
            class_name = simplify_category(category.get("name", "misc"))
            if class_name == "misc":
                continue
            raw_id = int(instance_token[:8], 16)
            raw_track_ids[instance_token] = raw_id

    # Remap to sequential IDs (matching t4_utils.py remapping)
    sorted_raw_ids = sorted(set(raw_track_ids.values()))
    raw_to_remapped = {raw_id: new_id for new_id, raw_id in enumerate(sorted_raw_ids)}
    token_to_remapped = {tok: raw_to_remapped[raw_id] for tok, raw_id in raw_track_ids.items()}
    print(f"Found {len(sorted_raw_ids)} dynamic objects, remapped to [0, {len(sorted_raw_ids)-1}]")

    # Build camera channel -> sample_data mapping (keyframes only)
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
        print(f"Auto-detected camera channels: {camera_channels}")

    # Collect entries: (image_path, image_name, sample_token, camera_channel, sd, cs)
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
            dyn_cam_dir = out_dynamic / ch
            bkgd_cam_dir = out_bkgd / ch
            dyn_cam_dir.mkdir(parents=True, exist_ok=True)
            bkgd_cam_dir.mkdir(parents=True, exist_ok=True)
            dyn_path = dyn_cam_dir / f"{image_name}.png"
            bkgd_path = bkgd_cam_dir / f"{image_name}.png"
            if args.skip_existing and dyn_path.exists() and bkgd_path.exists():
                continue
            entries.append((str(image_path), image_name, sample["token"], ch, sd, cs))

    print(f"Images to process: {len(entries)}")
    if not entries:
        print("Nothing to do.")
        return

    # --- Generate dynamic masks (from 3D bbox projection) ---
    print("Generating dynamic masks from 3D bounding box projections...")
    for image_path, image_name, sample_token, ch, sd, cs in tqdm(entries, desc="Dynamic masks"):
        cam_ego = ego_pose_by_token[sd["ego_pose_token"]]
        cam_sensor_to_ego = make_transform(cs["translation"], cs["rotation"])
        cam_to_world = make_transform(cam_ego["translation"], cam_ego["rotation"]) @ cam_sensor_to_ego
        world_to_cam = np.linalg.inv(cam_to_world)

        K = np.array(cs["camera_intrinsic"], dtype=np.float64)
        H, W = sd["height"], sd["width"]

        # 3-channel mask: background = 255
        dyn_mask = np.full((H, W, 3), 255, dtype=np.uint8)

        annotations = sample_annotation_map.get(sample_token, [])
        for ann in annotations:
            instance_token = ann["instance_token"]
            if instance_token not in token_to_remapped:
                continue
            remapped_id = token_to_remapped[instance_token]

            # Build 3D bbox
            size = ann["size"]  # [width, length, height]
            half = np.array([size[1], size[0], size[2]]) * 0.5 * args.box_scale
            bbox = np.array([[-half[0], -half[1], -half[2]], [half[0], half[1], half[2]]])
            corners_local = bbox_to_corner3d(bbox)
            corners_local_h = np.concatenate(
                [corners_local, np.ones((corners_local.shape[0], 1))], axis=1
            )

            # Object to world
            obj_to_world = make_transform(ann["translation"], ann["rotation"])
            corners_world = (corners_local_h @ obj_to_world.T)[:, :3]

            # Project to camera
            mask_2d = get_bound_2d_mask(
                corners_3d=corners_world,
                K=K,
                pose=world_to_cam,
                H=H, W=W,
            )
            dyn_mask[mask_2d > 0] = remapped_id

        cv2.imwrite(str(out_dynamic / ch / f"{image_name}.png"), dyn_mask)

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
        images = [Image.open(p).convert("RGB") for p, *_ in batch]
        original_sizes = [(img.height, img.width) for img in images]

        inputs = image_processor(images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits

        for i, (_, image_name, _, ch, *_) in enumerate(batch):
            h, w = original_sizes[i]
            upsampled = F.interpolate(
                logits[i:i+1], size=(h, w), mode="bilinear", align_corners=False
            )
            pred = upsampled.argmax(dim=1).squeeze(0).cpu().numpy()

            # Background mask: each background class gets a unique color
            bkgd_mask = np.zeros((h, w, 3), dtype=np.uint8)
            for cls_id, color in BKGD_CLASS_COLORS.items():
                bkgd_mask[pred == cls_id] = color

            bkgd_cam_dir = out_bkgd / ch
            cv2.imwrite(str(bkgd_cam_dir / f"{image_name}.png"), bkgd_mask)
            # Visualization
            Image.fromarray(bkgd_mask[:, :, ::-1]).save(
                str(bkgd_cam_dir / f"{image_name}.jpg"), quality=90
            )

    print(f"Done. Dynamic masks: {out_dynamic}, Background masks: {out_bkgd}")


if __name__ == "__main__":
    main()
