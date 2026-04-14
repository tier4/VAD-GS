"""Generate background segmentation masks (sam_bkgd_masks) using SAM
automatic mask generation.

Runs SAM in "segment everything" mode to partition each image into
visually distinct regions, then assigns each region a unique grayscale
ID (R=G=B).  Values (0,0,0) and (1,1,1) are reserved and not used,
since the trellis vacancy detection code skips them.

Usage:
    python script/t4/generate_bkgd_masks.py --config configs/example/t4_train_example.yaml
    python script/t4/generate_bkgd_masks.py --dataroot caf37e66-... --scene-index 0
    python script/t4/generate_bkgd_masks.py --image-dir /path/to/images --output-dir /path/to/out
"""

import argparse
import glob
import os
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

# SAM checkpoint URLs (Meta original)
_CHECKPOINT_URLS = {
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
}
_CACHE_DIR = Path.home() / ".cache" / "sam_checkpoints"

# Fixed random palette for colormap visualization (ID 0,1 = black)
np.random.seed(42)
_COLORMAP = np.random.randint(50, 255, (256, 3), dtype=np.uint8)
_COLORMAP[0] = [0, 0, 0]
_COLORMAP[1] = [0, 0, 0]

# IDs 0 and 1 are reserved (skipped by trellis), so usable range is [2, 255].
_MIN_ID = 2
_MAX_ID = 255


def masks_to_bkgd_image(masks, h, w):
    """Convert a list of SAM masks to a background mask image.

    Each mask is assigned a unique grayscale ID in [_MIN_ID, _MAX_ID].
    Masks are sorted by area (largest first) so that smaller segments
    paint on top of larger ones.

    Args:
        masks: List of dicts from SamAutomaticMaskGenerator, each with
            'segmentation' (H×W bool array) and 'area' (int).
        h, w: Image dimensions.

    Returns:
        (H, W, 3) uint8 array with R=G=B per-segment IDs.
    """
    label = np.zeros((h, w), dtype=np.uint8)

    # Sort by area descending so smaller masks overwrite larger ones
    sorted_masks = sorted(masks, key=lambda m: m["area"], reverse=True)

    for idx, mask_data in enumerate(sorted_masks):
        seg_id = _MIN_ID + (idx % (_MAX_ID - _MIN_ID + 1))
        label[mask_data["segmentation"]] = seg_id

    bkgd = np.stack([label, label, label], axis=2)
    return bkgd


def process_image(image_path, mask_generator, save_path):
    """Run automatic mask generation on one image and save the result."""
    image = cv2.imread(str(image_path))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]

    masks = mask_generator.generate(image_rgb)
    bkgd = masks_to_bkgd_image(masks, h, w)

    cv2.imwrite(str(save_path), bkgd)

    # Save colormap visualization
    colormap_path = str(save_path).replace(".png", "_colormap.jpeg")
    vis = _COLORMAP[bkgd[:, :, 0]]
    overlay = cv2.addWeighted(image, 0.4, vis, 0.6, 0)
    cv2.imwrite(colormap_path, overlay)


def resolve_checkpoint(model_type, checkpoint=None):
    """Return path to SAM checkpoint, downloading if necessary."""
    if checkpoint and os.path.exists(checkpoint):
        return checkpoint
    url = _CHECKPOINT_URLS[model_type]
    filename = url.rsplit("/", 1)[-1]
    cached = _CACHE_DIR / filename
    if cached.exists():
        print(f"Using cached checkpoint: {cached}")
        return str(cached)
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading SAM checkpoint ({model_type}) to {cached} ...")
    urllib.request.urlretrieve(url, str(cached))
    print("Download complete.")
    return str(cached)


def build_mask_generator(model_type, checkpoint, device, points_per_side=32):
    """Build a SamAutomaticMaskGenerator."""
    checkpoint = resolve_checkpoint(model_type, checkpoint)
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device=device)
    sam.eval()

    generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=points_per_side,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        min_mask_region_area=100,
    )
    return generator


def run_t4_mode(args):
    """Process a T4 dataset (uses config_utils / T4Dataset)."""
    from config_utils import add_config_arg, apply_config_defaults
    from t4_dataset import T4Dataset

    apply_config_defaults(args)
    if args.dataroot is None:
        raise ValueError("--dataroot is required (provide via --config or CLI)")
    if args.revision is None:
        args.revision = 0
    if args.scene_index is None:
        args.scene_index = 0

    ds = T4Dataset.from_args(args)
    dataroot = ds.dataroot
    print(f"Resolved dataroot: {dataroot}")

    output_dir = args.output_dir or (dataroot / "preprocessed" / "sam_bkgd_masks")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scene: {ds.scene.name}, {ds.num_samples} samples")

    # Collect frames, skip existing
    entries = []
    for frame in ds.iter_frames(args.camera_channels):
        cam_out_dir = output_dir / frame.camera_channel
        cam_out_dir.mkdir(parents=True, exist_ok=True)
        save_path = cam_out_dir / f"{frame.image_name}.png"
        if args.skip_existing and save_path.exists():
            continue
        entries.append((frame.image_path, save_path))

    return output_dir, entries


def run_image_dir_mode(args):
    """Process a flat directory of images."""
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir) if args.output_dir else image_dir.parent / "sam_bkgd_masks"
    output_dir.mkdir(parents=True, exist_ok=True)

    exts = ("*.png", "*.jpg", "*.jpeg")
    image_paths = []
    for ext in exts:
        image_paths.extend(sorted(glob.glob(str(image_dir / ext))))

    entries = []
    for img_path in image_paths:
        save_path = output_dir / (Path(img_path).stem + ".png")
        if args.skip_existing and save_path.exists():
            continue
        entries.append((img_path, save_path))

    return output_dir, entries


def main():
    parser = argparse.ArgumentParser(
        description="Generate background masks using SAM automatic mask generation"
    )

    # T4 mode arguments
    parser.add_argument("--config", type=str, default=None,
                        help="Path to training YAML config")
    parser.add_argument("--dataroot", type=str, default=None,
                        help="Dataset UUID or path (T4 mode)")
    parser.add_argument("--revision", type=int, default=None)
    parser.add_argument("--scene-index", type=int, default=None)
    parser.add_argument("--camera-channels", nargs="+", default=None)

    # Image directory mode arguments
    parser.add_argument("--image-dir", type=str, default=None,
                        help="Directory of images (non-T4 mode)")

    # Common arguments
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model-type", type=str, default="vit_h",
                        choices=["vit_h", "vit_l", "vit_b"],
                        help="SAM model type (default: vit_h)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to SAM checkpoint (.pth). Auto-downloaded if omitted.")
    parser.add_argument("--points-per-side", type=int, default=32,
                        help="Grid density for automatic mask generation (default: 32)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Determine mode
    if args.image_dir:
        output_dir, entries = run_image_dir_mode(args)
    elif args.dataroot or args.config:
        output_dir, entries = run_t4_mode(args)
    else:
        parser.error("Provide either --image-dir or --dataroot/--config")

    print(f"Images to process: {len(entries)}")
    if not entries:
        print("Nothing to do.")
        return

    print(f"Loading SAM ({args.model_type}) from {args.checkpoint}")
    mask_generator = build_mask_generator(
        args.model_type, args.checkpoint, device,
        points_per_side=args.points_per_side,
    )

    for image_path, save_path in tqdm(entries, desc="Background masks (SAM)"):
        process_image(image_path, mask_generator, save_path)

    print(f"Done. Background masks saved to {output_dir}")


if __name__ == "__main__":
    main()
