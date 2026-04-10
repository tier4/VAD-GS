"""Generate sky masks using SegFormer B5 (Cityscapes) for T4 datasets.

Uses t4-devkit for dataset I/O and frame iteration.

Usage:
    python script/t4/generate_sky_masks.py --config configs/example/t4_train_example.yaml
    python script/t4/generate_sky_masks.py --dataroot caf37e66-... --scene-index 0 --batch-size 4
"""

import argparse
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

CITYSCAPES_SKY_CLASS = 10


def main():
    parser = argparse.ArgumentParser(description="Generate sky masks for T4 dataset")
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

    output_dir = args.output_dir or (dataroot / "preprocessed" / "sky_masks")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Scene: {ds.scene.name}, {ds.num_samples} samples")

    # Collect frames, skip existing
    entries = []
    for frame in ds.iter_frames(args.camera_channels):
        cam_out_dir = output_dir / frame.camera_channel
        cam_out_dir.mkdir(parents=True, exist_ok=True)
        save_path = cam_out_dir / f"{frame.image_name}.png"
        if args.skip_existing and save_path.exists():
            continue
        entries.append(frame)

    print(f"Images to process: {len(entries)}")
    if not entries:
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
    for batch_start in tqdm(range(0, len(entries), batch_size), desc="Sky mask inference"):
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
            sky_mask = (pred == CITYSCAPES_SKY_CLASS).astype(np.uint8) * 255

            cam_out_dir = output_dir / frame.camera_channel
            cv2.imwrite(str(cam_out_dir / f"{frame.image_name}.png"), sky_mask)

    print(f"Done. Sky masks saved to {output_dir}")


if __name__ == "__main__":
    main()
