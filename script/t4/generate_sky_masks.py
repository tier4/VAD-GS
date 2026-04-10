"""Generate sky masks using SAM3 (Segment Anything Model 3) for T4 datasets.

Uses SAM3 with text prompt "sky" for open-vocabulary sky segmentation.
Pre-computes text embeddings once and reuses them across all images.

Usage:
    python script/t4/generate_sky_masks.py --config configs/example/t4_train_example.yaml
    python script/t4/generate_sky_masks.py --dataroot caf37e66-... --scene-index 0
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from transformers import Sam3Processor, Sam3Model

from config_utils import add_config_arg, apply_config_defaults
from t4_dataset import T4Dataset


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

    # Load SAM3
    model_id = "facebook/sam3"
    print(f"Loading model: {model_id}")
    processor = Sam3Processor.from_pretrained(model_id)
    model = Sam3Model.from_pretrained(model_id).to(device)
    model.eval()

    # Pre-compute text embeddings for "sky" (reused across all images)
    text_inputs = processor(text="sky", return_tensors="pt").to(device)
    with torch.no_grad():
        text_embeds = model.get_text_features(**text_inputs).pooler_output

    # Process images with pre-computed text embeddings
    for frame in tqdm(entries, desc="Sky mask (SAM3)"):
        image = Image.open(frame.image_path).convert("RGB")
        h, w = image.height, image.width

        img_inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(
                pixel_values=img_inputs.pixel_values,
                text_embeds=text_embeds,
                attention_mask=text_inputs.attention_mask,
            )

        results = processor.post_process_instance_segmentation(
            outputs, threshold=0.5, mask_threshold=0.5,
            target_sizes=img_inputs.get("original_sizes").tolist(),
        )[0]

        # Combine all detected sky instance masks into a single binary mask
        sky_mask = np.zeros((h, w), dtype=np.uint8)
        for mask in results["masks"]:
            sky_mask[mask.cpu().numpy().astype(bool)] = 255

        cam_out_dir = output_dir / frame.camera_channel
        cv2.imwrite(str(cam_out_dir / f"{frame.image_name}.png"), sky_mask)

    print(f"Done. Sky masks saved to {output_dir}")


if __name__ == "__main__":
    main()
