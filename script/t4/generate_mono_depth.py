"""Generate monocular depth maps using Depth Anything V2 Small for T4 datasets.

Uses t4-devkit for dataset I/O and frame iteration.

Usage:
    python script/t4/generate_mono_depth.py --config configs/example/t4_train_example.yaml
    python script/t4/generate_mono_depth.py \
        --dataroot /path/to/t4_dataset \
        --scene-index 0 --batch-size 4
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from config_utils import add_config_arg, apply_config_defaults
from t4_dataset import T4Dataset


def main():
    parser = argparse.ArgumentParser(
        description="Generate monocular depth maps (Depth Anything V2 Small) for T4 dataset"
    )
    add_config_arg(parser)
    parser.add_argument("--dataroot", type=str, default=None, help="Dataset UUID or path")
    parser.add_argument("--revision", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (default: <dataroot>/preprocessed/depth)")
    parser.add_argument("--scene-index", type=int, default=None)
    parser.add_argument("--camera-channels", nargs="+", default=None,
                        help="Camera channel names (auto-detected if omitted)")
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

    output_dir = args.output_dir or (dataroot / "preprocessed" / "depth")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Scene: {ds.scene.name}, {ds.num_samples} samples")

    # Collect frames, skip existing
    entries = []
    for frame in ds.iter_frames(args.camera_channels):
        cam_out_dir = output_dir / frame.camera_channel
        cam_out_dir.mkdir(parents=True, exist_ok=True)
        save_path = cam_out_dir / f"{frame.image_name}.npz"
        if args.skip_existing and save_path.exists():
            continue
        entries.append(frame)

    print(f"Images to process: {len(entries)}")
    if not entries:
        print("Nothing to do.")
        return

    # Load model
    model_id = "depth-anything/Depth-Anything-V2-Small-hf"
    print(f"Loading model: {model_id}")
    image_processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device)
    model.eval()

    # Run inference
    batch_size = args.batch_size
    for batch_start in tqdm(range(0, len(entries), batch_size), desc="Depth inference"):
        batch = entries[batch_start:batch_start + batch_size]
        images = [Image.open(f.image_path).convert("RGB") for f in batch]
        original_sizes = [(img.height, img.width) for img in images]

        inputs = image_processor(images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        post_processed = image_processor.post_process_depth_estimation(
            outputs, target_sizes=original_sizes,
        )

        for i, frame in enumerate(batch):
            depth_np = post_processed[i]["predicted_depth"].cpu().numpy().astype(np.float32)
            cam_out_dir = output_dir / frame.camera_channel
            np.savez_compressed(str(cam_out_dir / f"{frame.image_name}.npz"), depth=depth_np)
            # Visualization
            d_min, d_max = depth_np.min(), depth_np.max()
            if d_max - d_min > 1e-6:
                depth_vis = ((depth_np - d_min) / (d_max - d_min) * 255).astype(np.uint8)
            else:
                depth_vis = np.zeros_like(depth_np, dtype=np.uint8)
            Image.fromarray(depth_vis).save(
                str(cam_out_dir / f"{frame.image_name}.jpg"), quality=90
            )

    print(f"Done. Depth maps saved to {output_dir}")


if __name__ == "__main__":
    main()
