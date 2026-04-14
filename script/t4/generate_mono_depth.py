"""Generate metric depth maps using InfiniDepth (RGB + LiDAR) for T4 datasets.

The ``InfiniDepth_DepthSensor`` model variant takes an RGB image together with
a sparse sensor depth map and outputs a dense metric depth map.  This script
feeds the pre-generated LiDAR depth (from ``generate_lidar_depth.py``) as the
sparse sensor input.

Usage:
    python script/t4/generate_mono_depth.py --config configs/example/t4_train_example.yaml
    python script/t4/generate_mono_depth.py \
        --dataroot /path/to/t4_dataset \
        --scene-index 0
"""

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
from tqdm import tqdm

from config_utils import add_config_arg, apply_config_defaults
from t4_dataset import T4Dataset

# Make the vendored InfiniDepth submodule importable.
#   submodules/InfiniDepth/InfiniDepth/  <- Python package
#   submodules/InfiniDepth/inference_depth.py  <- top-level helpers
REPO_ROOT = Path(__file__).resolve().parents[2]
INFINIDEPTH_DIR = REPO_ROOT / "submodules" / "InfiniDepth"
if not INFINIDEPTH_DIR.exists():
    raise RuntimeError(
        f"InfiniDepth submodule not found at {INFINIDEPTH_DIR}. "
        "Run: git submodule update --init --recursive"
    )
sys.path.insert(0, str(INFINIDEPTH_DIR))


# HuggingFace repo / filename mapping for auto-download
_HF_CHECKPOINTS = {
    "infinidepth_depthsensor.ckpt": ("ritianyu/InfiniDepth", "infinidepth_depthsensor.ckpt"),
    "infinidepth.ckpt": ("ritianyu/InfiniDepth", "infinidepth.ckpt"),
    "model.pt": ("Ruicheng/moge-2-vitl-normal", "model.pt"),
}


def _ensure_checkpoint(path: Path) -> Path:
    """Download checkpoint from HuggingFace if it doesn't exist locally."""
    if path.exists():
        return path
    filename = path.name
    if filename not in _HF_CHECKPOINTS:
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\n"
            f"Unknown file '{filename}' — cannot auto-download."
        )
    repo_id, hf_filename = _HF_CHECKPOINTS[filename]
    print(f"Checkpoint not found at {path}, downloading from {repo_id}...")
    from huggingface_hub import hf_hub_download
    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=hf_filename,
        local_dir=str(path.parent),
    )
    print(f"Downloaded to {downloaded}")
    return path


def _import_infinidepth():
    """Deferred import so that --help works without InfiniDepth deps."""
    from inference_depth import (
        DepthInferenceArgs,
        load_depth_model,
        run_depth_inference,
    )
    return DepthInferenceArgs, load_depth_model, run_depth_inference


def load_lidar_depth_as_array(lidar_depth_path: Path) -> np.ndarray:
    """Load a LiDAR depth dict (``{'mask', 'value'}``) into a dense HxW array
    where invalid pixels are zero-filled."""
    data = np.load(str(lidar_depth_path), allow_pickle=True).item()
    mask = np.asarray(data["mask"], dtype=bool)
    value = np.asarray(data["value"], dtype=np.float32)
    depth = np.zeros(mask.shape, dtype=np.float32)
    depth[mask] = value[mask]
    return depth


def main():
    parser = argparse.ArgumentParser(
        description="Generate metric depth maps (InfiniDepth + LiDAR) for T4 dataset"
    )
    add_config_arg(parser)
    parser.add_argument("--dataroot", type=str, default=None, help="Dataset UUID or path")
    parser.add_argument("--revision", type=int, default=None)
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory (default: <dataroot>/preprocessed/depth)",
    )
    parser.add_argument(
        "--lidar-depth-dir", type=Path, default=None,
        help="LiDAR depth directory (default: <dataroot>/preprocessed/lidar_depth)",
    )
    parser.add_argument("--scene-index", type=int, default=None)
    parser.add_argument(
        "--camera-channels", nargs="+", default=None,
        help="Camera channel names (auto-detected if omitted)",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.add_argument(
        "--model-type", default="InfiniDepth_DepthSensor",
        choices=["InfiniDepth", "InfiniDepth_DepthSensor"],
    )
    parser.add_argument(
        "--depth-model-path", type=Path,
        default=INFINIDEPTH_DIR / "checkpoints/depth/infinidepth_depthsensor.ckpt",
        help="Path to InfiniDepth checkpoint (.ckpt)",
    )
    parser.add_argument(
        "--moge2-pretrained", type=Path,
        default=INFINIDEPTH_DIR / "checkpoints/moge-2-vitl-normal/model.pt",
    )
    parser.add_argument(
        "--input-size", nargs=2, type=int, default=(768, 1024),
        metavar=("H", "W"), help="Network input size (H W)",
    )
    parser.add_argument(
        "--output-resolution-mode", default="original",
        choices=["upsample", "original", "specific"],
        help="Resolution of the saved depth map (default: match input image)",
    )
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

    lidar_depth_dir = args.lidar_depth_dir or (dataroot / "preprocessed" / "lidar_depth")
    if args.model_type == "InfiniDepth_DepthSensor" and not lidar_depth_dir.exists():
        raise FileNotFoundError(
            f"LiDAR depth directory not found: {lidar_depth_dir}. "
            "Run generate_lidar_depth.py first."
        )

    print(f"Scene: {ds.scene.name}, {ds.num_samples} samples")

    # Collect frames
    entries = []
    missing_lidar = 0
    for frame in ds.iter_frames(args.camera_channels):
        cam_out_dir = output_dir / frame.camera_channel
        cam_out_dir.mkdir(parents=True, exist_ok=True)
        save_path = cam_out_dir / f"{frame.image_name}.npz"
        if args.skip_existing and save_path.exists():
            continue

        lidar_path: Path | None = None
        if args.model_type == "InfiniDepth_DepthSensor":
            lidar_path = lidar_depth_dir / frame.camera_channel / f"{frame.image_name}.npy"
            if not lidar_path.exists():
                missing_lidar += 1
                continue
        entries.append((frame, lidar_path))

    print(f"Images to process: {len(entries)}")
    if missing_lidar:
        print(f"[Warning] Skipped {missing_lidar} frames with missing LiDAR depth")
    if not entries:
        print("Nothing to do.")
        return

    # Auto-download checkpoints if missing
    args.depth_model_path = _ensure_checkpoint(args.depth_model_path)

    DepthInferenceArgs, load_depth_model, run_depth_inference = _import_infinidepth()

    depth_args = DepthInferenceArgs(
        input_image_path="",
        input_depth_path=None,
        model_type=args.model_type,
        depth_model_path=str(args.depth_model_path),
        moge2_pretrained=str(args.moge2_pretrained),
        input_size=tuple(args.input_size),
        output_resolution_mode=args.output_resolution_mode,
        save_pcd=False,
    )

    print(f"Loading InfiniDepth model: {args.model_type} ({args.depth_model_path})")
    model, device = load_depth_model(depth_args)
    model.eval()

    with tempfile.TemporaryDirectory(prefix="infinidepth_lidar_") as tmpdir:
        tmp_root = Path(tmpdir)
        for frame, lidar_path in tqdm(entries, desc="InfiniDepth (RGB+LiDAR)"):
            tmp_depth_file: str | None = None
            if lidar_path is not None:
                sparse = load_lidar_depth_as_array(lidar_path)
                tmp_depth_file = str(
                    tmp_root / f"{frame.camera_channel}_{frame.image_name}.npz"
                )
                np.savez_compressed(tmp_depth_file, depth=sparse)

            K = ds.get_camera_intrinsic(frame.calibrated_sensor_token)

            result = run_depth_inference(
                depth_args,
                model=model,
                device=device,
                input_image_path=frame.image_path,
                input_depth_path=tmp_depth_file,
                fx_org=float(K[0, 0]),
                fy_org=float(K[1, 1]),
                cx_org=float(K[0, 2]),
                cy_org=float(K[1, 2]),
            )

            depth_np = result.pred_depthmap.squeeze().detach().cpu().numpy().astype(np.float32)
            cam_out_dir = output_dir / frame.camera_channel
            np.savez_compressed(
                str(cam_out_dir / f"{frame.image_name}.npz"),
                depth=depth_np,
            )

    print(f"Done. Metric depth maps saved to {output_dir}")


if __name__ == "__main__":
    main()
