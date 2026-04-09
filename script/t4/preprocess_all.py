"""Run all T4 dataset preprocessing steps in sequence.

Generates: lidar_depth, depth (mono), sky_masks, sam_masks, sam_bkgd_masks, normal_img.
Skips steps whose output directories already contain data.

Usage:
    python script/t4/preprocess_all.py --dataroot caf37e66-... --scene-index 0 --batch-size 4
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

_ANNOTATION_DATASET_BASE = os.path.expanduser("~/.webauto/data/data/annotation_dataset")

SCRIPT_DIR = Path(__file__).parent


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


def dir_has_files(path, extensions=(".png", ".npy", ".npz")):
    """Check if directory exists and contains at least one file with given extensions."""
    if not path.exists():
        return False
    for f in path.iterdir():
        if f.suffix in extensions:
            return True
    return False


def run_step(name, script, args, output_dir, force=False):
    """Run a preprocessing step, skipping if output already exists."""
    if not force and dir_has_files(output_dir):
        print(f"[SKIP] {name}: output already exists at {output_dir}")
        return True

    print(f"\n{'='*60}")
    print(f"[RUN] {name}")
    print(f"{'='*60}")

    cmd = [sys.executable, str(script)] + args
    print(f"  Command: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"[FAIL] {name} failed with return code {result.returncode}")
        return False

    print(f"[DONE] {name}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run all T4 preprocessing steps")
    parser.add_argument("--dataroot", type=str, required=True, help="Dataset UUID or path")
    parser.add_argument("--revision", type=int, default=0)
    parser.add_argument("--scene-index", type=int, default=0)
    parser.add_argument("--camera-channels", nargs="+", default=None)
    parser.add_argument("--lidar-channel", default="LIDAR_CONCAT")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--force", action="store_true",
                        help="Force re-run all steps even if output exists")
    parser.add_argument("--steps", nargs="+", default=None,
                        help="Run only specific steps (lidar_depth, mono_depth, sky_masks, sam_masks, normal_maps)")
    args = parser.parse_args()

    dataroot = resolve_dataroot(args.dataroot, revision=args.revision)
    print(f"Resolved dataroot: {dataroot}")

    # Common args passed to all scripts
    common = ["--dataroot", args.dataroot, "--revision", str(args.revision),
              "--scene-index", str(args.scene_index)]
    if args.camera_channels:
        common += ["--camera-channels"] + args.camera_channels

    batch_args = ["--batch-size", str(args.batch_size)]
    if args.device:
        batch_args += ["--device", args.device]

    all_steps = ["lidar_depth", "mono_depth", "sky_masks", "sam_masks", "normal_maps"]
    steps = args.steps or all_steps

    results = {}

    # Step 1: LiDAR depth
    if "lidar_depth" in steps:
        results["lidar_depth"] = run_step(
            "LiDAR Depth",
            SCRIPT_DIR / "generate_lidar_depth.py",
            ["--dataroot", str(dataroot),
             "--scene-index", str(args.scene_index),
             "--camera-channels"] + (args.camera_channels or [
                "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
                "CAM_BACK_LEFT", "CAM_BACK_RIGHT"
             ]) + ["--lidar-channel", args.lidar_channel],
            dataroot / "lidar_depth",
            force=args.force,
        )

    # Step 2: Mono depth (Depth Anything V2 Small)
    if "mono_depth" in steps:
        results["mono_depth"] = run_step(
            "Mono Depth (Depth Anything V2 Small)",
            SCRIPT_DIR / "generate_mono_depth.py",
            common + batch_args,
            dataroot / "depth",
            force=args.force,
        )

    # Step 3: Sky masks (SegFormer B5 Cityscapes)
    if "sky_masks" in steps:
        results["sky_masks"] = run_step(
            "Sky Masks (SegFormer B5)",
            SCRIPT_DIR / "generate_sky_masks.py",
            common + batch_args,
            dataroot / "sky_masks",
            force=args.force,
        )

    # Step 4: Dynamic + Background masks
    if "sam_masks" in steps:
        results["sam_masks"] = run_step(
            "Dynamic + Background Masks",
            SCRIPT_DIR / "generate_sam_masks.py",
            common + batch_args,
            dataroot / "sam_masks",
            force=args.force,
        )

    # Step 5: Normal maps (from depth)
    if "normal_maps" in steps:
        results["normal_maps"] = run_step(
            "Normal Maps (from depth)",
            SCRIPT_DIR / "generate_normal_maps.py",
            common + batch_args,
            dataroot / "normal_img",
            force=args.force,
        )

    # Summary
    print(f"\n{'='*60}")
    print("Preprocessing Summary")
    print(f"{'='*60}")
    for step, success in results.items():
        status = "OK" if success else "FAILED"
        print(f"  {step}: {status}")

    failed = [s for s, ok in results.items() if not ok]
    if failed:
        print(f"\nFailed steps: {', '.join(failed)}")
        sys.exit(1)
    else:
        print("\nAll steps completed successfully.")


if __name__ == "__main__":
    main()
