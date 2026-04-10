"""Run all T4 dataset preprocessing steps in sequence.

Generates: lidar_depth, depth (mono), sky_masks, sam_masks, sam_bkgd_masks, normal_img.
Skips steps whose output directories already contain data.

Usage:
    python script/t4/preprocess_all.py --config configs/example/t4_train_example.yaml
    python script/t4/preprocess_all.py --config configs/example/t4_train_example.yaml --batch-size 8
    python script/t4/preprocess_all.py --dataroot caf37e66-... --scene-index 0 --batch-size 4
"""

import argparse
import subprocess
import sys
from pathlib import Path

from config_utils import add_config_arg, apply_config_defaults
from t4_dataset import T4Dataset

SCRIPT_DIR = Path(__file__).parent


def is_step_complete(output_dir, expected_counts, extension):
    """Check if all expected output files exist for a preprocessing step."""
    if not output_dir.exists() or not expected_counts:
        return False

    for camera, expected in expected_counts.items():
        cam_dir = output_dir / camera
        if not cam_dir.exists():
            return False
        actual = sum(1 for f in cam_dir.iterdir() if f.suffix == extension)
        if actual < expected:
            return False

    return True


def run_step(name, script, args, output_dir, force=False,
             expected_counts=None, extension=None):
    """Run a preprocessing step, skipping only if output is complete."""
    if not force and expected_counts and extension:
        if is_step_complete(output_dir, expected_counts, extension):
            print(f"[SKIP] {name}: output is complete at {output_dir}")
            return True
        elif output_dir.exists():
            print(f"[INCOMPLETE] {name}: output is incomplete, re-running")

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
    add_config_arg(parser)
    parser.add_argument("--dataroot", type=str, default=None, help="Dataset UUID or path")
    parser.add_argument("--revision", type=int, default=None)
    parser.add_argument("--scene-index", type=int, default=None)
    parser.add_argument("--camera-channels", nargs="+", default=None)
    parser.add_argument("--lidar-channel", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--force", action="store_true",
                        help="Force re-run all steps even if output exists")
    parser.add_argument("--steps", nargs="+", default=None,
                        help="Run only specific steps (lidar_depth, mono_depth, sky_masks, sam_masks, normal_maps)")
    args = parser.parse_args()

    # Apply config defaults, then hard defaults
    apply_config_defaults(args)
    if args.dataroot is None:
        parser.error("--dataroot is required (provide via --config or CLI)")
    if args.revision is None:
        args.revision = 0
    if args.scene_index is None:
        args.scene_index = 0
    if args.lidar_channel is None:
        args.lidar_channel = "LIDAR_CONCAT"

    # Use T4Dataset for frame counting
    ds = T4Dataset.from_args(args)
    dataroot = ds.dataroot
    print(f"Resolved dataroot: {dataroot}")

    # Common args passed to all sub-scripts
    common = ["--dataroot", str(args.dataroot), "--revision", str(args.revision),
              "--scene-index", str(args.scene_index)]
    if args.camera_channels:
        common += ["--camera-channels"] + args.camera_channels

    batch_args = ["--batch-size", str(args.batch_size)]
    if args.device:
        batch_args += ["--device", args.device]
    if args.force:
        batch_args += ["--no-skip-existing"]

    all_steps = ["lidar_depth", "mono_depth", "sky_masks", "sam_masks", "normal_maps"]
    steps = args.steps or all_steps

    results = {}

    prep = dataroot / "preprocessed"

    # Count expected frames per camera using t4-devkit
    lidar_cameras = args.camera_channels or [
        "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
        "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
    ]
    lidar_expected = ds.count_frames_per_camera(lidar_cameras)
    other_expected = ds.count_frames_per_camera(args.camera_channels)

    if lidar_expected:
        total = sum(lidar_expected.values())
        print(f"Expected frames (lidar cameras): {lidar_expected}  total={total}")
    if other_expected:
        total = sum(other_expected.values())
        print(f"Expected frames (all cameras):   {other_expected}  total={total}")

    # Step 1: LiDAR depth
    if "lidar_depth" in steps:
        results["lidar_depth"] = run_step(
            "LiDAR Depth",
            SCRIPT_DIR / "generate_lidar_depth.py",
            ["--dataroot", str(dataroot),
             "--scene-index", str(args.scene_index),
             "--camera-channels"] + lidar_cameras
            + ["--lidar-channel", args.lidar_channel],
            prep / "lidar_depth",
            force=args.force,
            expected_counts=lidar_expected,
            extension=".npy",
        )

    # Step 2: Mono depth (Depth Anything V2 Small)
    if "mono_depth" in steps:
        results["mono_depth"] = run_step(
            "Mono Depth (Depth Anything V2 Small)",
            SCRIPT_DIR / "generate_mono_depth.py",
            common + batch_args,
            prep / "depth",
            force=args.force,
            expected_counts=other_expected,
            extension=".npz",
        )

    # Step 3: Sky masks (SAM3 text prompt)
    if "sky_masks" in steps:
        results["sky_masks"] = run_step(
            "Sky Masks (SAM3)",
            SCRIPT_DIR / "generate_sky_masks.py",
            common + batch_args,
            prep / "sky_masks",
            force=args.force,
            expected_counts=other_expected,
            extension=".png",
        )

    # Step 4: Dynamic + Background masks (sam_masks + sam_bkgd_masks)
    if "sam_masks" in steps:
        sam_complete = (
            is_step_complete(prep / "sam_masks", other_expected, ".png")
            and is_step_complete(prep / "sam_bkgd_masks", other_expected, ".png")
        )
        if not args.force and sam_complete:
            print(f"[SKIP] Dynamic + Background Masks: output is complete")
            results["sam_masks"] = True
        else:
            if not args.force and (prep / "sam_masks").exists():
                print("[INCOMPLETE] Dynamic + Background Masks: output is incomplete, re-running")
            results["sam_masks"] = run_step(
                "Dynamic + Background Masks",
                SCRIPT_DIR / "generate_sam_masks.py",
                common + batch_args,
                prep / "sam_masks",
                force=True,
            )

    # Step 5: Normal maps (from depth)
    if "normal_maps" in steps:
        results["normal_maps"] = run_step(
            "Normal Maps (from depth)",
            SCRIPT_DIR / "generate_normal_maps.py",
            common + batch_args,
            prep / "normal_img",
            force=args.force,
            expected_counts=other_expected,
            extension=".png",
        )

    # Step 6: Visualize preprocessed data as MP4 videos (always runs)
    vis_args = ["--dataroot", str(dataroot),
                "--scene-index", str(args.scene_index)]
    if args.camera_channels:
        vis_args += ["--camera-channels"] + args.camera_channels
    results["visualize"] = run_step(
        "Visualize Preprocessed Data",
        SCRIPT_DIR / "visualize_preprocess.py",
        vis_args,
        dataroot / "preprocess_vis",
        force=True,
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

    # --- Final validation ---
    print(f"\n{'='*60}")
    print("Validation")
    print(f"{'='*60}")

    validation_specs = [
        ("lidar_depth", prep / "lidar_depth", lidar_expected, ".npy"),
        ("mono_depth",  prep / "depth",       other_expected, ".npz"),
        ("sky_masks",   prep / "sky_masks",   other_expected, ".png"),
        ("sam_masks",   prep / "sam_masks",   other_expected, ".png"),
        ("sam_bkgd_masks", prep / "sam_bkgd_masks", other_expected, ".png"),
        ("normal_maps", prep / "normal_img",  other_expected, ".png"),
    ]

    all_ok = True
    for step_name, output_dir, expected, ext in validation_specs:
        if not expected:
            print(f"  {step_name}: WARN (no expected frame counts available)")
            continue
        if not output_dir.exists():
            print(f"  {step_name}: MISSING (directory not found)")
            all_ok = False
            continue

        step_ok = True
        for camera, exp_count in expected.items():
            cam_dir = output_dir / camera
            if not cam_dir.exists():
                print(f"  {step_name}/{camera}: MISSING (0/{exp_count})")
                step_ok = False
                continue
            actual = sum(1 for f in cam_dir.iterdir() if f.suffix == ext)
            if actual < exp_count:
                print(f"  {step_name}/{camera}: INCOMPLETE ({actual}/{exp_count})")
                step_ok = False
            elif actual > exp_count:
                print(f"  {step_name}/{camera}: EXTRA ({actual}/{exp_count})")

        if step_ok:
            total = sum(expected.values())
            print(f"  {step_name}: OK ({total} files)")
        else:
            all_ok = False

    print(f"{'='*60}")
    if all_ok:
        print("All preprocessed data is complete.")
    else:
        print("Some preprocessed data is incomplete or missing.")
        sys.exit(1)

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
