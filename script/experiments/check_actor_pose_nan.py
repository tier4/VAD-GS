"""Diagnose the iter=50 NaN crash in the segmented pipeline.

We've already ruled out the "K_real=1 / duplicate timestamp" hypothesis by
simulating _batched_closest_indices in numpy. This script goes one step
further and constructs the real ActorPose module (CUDA tensors + opt_track
parameters) for each segment, then runs the batched rotation/translation
lookup at every training-camera timestamp and reports the first NaN/inf
hit. That catches anything dependent on opt_track init, val/train branch,
or the actual interpolation arithmetic — not just the index-padding case.

Usage:
    python script/experiments/check_actor_pose_nan.py --segments seg_00
    python script/experiments/check_actor_pose_nan.py    # all 13
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]


def diagnose_segment(name: str, cfg, generate_dataparser_outputs_t4):
    seg_yaml = REPO_ROOT / "configs/experiments/segmented" / f"{name}.yaml"
    if not seg_yaml.exists():
        print(f"[skip] {seg_yaml} not found")
        return None
    cfg.merge_from_file(str(seg_yaml))
    sel = list(cfg.data.get("selected_frames", []))
    print(f"\n=== {name}  selected_frames={sel} ===")

    out = generate_dataparser_outputs_t4(
        datadir=cfg.source_path,
        selected_frames=sel,
        build_pointcloud=False,
        cameras=cfg.data.get("cameras", None),
        camera_channels=cfg.data.get("camera_channels", None),
        lidar_channel=cfg.data.get("lidar_channel", None),
        scene_index=cfg.data.get("scene_index", 0),
    )

    obj_tracklets = out["obj_tracklets"]
    obj_info = out["obj_info"]
    tracklet_ts = np.asarray(out["tracklet_timestamps"], dtype=np.float64)
    cams_ts = np.asarray(out["cams_timestamps"], dtype=np.float64)

    # Build the camera_timestamps dict that ActorPose expects.
    cams = out["cams"]
    camera_timestamps: dict = {}
    for cam in sorted(set(cams)):
        camera_timestamps[cam] = {"train_timestamps": [], "test_timestamps": []}
    # Match the t4_readers split: split_test=4, so every 4th view is test.
    split_test = int(cfg.data.get("split_test", 4))
    for i, (cam, ts) in enumerate(zip(cams, cams_ts)):
        if split_test > 0 and i % split_test == 0:
            camera_timestamps[cam]["test_timestamps"].append(ts)
        else:
            camera_timestamps[cam]["train_timestamps"].append(ts)

    # Construct ActorPose with the actual data.
    from lib.models.actor_pose import ActorPose

    actor_pose = ActorPose(obj_tracklets, tracklet_ts, camera_timestamps, dict(obj_info))
    actor_pose.training_setup()

    track_ids = sorted(int(k) for k in obj_info.keys())
    print(f"  actors={len(track_ids)}  track_ids[:5]={track_ids[:5]}  opt_track={actor_pose.opt_track}")

    # Stub a viewpoint camera with just the meta dict the lookup uses.
    class _StubCam:
        def __init__(self, ts: float, cam: int, is_val: bool):
            self.meta = {"timestamp": ts, "cam": cam, "is_val": is_val}

    bad = []  # (kind, view_ts, cam, n_nan_in_result)
    n_views = len(cams)
    for i in range(n_views):
        for is_val in (False, True):
            stub = _StubCam(float(cams_ts[i]), int(cams[i]), is_val)
            try:
                rot = actor_pose.get_tracking_rotation_batched(track_ids, stub)
                trn = actor_pose.get_tracking_translation_batched(track_ids, stub)
            except Exception as exc:
                bad.append(("exception", float(cams_ts[i]), int(cams[i]), is_val, repr(exc)))
                continue
            n_rot_nan = int((~rot.isfinite()).sum().item())
            n_trn_nan = int((~trn.isfinite()).sum().item())
            if n_rot_nan + n_trn_nan > 0:
                # Find which actor row has the NaN
                rot_bad_rows = ((~rot.isfinite()).any(dim=-1)).nonzero(as_tuple=False).flatten().tolist()
                trn_bad_rows = ((~trn.isfinite()).any(dim=-1)).nonzero(as_tuple=False).flatten().tolist()
                bad.append(
                    (
                        "nan",
                        float(cams_ts[i]),
                        int(cams[i]),
                        is_val,
                        f"rot nan rows={rot_bad_rows[:5]}  trn nan rows={trn_bad_rows[:5]}",
                    )
                )

    if not bad:
        print("  OK: no NaN/inf in actor_pose lookup over any training camera")
    else:
        print(f"  ** {len(bad)} NaN/inf hits **")
        # Print the FIRST occurrence per kind
        seen = set()
        for entry in bad:
            tag = entry[0]
            if tag in seen:
                continue
            seen.add(tag)
            print(f"     {entry}")
        # Also report whether bad hits are concentrated in val branch
        n_val = sum(1 for e in bad if e[3])
        n_train = sum(1 for e in bad if not e[3])
        print(f"     train-branch hits: {n_train}  val-branch hits: {n_val}")

    return bad


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--segments",
        default="seg_00,seg_01,seg_02,seg_03,seg_04,seg_05,seg_06,seg_07,seg_08,seg_09,seg_10,seg_11,seg_12",
        help="Comma-separated list of segment yaml stems under configs/experiments/segmented/",
    )
    args = p.parse_args()

    seg_names = [s.strip() for s in args.segments.split(",") if s.strip()]

    os.chdir(REPO_ROOT)
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    sys.argv = [str(REPO_ROOT / "train.py"), "--config", "configs/experiments/segmented/seg_base.yaml"]
    from lib.config import cfg
    from lib.utils.t4_utils import generate_dataparser_outputs_t4

    summary = []
    for name in seg_names:
        try:
            bad = diagnose_segment(name, cfg, generate_dataparser_outputs_t4)
        except Exception as exc:
            print(f"[error] {name}: {exc}")
            bad = "err"
        summary.append((name, bad))

    print("\n=== SUMMARY ===")
    for name, bad in summary:
        if bad in ("err", None):
            print(f"  {name}: skipped/err")
        elif not bad:
            print(f"  {name}: OK")
        else:
            print(f"  {name}: {len(bad)} hits")


if __name__ == "__main__":
    main()
