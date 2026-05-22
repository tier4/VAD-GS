"""Backfill `<seg_model_path>/track_id_map.json` for segments trained
before `street_gaussian_model.setup_functions` started saving it.

For each segment's saved cfg dump (under `<seg_model_path>/configs/
config_*.yaml`), re-run the T4 actor-info parsing for that segment's
`data.selected_frames` and write the seq_id → original T4 track_id
map. Idempotent: skips segments that already have the file.

This only re-runs the cheap part of T4 data loading (object tracking
parsing + remap), not COLMAP / PLY building.

Usage:
    python script/experiments/build_track_id_maps.py
    python script/experiments/build_track_id_maps.py --force            # rebuild even if file exists
    python script/experiments/build_track_id_maps.py --seg-dir-glob 'output/t4_exp/t4_seg_*'
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from glob import glob
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seg-dir-glob", default="output/t4_exp/t4_seg_*",
                        help="Glob (relative to repo root) for segment model dirs.")
    parser.add_argument("--force", action="store_true",
                        help="Re-build mapping even if track_id_map.json already exists.")
    args = parser.parse_args()

    os.chdir(REPO_ROOT)
    # Ensure `from lib.config import cfg` resolves. Python's sys.path
    # only contains the script's own dir by default, not REPO_ROOT.
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    seg_dirs = sorted(glob(args.seg_dir_glob))
    if not seg_dirs:
        sys.exit(f"[build_maps] no segment dirs matched {args.seg_dir_glob}")

    # Import lazily so cfg side-effects don't happen unless we actually need them.
    # We have to set up sys.argv first so make_cfg() reads a sensible base config.
    # The cfg gets re-merged per segment yaml below.
    sys.argv = [str(REPO_ROOT / "train.py"), "--config", "configs/experiments/segmented/seg_base.yaml"]
    from lib.config import cfg  # triggers make_cfg with seg_base
    from lib.utils.t4_utils import generate_dataparser_outputs_t4

    for seg_dir in seg_dirs:
        seg_path = Path(seg_dir)
        out_path = seg_path / "track_id_map.json"
        if out_path.exists() and not args.force:
            print(f"[skip] {out_path} already present")
            continue

        # Find the per-segment cfg dump.
        cfg_dumps = sorted(seg_path.glob("configs/config_*.yaml"))
        if not cfg_dumps:
            print(f"[warn] {seg_path}: no cfg dump under configs/ — skipping")
            continue

        cfg.merge_from_file(str(cfg_dumps[0]))

        sel = cfg.data.get("selected_frames", [])
        print(f"[seg] {seg_path.name}: selected_frames={list(sel)} — re-parsing actors")

        try:
            output = generate_dataparser_outputs_t4(
                datadir=cfg.source_path,
                selected_frames=list(sel),
                build_pointcloud=False,
                cameras=cfg.data.get("cameras", None),
                camera_channels=cfg.data.get("camera_channels", None),
                lidar_channel=cfg.data.get("lidar_channel", None),
                scene_index=cfg.data.get("scene_index", 0),
            )
        except Exception as exc:
            print(f"[error] {seg_path.name}: T4 parse failed: {exc}")
            continue

        obj_info = output["obj_info"]
        mapping = {}
        for seq_id, meta in obj_info.items():
            orig = meta.get("original_instance_token_prefix") if isinstance(meta, dict) else None
            mapping[str(seq_id)] = orig if orig is not None else seq_id

        payload = {
            "seq_id_to_t4_track_id": mapping,
            "selected_frames": list(sel),
            "exp_name": cfg.exp_name,
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[done] {out_path} ({len(mapping)} actors)")


if __name__ == "__main__":
    main()
