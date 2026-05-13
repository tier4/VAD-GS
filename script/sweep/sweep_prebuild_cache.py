"""Prebuild the COLMAP + bkgd-PLY cache that sweep agents reuse.

Run this **once** before launching `script/sweep/sweep_launch.sh`. It
materialises:

    <cache_dir>/colmap/triangulated/sparse/model/*.bin
    <cache_dir>/input_ply/points3D_bkgd.{ply,npz}

`<cache_dir>` defaults to `output/t4_exp/<exp_name_in_base_yaml>` (i.e.
the model_path the base config would normally produce). Sweep agents then
symlink these subdirs into their per-run model_path — set
`VAD_GS_SWEEP_CACHE_FROM=<cache_dir>` (or pass `--cache-from` to
`sweep_run.py`) to wire it up.

Usage:
    python script/sweep/sweep_prebuild_cache.py --config configs/sweep/base.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs/sweep/base.yaml"),
        help="Base config used to derive source_path and model_path.",
    )
    parser.add_argument(
        "--exp-name",
        default=None,
        help="Override the cache exp_name (defaults to whatever the base yaml says).",
    )
    args = parser.parse_args()

    # Ensure REPO_ROOT is on sys.path so `from lib.config import cfg` works.
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    # Rewrite sys.argv so the import-time argparse in lib.config picks the
    # right config. Optionally override exp_name through yacs opts.
    new_argv = [sys.argv[0], "--config", args.config]
    if args.exp_name:
        new_argv.extend(["exp_name", args.exp_name])
    sys.argv = new_argv

    from lib.config import cfg
    from lib.datasets.dataset import Dataset

    print(f"[prebuild] model_path : {cfg.model_path}")
    print(f"[prebuild] source_path: {cfg.source_path}")

    # Dataset() materialises COLMAP + LiDAR PLY when missing.
    Dataset()

    ply = Path(cfg.model_path) / "input_ply" / "points3D_bkgd.ply"
    if not ply.is_file():
        raise SystemExit(f"[prebuild] FAILED: {ply} was not created")
    print(f"[prebuild] done. cache dir: {cfg.model_path}")
    print(f"[prebuild] export VAD_GS_SWEEP_CACHE_FROM={os.path.relpath(cfg.model_path, REPO_ROOT)}")


if __name__ == "__main__":
    main()
