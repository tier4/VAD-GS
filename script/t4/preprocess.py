"""Build COLMAP + pointcloud caches for a T4 dataset.

This is a single-process entry point that materialises the artifacts the
training script depends on:

  - ``<model_path>/colmap/triangulated/sparse/model/*.bin``  (COLMAP SfM)
  - ``<model_path>/input_ply/points3D_bkgd.ply``             (LiDAR pointcloud)
  - ``<model_path>/input_ply/points3D_bkgd.npz``             (voxel visibility)

Run this once before launching multi-GPU training so torchrun workers do not
race on the same SQLite DB / output directory.

Usage:
    python script/t4/preprocess.py --config <config>.yaml
"""

from __future__ import annotations

import os
import sys

sys.path.append(os.getcwd())

from lib.config import cfg
from lib.datasets.dataset import Dataset


def main() -> None:
    print(f"[preprocess] model_path : {cfg.model_path}")
    print(f"[preprocess] source_path: {cfg.source_path}")
    Dataset()
    print("[preprocess] done.")


if __name__ == "__main__":
    main()
