#!/usr/bin/env python3
"""Visualize PLY point clouds with Open3D.

Usage:
    python script/visualize_ply.py output/t4_exp/t4_scene_000/input_ply/points3D_bkgd.ply
    python script/visualize_ply.py output/t4_exp/t4_scene_000/input_ply/points3D_obj_000.ply
    python script/visualize_ply.py *.ply  # multiple files
"""

import argparse
import open3d as o3d


def main():
    parser = argparse.ArgumentParser(description="Visualize PLY point clouds")
    parser.add_argument("files", nargs="+", help="PLY file paths")
    parser.add_argument("--voxel-size", type=float, default=None,
                        help="Downsample voxel size (e.g. 0.3 for large files)")
    parser.add_argument("--point-size", type=float, default=1.0)
    args = parser.parse_args()

    geometries = []
    for path in args.files:
        pcd = o3d.io.read_point_cloud(path)
        n = len(pcd.points)
        if args.voxel_size:
            pcd = pcd.voxel_down_sample(args.voxel_size)
        print(f"{path}: {n} pts" + (f" -> {len(pcd.points)} after downsample" if args.voxel_size else ""))
        geometries.append(pcd)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="PLY Viewer")
    for g in geometries:
        vis.add_geometry(g)
    opt = vis.get_render_option()
    opt.point_size = args.point_size
    opt.background_color = [0.1, 0.1, 0.1]
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
