"""COLMAP runner for T4 datasets.

Adapted from script/waymo/colmap_drivestudio_full.py to handle T4 image
filenames (e.g. ``data/CAM_FRONT/00042.jpg``) instead of DriveStudio
format (``{frame}_{cam}.jpg``).
"""

import sqlite3
import os
import numpy as np
import cv2
import shutil
import subprocess
import sys
import json

sys.path.append(os.getcwd())

from scipy.spatial.transform import Rotation as R
from lib.config import cfg
from lib.utils.data_utils import get_val_frames


def run_colmap_command(args):
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = env.get("QT_QPA_PLATFORM", "offscreen")
    env.pop("QT_PLUGIN_PATH", None)
    env.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
    subprocess.run(args, check=True, env=env)


def _t4_image_to_colmap_name(image_path, cam_idx, frame_idx):
    """Convert a T4 image path to COLMAP naming: ``cam_{cam_idx}/{frame_idx:06d}.png``."""
    return f"cam_{cam_idx}/{frame_idx:06d}.png"


image_filename_to_cam = lambda x: int(x.split("/")[0].split("_")[1])  # cam_{id}/...


def run_colmap_t4(result, extrinsics_list):
    """Run COLMAP triangulation for a T4 dataset.

    Args:
        result: Output dict from ``generate_dataparser_outputs_t4``.
        extrinsics_list: Per-camera extrinsic matrices (camera-to-lidar at
            frame 0), indexed by camera integer ID.
    """
    model_path = cfg.model_path
    data_path = cfg.source_path
    colmap_dir = os.path.join(model_path, "colmap")
    os.makedirs(colmap_dir, exist_ok=True)
    print("Running COLMAP for T4, colmap dir:", colmap_dir)

    unique_cams = sorted(set(result["cams"]))
    print("cameras:", unique_cams)

    for cam in unique_cams:
        os.makedirs(os.path.join(colmap_dir, "train_imgs", f"cam_{cam}"), exist_ok=True)
        os.makedirs(os.path.join(colmap_dir, "test_imgs", f"cam_{cam}"), exist_ok=True)
        os.makedirs(os.path.join(colmap_dir, "mask", f"cam_{cam}"), exist_ok=True)

    train_images_dir = os.path.join(colmap_dir, "train_imgs")
    mask_images_dir = os.path.join(colmap_dir, "mask")

    image_filenames = result["image_filenames"]
    c2ws = result["c2ws"]
    ixts = result["ixts"]
    frames_idx = result["frames_idx"]
    cams = result["cams"]

    split_test = cfg.data.get("split_test", -1)
    split_train = cfg.data.get("split_train", -1)
    num_frames = result["num_frames"]
    train_frames, test_frames = get_val_frames(
        num_frames,
        test_every=split_test if split_test > 0 else None,
        train_every=split_train if split_train > 0 else None,
    )

    # Build COLMAP-compatible names and copy images
    c2w_dict = {}
    for i, image_filename in enumerate(image_filenames):
        frame_idx = frames_idx[i]
        cam = cams[i]
        colmap_name = _t4_image_to_colmap_name(image_filename, cam, frame_idx)
        c2w_dict[colmap_name] = c2ws[i]

        dst = os.path.join(train_images_dir, colmap_name)
        if not os.path.exists(dst):
            shutil.copyfile(image_filename, dst)

        # Create empty mask (no dynamic mask pre-computed for T4)
        mask_dst = os.path.join(mask_images_dir, f"cam_{cam}", f"{frame_idx:06d}.png.png")
        if not os.path.exists(mask_dst):
            img = cv2.imread(image_filename)
            if img is not None:
                empty_mask = np.zeros(img.shape[:2], dtype=np.uint8)
                cv2.imwrite(mask_dst, empty_mask)

    # Feature extraction
    run_colmap_command([
        "colmap", "feature_extractor",
        "--ImageReader.mask_path", mask_images_dir,
        "--ImageReader.camera_model", "SIMPLE_PINHOLE",
        "--ImageReader.single_camera_per_folder", "1",
        "--database_path", f"{colmap_dir}/database.db",
        "--image_path", train_images_dir,
    ])

    # Load intrinsics per camera
    camera_infos = {}
    for cam in unique_cams:
        for i, c in enumerate(cams):
            if c == cam:
                break
        sample_img = cv2.imread(image_filenames[i])
        img_h, img_w = sample_img.shape[:2]
        camera_infos[cam] = {"ixt": ixts[i], "img_h": img_h, "img_w": img_w}

    # Read database image IDs
    conn = sqlite3.connect(f"{colmap_dir}/database.db")
    c = conn.cursor()
    c.execute("SELECT * FROM images")
    db_images = c.fetchall()
    conn.close()

    cam_to_db_id = {}
    for row in db_images:
        name = row[1]
        cam = image_filename_to_cam(name)
        cam_to_db_id[cam] = row[2]

    with open(f"{colmap_dir}/id_names.txt", "w") as f:
        for row in db_images:
            f.write(f"{row[0]} {row[1]}\n")

    id_names = [(row[0], row[1]) for row in db_images]

    # Create sparse model
    model_dir = f"{colmap_dir}/created/sparse/model"
    os.makedirs(model_dir, exist_ok=True)

    # images.txt
    with open(f"{model_dir}/images.txt", "w") as f:
        for id_, name in id_names:
            transform = np.linalg.inv(c2w_dict[name])
            r = R.from_matrix(transform[:3, :3])
            rquat = r.as_quat()  # [x, y, z, w]
            rquat = [rquat[3], rquat[0], rquat[1], rquat[2]]  # [w, x, y, z]
            out = np.concatenate([rquat, transform[:3, 3]])
            cam = image_filename_to_cam(name)
            db_cam_id = cam_to_db_id[cam]
            f.write(f"{id_} {' '.join(str(a) for a in out.tolist())} {db_cam_id} {name}\n\n")

    # cameras.txt
    with open(f"{model_dir}/cameras.txt", "w") as f:
        for cam in unique_cams:
            db_cam_id = cam_to_db_id[cam]
            info = camera_infos[cam]
            ixt = info["ixt"]
            f.write(
                f"{db_cam_id} SIMPLE_PINHOLE {info['img_w']} {info['img_h']} "
                f"{ixt[0, 0]} {ixt[0, 2]} {ixt[1, 2]}\n"
            )

    # Update database intrinsics
    conn = sqlite3.connect(f"{colmap_dir}/database.db")
    c = conn.cursor()
    for cam in unique_cams:
        cam_id = cam_to_db_id[cam]
        ixt = camera_infos[cam]["ixt"]
        params = np.array([ixt[0, 0], ixt[0, 2], ixt[1, 2]], dtype=np.float64)
        c.execute(
            "UPDATE cameras SET params = ? WHERE camera_id = ?",
            (params.tobytes(), cam_id),
        )
    conn.commit()
    conn.close()

    # points3D.txt (empty)
    open(f"{model_dir}/points3D.txt", "a").close()

    # Rigid camera config
    ref_camera_id = unique_cams[0]
    rigid_cam_list = []
    for cam_id in unique_cams:
        ref_ext = extrinsics_list[ref_camera_id]
        cur_ext = extrinsics_list[cam_id]
        rel_ext = np.linalg.inv(cur_ext) @ ref_ext
        r = R.from_matrix(rel_ext[:3, :3])
        qvec = r.as_quat()
        rigid_cam_list.append({
            "camera_id": cam_id,
            "image_prefix": f"cam_{cam_id}",
            "cam_from_rig_rotation": [qvec[3], qvec[0], qvec[1], qvec[2]],
            "cam_from_rig_translation": rel_ext[:3, 3].tolist(),
        })

    rigid_config_path = os.path.join(colmap_dir, "cam_rigid_config.json")
    with open(rigid_config_path, "w") as f:
        json.dump([{"ref_camera_id": ref_camera_id, "cameras": rigid_cam_list}], f, indent=4)

    # Matching
    run_colmap_command([
        "colmap", "exhaustive_matcher",
        "--database_path", f"{colmap_dir}/database.db",
    ])

    # Triangulation
    triangulated_dir = os.path.join(colmap_dir, "triangulated/sparse/model")
    os.makedirs(triangulated_dir, exist_ok=True)
    run_colmap_command([
        "colmap", "point_triangulator",
        "--database_path", f"{colmap_dir}/database.db",
        "--image_path", train_images_dir,
        "--input_path", model_dir,
        "--output_path", triangulated_dir,
        "--Mapper.ba_refine_focal_length", "0",
        "--Mapper.ba_refine_principal_point", "0",
        "--Mapper.max_extra_param", "0",
        "--clear_points", "0",
        "--Mapper.ba_global_max_num_iterations", "30",
        "--Mapper.filter_max_reproj_error", "4",
        "--Mapper.filter_min_tri_angle", "0.5",
        "--Mapper.tri_min_angle", "0.5",
        "--Mapper.tri_ignore_two_view_tracks", "1",
        "--Mapper.tri_complete_max_reproj_error", "4",
        "--Mapper.tri_continue_max_angle_error", "4",
    ])

    if cfg.data.get("use_colmap_pose", False):
        run_colmap_command([
            "colmap", "rig_bundle_adjuster",
            "--input_path", triangulated_dir,
            "--output_path", triangulated_dir,
            "--rig_config_path", rigid_config_path,
            "--estimate_rig_relative_poses", "0",
            "--RigBundleAdjustment.refine_relative_poses", "1",
            "--BundleAdjustment.max_num_iterations", "50",
            "--BundleAdjustment.refine_focal_length", "0",
            "--BundleAdjustment.refine_principal_point", "0",
            "--BundleAdjustment.refine_extra_params", "0",
        ])

    # Cleanup temp images
    os.system(f"rm -rf {train_images_dir}")
    os.system(f"rm -rf {os.path.join(colmap_dir, 'test_imgs')}")
    os.system(f"rm -rf {mask_images_dir}")
