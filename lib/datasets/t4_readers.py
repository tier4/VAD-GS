"""T4 dataset reader for VAD-GS.

Reads TIER IV (nuScenes-compatible) T4 format datasets directly
from their JSON annotation files. Mirrors the structure of
my_drivestudio_readers.py for compatibility with the VAD-GS pipeline.
"""

from __future__ import annotations

from typing import Any

from lib.utils.t4_utils import generate_dataparser_outputs_t4, resolve_t4_dataset_path
from lib.utils.graphics_utils import focal2fov, BasicPointCloud
from lib.utils.data_utils import get_val_frames
from lib.datasets.base_readers import (
    CameraInfo, SceneInfo, getNerfppNorm, fetchPly, get_Sphere_Norm,
)
from lib.config import cfg
from tqdm import tqdm
from PIL import Image
import os
import numpy as np
import cv2
import sys

sys.path.append(os.getcwd())


def readT4Info(path: str, images: str = "images", split_train: int = -1, split_test: int = -1, **kwargs: Any) -> SceneInfo:
    """Read T4 dataset and return SceneInfo for VAD-GS training/evaluation."""

    # Resolve dataset ID to filesystem path (supports UUID-based lookup)
    revision = cfg.data.get("revision", 0)
    path = resolve_t4_dataset_path(path, revision=revision)
    print(f"T4 dataset path: {path}")

    selected_frames = cfg.data.get("selected_frames", None)
    if cfg.debug:
        selected_frames = [0, 0]

    # Load pre-computed point cloud if specified
    if cfg.data.get("load_pcd_from", False) and (cfg.mode == "train"):
        import shutil

        load_dir = os.path.join(cfg.workspace, cfg.data.load_pcd_from, "input_ply")
        save_dir = os.path.join(cfg.model_path, "input_ply")
        os.system(f"rm -rf {save_dir}")
        shutil.copytree(load_dir, save_dir)

        colmap_dir = os.path.join(cfg.workspace, cfg.data.load_pcd_from, "colmap")
        save_dir = os.path.join(cfg.model_path, "colmap")
        os.system(f"rm -rf {save_dir}")
        shutil.copytree(colmap_dir, save_dir)

    bkgd_ply_path = os.path.join(cfg.model_path, "input_ply/points3D_bkgd.ply")
    build_pointcloud = (cfg.mode == "train") and (
        not os.path.exists(bkgd_ply_path) or cfg.data.get("regenerate_pcd", False)
    )

    # Guidance data directories — prefer preprocessed/ subdirectory, fall back to legacy flat layout
    prep = os.path.join(path, "preprocessed")

    def _resolve_guidance_dir(name: str, *legacy_names: str) -> str | None:
        """Return the first existing directory: preprocessed/<name>, <path>/<name>, <path>/<legacy>."""
        candidate = os.path.join(prep, name)
        if os.path.exists(candidate):
            return candidate
        candidate = os.path.join(path, name)
        if os.path.exists(candidate):
            return candidate
        for ln in legacy_names:
            candidate = os.path.join(path, ln)
            if os.path.exists(candidate):
                return candidate
        return None

    dynamic_mask_dir = _resolve_guidance_dir("sam_masks") or os.path.join(prep, "sam_masks")
    bkgd_mask_dir = _resolve_guidance_dir("sam_bkgd_masks") or os.path.join(prep, "sam_bkgd_masks")
    load_dynamic_mask = os.path.exists(dynamic_mask_dir)
    load_seg_bkgd = cfg.data.get("use_seg_bkgd", True) and os.path.exists(bkgd_mask_dir)

    sky_mask_dir = _resolve_guidance_dir("sky_masks") or os.path.join(prep, "sky_masks")
    load_sky_mask = (cfg.mode == "train") and os.path.exists(sky_mask_dir)

    lidar_depth_dir = _resolve_guidance_dir("lidar_depth") or os.path.join(prep, "lidar_depth")
    load_lidar_depth = (cfg.mode == "train") and os.path.exists(lidar_depth_dir)

    mono_depth_dir = _resolve_guidance_dir("depth", "depth_v2") or os.path.join(prep, "depth")
    load_mono_depth = os.path.exists(mono_depth_dir)

    normal_dir = _resolve_guidance_dir("normal_img") or os.path.join(prep, "normal_img")
    load_normal = os.path.exists(normal_dir)

    # T4-specific config
    camera_channels = cfg.data.get("camera_channels", None)
    lidar_channel = cfg.data.get("lidar_channel", None)
    scene_index = cfg.data.get("scene_index", 0)

    # Generate dataparser outputs
    output = generate_dataparser_outputs_t4(
        datadir=path,
        selected_frames=selected_frames,
        build_pointcloud=build_pointcloud,
        cameras=cfg.data.get("cameras", None),
        camera_channels=camera_channels,
        lidar_channel=lidar_channel,
        scene_index=scene_index,
    )

    exts = output["exts"]
    ixts = output["ixts"]
    poses = output["poses"]
    c2ws = output["c2ws"]
    image_filenames = output["image_filenames"]
    obj_tracklets = output["obj_tracklets"]
    obj_info = output["obj_info"]
    frames, cams = output["frames"], output["cams"]
    frames_idx = output["frames_idx"]
    num_frames = output["num_frames"]
    cams_timestamps = output["cams_timestamps"]
    tracklet_timestamps = output["tracklet_timestamps"]
    obj_bounds = output["obj_bounds"]
    ego_frame_poses = output["ego_frame_poses"]
    obj_view_dict = output["obj_view_dict"]

    train_frames, test_frames = get_val_frames(
        num_frames,
        test_every=split_test if split_test > 0 else None,
        train_every=split_train if split_train > 0 else None,
    )

    # Build scene metadata
    num_cams = len(set(cams))
    scene_metadata = dict()
    scene_metadata["obj_tracklets"] = obj_tracklets
    scene_metadata["tracklet_timestamps"] = tracklet_timestamps
    scene_metadata["obj_meta"] = obj_info
    scene_metadata["num_images"] = len(exts)
    scene_metadata["num_cams"] = num_cams
    scene_metadata["num_frames"] = num_frames
    scene_metadata["ego_frame_poses"] = ego_frame_poses
    scene_metadata["obj_view_dict"] = obj_view_dict
    scene_metadata["c2ws"] = c2ws
    scene_metadata["ixts"] = ixts

    camera_timestamps = dict()
    for cam in sorted(set(cams)):
        camera_timestamps[cam] = dict()
        camera_timestamps[cam]["train_timestamps"] = []
        camera_timestamps[cam]["test_timestamps"] = []

    def _find_guidance_file(base_dir: str, cam_channel: str, image_name: str, ext: str) -> str | None:
        """Find guidance file, checking camera subdirectory first then flat."""
        cam_path = os.path.join(base_dir, cam_channel, f"{image_name}{ext}")
        if os.path.exists(cam_path):
            return cam_path
        flat_path = os.path.join(base_dir, f"{image_name}{ext}")
        if os.path.exists(flat_path):
            return flat_path
        return None

    # Build CameraInfo list
    cam_infos = []
    for i in tqdm(range(len(exts)), desc="Loading T4 cameras"):
        ext = exts[i]
        ixt = ixts[i]
        c2w = c2ws[i]
        pose = poses[i]
        image_path = image_filenames[i]
        image_name = os.path.basename(image_path).split(".")[0]
        # Camera channel = parent directory name (e.g. "CAM_FRONT")
        cam_channel = os.path.basename(os.path.dirname(image_path))
        # Get dimensions from header only; defer pixel loading to loadCam
        with Image.open(image_path) as img:
            width, height = img.size
        image = None
        fx, fy = ixt[0, 0], ixt[1, 1]
        FovY = focal2fov(fx, height)
        FovX = focal2fov(fy, width)

        RT = np.linalg.inv(c2w)
        R = RT[:3, :3].T
        T = RT[:3, 3]
        K = ixt.copy()

        metadata = dict()
        metadata["frame"] = frames[i]
        metadata["cam"] = cams[i]
        metadata["frame_idx"] = frames_idx[i]
        metadata["ego_pose"] = pose
        metadata["extrinsic"] = ext
        metadata["timestamp"] = cams_timestamps[i]

        if frames_idx[i] in train_frames:
            metadata["is_val"] = False
            camera_timestamps[cams[i]]["train_timestamps"].append(cams_timestamps[i])
        else:
            metadata["is_val"] = True
            camera_timestamps[cams[i]]["test_timestamps"].append(cams_timestamps[i])

        guidance = dict()

        # Store paths for deferred loading (loaded in loadguidance)
        if load_dynamic_mask:
            dynamic_mask_path = _find_guidance_file(dynamic_mask_dir, cam_channel, image_name, ".png")
            if dynamic_mask_path is not None:
                guidance["dynamic_mask"] = dynamic_mask_path

                if load_seg_bkgd:
                    seg_bkgd_mask_path = _find_guidance_file(bkgd_mask_dir, cam_channel, image_name, ".png")
                    if seg_bkgd_mask_path is not None:
                        guidance["seg_bkgd"] = seg_bkgd_mask_path

                # Save obj_bound to disk to avoid ~700MB of PIL Images in RAM
                # Use cam_channel prefix to avoid filename collisions between cameras
                # (T4 images share the same basename across cameras, e.g. 00193.jpg)
                obj_bound_dir = os.path.join(cfg.model_path, "obj_bounds")
                os.makedirs(obj_bound_dir, exist_ok=True)
                obj_bound_path = os.path.join(obj_bound_dir, f"{cam_channel}_{image_name}.png")
                if not os.path.exists(obj_bound_path):
                    cv2.imwrite(obj_bound_path, obj_bounds[i].astype(np.uint8) * 255)
                guidance["obj_bound"] = obj_bound_path

        if load_lidar_depth:
            depth_path = _find_guidance_file(lidar_depth_dir, cam_channel, image_name, ".npy")
            if depth_path is not None:
                guidance["lidar_depth"] = depth_path

        if load_sky_mask:
            sky_mask_path = _find_guidance_file(sky_mask_dir, cam_channel, image_name, ".png")
            if sky_mask_path is not None:
                guidance["sky_mask"] = sky_mask_path

        if load_mono_depth:
            mono_depth_path = _find_guidance_file(mono_depth_dir, cam_channel, image_name, ".npz")
            if mono_depth_path is None:
                mono_depth_path = _find_guidance_file(mono_depth_dir, cam_channel, image_name, ".png")
            if mono_depth_path is not None:
                guidance["mono_depth"] = mono_depth_path

        if load_normal:
            normal_img_path = _find_guidance_file(normal_dir, cam_channel, image_name, ".png")
            if normal_img_path is not None:
                guidance["mono_normal"] = normal_img_path

        cam_info = CameraInfo(
            uid=i, R=R, T=T, FovY=FovY, FovX=FovX, K=K,
            image=image, image_path=image_path, image_name=image_name,
            width=width, height=height,
            metadata=metadata,
            guidance=guidance,
        )
        cam_infos.append(cam_info)

    train_cam_infos = [c for c in cam_infos if not c.metadata["is_val"]]
    test_cam_infos = [c for c in cam_infos if c.metadata["is_val"]]

    for cam in sorted(set(cams)):
        camera_timestamps[cam]["train_timestamps"] = sorted(camera_timestamps[cam]["train_timestamps"])
        camera_timestamps[cam]["test_timestamps"] = sorted(camera_timestamps[cam]["test_timestamps"])
    scene_metadata["camera_timestamps"] = camera_timestamps

    novel_view_cam_infos = []

    # Scene extent
    if cfg.mode == "novel_view":
        nerf_normalization = getNerfppNorm(novel_view_cam_infos)
    else:
        nerf_normalization = getNerfppNorm(train_cam_infos)

    nerf_normalization["radius"] = max(nerf_normalization["radius"], 10)

    if cfg.data.get("extent", False):
        nerf_normalization["radius"] = cfg.data.extent

    cfg.data.extent = float(nerf_normalization["radius"])

    scene_metadata["scene_center"] = nerf_normalization["center"]
    scene_metadata["scene_radius"] = nerf_normalization["radius"]
    print(f"Scene extent: {nerf_normalization['radius']}")

    # Sphere normalization
    lidar_ply_path = os.path.join(cfg.model_path, "input_ply/points3D_lidar.ply")
    if os.path.exists(lidar_ply_path):
        sphere_pcd = fetchPly(lidar_ply_path)
    elif os.path.exists(bkgd_ply_path):
        sphere_pcd = fetchPly(bkgd_ply_path)
    else:
        cam_centers = []
        for cam_info in train_cam_infos:
            RT = np.eye(4)
            RT[:3, :3] = cam_info.R.T
            RT[:3, 3] = cam_info.T
            cam_centers.append(np.linalg.inv(RT)[:3, 3])
        cam_centers = np.asarray(cam_centers, dtype=np.float32)
        sphere_pcd = BasicPointCloud(
            points=cam_centers,
            colors=np.zeros_like(cam_centers),
            normals=np.zeros_like(cam_centers),
        )

    sphere_normalization = get_Sphere_Norm(sphere_pcd.points)
    scene_metadata["sphere_center"] = sphere_normalization["center"]
    scene_metadata["sphere_radius"] = sphere_normalization["radius"]
    print(f"Sphere extent: {sphere_normalization['radius']}")

    pcd = fetchPly(bkgd_ply_path) if os.path.exists(bkgd_ply_path) else None
    if cfg.mode == "train":
        if pcd is None:
            raise FileNotFoundError(
                f"Missing generated background point cloud: {bkgd_ply_path}"
            )
        point_cloud = pcd
    else:
        point_cloud = None
        bkgd_ply_path = None

    scene_info = SceneInfo(
        point_cloud=point_cloud,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=bkgd_ply_path,
        metadata=scene_metadata,
        novel_view_cameras=novel_view_cam_infos,
    )

    return scene_info
