"""T4 dataset reader for VAD-GS.

Reads TIER IV (nuScenes-compatible) T4 format datasets directly
from their JSON annotation files. Mirrors the structure of
my_drivestudio_readers.py for compatibility with the VAD-GS pipeline.
"""

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


def readT4Info(path, images="images", split_train=-1, split_test=-1, **kwargs):
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

    # Guidance data directories (may or may not exist for T4 datasets)
    dynamic_mask_dir = os.path.join(path, "sam_masks")
    bkgd_mask_dir = os.path.join(path, "sam_bkgd_masks")
    load_dynamic_mask = os.path.exists(dynamic_mask_dir) and os.path.exists(bkgd_mask_dir)

    sky_mask_dir = os.path.join(path, "sky_masks")
    load_sky_mask = (cfg.mode == "train") and os.path.exists(sky_mask_dir)

    lidar_depth_dir = os.path.join(path, "lidar_depth")
    load_lidar_depth = (cfg.mode == "train") and os.path.exists(lidar_depth_dir)

    mono_depth_dir = os.path.join(path, "depth")
    if not os.path.exists(mono_depth_dir):
        mono_depth_dir = os.path.join(path, "depth_v2")
    load_mono_depth = os.path.exists(mono_depth_dir)

    normal_dir = os.path.join(path, "normal_img")
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

    # Build CameraInfo list
    cam_infos = []
    for i in tqdm(range(len(exts)), desc="Loading T4 cameras"):
        ext = exts[i]
        ixt = ixts[i]
        c2w = c2ws[i]
        pose = poses[i]
        image_path = image_filenames[i]
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        width, height = image.size
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

        # Load dynamic mask
        if load_dynamic_mask:
            dynamic_mask_path = os.path.join(dynamic_mask_dir, f"{image_name}.png")
            if os.path.exists(dynamic_mask_path):
                dynamic_mask = cv2.imread(dynamic_mask_path)
                guidance["dynamic_mask"] = dynamic_mask

                seg_bkgd_mask_path = os.path.join(bkgd_mask_dir, f"{image_name}.png")
                if os.path.exists(seg_bkgd_mask_path):
                    seg_bkgd_mask = cv2.imread(seg_bkgd_mask_path)
                    guidance["seg_bkgd"] = seg_bkgd_mask

                guidance["obj_bound"] = Image.fromarray(obj_bounds[i])

        # Load lidar depth
        if load_lidar_depth:
            depth_path = os.path.join(lidar_depth_dir, f"{image_name}.npy")
            if os.path.exists(depth_path):
                depth = np.load(depth_path, allow_pickle=True)
                depth = dict(depth.item())
                mask = depth["mask"]
                value = depth["value"]
                depth_arr = np.zeros_like(mask).astype(np.float32)
                depth_arr[mask] = value
                guidance["lidar_depth"] = depth_arr

        # Load sky mask
        if load_sky_mask:
            sky_mask_path = os.path.join(sky_mask_dir, f"{image_name}.png")
            if os.path.exists(sky_mask_path):
                sky_mask = (cv2.imread(sky_mask_path)[..., 0]) > 0.0
                guidance["sky_mask"] = Image.fromarray(sky_mask)

        # Load mono depth (prefer .npz from Depth Anything V2, fallback to .png)
        if load_mono_depth:
            depth_npz_path = os.path.join(mono_depth_dir, f"{image_name}.npz")
            depth_png_path = os.path.join(mono_depth_dir, f"{image_name}.png")
            if os.path.exists(depth_npz_path):
                mono_depth_raw = np.load(depth_npz_path)["depth"]
                # Normalize to [0, 255] uint8 for PIL (higher value = farther)
                d_min, d_max = mono_depth_raw.min(), mono_depth_raw.max()
                if d_max - d_min > 1e-6:
                    mono_depth = ((mono_depth_raw - d_min) / (d_max - d_min) * 255).astype(np.uint8)
                else:
                    mono_depth = np.zeros_like(mono_depth_raw, dtype=np.uint8)
                guidance["mono_depth"] = Image.fromarray(mono_depth)
            elif os.path.exists(depth_png_path):
                mono_depth = 255 - cv2.imread(depth_png_path)[:, :, 0]
                guidance["mono_depth"] = Image.fromarray(mono_depth)

        # Load normal map
        if load_normal:
            normal_img_path = os.path.join(normal_dir, f"{image_name}.png")
            if os.path.exists(normal_img_path):
                tmp = cv2.imread(normal_img_path) / 255.0 * 2 - 1
                ref_norm = np.zeros(tmp.shape)
                ref_norm[:, :, 0] = -tmp[:, :, 2]
                ref_norm[:, :, 1] = -tmp[:, :, 1]
                ref_norm[:, :, 2] = -tmp[:, :, 0]
                guidance["mono_normal"] = ref_norm

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
