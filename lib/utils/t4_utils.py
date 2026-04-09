"""T4 dataset utilities for loading TIER IV (nuScenes-compatible) format data.

This module directly parses T4 JSON annotation files without requiring
the t4_devkit library. The output format matches drivestudio_utils.py's
generate_dataparser_outputs() for compatibility with the VAD-GS pipeline.
"""

import os
import json
import math
import numpy as np
import cv2
import torch
import open3d as o3d
from glob import glob
from tqdm import tqdm
from lib.config import cfg
from lib.utils.box_utils import bbox_to_corner3d, inbbox_points, get_bound_2d_mask
from lib.utils.colmap_utils import read_points3D_binary, read_extrinsics_binary
from lib.utils.data_utils import get_val_frames
from lib.utils.graphics_utils import get_rays, sphere_intersection
from lib.utils.general_utils import matrix_to_quaternion, quaternion_to_matrix_numpy
from lib.datasets.base_readers import storePly, get_Sphere_Norm

# Class name mapping (same as drivestudio_utils.py)
waymo_track2label = {
    "vehicle": 0, "Vehicle": 0,
    "pedestrian": 1, "Pedestrian": 1,
    "cyclist": 2,
    "sign": 3,
    "misc": -1,
}

# Default T4 camera channel names
T4_CAMERA_CHANNELS_DEFAULT = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

T4_LIDAR_CHANNEL_DEFAULT = "LIDAR_CONCAT"

# Re-export the lightweight resolver that lives in cfg_utils (no circular import)
from lib.utils.cfg_utils import _resolve_t4_dataset_path as resolve_t4_dataset_path


# ---------------------------------------------------------------------------
# T4 JSON table loading
# ---------------------------------------------------------------------------

def load_t4_tables(annotation_dir):
    """Load all T4 JSON annotation tables from the annotation/ directory."""
    tables = {}
    table_names = [
        "scene", "sample", "sample_data", "sample_annotation",
        "calibrated_sensor", "ego_pose", "sensor", "instance", "category",
    ]
    for name in table_names:
        path = os.path.join(annotation_dir, f"{name}.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                tables[name] = json.load(f)
        else:
            tables[name] = []
    return tables


def index_by_token(items):
    """Create a dict mapping token -> item for a list of T4 records."""
    return {item["token"]: item for item in items}


def sample_chain(scene, sample_by_token):
    """Return ordered list of samples for a scene."""
    ordered = []
    token = scene["first_sample_token"]
    while token:
        sample = sample_by_token[token]
        ordered.append(sample)
        token = sample.get("next", "")
    return ordered


def build_sample_channel_map(sample_data_items, calibrated_sensor_by_token, sensor_by_token):
    """Build mapping: sample_token -> {channel_name: (sample_data, calibrated_sensor)}."""
    mapping = {}
    for sd in sample_data_items:
        cs = calibrated_sensor_by_token.get(sd["calibrated_sensor_token"])
        if cs is None:
            continue
        sensor = sensor_by_token.get(cs["sensor_token"])
        if sensor is None:
            continue
        mapping.setdefault(sd["sample_token"], {})[sensor["channel"]] = (sd, cs)
    return mapping


def build_sample_annotation_map(sample_annotation_items):
    """Build mapping: sample_token -> [annotations]."""
    mapping = {}
    for ann in sample_annotation_items:
        mapping.setdefault(ann["sample_token"], []).append(ann)
    return mapping


# ---------------------------------------------------------------------------
# Quaternion / transform helpers
# ---------------------------------------------------------------------------

def quat_wxyz_to_rotmat(quat):
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
    w, x, y, z = quat
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float64,
    )


def make_transform(translation, rotation_wxyz):
    """Create 4x4 transform matrix from translation and quaternion [w,x,y,z]."""
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = quat_wxyz_to_rotmat(rotation_wxyz)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64)
    return transform


def simplify_category(name):
    """Simplify T4/nuScenes category name to vehicle/pedestrian/cyclist/misc."""
    name = name.lower()
    if "vehicle" in name:
        return "vehicle"
    if "pedestrian" in name:
        return "pedestrian"
    if "cycle" in name or "bicycle" in name or "motorcycle" in name:
        return "cyclist"
    return "misc"


# ---------------------------------------------------------------------------
# Camera calibration loading
# ---------------------------------------------------------------------------

def load_camera_info_t4(
    tables,
    dataset_root,
    samples,
    camera_channels,
    lidar_channel,
):
    """Extract camera calibration and ego poses from T4 tables.

    Returns:
        intrinsics: list of 3x3 np.ndarray per camera channel
        extrinsics: list of 4x4 np.ndarray per camera (cam-to-lidar at frame 0)
        ego_frame_poses: np.ndarray [num_frames, 4, 4] (lidar-to-world, normalized)
        ego_cam_poses: np.ndarray [num_cams, num_frames, 4, 4] (cam-to-world, normalized)
        image_sizes: list of (height, width) per camera channel
    """
    calibrated_sensor_by_token = index_by_token(tables["calibrated_sensor"])
    sensor_by_token = index_by_token(tables["sensor"])
    ego_pose_by_token = index_by_token(tables["ego_pose"])

    # Build channel -> calibrated_sensor mapping from sample_data
    # We need to find the calibrated_sensor for each camera channel
    channel_to_cs = {}
    channel_to_sensor_to_ego = {}
    for sd in tables["sample_data"]:
        cs = calibrated_sensor_by_token.get(sd["calibrated_sensor_token"])
        if cs is None:
            continue
        sensor = sensor_by_token.get(cs["sensor_token"])
        if sensor is None:
            continue
        ch = sensor["channel"]
        if ch not in channel_to_cs:
            channel_to_cs[ch] = cs
            channel_to_sensor_to_ego[ch] = make_transform(cs["translation"], cs["rotation"])

    # Build sample_data lookup: sample_token -> {channel: sample_data}
    sample_channel_map = build_sample_channel_map(
        tables["sample_data"], calibrated_sensor_by_token, sensor_by_token
    )

    num_cams = len(camera_channels)
    num_frames = len(samples)

    # Extract intrinsics per camera
    intrinsics = []
    for ch in camera_channels:
        cs = channel_to_cs.get(ch)
        if cs is None:
            raise ValueError(f"Camera channel {ch} not found in calibrated_sensor")
        K = np.array(cs["camera_intrinsic"], dtype=np.float64)
        intrinsics.append(K)

    # Extract image sizes from first sample's sample_data
    image_sizes = []
    first_sample_data = sample_channel_map.get(samples[0]["token"], {})
    for ch in camera_channels:
        if ch in first_sample_data:
            sd, _ = first_sample_data[ch]
            image_sizes.append((sd["height"], sd["width"]))
        else:
            image_sizes.append((900, 1600))  # fallback

    # Compute lidar_to_world for first frame (reference)
    first_sample_data = sample_channel_map.get(samples[0]["token"], {})
    if lidar_channel in first_sample_data:
        lidar_sd, lidar_cs = first_sample_data[lidar_channel]
        lidar_ego = ego_pose_by_token[lidar_sd["ego_pose_token"]]
        lidar_sensor_to_ego = make_transform(lidar_cs["translation"], lidar_cs["rotation"])
        lidar_to_world_start = make_transform(lidar_ego["translation"], lidar_ego["rotation"]) @ lidar_sensor_to_ego
    else:
        # Fallback: use ego_pose directly
        first_sd = list(first_sample_data.values())[0][0]
        ego = ego_pose_by_token[first_sd["ego_pose_token"]]
        lidar_to_world_start = make_transform(ego["translation"], ego["rotation"])
        lidar_sensor_to_ego = np.eye(4)

    # Compute per-frame ego poses and camera poses
    ego_frame_poses = []
    ego_cam_poses = [[] for _ in range(num_cams)]

    for frame_idx, sample in enumerate(samples):
        frame_data = sample_channel_map.get(sample["token"], {})

        # Lidar ego pose
        if lidar_channel in frame_data:
            lidar_sd, lidar_cs = frame_data[lidar_channel]
            lidar_ego = ego_pose_by_token[lidar_sd["ego_pose_token"]]
            l_sensor_to_ego = make_transform(lidar_cs["translation"], lidar_cs["rotation"])
            lidar_to_world = make_transform(lidar_ego["translation"], lidar_ego["rotation"]) @ l_sensor_to_ego
        else:
            # Use any available sensor's ego pose
            any_sd = list(frame_data.values())[0][0] if frame_data else None
            if any_sd:
                ego = ego_pose_by_token[any_sd["ego_pose_token"]]
                lidar_to_world = make_transform(ego["translation"], ego["rotation"])
            else:
                lidar_to_world = np.eye(4)

        # Normalize to first frame
        ego_pose_normalized = np.linalg.inv(lidar_to_world_start) @ lidar_to_world
        ego_frame_poses.append(ego_pose_normalized)

        # Camera poses
        for cam_idx, ch in enumerate(camera_channels):
            if ch in frame_data:
                cam_sd, cam_cs = frame_data[ch]
                cam_ego = ego_pose_by_token[cam_sd["ego_pose_token"]]
                cam_sensor_to_ego = make_transform(cam_cs["translation"], cam_cs["rotation"])
                cam_to_world = make_transform(cam_ego["translation"], cam_ego["rotation"]) @ cam_sensor_to_ego
                cam_to_world_normalized = np.linalg.inv(lidar_to_world_start) @ cam_to_world
            else:
                # Fallback: use ego_pose @ static calibration
                cam_to_world_normalized = ego_pose_normalized @ channel_to_sensor_to_ego.get(ch, np.eye(4))
            ego_cam_poses[cam_idx].append(cam_to_world_normalized)

    ego_frame_poses = np.array(ego_frame_poses)
    ego_cam_poses = np.array(ego_cam_poses)

    # Compute extrinsics (cam-to-lidar at frame 0)
    extrinsics = []
    for cam_idx in range(num_cams):
        cam_to_world_frame0 = ego_cam_poses[cam_idx, 0]
        # extrinsic = inv(lidar_to_world_start) @ cam_to_world_frame0
        # But since ego_cam_poses is already normalized, this is just cam_to_world_frame0
        # relative to the first lidar frame
        ext = cam_to_world_frame0.copy()
        extrinsics.append(ext)

    return intrinsics, extrinsics, ego_frame_poses, ego_cam_poses, image_sizes, lidar_to_world_start


# ---------------------------------------------------------------------------
# Object tracking
# ---------------------------------------------------------------------------

def make_obj_pose_ds(ego_pose_world, obj_pose_world):
    """Compute object pose in vehicle and world frames (7D: xyz + quaternion)."""
    obj_pose_vehicle = np.matmul(np.linalg.inv(ego_pose_world), obj_pose_world)

    obj_rotation_vehicle = torch.from_numpy(obj_pose_vehicle[:3, :3]).float().unsqueeze(0)
    obj_quaternion_vehicle = matrix_to_quaternion(obj_rotation_vehicle).squeeze(0).numpy()
    obj_quaternion_vehicle = obj_quaternion_vehicle / np.linalg.norm(obj_quaternion_vehicle)
    obj_position_vehicle = obj_pose_vehicle[:3, 3]
    obj_pose_vehicle_7d = np.concatenate([obj_position_vehicle, obj_quaternion_vehicle])

    obj_rotation_world = torch.from_numpy(obj_pose_world[:3, :3]).float().unsqueeze(0)
    obj_quaternion_world = matrix_to_quaternion(obj_rotation_world).squeeze(0).numpy()
    obj_quaternion_world = obj_quaternion_world / np.linalg.norm(obj_quaternion_world)
    obj_position_world = obj_pose_world[:3, 3]
    obj_pose_world_7d = np.concatenate([obj_position_world, obj_quaternion_world])

    return obj_pose_vehicle_7d, obj_pose_world_7d


def get_obj_pose_tracking_t4(
    tables,
    samples,
    selected_frames,
    ego_frame_poses,
    lidar_to_world_start,
    cameras=None,
):
    """Extract object tracking data from T4 annotations.

    Returns:
        objects_tracklets_world: np.ndarray [num_frames, max_obj, 8]
        objects_tracklets_vehicle: np.ndarray [num_frames, max_obj, 8]
        objects_info: dict with track metadata
    """
    instance_by_token = index_by_token(tables["instance"])
    category_by_token = index_by_token(tables["category"])
    sample_annotation_map = build_sample_annotation_map(tables["sample_annotation"])

    start_frame, end_frame = selected_frames[0], selected_frames[1]
    num_frames = end_frame - start_frame + 1

    objects_info = {}
    n_obj_in_frame = np.zeros(len(samples))

    # First pass: count objects and build info
    for frame_idx, sample in enumerate(samples):
        annotations = sample_annotation_map.get(sample["token"], [])
        n_obj_in_frame[frame_idx] = len(annotations)

        for ann in annotations:
            instance_token = ann["instance_token"]
            instance = instance_by_token.get(instance_token, {})
            category = category_by_token.get(instance.get("category_token", ""), {})
            class_name = simplify_category(category.get("name", "misc"))
            if class_name == "misc":
                continue

            track_id = int(instance_token[:8], 16)
            # T4/nuScenes size = [width, length, height]
            size = ann["size"]
            box_length = size[1]
            box_width = size[0]
            box_height = size[2]

            if track_id not in objects_info:
                objects_info[track_id] = {
                    "track_id": track_id,
                    "class": class_name,
                    "class_label": waymo_track2label.get(class_name, -1),
                    "height": box_height,
                    "width": box_width,
                    "length": box_length,
                }
            else:
                objects_info[track_id]["height"] = max(objects_info[track_id]["height"], box_height)
                objects_info[track_id]["width"] = max(objects_info[track_id]["width"], box_width)
                objects_info[track_id]["length"] = max(objects_info[track_id]["length"], box_length)

    # Second pass: build tracklets
    max_obj_per_frame = int(n_obj_in_frame[start_frame:end_frame + 1].max()) if num_frames > 0 else 1
    max_obj_per_frame = max(max_obj_per_frame, 1)

    visible_objects_ids = np.ones([num_frames, max_obj_per_frame]) * -1.0
    visible_objects_pose_vehicle = np.ones([num_frames, max_obj_per_frame, 7]) * -1.0
    visible_objects_pose_world = np.ones([num_frames, max_obj_per_frame, 7]) * -1.0

    for frame_idx, sample in enumerate(samples):
        if frame_idx < start_frame or frame_idx > end_frame:
            continue
        relative_idx = frame_idx - start_frame

        annotations = sample_annotation_map.get(sample["token"], [])
        for ann in annotations:
            instance_token = ann["instance_token"]
            instance = instance_by_token.get(instance_token, {})
            category = category_by_token.get(instance.get("category_token", ""), {})
            class_name = simplify_category(category.get("name", "misc"))
            if class_name == "misc":
                continue

            track_id = int(instance_token[:8], 16)
            if track_id not in objects_info:
                continue

            # Object pose in global coordinates
            obj_to_world = make_transform(ann["translation"], ann["rotation"])
            # Normalize to first frame
            obj_to_world_normalized = np.linalg.inv(lidar_to_world_start) @ obj_to_world

            ego_pose_world = ego_frame_poses[frame_idx]
            obj_pose_vehicle, obj_pose_world = make_obj_pose_ds(ego_pose_world, obj_to_world_normalized)

            free_slots = np.argwhere(visible_objects_ids[relative_idx, :] < 0)
            if len(free_slots) == 0:
                continue
            obj_column = free_slots.min()

            visible_objects_ids[relative_idx, obj_column] = track_id
            visible_objects_pose_vehicle[relative_idx, obj_column] = obj_pose_vehicle
            visible_objects_pose_world[relative_idx, obj_column] = obj_pose_world

    # Remove static objects
    print("Removing static objects")
    for key in list(objects_info.keys()):
        all_obj_idx = np.where(visible_objects_ids == key)
        if len(all_obj_idx[0]) > 0:
            obj_world_positions = visible_objects_pose_world[all_obj_idx][:, :3]
            distance = np.linalg.norm(obj_world_positions[0] - obj_world_positions[-1])
            dynamic = np.any(np.std(obj_world_positions, axis=0) > 0.5) or distance > 2
            if not dynamic:
                visible_objects_ids[all_obj_idx] = -1.0
                visible_objects_pose_vehicle[all_obj_idx] = -1.0
                visible_objects_pose_world[all_obj_idx] = -1.0
                objects_info.pop(key)
        else:
            objects_info.pop(key)

    # Clip max_num_obj
    mask = visible_objects_ids >= 0
    max_obj_per_frame_new = int(np.sum(mask, axis=1).max())
    print("Max obj per frame:", max_obj_per_frame_new)

    if max_obj_per_frame_new == 0:
        print("No moving obj in current sequence, make dummy visible objects")
        visible_objects_ids = np.ones([num_frames, 1]) * -1.0
        visible_objects_pose_world = np.ones([num_frames, 1, 7]) * -1.0
        visible_objects_pose_vehicle = np.ones([num_frames, 1, 7]) * -1.0
    elif max_obj_per_frame_new < max_obj_per_frame:
        ids_new = np.ones([num_frames, max_obj_per_frame_new]) * -1.0
        pose_vehicle_new = np.ones([num_frames, max_obj_per_frame_new, 7]) * -1.0
        pose_world_new = np.ones([num_frames, max_obj_per_frame_new, 7]) * -1.0
        for fi in range(num_frames):
            for y in range(max_obj_per_frame):
                obj_id = visible_objects_ids[fi, y]
                if obj_id >= 0:
                    obj_col = np.argwhere(ids_new[fi, :] < 0).min()
                    ids_new[fi, obj_col] = obj_id
                    pose_vehicle_new[fi, obj_col] = visible_objects_pose_vehicle[fi, y]
                    pose_world_new[fi, obj_col] = visible_objects_pose_world[fi, y]
        visible_objects_ids = ids_new
        visible_objects_pose_vehicle = pose_vehicle_new
        visible_objects_pose_world = pose_world_new

    # Remap track_ids to sequential small integers (0, 1, 2, ...)
    # Required because dynamic masks store track_id as uint8 pixel values (0-254)
    old_to_new = {}
    for new_id, old_id in enumerate(sorted(objects_info.keys())):
        old_to_new[old_id] = new_id

    if old_to_new:
        remapped_info = {}
        for old_id, new_id in old_to_new.items():
            info = objects_info[old_id]
            info["track_id"] = new_id
            info["original_instance_token_prefix"] = old_id
            remapped_info[new_id] = info
        objects_info = remapped_info

        # Remap visible_objects_ids
        for fi in range(visible_objects_ids.shape[0]):
            for col in range(visible_objects_ids.shape[1]):
                old_id = int(visible_objects_ids[fi, col])
                if old_id >= 0 and old_id in old_to_new:
                    visible_objects_ids[fi, col] = old_to_new[old_id]

        print(f"Remapped {len(old_to_new)} track_ids to sequential [0, {len(old_to_new)-1}]")

    box_scale = cfg.data.get("box_scale", 1.0)
    print("box scale:", box_scale)

    frames = np.arange(start_frame, end_frame + 1, dtype=np.int32)

    # Postprocess object_info
    for key in objects_info.keys():
        obj = objects_info[key]
        obj["deformable"] = obj["class"] == "pedestrian"
        obj["width"] = obj["width"] * box_scale
        obj["length"] = obj["length"] * box_scale

        obj_frame_idx = np.argwhere(visible_objects_ids == key)[:, 0].astype(np.int32)
        obj_frames = frames[obj_frame_idx]
        obj["start_frame"] = int(np.min(obj_frames))
        obj["end_frame"] = int(np.max(obj_frames))

    # Build tracklet arrays [num_frames, max_obj, 8] = [track_id, x, y, z, qw, qx, qy, qz]
    objects_tracklets_world = np.concatenate(
        [visible_objects_ids[..., None], visible_objects_pose_world], axis=-1
    )
    objects_tracklets_vehicle = np.concatenate(
        [visible_objects_ids[..., None], visible_objects_pose_vehicle], axis=-1
    )

    return objects_tracklets_world, objects_tracklets_vehicle, objects_info


# ---------------------------------------------------------------------------
# LiDAR data loading
# ---------------------------------------------------------------------------

def read_lidar_t4(path):
    """Read T4 LiDAR file (.pcd.bin or .bin). Returns (N, 3) xyz in sensor frame."""
    raw = np.fromfile(path, dtype=np.float32)
    # T4: (N, 5) = [x, y, z, intensity, ring_idx]
    # nuScenes: (N, 5) = [x, y, z, intensity, ring_idx]
    if raw.size % 5 == 0:
        raw = raw.reshape(-1, 5)
    elif raw.size % 4 == 0:
        raw = raw.reshape(-1, 4)
    else:
        raw = raw.reshape(-1, 5)
    return raw[:, :3]


def project_lidar_to_cameras(points_xyz, cam_to_worlds, intrinsics_list, image_sizes, cameras):
    """Project LiDAR points (in world frame) to multiple cameras.

    Returns projection array matching DriveStudio's pointcloud.npz format:
    (N, 6) = [cam1, u1, v1, cam2, u2, v2]
    """
    N = points_xyz.shape[0]
    projection = np.full((N, 6), -1, dtype=np.int16)
    pts_h = np.concatenate([points_xyz, np.ones((N, 1), dtype=np.float64)], axis=1)

    for cam_idx, cam_id in enumerate(cameras):
        c2w = cam_to_worlds[cam_idx]
        K = intrinsics_list[cam_idx]
        H, W = image_sizes[cam_idx]

        w2c = np.linalg.inv(c2w)
        pts_cam = (pts_h @ w2c.T)[:, :3]
        depth = pts_cam[:, 2]
        valid = depth > 1e-6

        uvw = pts_cam @ K.T
        uv = np.zeros((N, 2), dtype=np.float64)
        uv[valid, 0] = uvw[valid, 0] / depth[valid]
        uv[valid, 1] = uvw[valid, 1] / depth[valid]

        visible = valid & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)

        first_slot = visible & (projection[:, 0] < 0)
        second_slot = visible & (projection[:, 0] >= 0) & (projection[:, 3] < 0)
        projection[first_slot, 0] = cam_id
        projection[first_slot, 1] = np.round(uv[first_slot, 0]).astype(np.int16)
        projection[first_slot, 2] = np.round(uv[first_slot, 1]).astype(np.int16)
        projection[second_slot, 3] = cam_id
        projection[second_slot, 4] = np.round(uv[second_slot, 0]).astype(np.int16)
        projection[second_slot, 5] = np.round(uv[second_slot, 1]).astype(np.int16)

    return projection


# ---------------------------------------------------------------------------
# Main data parser
# ---------------------------------------------------------------------------

def generate_dataparser_outputs_t4(
    datadir,
    selected_frames=None,
    build_pointcloud=True,
    cameras=None,
    camera_channels=None,
    lidar_channel=None,
    scene_index=0,
):
    """Parse T4 dataset and produce outputs matching DriveStudio format.

    Args:
        datadir: Root directory of the T4 dataset (containing annotation/ and data/)
        selected_frames: [start, end] frame range or None for all
        build_pointcloud: Whether to build point cloud PLY files
        cameras: List of camera indices to use [0, 1, 2, ...]
        camera_channels: List of camera channel names
        lidar_channel: LiDAR channel name
        scene_index: Index of the scene to load

    Returns:
        dict with same keys as drivestudio_utils.generate_dataparser_outputs()
    """
    # Resolve dataset ID to path if needed
    revision = cfg.data.get("revision", 0)
    datadir = resolve_t4_dataset_path(datadir, revision=revision)

    if camera_channels is None:
        camera_channels = cfg.data.get("camera_channels", T4_CAMERA_CHANNELS_DEFAULT)
    if lidar_channel is None:
        lidar_channel = cfg.data.get("lidar_channel", T4_LIDAR_CHANNEL_DEFAULT)
    if cameras is None:
        cameras = list(range(len(camera_channels)))

    num_cameras = len(cameras)

    # Find annotation directory
    annotation_dir = os.path.join(datadir, "annotation")
    if not os.path.exists(annotation_dir):
        # Try looking for JSON files directly in datadir
        if os.path.exists(os.path.join(datadir, "sample.json")):
            annotation_dir = datadir
        else:
            raise FileNotFoundError(
                f"Cannot find T4 annotation directory at {annotation_dir}"
            )

    print(f"Loading T4 tables from {annotation_dir}")
    tables = load_t4_tables(annotation_dir)

    # Index tables
    sample_by_token = index_by_token(tables["sample"])
    calibrated_sensor_by_token = index_by_token(tables["calibrated_sensor"])
    sensor_by_token = index_by_token(tables["sensor"])
    ego_pose_by_token = index_by_token(tables["ego_pose"])

    # Get scene
    if len(tables["scene"]) == 0:
        raise ValueError("No scenes found in T4 dataset")
    scene = tables["scene"][min(scene_index, len(tables["scene"]) - 1)]
    samples = sample_chain(scene, sample_by_token)
    print(f"Scene: {scene.get('name', 'unknown')}, {len(samples)} samples")

    # Auto-detect camera channels if needed
    sample_channel_map = build_sample_channel_map(
        tables["sample_data"], calibrated_sensor_by_token, sensor_by_token
    )

    # Determine available camera channels
    first_frame_data = sample_channel_map.get(samples[0]["token"], {})
    available_cameras = [ch for ch in camera_channels if ch in first_frame_data]
    if not available_cameras:
        # Try to auto-detect from sensor.json
        all_cam_channels = sorted(
            sensor["channel"]
            for sensor in tables["sensor"]
            if sensor.get("modality", "") == "camera"
        )
        available_cameras = all_cam_channels[:len(cameras)]
        if not available_cameras:
            raise ValueError(f"No camera channels found. Available: {list(first_frame_data.keys())}")
        print(f"Auto-detected camera channels: {available_cameras}")
    camera_channels = available_cameras
    cameras = list(range(len(camera_channels)))
    num_cameras = len(cameras)

    # Frame range
    num_frames_all = len(samples)
    if selected_frames is None:
        start_frame = 0
        end_frame = num_frames_all - 1
    else:
        start_frame = max(0, selected_frames[0])
        end_frame = min(num_frames_all - 1, selected_frames[1])
    selected_frames = [start_frame, end_frame]
    num_frames = end_frame - start_frame + 1

    # Load camera calibration
    print("Loading camera calibration...")
    intrinsics, extrinsics, ego_frame_poses, ego_cam_poses, image_sizes, lidar_to_world_start = \
        load_camera_info_t4(tables, datadir, samples, camera_channels, lidar_channel)

    # Build per-image data
    frames = []
    frames_idx = []
    cams = []
    image_filenames = []
    ixts = []
    exts = []
    poses = []
    c2ws = []
    cams_timestamps = []
    frames_timestamps = list(range(start_frame, end_frame + 1))

    # Map image dimensions per camera
    image_heights = [sz[0] for sz in image_sizes]
    image_widths = [sz[1] for sz in image_sizes]

    for frame_idx in range(start_frame, end_frame + 1):
        sample = samples[frame_idx]
        frame_data = sample_channel_map.get(sample["token"], {})

        for cam_idx, ch in enumerate(camera_channels):
            cam = cameras[cam_idx]
            if ch not in frame_data:
                continue

            sd, cs = frame_data[ch]
            image_path = os.path.join(datadir, sd["filename"])
            if not os.path.exists(image_path):
                continue

            ixt = intrinsics[cam_idx]
            ext = extrinsics[cam_idx]
            pose = ego_frame_poses[frame_idx]
            c2w = ego_cam_poses[cam_idx, frame_idx]

            frames.append(frame_idx)
            frames_idx.append(frame_idx - start_frame)
            cams.append(cam)
            image_filenames.append(image_path)
            ixts.append(ixt)
            exts.append(ext)
            poses.append(pose)
            c2ws.append(c2w)
            cams_timestamps.append(frame_idx)  # Use frame index as timestamp

    exts = np.stack(exts, axis=0)
    ixts = np.stack(ixts, axis=0)
    poses = np.stack(poses, axis=0)
    c2ws = np.stack(c2ws, axis=0)

    timestamp_offset = 0
    cams_timestamps = np.array(cams_timestamps) - timestamp_offset
    frames_timestamps = np.array(frames_timestamps) - timestamp_offset
    min_timestamp = min(cams_timestamps.min(), frames_timestamps.min())
    max_timestamp = max(cams_timestamps.max(), frames_timestamps.max())

    # Object tracking
    print("Loading object tracking...")
    _, object_tracklets_vehicle, object_info = get_obj_pose_tracking_t4(
        tables, samples, selected_frames, ego_frame_poses, lidar_to_world_start, cameras
    )

    for track_id in object_info.keys():
        object_start_frame = object_info[track_id]["start_frame"]
        object_end_frame = object_info[track_id]["end_frame"]
        object_info[track_id]["start_timestamp"] = max(object_start_frame, min_timestamp)
        object_info[track_id]["end_timestamp"] = min(object_end_frame, max_timestamp)

    result = dict()
    result["num_frames"] = num_frames
    result["exts"] = exts
    result["ixts"] = ixts
    result["poses"] = poses
    result["c2ws"] = c2ws
    result["obj_tracklets"] = object_tracklets_vehicle
    result["obj_info"] = object_info
    result["frames"] = frames
    result["cams"] = cams
    result["frames_idx"] = frames_idx
    result["image_filenames"] = image_filenames
    result["cams_timestamps"] = cams_timestamps
    result["tracklet_timestamps"] = frames_timestamps
    result["ego_frame_poses"] = ego_frame_poses

    # Compute object bounding masks
    print("Computing object bounding masks...")
    obj_bounds = []
    obj_view_dict = {}
    for i, image_filename in tqdm(enumerate(image_filenames)):
        cam = cams[i]
        h, w = image_heights[cam], image_widths[cam]
        obj_bound = np.zeros((h, w), dtype=np.uint8)
        obj_tracklets = object_tracklets_vehicle[frames_idx[i]]
        ixt, ext = ixts[i], exts[i]

        for obj_tracklet in obj_tracklets:
            track_id = int(obj_tracklet[0])
            if track_id >= 0 and track_id in object_info:
                obj_pose_vehicle = np.eye(4)
                obj_pose_vehicle[:3, :3] = quaternion_to_matrix_numpy(obj_tracklet[4:8])
                obj_pose_vehicle[:3, 3] = obj_tracklet[1:4]
                obj_length = object_info[track_id]["length"]
                obj_width = object_info[track_id]["width"]
                obj_height = object_info[track_id]["height"]
                bbox = np.array(
                    [[-obj_length, -obj_width, -obj_height],
                     [obj_length, obj_width, obj_height]]
                ) * 0.5
                corners_local = bbox_to_corner3d(bbox)
                corners_local = np.concatenate(
                    [corners_local, np.ones_like(corners_local[..., :1])], axis=-1
                )
                corners_vehicle = corners_local @ obj_pose_vehicle.T
                mask = get_bound_2d_mask(
                    corners_3d=corners_vehicle[..., :3],
                    K=ixt,
                    pose=np.linalg.inv(ext),
                    H=h, W=w,
                )
                obj_bound = np.logical_or(obj_bound, mask)

                if mask.sum() < 20 * 20 * 10:
                    continue
                if track_id not in obj_view_dict:
                    obj_view_dict[track_id] = {}
                obj_view_dict[track_id][i] = [mask.sum(), obj_pose_vehicle]

        obj_bounds.append(obj_bound)

    result["obj_bounds"] = obj_bounds
    result["obj_view_dict"] = obj_view_dict

    # Build point cloud
    if build_pointcloud:
        _build_pointcloud_t4(
            result, datadir, tables, samples, sample_channel_map,
            ego_pose_by_token, calibrated_sensor_by_token, sensor_by_token,
            camera_channels, lidar_channel, cameras,
            start_frame, end_frame, num_cameras, num_frames,
            intrinsics, extrinsics, ego_frame_poses, ego_cam_poses,
            image_filenames, image_heights, image_widths,
            object_tracklets_vehicle, object_info, lidar_to_world_start,
        )

    return result


def _build_pointcloud_t4(
    result, datadir, tables, samples, sample_channel_map,
    ego_pose_by_token, calibrated_sensor_by_token, sensor_by_token,
    camera_channels, lidar_channel, cameras,
    start_frame, end_frame, num_cameras, num_frames,
    intrinsics, extrinsics, ego_frame_poses, ego_cam_poses,
    image_filenames, image_heights, image_widths,
    object_tracklets_vehicle, object_info, lidar_to_world_start,
):
    """Build point cloud from T4 LiDAR data (mirrors drivestudio_utils.py logic)."""
    # Run COLMAP first (use T4-specific COLMAP runner)
    colmap_basedir = os.path.join(f"{cfg.model_path}/colmap")
    if not os.path.exists(os.path.join(colmap_basedir, "triangulated/sparse/model")):
        from script.t4.colmap_t4 import run_colmap_t4
        run_colmap_t4(result, extrinsics_list=extrinsics)

    print("Building point cloud from T4 LiDAR data...")
    pointcloud_dir = os.path.join(cfg.model_path, "input_ply")
    os.makedirs(pointcloud_dir, exist_ok=True)

    points_xyz_dict = {"bkgd": []}
    points_rgb_dict = {"bkgd": []}
    points_normal_dict = {"bkgd": []}
    points_view_dict = {"bkgd": []}

    for track_id in object_info.keys():
        points_xyz_dict[f"obj_{track_id:03d}"] = []
        points_rgb_dict[f"obj_{track_id:03d}"] = []
        points_normal_dict[f"obj_{track_id:03d}"] = []
        points_view_dict[f"obj_{track_id:03d}"] = []

    # Load COLMAP points
    points_colmap_path = os.path.join(colmap_basedir, "triangulated/sparse/model/points3D.bin")
    try:
        points_colmap_xyz, points_colmap_rgb, points_colmap_error, points_colmap_tracks = read_points3D_binary(points_colmap_path)
        points_colmap_rgb = points_colmap_rgb / 255.0
        colmap_images = read_extrinsics_binary(points_colmap_path.replace("points3D", "images"))
        _mask = points_colmap_error[:, 0] < 0.6
        points_colmap_xyz = points_colmap_xyz[_mask]
        points_colmap_rgb = points_colmap_rgb[_mask]
        points_colmap_tracks = [arr for arr, m in zip(points_colmap_tracks, _mask) if m]
        has_colmap = True
    except Exception:
        print("No COLMAP point cloud available")
        has_colmap = False

    # Check for normal maps
    normal_dir = os.path.join(datadir, "normal_img")
    has_normals = os.path.exists(normal_dir)
    if not has_normals:
        print("Warning: normal_img/ not found. Using zero normals for point cloud.")

    c2ws_all = result["c2ws"]
    ixts_all = result["ixts"]
    cams_list = result["cams"]
    N_VIEWS = num_frames * num_cameras

    normals_world_all = []

    # Process each frame
    for i, frame in tqdm(enumerate(range(start_frame, end_frame + 1)), desc="Building point cloud"):
        idxs = list(range(i * num_cameras, (i + 1) * num_cameras))
        if max(idxs) >= len(image_filenames):
            break
        cams_frame = [cams_list[idx] for idx in idxs]
        image_filenames_frame = [image_filenames[idx] for idx in idxs]

        # Load LiDAR points for this frame
        sample = samples[frame]
        frame_data = sample_channel_map.get(sample["token"], {})

        if lidar_channel not in frame_data:
            continue

        lidar_sd, lidar_cs = frame_data[lidar_channel]
        lidar_path = os.path.join(datadir, lidar_sd["filename"])
        if not os.path.exists(lidar_path):
            continue

        lidar_points_sensor = read_lidar_t4(lidar_path)

        # Transform LiDAR points: sensor -> ego -> world (normalized)
        lidar_ego_pose = ego_pose_by_token[lidar_sd["ego_pose_token"]]
        l_sensor_to_ego = make_transform(lidar_cs["translation"], lidar_cs["rotation"])
        lidar_to_world = make_transform(lidar_ego_pose["translation"], lidar_ego_pose["rotation"]) @ l_sensor_to_ego
        lidar_to_world_norm = np.linalg.inv(lidar_to_world_start) @ lidar_to_world

        # Project to cameras to filter visible points
        cam_to_worlds_frame = [ego_cam_poses[cam_idx, frame] for cam_idx in range(num_cameras)]
        cam_intrinsics_frame = [intrinsics[cam_idx] for cam_idx in range(num_cameras)]
        cam_sizes_frame = [(image_heights[cam_idx], image_widths[cam_idx]) for cam_idx in range(num_cameras)]

        # Transform points to world coordinates
        pts_h = np.concatenate([lidar_points_sensor, np.ones((lidar_points_sensor.shape[0], 1))], axis=1)
        points_xyz_world = pts_h @ lidar_to_world_norm.T

        # Filter: keep points visible by at least one camera
        projection = project_lidar_to_cameras(
            points_xyz_world[:, :3], cam_to_worlds_frame, cam_intrinsics_frame, cam_sizes_frame, cameras
        )
        mask = np.array([projection[j, 0] in cameras or projection[j, 3] in cameras for j in range(projection.shape[0])]).astype(bool)

        points_xyz_vehicle = pts_h[mask]  # Keep in homogeneous sensor coords for consistency
        points_xyz_world_filtered = points_xyz_world[mask]

        ego_pose = ego_frame_poses[frame]

        points_rgb = np.ones_like(points_xyz_vehicle[:, :3])
        points_normal = np.zeros_like(points_xyz_vehicle[:, :3])
        points_visibility = np.zeros([points_xyz_vehicle.shape[0], N_VIEWS], dtype=bool)

        for cam, image_filename, idx in zip(cams_frame, image_filenames_frame, idxs):
            image = cv2.imread(image_filename)[..., [2, 1, 0]] / 255.0

            if has_normals:
                normal_filename = image_filename.replace("images", "normal_img")
                # Handle different extensions
                for ext_try in [".png", ".jpg"]:
                    nf = os.path.splitext(normal_filename)[0] + ext_try
                    if os.path.exists(nf):
                        normal_filename = nf
                        break
                if os.path.exists(normal_filename):
                    normal_dsine = cv2.imread(normal_filename) / 255.0 * 2 - 1
                    normals_transformed = np.zeros_like(normal_dsine)
                    normals_transformed[..., 0] = -normal_dsine[..., 2]
                    normals_transformed[..., 1] = -normal_dsine[..., 1]
                    normals_transformed[..., 2] = -normal_dsine[..., 0]
                else:
                    normals_transformed = np.zeros_like(image)
            else:
                normals_transformed = np.zeros_like(image)

            ixt = ixts_all[idx]
            c2w = c2ws_all[idx]
            normals_world = normals_transformed @ c2w[:3, :3].T
            normals_world_all.append(normals_world)

            # Project points to this camera
            view_pos_world = np.concatenate(
                [points_xyz_world_filtered[:, :3], np.ones_like(points_xyz_world_filtered[:, :1])], axis=-1
            )
            view_pos_cam = view_pos_world @ np.linalg.inv(c2w).T
            tmp = view_pos_cam[:, :3] @ ixt.T
            us = tmp[:, 0] / tmp[:, 2]
            vs = tmp[:, 1] / tmp[:, 2]

            vis_mask = (us >= 0) & (us < image.shape[1]) & (vs >= 0) & (vs < image.shape[0]) & (tmp[:, 2] > 2)

            mask_projw = us.astype(np.int16)[vis_mask]
            mask_projh = vs.astype(np.int16)[vis_mask]

            mask_rgb = image[mask_projh, mask_projw]
            mask_normal = normals_world[mask_projh, mask_projw]

            points_rgb[vis_mask] = mask_rgb
            points_normal[vis_mask] = mask_normal
            points_visibility[vis_mask, idx] = True

        # Filter points in object bounding boxes
        points_xyz_obj_mask = np.zeros(points_xyz_vehicle.shape[0], dtype=bool)

        for tracklet in object_tracklets_vehicle[i]:
            track_id = int(tracklet[0])
            if track_id >= 0 and track_id in object_info:
                obj_pose_vehicle = np.eye(4)
                obj_pose_vehicle[:3, :3] = quaternion_to_matrix_numpy(tracklet[4:8])
                obj_pose_vehicle[:3, 3] = tracklet[1:4]
                vehicle2local = np.linalg.inv(obj_pose_vehicle)

                # Note: points_xyz_vehicle here is in sensor/ego frame
                # Transform world points to object local frame via vehicle frame
                pts_world_h = np.concatenate(
                    [points_xyz_world_filtered[:, :3], np.ones_like(points_xyz_world_filtered[:, :1])], axis=-1
                )
                pts_vehicle_h = pts_world_h @ np.linalg.inv(ego_pose).T
                points_xyz_obj = pts_vehicle_h @ vehicle2local.T
                points_xyz_obj = points_xyz_obj[..., :3]

                if has_normals:
                    points_normal_obj = points_normal @ np.linalg.inv(ego_pose[:3, :3]).T @ vehicle2local[:3, :3].T
                else:
                    points_normal_obj = np.zeros_like(points_normal)

                length = object_info[track_id]["length"]
                width = object_info[track_id]["width"]
                height = object_info[track_id]["height"] * 1.2
                bbox = [[-length / 2, -width / 2, -height / 2], [length / 2, width / 2, height / 2]]
                obj_corners_3d_local = bbox_to_corner3d(bbox)

                points_xyz_inbbox = inbbox_points(points_xyz_obj, obj_corners_3d_local)
                points_xyz_obj_mask = np.logical_or(points_xyz_obj_mask, points_xyz_inbbox)

                points_xyz_dict[f"obj_{track_id:03d}"].append(points_xyz_obj[points_xyz_inbbox])
                points_rgb_dict[f"obj_{track_id:03d}"].append(points_rgb[points_xyz_inbbox])
                points_normal_dict[f"obj_{track_id:03d}"].append(points_normal_obj[points_xyz_inbbox])
                points_view_dict[f"obj_{track_id:03d}"].append(points_visibility[points_xyz_inbbox])

        points_lidar_xyz = points_xyz_world_filtered[~points_xyz_obj_mask][..., :3]
        points_lidar_rgb = points_rgb[~points_xyz_obj_mask]
        points_lidar_normal = points_normal[~points_xyz_obj_mask]
        points_lidar_visibility = points_visibility[~points_xyz_obj_mask]

        points_xyz_dict["bkgd"].append(points_lidar_xyz)
        points_rgb_dict["bkgd"].append(points_lidar_rgb)
        points_normal_dict["bkgd"].append(points_lidar_normal)
        points_view_dict["bkgd"].append(points_lidar_visibility)

    # Voxelize and save
    if not points_xyz_dict["bkgd"]:
        print("Warning: No LiDAR points were loaded. Skipping point cloud generation.")
        return

    initial_num_obj = 20000
    voxel_size = 0.15

    # Background points
    points_bkgd_lidar_xyz = np.concatenate(points_xyz_dict["bkgd"], axis=0)
    points_bkgd_lidar_rgb = np.concatenate(points_rgb_dict["bkgd"], axis=0)
    points_bkgd_lidar_normal = np.concatenate(points_normal_dict["bkgd"], axis=0)
    points_bkgd_lidar_view = np.concatenate(points_view_dict["bkgd"], axis=0)

    lidar_sphere_normalization = get_Sphere_Norm(points_bkgd_lidar_xyz)
    sphere_center = lidar_sphere_normalization["center"]
    sphere_radius = lidar_sphere_normalization["radius"]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_bkgd_lidar_xyz[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(points_bkgd_lidar_rgb)

    downsampled_pcd, _, point_indices_for_each_voxel = pcd.voxel_down_sample_and_trace(
        voxel_size=voxel_size,
        min_bound=(pcd.get_min_bound() // voxel_size) * voxel_size,
        max_bound=(pcd.get_max_bound() // voxel_size + 1) * voxel_size,
    )
    downsample_outlier_pcd, downsample_outlier_indice = downsampled_pcd.remove_radius_outlier(
        nb_points=10, radius=0.5
    )

    views_for_voxels = []
    normals_for_voxels = []
    for _tmp in downsample_outlier_indice:
        indices = point_indices_for_each_voxel[_tmp]
        views_voxel = np.zeros([N_VIEWS], dtype=bool)
        normals_voxel = []
        for p3d_idx in indices:
            views_voxel = np.logical_or(views_voxel, points_bkgd_lidar_view[p3d_idx])
            normals_voxel.append(points_bkgd_lidar_normal[p3d_idx])
        views_for_voxels.append(views_voxel)
        normals_for_voxels.append(np.mean(np.array(normals_voxel).reshape(-1, 3), axis=0))

    # Combine with COLMAP points
    try:
        if has_colmap and cfg.data.get("filter_colmap", True):
            points_colmap_mask = np.ones(points_colmap_xyz.shape[0], dtype=bool)
            for ii, ext in enumerate(result["exts"]):
                camera_position = c2ws_all[ii][:3, 3]
                radius = np.linalg.norm(points_colmap_xyz - camera_position, axis=-1)
                m = np.logical_or(radius < cfg.data.get("extent", 10), points_colmap_xyz[:, 2] < camera_position[2])
                points_colmap_mask = np.logical_and(points_colmap_mask, ~m)

            points_colmap_dist = np.linalg.norm(points_colmap_xyz - sphere_center, axis=-1)
            m = points_colmap_dist < 2 * sphere_radius
            points_colmap_mask = np.logical_and(points_colmap_mask, m)

            points_colmap_xyz = points_colmap_xyz[points_colmap_mask]
            points_colmap_rgb = points_colmap_rgb[points_colmap_mask]

            for p3d_colmap_id in range(len(points_colmap_tracks)):
                if not points_colmap_mask[p3d_colmap_id]:
                    continue
                views_voxel = np.zeros([N_VIEWS], dtype=bool)
                normals_voxel = []
                p_tracks = points_colmap_tracks[p3d_colmap_id]
                for track_id_idx in range(len(p_tracks)):
                    colmap_view_id = p_tracks[track_id_idx][0]
                    point2d_id = p_tracks[track_id_idx][1]
                    v, u = colmap_images[colmap_view_id].xys[point2d_id].astype(np.int16)
                    _cam_id = (colmap_view_id - 1) // num_frames
                    _stamp_id = (colmap_view_id - 1) % num_frames
                    _my_view_id = _stamp_id * num_cameras + _cam_id
                    if start_frame <= _stamp_id <= end_frame and _my_view_id < len(normals_world_all):
                        p_n = normals_world_all[_my_view_id][u, v]
                        views_voxel[_my_view_id] = True
                        normals_voxel.append(p_n)

                if not normals_voxel:
                    normals_voxel = [np.zeros(3)]
                views_for_voxels.append(views_voxel)
                normals_for_voxels.append(np.mean(np.array(normals_voxel).reshape(-1, 3), axis=0))

            points_bkgd_xyz = np.concatenate(
                [np.array(downsample_outlier_pcd.points), points_colmap_xyz], axis=0
            )
            points_bkgd_rgb = np.concatenate(
                [np.array(downsample_outlier_pcd.colors), points_colmap_rgb], axis=0
            )
        else:
            raise Exception("Skip COLMAP")
    except Exception:
        print("Using LiDAR-only point cloud")
        points_bkgd_xyz = np.array(downsample_outlier_pcd.points)
        points_bkgd_rgb = np.array(downsample_outlier_pcd.colors)

    normals = np.array(normals_for_voxels).reshape(-1, 3)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    points_bkgd_normal = normals / norms

    voxels_xyz_dict = {"bkgd": points_bkgd_xyz}
    voxels_rgb_dict = {"bkgd": points_bkgd_rgb}
    voxels_normal_sh_dict = {"bkgd": points_bkgd_normal}
    voxels_view_dict = {"bkgd": np.array(views_for_voxels)}

    # Object points
    for k, v in points_xyz_dict.items():
        if len(v) == 0 or k == "bkgd":
            continue
        if k.startswith("obj"):
            points_obj_xyz = np.concatenate(points_xyz_dict[k], axis=0)
            points_obj_rgb = np.concatenate(points_rgb_dict[k], axis=0)
            points_obj_view = np.concatenate(points_view_dict[k], axis=0)
            points_obj_normal = np.concatenate(points_normal_dict[k], axis=0)

            # Normal-based filtering (skip if no normals available)
            if has_normals:
                _mask_a = points_obj_normal[:, 1] * points_obj_xyz[:, 1] > 0
                _mask_gnd = np.logical_and(points_obj_normal[:, 2] > 0.9, points_obj_xyz[:, 2] < 0)
                if _mask_gnd.sum() > 100:
                    gnd_height = points_obj_xyz[:, 2].min() + (points_obj_xyz[_mask_gnd, 2].mean() - points_obj_xyz[:, 2].min()) * 2
                    _mask_b = np.logical_or(points_obj_xyz[:, 2] > gnd_height, points_obj_normal[:, 2] < 0.7)
                    mask_right_direction = np.logical_and(_mask_a, _mask_b)
                    points_obj_xyz = points_obj_xyz[mask_right_direction]
                    points_obj_rgb = points_obj_rgb[mask_right_direction]
                    points_obj_view = points_obj_view[mask_right_direction]
                    points_obj_normal = points_obj_normal[mask_right_direction]

            pcd_obj = o3d.geometry.PointCloud()
            pcd_obj.points = o3d.utility.Vector3dVector(points_obj_xyz)
            pcd_obj.colors = o3d.utility.Vector3dVector(points_obj_rgb)

            if points_obj_xyz.shape[0] > initial_num_obj:
                downsampled_pcd_obj, _, point_indices_for_each_voxel_obj = pcd_obj.voxel_down_sample_and_trace(
                    voxel_size=voxel_size,
                    min_bound=(pcd_obj.get_min_bound() // voxel_size) * voxel_size,
                    max_bound=(pcd_obj.get_max_bound() // voxel_size + 1) * voxel_size,
                )
                points_xyz_ds = np.array(downsampled_pcd_obj.points)
                points_rgb_ds = np.array(downsampled_pcd_obj.colors)

                views_for_voxels_obj = []
                normals_for_voxels_obj = []
                for indices in point_indices_for_each_voxel_obj:
                    views_voxel = np.zeros([N_VIEWS], dtype=bool)
                    normals_voxel = []
                    for p3d_idx in indices:
                        views_voxel = np.logical_or(views_voxel, points_obj_view[p3d_idx])
                        normals_voxel.append(points_obj_normal[p3d_idx])
                    views_for_voxels_obj.append(views_voxel)
                    normals_for_voxels_obj.append(np.mean(normals_voxel, axis=0))

                point_view = np.array(views_for_voxels_obj)
                point_normal = np.stack(normals_for_voxels_obj)
                norms = np.linalg.norm(point_normal, axis=1, keepdims=True)
                norms = np.where(norms < 1e-8, 1.0, norms)
                point_normal = point_normal / norms
            else:
                points_xyz_ds = points_obj_xyz
                points_rgb_ds = points_obj_rgb
                point_view = np.zeros_like(points_obj_view)
                point_view[:, np.where(points_obj_view.sum(0) > 0)] = True
                point_normal = points_obj_normal

            voxels_xyz_dict[k] = points_xyz_ds
            voxels_rgb_dict[k] = points_rgb_ds
            voxels_view_dict[k] = point_view
            voxels_normal_sh_dict[k] = point_normal

    # Save PLY files
    for k in voxels_xyz_dict.keys():
        pts_xyz = voxels_xyz_dict[k]
        pts_rgb = voxels_rgb_dict[k]
        pts_normal = voxels_normal_sh_dict[k]
        pts_vis = voxels_view_dict[k]

        if np.isnan(pts_normal).sum():
            print(f"NaN in normals for {k}")

        if pts_xyz.shape[0] < 100:
            continue

        ply_path = os.path.join(pointcloud_dir, f"points3D_{k}.ply")
        try:
            storePly(ply_path, pts_xyz, pts_rgb, normals=pts_normal[:, 0:])
            np.save(ply_path[:-3] + "npy", pts_vis)
            print(f"Saved point cloud for {k}: {pts_xyz.shape[0]} points")
        except Exception as e:
            print(f"Failed to save point cloud for {k}: {e}")

    # Also save a lidar-only PLY for sphere normalization reference
    lidar_ply_path = os.path.join(pointcloud_dir, "points3D_lidar.ply")
    if not os.path.exists(lidar_ply_path) and len(points_xyz_dict["bkgd"]) > 0:
        lidar_xyz = np.concatenate(points_xyz_dict["bkgd"], axis=0) if isinstance(points_xyz_dict["bkgd"], list) else points_xyz_dict["bkgd"]
        lidar_rgb = np.concatenate(points_rgb_dict["bkgd"], axis=0) if isinstance(points_rgb_dict["bkgd"], list) else points_rgb_dict["bkgd"]
        storePly(lidar_ply_path, lidar_xyz[:, :3], lidar_rgb)
        print(f"Saved LiDAR reference point cloud: {lidar_xyz.shape[0]} points")
