import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np


CAMERA_CHANNELS = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]
LIDAR_CHANNEL = "LIDAR_TOP"


def quat_wxyz_to_rotmat(quat):
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
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = quat_wxyz_to_rotmat(rotation_wxyz)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64)
    return transform


def simplify_category(name):
    name = name.lower()
    if "vehicle" in name:
        return "vehicle"
    if "pedestrian" in name:
        return "pedestrian"
    if "cycle" in name or "bicycle" in name or "motorcycle" in name:
        return "cyclist"
    return "misc"


def load_tables(version_root):
    tables = {}
    for name in [
        "scene",
        "sample",
        "sample_data",
        "sample_annotation",
        "ego_pose",
        "calibrated_sensor",
        "sensor",
        "instance",
        "category",
    ]:
        with open(version_root / f"{name}.json", "r") as f:
            tables[name] = json.load(f)
    return tables


def index_by_token(items):
    return {item["token"]: item for item in items}


def build_sample_channel_map(sample_data_items, calibrated_sensor_by_token, sensor_by_token):
    mapping = {}
    for sd in sample_data_items:
        cs = calibrated_sensor_by_token[sd["calibrated_sensor_token"]]
        sensor = sensor_by_token[cs["sensor_token"]]
        mapping.setdefault(sd["sample_token"], {})[sensor["channel"]] = (sd, cs)
    return mapping


def build_sample_annotation_map(sample_annotation_items):
    mapping = {}
    for ann in sample_annotation_items:
        mapping.setdefault(ann["sample_token"], []).append(ann)
    return mapping


def link_or_copy(src, dst, copy_files=False):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy_files:
        shutil.copy2(src, dst)
    else:
        os.symlink(src, dst)


def sample_chain(scene, sample_by_token):
    ordered = []
    token = scene["first_sample_token"]
    while token:
        sample = sample_by_token[token]
        ordered.append(sample)
        token = sample["next"]
    return ordered


def read_lidar_bin(path):
    points = np.fromfile(path, dtype=np.float32)
    points = points.reshape(-1, 5)
    return points[:, :4]


def project_points(points_lidar_xyz, cam_to_world, intrinsic):
    pts_h = np.concatenate([points_lidar_xyz, np.ones((points_lidar_xyz.shape[0], 1), dtype=np.float64)], axis=1)
    world_to_cam = np.linalg.inv(cam_to_world)
    pts_cam = (pts_h @ world_to_cam.T)[:, :3]
    depth = pts_cam[:, 2]
    valid = depth > 1e-6
    uvw = pts_cam @ intrinsic.T
    uv = np.zeros((pts_cam.shape[0], 2), dtype=np.float64)
    uv[valid, 0] = uvw[valid, 0] / depth[valid]
    uv[valid, 1] = uvw[valid, 1] / depth[valid]
    return uv, depth, valid


def build_instance_jsons(samples, sample_annotations_by_sample, instance_by_token, category_by_token, out_dir):
    frame_instances = {}
    instances_info = {}

    for frame_idx, sample in enumerate(samples):
        frame_key = str(frame_idx)
        frame_instances[frame_key] = []
        for ann in sample_annotations_by_sample.get(sample["token"], []):
            instance_token = ann["instance_token"]
            instance = instance_by_token[instance_token]
            category = category_by_token[instance["category_token"]]["name"]
            class_name = simplify_category(category)
            if class_name == "misc":
                continue

            track_id = int(instance_token[:8], 16)
            obj_to_world = make_transform(ann["translation"], ann["rotation"]).tolist()
            size = ann["size"]
            box_size = [size[1], size[0], size[2]]

            frame_instances[frame_key].append(track_id)

            if str(track_id) not in instances_info:
                instances_info[str(track_id)] = {
                    "id": track_id,
                    "class_name": class_name,
                    "frame_annotations": {
                        "frame_idx": [],
                        "obj_to_world": [],
                        "box_size": [],
                    },
                }

            entry = instances_info[str(track_id)]["frame_annotations"]
            entry["frame_idx"].append(frame_idx)
            entry["obj_to_world"].append(obj_to_world)
            entry["box_size"].append(box_size)

    instances_dir = out_dir / "instances"
    instances_dir.mkdir(parents=True, exist_ok=True)
    with open(instances_dir / "frame_instances.json", "w") as f:
        json.dump(frame_instances, f)
    with open(instances_dir / "instances_info.json", "w") as f:
        json.dump(instances_info, f)


def convert_scene(dataset_root, version, scene_index, out_dir, copy_files=False):
    version_root = dataset_root / version
    tables = load_tables(version_root)
    scene = tables["scene"][scene_index]

    sample_by_token = index_by_token(tables["sample"])
    ego_pose_by_token = index_by_token(tables["ego_pose"])
    calibrated_sensor_by_token = index_by_token(tables["calibrated_sensor"])
    sensor_by_token = index_by_token(tables["sensor"])
    instance_by_token = index_by_token(tables["instance"])
    category_by_token = index_by_token(tables["category"])
    sample_annotation_by_token = index_by_token(tables["sample_annotation"])
    sample_channel_map = build_sample_channel_map(
        tables["sample_data"], calibrated_sensor_by_token, sensor_by_token
    )
    sample_annotation_map = build_sample_annotation_map(tables["sample_annotation"])

    samples = sample_chain(scene, sample_by_token)

    out_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ["images", "intrinsics", "extrinsics", "lidar", "lidar_pose"]:
        (out_dir / subdir).mkdir(parents=True, exist_ok=True)

    first_sample_data = None
    for sample in samples:
        frame_data = sample_channel_map.get(sample["token"], {})
        if all(channel in frame_data for channel in CAMERA_CHANNELS + [LIDAR_CHANNEL]):
            first_sample_data = frame_data
            break
    if first_sample_data is None:
        raise RuntimeError("Scene does not contain the required camera and lidar channels")

    for cam_idx, channel in enumerate(CAMERA_CHANNELS):
        sd, cs = first_sample_data[channel]
        intrinsic = cs["camera_intrinsic"]
        fx = intrinsic[0][0]
        fy = intrinsic[1][1]
        cx = intrinsic[0][2]
        cy = intrinsic[1][2]
        values = np.array([fx, fy, cx, cy, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        np.savetxt(out_dir / "intrinsics" / f"{cam_idx}.txt", values[None], fmt="%.8f")

    pointcloud = {}
    camera_projection = {}

    for frame_idx, sample in enumerate(samples):
        frame_data = sample_channel_map.get(sample["token"], {})

        if not all(channel in frame_data for channel in CAMERA_CHANNELS + [LIDAR_CHANNEL]):
            continue

        lidar_sd, lidar_cs = frame_data[LIDAR_CHANNEL]
        lidar_pose = ego_pose_by_token[lidar_sd["ego_pose_token"]]
        lidar_to_world = make_transform(lidar_pose["translation"], lidar_pose["rotation"]) @ make_transform(lidar_cs["translation"], lidar_cs["rotation"])
        np.savetxt(out_dir / "lidar_pose" / f"{frame_idx:03d}.txt", lidar_to_world, fmt="%.8f")

        lidar_src = dataset_root / lidar_sd["filename"]
        lidar_points = read_lidar_bin(lidar_src)
        lidar_dst = out_dir / "lidar" / f"{frame_idx:03d}.bin"
        lidar_points.astype(np.float32).tofile(lidar_dst)
        pointcloud[frame_idx] = lidar_points[:, :3].astype(np.float32)

        projection = np.full((lidar_points.shape[0], 6), -1, dtype=np.int16)
        pts_xyz = lidar_points[:, :3].astype(np.float64)

        for cam_idx, channel in enumerate(CAMERA_CHANNELS):
            cam_sd, cam_cs = frame_data[channel]
            cam_pose = ego_pose_by_token[cam_sd["ego_pose_token"]]
            cam_to_world = make_transform(cam_pose["translation"], cam_pose["rotation"]) @ make_transform(cam_cs["translation"], cam_cs["rotation"])
            np.savetxt(out_dir / "extrinsics" / f"{frame_idx:03d}_{cam_idx}.txt", cam_to_world, fmt="%.8f")

            image_src = dataset_root / cam_sd["filename"]
            image_dst = out_dir / "images" / f"{frame_idx:03d}_{cam_idx}.jpg"
            link_or_copy(image_src, image_dst, copy_files=copy_files)

            intrinsic = np.array(cam_cs["camera_intrinsic"], dtype=np.float64)
            uv, depth, valid = project_points(pts_xyz, cam_to_world, intrinsic)
            width = cam_sd["width"]
            height = cam_sd["height"]
            visible = valid & (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)

            first_slot = visible & (projection[:, 0] < 0)
            second_slot = visible & (projection[:, 0] >= 0) & (projection[:, 3] < 0)
            projection[first_slot, 0] = cam_idx
            projection[first_slot, 1] = np.round(uv[first_slot, 0]).astype(np.int16)
            projection[first_slot, 2] = np.round(uv[first_slot, 1]).astype(np.int16)
            projection[second_slot, 3] = cam_idx
            projection[second_slot, 4] = np.round(uv[second_slot, 0]).astype(np.int16)
            projection[second_slot, 5] = np.round(uv[second_slot, 1]).astype(np.int16)

        camera_projection[frame_idx] = projection

    np.savez_compressed(out_dir / "pointcloud.npz", pointcloud=pointcloud, camera_projection=camera_projection)

    sample_annotations_by_sample = {ann["token"]: ann for ann in tables["sample_annotation"]}
    build_instance_jsons(
        samples=samples,
        sample_annotations_by_sample=sample_annotation_map,
        instance_by_token=instance_by_token,
        category_by_token=category_by_token,
        out_dir=out_dir,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--version", type=str, default="v1.0-mini")
    parser.add_argument("--scene-index", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--copy-files", action="store_true")
    args = parser.parse_args()

    convert_scene(
        dataset_root=args.dataset_root,
        version=args.version,
        scene_index=args.scene_index,
        out_dir=args.output_dir,
        copy_files=args.copy_files,
    )


if __name__ == "__main__":
    main()
