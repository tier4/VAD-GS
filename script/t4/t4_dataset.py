"""Shared T4 dataset access layer using t4-devkit.

Wraps ``t4_devkit.Tier4`` so that all preprocessing scripts share
a single, well-tested path for dataset loading, frame iteration,
coordinate transforms and 3D-to-2D projection.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from t4_devkit import Tier4
from t4_devkit.common.geometry import view_points
from t4_devkit.dataclass import LidarPointCloud, HomogeneousMatrix

from config_utils import resolve_dataroot


# ---------------------------------------------------------------------------
# Data containers returned by iteration helpers
# ---------------------------------------------------------------------------

@dataclass
class FrameInfo:
    """Per-camera-per-sample information used by preprocessing scripts."""

    sample_idx: int
    sample_token: str
    camera_channel: str
    image_path: str
    image_name: str
    width: int
    height: int
    # Tokens for lazy access
    sample_data_token: str
    ego_pose_token: str
    calibrated_sensor_token: str


# ---------------------------------------------------------------------------
# Main wrapper
# ---------------------------------------------------------------------------

class T4Dataset:
    """Thin wrapper around ``t4_devkit.Tier4`` for preprocessing scripts.

    Usage::

        ds = T4Dataset.from_args(args)          # uses --dataroot/--revision/--scene-index
        for frame in ds.iter_frames(["CAM_FRONT", "CAM_BACK_LEFT"]):
            print(frame.image_path)
    """

    def __init__(self, dataroot: str | Path, revision: int = 0, scene_index: int = 0):
        self.dataroot = resolve_dataroot(dataroot, revision=revision)
        rev_str = str(revision) if revision else None
        self._t4 = Tier4(str(self.dataroot), revision=rev_str, verbose=False)
        self._scene = self._t4.scene[scene_index]

        # Build ordered sample list
        self._samples: list = []
        tok = self._scene.first_sample_token
        while tok:
            s = self._t4.get("sample", tok)
            self._samples.append(s)
            tok = s.next

    # -- convenience properties ------------------------------------------------

    @property
    def t4(self) -> Tier4:
        return self._t4

    @property
    def scene(self):
        return self._scene

    @property
    def samples(self):
        return self._samples

    @property
    def num_samples(self) -> int:
        return len(self._samples)

    # -- construction helpers --------------------------------------------------

    @classmethod
    def from_args(cls, args) -> "T4Dataset":
        """Create from an argparse Namespace (expects dataroot, revision, scene_index)."""
        return cls(
            dataroot=args.dataroot,
            revision=getattr(args, "revision", 0) or 0,
            scene_index=getattr(args, "scene_index", 0) or 0,
        )

    # -- frame iteration -------------------------------------------------------

    def iter_frames(
        self,
        camera_channels: list[str] | None = None,
    ):
        """Yield :class:`FrameInfo` for every (sample, camera) pair.

        If *camera_channels* is ``None``, auto-detect from the first sample.
        Yields frames in sample-order, cameras in the given order.
        """
        if camera_channels is None:
            camera_channels = self._auto_detect_cameras()

        for idx, sample in enumerate(self._samples):
            for ch in camera_channels:
                sd_token = sample.data.get(ch)
                if sd_token is None:
                    continue
                sd = self._t4.get("sample_data", sd_token)
                image_path = self._t4.get_sample_data_path(sd_token)
                if not os.path.exists(image_path):
                    continue
                yield FrameInfo(
                    sample_idx=idx,
                    sample_token=sample.token,
                    camera_channel=ch,
                    image_path=image_path,
                    image_name=Path(image_path).stem,
                    width=sd.width,
                    height=sd.height,
                    sample_data_token=sd_token,
                    ego_pose_token=sd.ego_pose_token,
                    calibrated_sensor_token=sd.calibrated_sensor_token,
                )

    def count_frames_per_camera(
        self,
        camera_channels: list[str] | None = None,
    ) -> dict[str, int]:
        """Count expected frames per camera channel."""
        counts: dict[str, int] = {}
        for frame in self.iter_frames(camera_channels):
            counts[frame.camera_channel] = counts.get(frame.camera_channel, 0) + 1
        return counts

    def _auto_detect_cameras(self) -> list[str]:
        chs = []
        for sensor in self._t4.sensor:
            if sensor.modality is not None and sensor.modality.value == "camera":
                chs.append(sensor.channel)
        return sorted(chs)

    # -- calibration / transform helpers ---------------------------------------

    def get_camera_intrinsic(self, calibrated_sensor_token: str) -> np.ndarray:
        """Return 3x3 camera intrinsic matrix."""
        cs = self._t4.get("calibrated_sensor", calibrated_sensor_token)
        return np.array(cs.camera_intrinsic, dtype=np.float64)

    def get_camera_distortion(self, calibrated_sensor_token: str) -> np.ndarray:
        """Return camera distortion coefficients."""
        cs = self._t4.get("calibrated_sensor", calibrated_sensor_token)
        return np.array(cs.camera_distortion, dtype=np.float64)

    def get_world_to_sensor(self, frame: FrameInfo) -> np.ndarray:
        """Compute map→sensor 4x4 transform matrix for a frame."""
        return np.linalg.inv(self.get_sensor_to_world(frame))

    def get_sensor_to_world(self, frame: FrameInfo) -> np.ndarray:
        """Compute sensor→map 4x4 transform matrix for a frame."""
        ego = self._t4.get("ego_pose", frame.ego_pose_token)
        cs = self._t4.get("calibrated_sensor", frame.calibrated_sensor_token)

        sensor_to_ego = HomogeneousMatrix(
            cs.translation, cs.rotation,
            src=frame.camera_channel, dst="base_link",
        )
        ego_to_map = HomogeneousMatrix(
            ego.translation, ego.rotation,
            src="base_link", dst="map",
        )
        return ego_to_map.matrix @ sensor_to_ego.matrix

    # -- 3D annotation helpers -------------------------------------------------

    def get_boxes_in_sensor(self, sample_data_token: str):
        """Return (boxes_in_sensor_coord, cam_intrinsic) via t4-devkit.

        Each box is a ``Box3D`` with position/rotation in the sensor frame.
        """
        _, boxes, cam_intrinsic = self._t4.get_sample_data(
            sample_data_token, as_3d=True, as_sensor_coord=True,
        )
        return boxes, cam_intrinsic

    # -- LiDAR helpers ---------------------------------------------------------

    def get_lidar_sample_data_token(
        self,
        sample_token: str,
        lidar_channel: str = "LIDAR_CONCAT",
    ) -> str | None:
        """Return the sample_data token for a LiDAR channel in a sample."""
        sample = self._t4.get("sample", sample_token)
        return sample.data.get(lidar_channel)

    def load_lidar_points(self, sample_data_token: str) -> np.ndarray:
        """Load LiDAR points as (N, 3) xyz array in sensor frame."""
        path = self._t4.get_sample_data_path(sample_data_token)
        pcd = LidarPointCloud.from_file(path)
        return pcd.points[:3].T  # (N, 3)

    def transform_lidar_to_camera(
        self,
        lidar_points: np.ndarray,
        lidar_sd_token: str,
        camera_sd_token: str,
    ) -> np.ndarray:
        """Transform LiDAR points (N, 3) from lidar frame to camera frame.

        Returns (N, 3) array in camera coordinates.
        """
        lidar_sd = self._t4.get("sample_data", lidar_sd_token)
        cam_sd = self._t4.get("sample_data", camera_sd_token)

        # lidar sensor → map  (sensor_to_ego @ ego_to_map)
        lidar_ego = self._t4.get("ego_pose", lidar_sd.ego_pose_token)
        lidar_cs = self._t4.get("calibrated_sensor", lidar_sd.calibrated_sensor_token)
        lidar_sensor_to_ego = HomogeneousMatrix(
            lidar_cs.translation, lidar_cs.rotation,
            src="lidar", dst="base_link",
        )
        lidar_ego_to_map = HomogeneousMatrix(
            lidar_ego.translation, lidar_ego.rotation,
            src="base_link", dst="map",
        )
        lidar_to_map = lidar_ego_to_map.matrix @ lidar_sensor_to_ego.matrix

        # map → camera sensor  (inv(ego_to_map @ sensor_to_ego))
        cam_ego = self._t4.get("ego_pose", cam_sd.ego_pose_token)
        cam_cs = self._t4.get("calibrated_sensor", cam_sd.calibrated_sensor_token)
        cam_sensor_to_ego = HomogeneousMatrix(
            cam_cs.translation, cam_cs.rotation,
            src="camera", dst="base_link",
        )
        cam_ego_to_map = HomogeneousMatrix(
            cam_ego.translation, cam_ego.rotation,
            src="base_link", dst="map",
        )
        cam_to_map = cam_ego_to_map.matrix @ cam_sensor_to_ego.matrix
        map_to_cam = np.linalg.inv(cam_to_map)

        # Full: lidar → map → camera
        full = map_to_cam @ lidar_to_map

        pts_h = np.hstack([lidar_points, np.ones((len(lidar_points), 1))])
        pts_cam = (full @ pts_h.T).T[:, :3]
        return pts_cam

    # -- Projection helper -----------------------------------------------------

    @staticmethod
    def project_points_to_image(
        points_3d: np.ndarray,
        intrinsic: np.ndarray,
        distortion: np.ndarray | None = None,
        image_size: tuple[int, int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Project 3D points (N, 3) in camera frame to 2D pixel coords.

        Returns:
            uv: (N, 2) pixel coordinates
            valid: (N,) bool mask — True if in front of camera (and optionally in image)
        """
        depths = points_3d[:, 2]
        valid = depths > 0

        # Use t4-devkit's view_points (expects (3, N) input)
        pts_3xN = points_3d.T  # (3, N)
        dist = distortion if distortion is not None and len(distortion) > 0 else None
        pts_2d = view_points(pts_3xN, intrinsic, distortion=dist, normalize=True)
        uv = pts_2d[:2].T  # (N, 2)

        if image_size is not None:
            w, h = image_size
            valid = valid & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)

        return uv, valid
