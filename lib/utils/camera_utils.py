from __future__ import annotations

from typing import Any, Generator

import numpy as np
import torch
import copy
import torch.nn as nn
import cv2
import math
from PIL import Image
from tqdm import tqdm
from lib.utils.general_utils import PILtoTorch, NumpytoTorch, matrix_to_quaternion
from lib.utils.graphics_utils import fov2focal, getProjectionMatrix, getWorld2View2, getProjectionMatrixK
from lib.datasets.base_readers import CameraInfo
from lib.config import cfg
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

# if training, put everything to cuda
# image_to_cuda = (cfg.mode == 'train') 

class Camera(nn.Module):
    def __init__(
        self,
        id: int,
        R: np.ndarray, T: np.ndarray,
        FoVx: float, FoVy: float, K: np.ndarray | None,
        image: torch.Tensor | None, image_name: str,
        trans: np.ndarray = np.array([0.0, 0.0, 0.0]),
        scale: float = 1.0,
        metadata: dict[str, Any] = dict(),
        guidance: dict[str, Any] | LazyGuidanceDict = dict(),
        image_path: str | None = None,
        resolution: tuple[int, int] | None = None,
    ) -> None:
        super(Camera, self).__init__()

        self.id: int = id
        self.R: np.ndarray = R
        self.T: np.ndarray = T
        self.FoVx: float = FoVx
        self.FoVy: float = FoVy
        self.K: np.ndarray | torch.Tensor | None = K
        self.image_name: str = image_name
        self.trans: np.ndarray = trans
        self.scale: float = scale

        # metadata
        self.meta: dict[str, Any] = metadata

        # guidance
        self.guidance: dict[str, Any] | LazyGuidanceDict = guidance

        # Lazy image loading: store path/resolution, load on demand
        self._image_path: str | None = image_path
        self._image_resolution: tuple[int, int] | None = resolution
        self._image_device: str = 'cpu'
        if image is not None:
            self._original_image: torch.Tensor | None = image.clamp(0, 1)
            self.image_height: int = self._original_image.shape[1]
            self.image_width: int = self._original_image.shape[2]
        else:
            self._original_image = None
            # Compute dimensions from resolution without loading image
            self.image_width, self.image_height = resolution

        self.zfar: float = 1000.0
        self.znear: float = 0.001
        self.world_view_transform: torch.Tensor = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()

        if self.K is not None:
            self.projection_matrix: torch.Tensor = getProjectionMatrixK(znear=self.znear, zfar=self.zfar, K=self.K, H=self.image_height, W=self.image_width).transpose(0,1).cuda()
            self.K = torch.from_numpy(self.K).float().cuda()
        else:
            self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()

        self.full_proj_transform: torch.Tensor = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center: torch.Tensor = self.world_view_transform.inverse()[3, :3]

        if 'ego_pose' in self.meta.keys():
            self.ego_pose: torch.Tensor = torch.from_numpy(self.meta['ego_pose']).float().cuda()
            del self.meta['ego_pose']

        if 'extrinsic' in self.meta.keys():
            self.extrinsic: torch.Tensor = torch.from_numpy(self.meta['extrinsic']).float().cuda()
            del self.meta['extrinsic']

        self.normal: torch.Tensor | None = None

    @property
    def original_image(self) -> torch.Tensor:
        if self._original_image is None:
            self._load_image()
        assert self._original_image is not None
        return self._original_image

    @original_image.setter
    def original_image(self, value: torch.Tensor) -> None:
        self._original_image = value

    def _load_image(self) -> None:
        """Load image from disk on demand."""
        pil_image = Image.open(self._image_path)
        image = PILtoTorch(pil_image, self._image_resolution, resize_mode=Image.BILINEAR)[:3, ...]
        pil_image.close()
        self._original_image = image.clamp(0, 1)
        if self._image_device != 'cpu':
            self._original_image = self._original_image.to(self._image_device)

    def unload_image(self) -> None:
        """Free image tensor to save memory. Will be reloaded on next access."""
        if self._image_path is not None:
            self._original_image = None

    def set_extrinsic(self, c2w: np.ndarray) -> None:
        w2c: np.ndarray = np.linalg.inv(c2w)
        R: np.ndarray = w2c[:3, :3].T
        T: np.ndarray = w2c[:3, 3]

        # set R, T
        self.R = R
        self.T = T

        # change attributes associated with R, T
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, self.trans, self.scale)).transpose(0, 1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def set_intrinsic(self, K: np.ndarray) -> None:
        self.K = torch.from_numpy(K).float().cuda()
        self.projection_matrix = getProjectionMatrixK(znear=self.znear, zfar=self.zfar, K=self.K, H=self.image_height, W=self.image_width).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)

    def get_extrinsic(self) -> np.ndarray:
        w2c: np.ndarray = np.eye(4)
        w2c[:3, :3] = self.R.T
        w2c[:3, 3] = self.T
        c2w: np.ndarray = np.linalg.inv(w2c)
        return c2w

    def get_intrinsic(self) -> np.ndarray:
        ixt: np.ndarray = self.K.cpu().numpy()
        return ixt

    def set_device(self, device: str) -> None:
        self._image_device = device
        if self._original_image is not None:
            self._original_image = self._original_image.to(device)
        if isinstance(self.guidance, LazyGuidanceDict):
            self.guidance.to(device, non_blocking=True)
        else:
            for k, v in self.guidance.items():
                self.guidance[k] = v.to(device, non_blocking=True)

        
class MiniCam:
    def __init__(self, width: int, height: int, fovy: float, fovx: float, znear: float, zfar: float, world_view_transform: torch.Tensor, full_proj_transform: torch.Tensor) -> None:
        self.image_width: int = width
        self.image_height: int = height
        self.FoVy: float = fovy
        self.FoVx: float = fovx
        self.znear: float = znear
        self.zfar: float = zfar
        self.world_view_transform: torch.Tensor = world_view_transform
        self.full_proj_transform: torch.Tensor = full_proj_transform
        view_inv: torch.Tensor = torch.inverse(self.world_view_transform)
        self.camera_center: torch.Tensor = view_inv[3][:3]

def _load_guidance_from_path(k: str, path: str) -> Image.Image | np.ndarray | None:
    """Load guidance data from a file path (deferred loading)."""
    if k == 'obj_bound':
        return Image.fromarray(cv2.imread(path, cv2.IMREAD_GRAYSCALE))
    elif k == 'dynamic_mask':
        return cv2.imread(path)
    elif k == 'seg_bkgd':
        return cv2.imread(path)
    elif k == 'sky_mask':
        return Image.fromarray((cv2.imread(path)[..., 0]) > 0.0)
    elif k == 'lidar_depth':
        depth = np.load(path, allow_pickle=True)
        depth = dict(depth.item())
        mask = depth["mask"]
        value = depth["value"]
        depth_arr = np.zeros_like(mask).astype(np.float32)
        depth_arr[mask] = value
        return depth_arr
    elif k == 'mono_depth':
        if path.endswith('.npz'):
            mono_depth_raw = np.load(path)["depth"]
            d_min, d_max = mono_depth_raw.min(), mono_depth_raw.max()
            if d_max - d_min > 1e-6:
                mono_depth = ((mono_depth_raw - d_min) / (d_max - d_min) * 255).astype(np.uint8)
            else:
                mono_depth = np.zeros_like(mono_depth_raw, dtype=np.uint8)
            return Image.fromarray(mono_depth)
        else:
            return Image.fromarray(255 - cv2.imread(path)[:, :, 0])
    elif k == 'mono_normal':
        tmp = cv2.imread(path).astype(np.float32) / 255.0 * 2 - 1
        ref_norm = np.empty(tmp.shape, dtype=np.float32)
        ref_norm[:, :, 0] = -tmp[:, :, 2]
        ref_norm[:, :, 1] = -tmp[:, :, 1]
        ref_norm[:, :, 2] = -tmp[:, :, 0]
        return ref_norm
    return None


def _convert_guidance_to_tensor(k: str, raw_data: Image.Image | np.ndarray, resolution: tuple[int, int]) -> torch.Tensor | Image.Image | np.ndarray:
    """Convert raw guidance data (numpy/PIL) to a tensor at the target resolution."""
    if k == 'mask':
        return PILtoTorch(raw_data, resolution, resize_mode=Image.NEAREST).bool()
    elif k == 'acc_mask':
        return PILtoTorch(raw_data, resolution, resize_mode=Image.NEAREST).bool()
    elif k == 'sky_mask':
        return PILtoTorch(raw_data, resolution, resize_mode=Image.NEAREST).bool()
    elif k == 'obj_bound':
        return PILtoTorch(raw_data, resolution, resize_mode=Image.NEAREST).bool()
    elif k == 'lidar_depth':
        return NumpytoTorch(raw_data, resolution, resize_mode=Image.NEAREST).float()
    elif k == 'dynamic_mask':
        return NumpytoTorch(raw_data, resolution, resize_mode=Image.NEAREST).int()
    elif k == 'mono_depth':
        return PILtoTorch(raw_data, resolution, resize_mode=Image.NEAREST).to(torch.float)
    elif k == 'mono_normal':
        return NumpytoTorch(raw_data, resolution, resize_mode=Image.NEAREST).to(torch.float)
    elif k == 'seg_bkgd':
        return NumpytoTorch(raw_data, resolution, resize_mode=Image.NEAREST).int()
    return raw_data


class LazyGuidanceDict:
    """Dict-like container that lazily loads guidance from file paths.

    File-backed entries are loaded and converted to tensors only on first
    access, keeping peak memory low.  Values written at runtime (e.g.
    ``bkgd_voxel``) are marked *persistent* and survive ``unload()``.
    """

    def __init__(self, sources: dict[str, str | Image.Image | np.ndarray], resolution: tuple[int, int]) -> None:
        self._sources: dict[str, str | Image.Image | np.ndarray] = dict(sources)   # key -> file path (str) or PIL Image
        self._resolution: tuple[int, int] = resolution
        self._cache: dict[str, torch.Tensor | None] = {}                # key -> loaded tensor / value
        self._persistent: set[str] = set()        # keys set via __setitem__

    # --- dict protocol ---------------------------------------------------
    def __contains__(self, key: str) -> bool:
        return key in self._cache or key in self._sources

    def __getitem__(self, key: str) -> Any:
        if key in self._cache:
            return self._cache[key]
        if key not in self._sources:
            raise KeyError(key)
        value = self._load(key)
        self._cache[key] = value
        return value

    def __setitem__(self, key: str, value: torch.Tensor | Any) -> None:
        self._cache[key] = value
        self._persistent.add(key)

    def get(self, key: str, default: Any = None) -> torch.Tensor | Any:
        if key in self:
            return self[key]
        return default

    def keys(self) -> set[str]:
        return set(self._sources) | set(self._cache)

    def items(self) -> Generator[tuple[str, torch.Tensor | None], None, None]:
        for k in self.keys():
            yield k, self[k]

    # --- lazy load / unload ----------------------------------------------
    def _load(self, key: str) -> torch.Tensor | Image.Image | np.ndarray | None:
        source: str | Image.Image | np.ndarray = self._sources[key]
        if isinstance(source, str):
            raw: Image.Image | np.ndarray | None = _load_guidance_from_path(key, source)
        else:
            raw = source
        if raw is None:
            return None
        return _convert_guidance_to_tensor(key, raw, self._resolution)

    def unload(self) -> None:
        """Free lazily-loaded tensors. Runtime-set values are kept."""
        for key in list(self._cache):
            if key not in self._persistent:
                del self._cache[key]

    def to(self, device: str, non_blocking: bool = False) -> None:
        """Move all currently-cached tensors to *device*."""
        for key in list(self._cache):
            v = self._cache[key]
            if hasattr(v, 'to'):
                self._cache[key] = v.to(device, non_blocking=non_blocking)


def loadguidance(guidance: dict[str, Any], resolution: tuple[int, int]) -> dict[str, Any]:
    new_guidance: dict[str, Any] = dict()
    for k, v in guidance.items():
        # Deferred loading: if value is a file path string, load from disk
        if isinstance(v, str):
            v = _load_guidance_from_path(k, v)
            if v is None:
                continue

        new_guidance[k] = _convert_guidance_to_tensor(k, v, resolution)

    return new_guidance
        
WARNED: bool = False
def loadCam(cam_info: CameraInfo, resolution_scale: float, scale: float = 1.0) -> Camera:
    orig_w: int = cam_info.width
    orig_h: int = cam_info.height
    scale = min(scale, 960 / orig_w)
    scale = scale / resolution_scale
    resolution: tuple[int, int] = (int(orig_w * scale), int(orig_h * scale))

    K: np.ndarray = copy.deepcopy(cam_info.K)
    K[:2] *= scale

    # Lazy loading: if image_path is available, defer loading to save ~10GB RAM
    image: torch.Tensor | None
    image_path: str | None
    if cam_info.image is not None:
        image = PILtoTorch(cam_info.image, resolution, resize_mode=Image.BILINEAR)[:3, ...]
        image_path = None
    else:
        image = None
        image_path = cam_info.image_path

    guidance: LazyGuidanceDict = LazyGuidanceDict(cam_info.guidance, resolution)
    cam_info.guidance.clear()

    cam: Camera = Camera(
        id=cam_info.uid,
        R=cam_info.R,
        T=cam_info.T,
        FoVx=cam_info.FovX,
        FoVy=cam_info.FovY,
        K=K,
        image=image,
        image_name=cam_info.image_name,
        metadata=cam_info.metadata,
        guidance=guidance,
        image_path=image_path,
        resolution=resolution,
    )
    return cam


def cameraList_from_camInfos(cam_infos: list[CameraInfo], resolution_scale: float) -> list[Camera]:
    camera_list: list[Camera] = []

    for i, cam_info in tqdm(enumerate(cam_infos)):
        camera_list.append(loadCam(cam_info, resolution_scale))

    return camera_list

def camera_to_JSON(id: int, camera: CameraInfo) -> dict[str, Any]:
    Rt: np.ndarray = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C: np.ndarray = np.linalg.inv(Rt)
    pos: np.ndarray = W2C[:3, 3]
    rot: np.ndarray = W2C[:3, :3]
    serializable_array_2d: list[list[float]] = [x.tolist() for x in rot]
    camera_entry: dict[str, Any] = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry

def make_rasterizer(
    viewpoint_camera: Camera,
    active_sh_degree: int = 0,
    bg_color: torch.Tensor | None = None,
    scaling_modifier: float | None = None,
) -> GaussianRasterizer:
    if bg_color is None:
        bg_list = [1, 1, 1] if cfg.data.white_background else [0, 0, 0]
        bg_color = torch.tensor(bg_list).float().cuda()
    if scaling_modifier is None:
        scaling_modifier = cfg.render.scaling_modifier
    debug: bool = cfg.render.debug

    # Set up rasterization configuration
    tanfovx: float = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy: float = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings: GaussianRasterizationSettings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=debug,
    )    
            
    rasterizer: GaussianRasterizer = GaussianRasterizer(raster_settings=raster_settings)
    return rasterizer
