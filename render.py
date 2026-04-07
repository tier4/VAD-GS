import torch
import os
import json
import collections
from tqdm import tqdm
import numpy as np
import imageio
from lib.models.street_gaussian_model import StreetGaussianModel
from lib.models.street_gaussian_renderer import StreetGaussianRenderer
from lib.datasets.dataset import Dataset
from lib.models.scene import Scene
from lib.utils.general_utils import safe_state
from lib.config import cfg
from lib.visualizers.base_visualizer import BaseVisualizer as Visualizer
from lib.visualizers.street_gaussian_visualizer import StreetGaussianVisualizer
from lib.utils.img_utils import visualize_depth_numpy
import time


def _compose_log_frame(gt_image, result, result_obj):
    """Compose a 2-row frame matching the log_images layout.

    Row 0: GT | Rendered | Depth (colored)
    Row 1: Diff | Object render | Normal

    All inputs are torch tensors on GPU (CHW format).
    Returns: numpy uint8 array (HWC) for video writing.
    """
    rgb = result['rgb']
    depth = result['depth']

    # Depth colored
    depth_np = depth.detach().cpu().numpy().squeeze(0)  # [H, W]
    depth_colored, _ = visualize_depth_numpy(depth_np)
    depth_colored = depth_colored[..., [2, 1, 0]] / 255.
    depth_colored = torch.from_numpy(depth_colored).permute(2, 0, 1).float().cuda()

    row0 = torch.cat([gt_image, rgb, depth_colored], dim=2)

    # Diff
    diff = ((rgb.detach().cpu() - gt_image.cpu()) ** 2).sum(dim=0, keepdim=True)  # [1, H, W]
    diff_colored, _ = visualize_depth_numpy(diff.numpy().squeeze(0), cmap=8)  # COLORMAP_TURBO=8
    diff_colored = diff_colored[..., [2, 1, 0]] / 255.
    diff_colored = torch.from_numpy(diff_colored).permute(2, 0, 1).float().cuda()

    # Object render
    image_obj = result_obj['rgb']

    # Normal
    if 'normals' in result:
        normal = -result['normals'] / 2 + 0.5
    else:
        normal = torch.zeros_like(rgb)

    row1 = torch.cat([diff_colored, image_obj, normal], dim=2)

    frame = torch.cat([row0, row1], dim=1)
    frame = torch.clamp(frame, 0.0, 1.0)
    frame = (frame.detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    return frame


def _render_and_collect(split_name, cameras, renderer, gaussians, visualizer):
    """Render cameras, save images, and collect per-camera video frames."""
    times = []
    cam_frames = collections.defaultdict(list)

    for idx, camera in enumerate(tqdm(cameras, desc=f"Rendering {split_name}")):
        torch.cuda.synchronize()
        start_time = time.time()

        result = renderer.render(camera, gaussians)
        result_obj = renderer.render_object(camera, gaussians)

        torch.cuda.synchronize()
        end_time = time.time()
        times.append((end_time - start_time) * 1000)

        visualizer.visualize(result, camera)

        gt_image = camera.original_image[:3].cuda()
        frame = _compose_log_frame(gt_image, result, result_obj)
        cam_id = camera.meta['cam']
        cam_frames[cam_id].append(frame)

    return times, cam_frames


def _save_per_camera_videos(cam_frames, save_dir, fps):
    """Write per-camera MP4 videos from collected frames."""
    os.makedirs(save_dir, exist_ok=True)
    for cam_id in sorted(cam_frames.keys()):
        frames = cam_frames[cam_id]
        out_path = os.path.join(save_dir, f'log_video_cam{cam_id}.mp4')
        imageio.mimwrite(out_path, frames, fps=fps)
        print(f"Saved {out_path} ({len(frames)} frames)")


def render_sets():
    cfg.render.save_image = True
    cfg.render.save_video = False

    fps = cfg.render.get('fps', 24)

    with torch.no_grad():
        dataset = Dataset()
        gaussians = StreetGaussianModel(dataset.scene_info.metadata)
        scene = Scene(gaussians=gaussians, dataset=dataset)
        renderer = StreetGaussianRenderer()

        times = []
        if not cfg.eval.skip_train:
            save_dir = os.path.join(cfg.model_path, 'train', "ours_{}".format(scene.loaded_iter))
            visualizer = Visualizer(save_dir)
            cameras = scene.getTrainCameras()
            t, cam_frames = _render_and_collect("Training View", cameras, renderer, gaussians, visualizer)
            times.extend(t)
            _save_per_camera_videos(cam_frames, save_dir, fps)

        if not cfg.eval.skip_test:
            save_dir = os.path.join(cfg.model_path, 'test', "ours_{}".format(scene.loaded_iter))
            visualizer = Visualizer(save_dir)
            cameras = scene.getTestCameras()
            t, cam_frames = _render_and_collect("Testing View", cameras, renderer, gaussians, visualizer)
            times.extend(t)
            _save_per_camera_videos(cam_frames, save_dir, fps)

        print(times)
        print('average rendering time: ', sum(times[1:]) / len(times[1:]))
                
def render_trajectory():
    cfg.render.save_image = False
    cfg.render.save_video = True
    
    with torch.no_grad():
        dataset = Dataset()        
        gaussians = StreetGaussianModel(dataset.scene_info.metadata)

        scene = Scene(gaussians=gaussians, dataset=dataset)
        renderer = StreetGaussianRenderer()
        
        save_dir = os.path.join(cfg.model_path, 'trajectory', "ours_{}".format(scene.loaded_iter))
        visualizer = StreetGaussianVisualizer(save_dir)
        
        train_cameras = scene.getTrainCameras()
        test_cameras = scene.getTestCameras()
        cameras = train_cameras + test_cameras
        cameras = list(sorted(cameras, key=lambda x: x.id))

        for idx, camera in enumerate(tqdm(cameras, desc="Rendering Trajectory")):
            result = renderer.render_all(camera, gaussians)  
            visualizer.visualize(result, camera)

        visualizer.summarize()
            
if __name__ == "__main__":
    print("Rendering " + cfg.model_path)
    safe_state(cfg.eval.quiet)
    
    if cfg.mode == 'evaluate':
        render_sets()
    elif cfg.mode == 'trajectory':
        render_trajectory()
    else:
        raise NotImplementedError()
