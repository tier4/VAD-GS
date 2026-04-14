import torch
import os
import json
import collections
from tqdm import tqdm
import numpy as np
import cv2
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
            
def render_background_gt():
    """Render only background Gaussians along the GT trajectory.

    Uses train + test cameras sorted by id (the full GT trajectory) and
    renders only the background Gaussian model (no foreground objects).
    Sky blending is applied when the sky model is available.
    Per-camera MP4 videos are saved.
    """
    cfg.render.save_image = False
    cfg.render.save_video = False

    fps = cfg.render.get('fps', 24)

    with torch.no_grad():
        dataset = Dataset()
        gaussians = StreetGaussianModel(dataset.scene_info.metadata)
        scene = Scene(gaussians=gaussians, dataset=dataset)
        renderer = StreetGaussianRenderer()

        train_cameras = scene.getTrainCameras()
        test_cameras = scene.getTestCameras()
        cameras = train_cameras + test_cameras
        cameras = sorted(cameras, key=lambda x: x.id)

        cam_frames = collections.defaultdict(list)

        for idx, camera in enumerate(tqdm(cameras, desc="Rendering Background GT")):
            # Render background only
            result_bkgd = renderer.render_background(camera, gaussians)
            rgb = result_bkgd['rgb']

            # Blend sky if available
            if gaussians.include_sky:
                sky_color = gaussians.sky_cubemap(camera, result_bkgd['acc'].detach())
                rgb = rgb + sky_color * (1 - result_bkgd['acc'])

            rgb = torch.clamp(rgb, 0.0, 1.0)

            rgb_np = (rgb.detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            cam_id = camera.meta['cam']
            cam_frames[cam_id].append(rgb_np)

        save_dir = os.path.join(cfg.model_path, 'background_gt',
                                f"ours_{scene.loaded_iter}")
        _save_per_camera_videos(cam_frames, save_dir, fps)


def render_lateral_shift():
    """Render front camera shifted laterally from GT trajectory.

    Shifts the front camera (cam==0) along its local right axis in 2 m
    increments from -6 m to +6 m.  Positive values move the camera to the
    RIGHT when viewed from the GT pose.  The seven views are concatenated
    horizontally in the order [+6, +4, +2, 0, -2, -4, -6] m and written
    as a single MP4 video.
    """
    cfg.render.save_image = False
    cfg.render.save_video = False

    fps = cfg.render.get('fps', 24)
    shifts = [6, 4, 2, 0, -2, -4, -6]  # metres; positive = right

    with torch.no_grad():
        dataset = Dataset()
        gaussians = StreetGaussianModel(dataset.scene_info.metadata)
        scene = Scene(gaussians=gaussians, dataset=dataset)
        renderer = StreetGaussianRenderer()

        # Collect front cameras from both train and test splits, sorted by id
        train_cameras = scene.getTrainCameras()
        test_cameras = scene.getTestCameras()
        cameras = train_cameras + test_cameras
        cameras = sorted(cameras, key=lambda x: x.id)
        front_cameras = [c for c in cameras if c.meta['cam'] == 0]

        frames = []
        for camera in tqdm(front_cameras, desc="Rendering lateral shifts"):
            c2w_orig = camera.get_extrinsic()          # 4x4 numpy
            right_dir = c2w_orig[:3, 0]                 # camera local x-axis (right)

            shift_images = []
            for shift in shifts:
                # --- shift camera along its right axis ---
                c2w_shifted = c2w_orig.copy()
                c2w_shifted[:3, 3] += shift * right_dir
                camera.set_extrinsic(c2w_shifted)

                result = renderer.render(camera, gaussians)
                rgb = result['rgb']                     # [3, H, W]
                rgb_np = np.ascontiguousarray(
                    (torch.clamp(rgb, 0, 1)
                     .detach().cpu().numpy()
                     .transpose(1, 2, 0) * 255).astype(np.uint8))

                # Draw label (white text with black outline for readability)
                label = f"{shift:+d}m" if shift != 0 else "0m (GT)"
                cv2.putText(rgb_np, label, (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                            (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(rgb_np, label, (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                            (255, 255, 255), 2, cv2.LINE_AA)

                shift_images.append(rgb_np)

            # Restore original pose
            camera.set_extrinsic(c2w_orig)

            # Concatenate the 7 views horizontally: +6 | +4 | +2 | 0 | -2 | -4 | -6
            frame = np.concatenate(shift_images, axis=1)
            frames.append(frame)

        # Save video
        save_dir = os.path.join(cfg.model_path, 'lateral_shift',
                                f"ours_{scene.loaded_iter}")
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, 'lateral_shift_front.mp4')
        imageio.mimwrite(out_path, frames, fps=fps)
        print(f"Saved {out_path} ({len(frames)} frames)")


if __name__ == "__main__":
    print("Rendering " + cfg.model_path)
    safe_state(cfg.eval.quiet)

    if cfg.mode == 'evaluate':
        render_sets()
    elif cfg.mode == 'trajectory':
        render_trajectory()
    elif cfg.mode == 'lateral_shift':
        render_lateral_shift()
    elif cfg.mode == 'background_gt':
        render_background_gt()
    else:
        raise NotImplementedError()
