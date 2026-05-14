from __future__ import annotations

import os
import torch
from torch.amp import autocast, GradScaler
import patchmatch_cuda

# Cap PyTorch's intra-op thread pool when running as a sweep agent.
# Multiple PyTorch processes default to grabbing every CPU core each, which
# oversubscribes the machine into a load-avg-300 thread-contention spiral
# (GPU goes ~idle while CPUs thrash). The launcher sets VAD_GS_NUM_THREADS
# to floor(nproc / NUM_AGENTS); honour it here as a safety net even when
# users invoke train.py outside the launcher.
_vad_gs_threads = os.environ.get("VAD_GS_NUM_THREADS")
if _vad_gs_threads:
    try:
        _n = max(1, int(_vad_gs_threads))
        torch.set_num_threads(_n)
        torch.set_num_interop_threads(min(_n, 4))
    except (RuntimeError, ValueError):
        pass

import random
from random import randint
from lib.utils.loss_utils import l1_loss, l2_loss, psnr, ssim, patch_norm_mse_loss, patch_norm_mse_loss_global
from lib.utils.img_utils import save_img_torch, visualize_depth_numpy
from lib.models.street_gaussian_renderer import StreetGaussianRenderer
from lib.models.street_gaussian_model import StreetGaussianModel
from lib.utils.general_utils import safe_state
from lib.utils.perf_timer import perf_section, perf_iter_end
from lib.utils.camera_utils import Camera
from lib.utils.cfg_utils import save_cfg
from lib.models.scene import Scene
from lib.datasets.dataset import Dataset
from lib.config import cfg
import torch.distributed as dist
from lib.utils.dist_utils import (
    setup_distributed, cleanup_distributed, is_distributed, is_main_process,
    all_reduce_gradients, sync_densification_stats, broadcast_model_state,
    sync_grad_scaler,
)
from lib.models.mvs import depth_propagation, check_geometric_consistency, read_propagted_depth, depth_propagation_old
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from lib.utils.system_utils import searchForMaxIteration
import numpy as np
import cv2
from lib.models.trellis import parallel_rasterize
from lib.models.street_gaussian_model import quaternion_raw_multiply, matrix_to_quaternion, quaternion_to_matrix
from lib.models.mvs import densify_bkgd_by_viewpoint
############################
import time
import matplotlib.pyplot as plt
import threading
import psutil
from pynvml import *
import queue

marker_queue: queue.Queue[str] = queue.Queue()

############################
import gc
import shutil
from collections import OrderedDict
from plyfile import PlyData, PlyElement
inverse_opacity = lambda x: np.log(x/(1-x))
inverse_scale = lambda x: np.log(x)

# LRU cap for the bkgd_voxel_depth guidance cache. Each entry is ~3-15 MB
# (float16 depth + int32 source arrays scaled with image resolution). Keeping
# all views in RAM (e.g. 1250 views for T4) grows the cache to several GB.
# Holding only the most recently used views trades a small amount of recompute
# for bounded CPU RAM.
BKGD_VOXEL_CACHE_MAX = 128


def _lru_touch_bkgd_voxel_cache(tracker: OrderedDict, cam, max_size: int = BKGD_VOXEL_CACHE_MAX) -> None:
    """Record that *cam* now holds a bkgd_voxel_depth entry and evict the
    oldest tracked camera's cached entry when the tracker exceeds max_size."""
    cam_id = id(cam)
    if cam_id in tracker:
        tracker.move_to_end(cam_id)
    else:
        tracker[cam_id] = cam
    while len(tracker) > max_size:
        _, evicted_cam = tracker.popitem(last=False)
        guidance = getattr(evicted_cam, 'guidance', None)
        if guidance is None:
            continue
        # LazyGuidanceDict stores user-set values in _cache and marks them
        # _persistent; plain dict guidance just needs a del.
        cache = getattr(guidance, '_cache', None)
        persistent = getattr(guidance, '_persistent', None)
        if cache is not None:
            cache.pop('bkgd_voxel_depth', None)
            if persistent is not None:
                persistent.discard('bkgd_voxel_depth')
        else:
            try:
                del guidance['bkgd_voxel_depth']
            except (KeyError, TypeError):
                pass


try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    import wandb as _wandb
    WANDB_FOUND = True
except ImportError:
    _wandb = None
    WANDB_FOUND = False


def _wandb_active() -> bool:
    """True when a wandb run has been initialised (by the sweep wrapper)."""
    return WANDB_FOUND and getattr(_wandb, "run", None) is not None


def _wandb_log(payload: dict, step: int | None = None) -> None:
    if not _wandb_active():
        return
    try:
        if step is not None:
            _wandb.log(payload, step=step)
        else:
            _wandb.log(payload)
    except Exception as exc:
        print(f"[wandb] log failed: {exc}")


def monitor_resources(
    interval: float = 1.0,
    log_file: str | None = None,
    gpu_id: int = 0,
    stop_event: threading.Event | None = None,
    marker_queue: queue.Queue[str] | None = None,
) -> None:
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(gpu_id)

    current_stage = "INIT"

    while stop_event is None or not stop_event.is_set():
        # ====== 处理阶段 marker ======
        while marker_queue is not None and not marker_queue.empty():
            current_stage = marker_queue.get()
            marker_line = (
                f"\n===== STAGE: {current_stage} "
                f"@ {time.strftime('%H:%M:%S')} ====="
            )
            # print(marker_line)
            if log_file:
                with open(log_file, "a") as f:
                    f.write(marker_line + "\n")

        # ====== 资源采样 ======
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory().percent
        info = nvmlDeviceGetMemoryInfo(handle)

        util = nvmlDeviceGetUtilizationRates(handle)
        gpu = util.gpu
        mem_total = round((info.total // 1048576) / 1024)
        mem_process_used = round((info.used // 1048576) / 1024)

        line = (
            f"[{time.strftime('%H:%M:%S')}] "
            f"{current_stage:<20} | "
            f"CPU: {cpu:5.1f}% | "
            f"RAM: {mem:5.1f}% | "
            f"GPU: {gpu:5.1f}% | "
            f"GPU-USED: {mem_process_used:5.1f}G |"
            f"GPU-TOTAL: {mem_total:5.1f}G |"
        )

        # print(line)
        if log_file:
            with open(log_file, "a") as f:
                f.write(line + "\n")

        time.sleep(interval)

    nvmlShutdown()



def training(rank: int = 0, world_size: int = 1) -> None:
    training_args = cfg.train
    optim_args = cfg.optim
    data_args = cfg.data

    start_iter = 0
    tb_writer = prepare_output_and_logger() if is_main_process() else None

    marker_queue.put("Loading Dataset")
    if is_distributed():
        bkgd_ply = os.path.join(cfg.model_path, "input_ply", "points3D_bkgd.ply")
        if not os.path.exists(bkgd_ply):
            raise RuntimeError(
                f"Preprocessing cache missing: {bkgd_ply}\n"
                f"Run `python script/t4/preprocess.py --config <config>.yaml` "
                f"before launching distributed training."
            )
    dataset = Dataset()
    marker_queue.put("Lolding Model")
    gaussians = StreetGaussianModel(dataset.scene_info.metadata)
    marker_queue.put("Lolding Scene")
    scene = Scene(gaussians=gaussians, dataset=dataset)
    marker_queue.put("Start Training")

    cams_per_frame = len(data_args.get("cameras", [0, 1, 2]))

    gaussians.training_setup()
    try:
        if cfg.loaded_iter == -1:
            loaded_iter = searchForMaxIteration(cfg.trained_model_dir)
        else:
            loaded_iter = cfg.loaded_iter
        ckpt_path = os.path.join(cfg.trained_model_dir, f'iteration_{loaded_iter}.pth')
        state_dict = torch.load(ckpt_path)
        start_iter = state_dict['iter']
        print(f'Loading model from {ckpt_path}')
        gaussians.load_state_dict(state_dict)
    except:
        pass

    print(f'Starting from {start_iter}')
    save_cfg(cfg, cfg.model_path, epoch=start_iter)

    gaussians_renderer = StreetGaussianRenderer()

    use_amp = optim_args.use_amp
    scaler = GradScaler('cuda', enabled=use_amp)
    print(f'AMP (Automatic Mixed Precision): {"ON" if use_amp else "OFF"}')

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0
    psnr_dict = {}
    progress_bar = tqdm(range(start_iter, training_args.iterations), disable=not is_main_process())
    start_iter += 1

    # LRU tracker for bkgd_voxel_depth guidance cache (see module-level helper).
    bkgd_voxel_cache_tracker: OrderedDict[int, Camera] = OrderedDict()

    viewpoint_full_stack = [] # for view ID consistency. Test visibility would not be set to zero during training.
    l1 = scene.getTrainCameras().copy()
    l2 = scene.getTestCameras().copy()
    i,j = 0,0
    while (i < len(l1) and j < len(l2)):
        if l1[i].id < l2[j].id:
            viewpoint_full_stack.append(l1[i])
            i = i + 1     
        else:   
            viewpoint_full_stack.append(l2[j])
            j = j + 1
    while i < len(l1):
        viewpoint_full_stack.append(l1[i])
        i = i + 1   
    while j < len(l2):
        viewpoint_full_stack.append(l2[j])
        j = j + 1


    FULL_STACK_LENGTH = len(viewpoint_full_stack)
    DIVERSITY_THRES = 1e-5
    view_stack_iter = start_iter // len(scene.getTrainCameras())

    include_list = list(set(gaussians.model_name_id.keys()) - set(["background"]))

    for obj_name in gaussians.model_name_id:
        obj_model = getattr(gaussians, obj_name)
        if obj_model.grape_trellis is not None:
            selected_frames = data_args.selected_frames
            if selected_frames is None:
                selected_frames = [0, dataset.scene_info.metadata["num_frames"] - 1]
            obj_model.grape_trellis.set_param(dataset.scene_info.metadata["c2ws"], dataset.scene_info.metadata["ixts"], selected_frames, cams_per_frame=cams_per_frame)
    N_bkgd_init = gaussians.background.get_xyz.shape[0]

    # --- View preload to VRAM ---------------------------------------------
    # Distributed always preloads (it is the only way data-parallel ranks
    # stay aligned). Single-GPU preloads when cfg.train.preload_vram is set,
    # which is the default for sweep runs to avoid 8 parallel agents each
    # re-decoding jpegs and reading guidance off disk every iteration.
    preload_vram = is_distributed() or training_args.get("preload_vram", False)
    if preload_vram:
        all_train_cams = scene.getTrainCameras()
        all_train_cams_sorted = sorted(all_train_cams, key=lambda c: c.id)
        if is_distributed():
            local_cameras = all_train_cams_sorted[rank::world_size]
            preload_label = f"{len(local_cameras)} views per GPU ({len(all_train_cams)} total / {world_size} GPUs)"
        else:
            local_cameras = all_train_cams_sorted
            preload_label = f"{len(local_cameras)} views (single-GPU)"

        if is_main_process():
            print(f"Preloading {preload_label} to VRAM")

        _preload_bar = tqdm(
            local_cameras,
            desc="Preload (img + guidance)",
            unit="view",
            disable=not is_main_process(),
        )
        for cam in _preload_bar:
            # Trigger lazy loading of image and all guidance data
            _ = cam.original_image
            for key in list(cam.guidance.keys()):
                _ = cam.guidance[key]
            # Move everything to VRAM
            cam.set_device('cuda')
            # Re-assign via __setitem__ to mark entries as persistent
            # so they survive unload() calls.
            for key in list(cam.guidance.keys()):
                cam.guidance[key] = cam.guidance[key]

        # The per-camera bkgd_voxel_depth is the dominant per-iter cost
        # (~1.6s on this dataset). It is deterministic for a given
        # (camera, image_shape) under the current trellis state, so
        # precompute it for every preloaded view here. The cache cap is
        # bumped to len(local_cameras) below, so the LRU never evicts.
        _trellis = gaussians.background.grape_trellis
        _vox_bar = tqdm(
            local_cameras,
            desc="Preload (bkgd_voxel_depth)",
            unit="view",
            disable=not is_main_process(),
        )
        for cam in _vox_bar:
            if "bkgd_voxel_depth" in cam.guidance:
                continue
            img = cam.original_image
            img_H, img_W = img.shape[1], img.shape[2]
            scaled_K = cam.K.detach().cpu().numpy() if hasattr(cam.K, "detach") else cam.K.cpu().numpy()
            v_val, v_src, _mask, _uvs = _trellis.render_voxel_depth(
                cam.id, img_H, img_W, scaled_K=scaled_K,
            )
            cam.guidance["bkgd_voxel_depth"] = (
                v_val.astype(np.float16),
                v_src.astype(np.int32),
            )
            del _mask, _uvs

        if is_main_process():
            print(f"VRAM preload complete for {len(local_cameras)} views")
    else:
        local_cameras = None  # lazy-load every iter

    viewpoint_stack = None
    check_interval = 0
    check_history = 0
    for iteration in range(start_iter, training_args.iterations + 1):

        iter_start.record()
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        with perf_section("camera_pick"):
            if preload_vram:
                if not viewpoint_stack:
                    viewpoint_stack = list(local_cameras)
                    random.shuffle(viewpoint_stack)
                    view_stack_iter += 1
            else:
                if not viewpoint_stack:
                    viewpoint_stack = scene.getTrainCameras().copy()
                    view_stack_iter += 1

            viewpoint_cam: Camera = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
            randidx = viewpoint_cam.id

        with perf_section("guidance_load"):
            gt_image = viewpoint_cam.original_image
            gt_image = gt_image.cuda(non_blocking=True) if not gt_image.is_cuda else gt_image
            loss_mask = viewpoint_cam.guidance['mask'] if 'mask' in viewpoint_cam.guidance else torch.ones_like(gt_image[0:1]).bool()
            loss_mask = loss_mask.cuda(non_blocking=True) if not loss_mask.is_cuda else loss_mask
            lidar_depth = None
            sky_mask = None
            dynamic_mask = None
            seg_bkgd_mask = None
            mono_depth = None
            mono_normal = None
            if 'lidar_depth' in viewpoint_cam.guidance:
                lidar_depth = viewpoint_cam.guidance['lidar_depth']
                lidar_depth = lidar_depth.cuda(non_blocking=True) if not lidar_depth.is_cuda else lidar_depth
            if 'sky_mask' in viewpoint_cam.guidance:
                sky_mask = viewpoint_cam.guidance['sky_mask']
                sky_mask = sky_mask.cuda(non_blocking=True) if not sky_mask.is_cuda else sky_mask
            # if 'obj_bound' in viewpoint_cam.guidance:
            #     obj_bound = viewpoint_cam.guidance['obj_bound']
            #     obj_bound = obj_bound.cuda(non_blocking=True) if not obj_bound.is_cuda else obj_bound
            if 'dynamic_mask' in viewpoint_cam.guidance:
                dynamic_mask = viewpoint_cam.guidance['dynamic_mask']
                dynamic_mask = dynamic_mask.cuda(non_blocking=True) if not dynamic_mask.is_cuda else dynamic_mask
            if "seg_bkgd" in viewpoint_cam.guidance:
                seg_bkgd_mask = viewpoint_cam.guidance['seg_bkgd']

            if "mono_depth" in viewpoint_cam.guidance:
                mono_depth = viewpoint_cam.guidance['mono_depth']
                mono_depth = mono_depth.cuda(non_blocking=True) if not mono_depth.is_cuda else mono_depth
            if "mono_normal" in viewpoint_cam.guidance:
                mono_normal = viewpoint_cam.guidance['mono_normal']
                mono_normal = mono_normal.cuda(non_blocking=True) if not mono_normal.is_cuda else mono_normal
                # if not viewpoint_cam.guidance['mono_normal'].is_cuda:
                #     viewpoint_cam.guidance['mono_normal'] = viewpoint_cam.guidance['mono_normal'].cuda(non_blocking=True)

        with perf_section("voxel_depth_cache"):
            current_view, img_H, img_W = randidx, gt_image.shape[1], gt_image.shape[2]
            if "bkgd_voxel_depth" not in viewpoint_cam.guidance:
                voxel_depth_value, voxel_depth_source, mask_visible, uvs = gaussians.background.grape_trellis.render_voxel_depth(current_view, img_H, img_W, scaled_K=viewpoint_cam.K.cpu().numpy())
                # Cache only pixel-level data (~3 MB); voxel-level data (mask_visible, uvs)
                # is ~37 MB per camera and only needed for propagation — recomputed on demand.
                viewpoint_cam.guidance["bkgd_voxel_depth"] = (voxel_depth_value.astype(np.float16), voxel_depth_source.astype(np.int32))
                del mask_visible, uvs
            # Bound the per-camera cache via LRU eviction across all previously-visited cameras.
            # In multi-GPU mode, each rank holds few views — no eviction needed.
            if not is_distributed():
                # When we have already paid the RAM cost of preloading every
                # train view into VRAM, holding their (much smaller) voxel
                # depth caches is free — bump the cap so we get a 100% hit
                # rate instead of 1/4 (which was costing ~1.6 s/iter).
                _bkgd_cache_cap = (
                    len(local_cameras) if (preload_vram and local_cameras is not None)
                    else BKGD_VOXEL_CACHE_MAX
                )
                _lru_touch_bkgd_voxel_cache(bkgd_voxel_cache_tracker, viewpoint_cam, max_size=_bkgd_cache_cap)
        

        flag_global_reconstruct = False
        flag_local_reconstruct = False
        flag_actor_reconstruct = False
        _propagation_ran = False

        ###################### hard depth #######################
        # In multi-GPU mode, only rank 0 runs depth propagation.  Other ranks
        # skip this block entirely; model params are broadcast afterwards.
        _skip_propagation = is_distributed() and rank != 0
        # check_views = [5, 10, 20, 40, 60, 80, 100]
        # if view_stack_iter in check_views : # and iteration % optim_args.propagation_interval == 0:
        if not _skip_propagation and view_stack_iter % optim_args.propagation_interval == 0 and iteration > optim_args.propagated_iteration_begin and iteration < optim_args.propagated_iteration_end:
            _propagation_ran = True
            with perf_section("propagation_probe"):
                soft_render_pkg = gaussians_renderer.render(viewpoint_cam, gaussians)
                image = soft_render_pkg["rgb"]
                _similarity = ssim(image, gt_image, mask=loss_mask)
            if _similarity < 0.8:
            # if _similarity > 0.8:
            #     continue

                # ssim(image, gt_image, mask=loss_mask)
                # loss_hard = 0

                hard_render_pkg = gaussians_renderer.render(viewpoint_cam, gaussians, render_type="hard_depth")
                hard_depth = hard_render_pkg["depth"][0]
                del hard_render_pkg

                voxel_depth_value, voxel_depth_source = viewpoint_cam.guidance["bkgd_voxel_depth"]
                voxel_depth_tensor = torch.from_numpy(voxel_depth_value).cuda()

                m1 = (voxel_depth_tensor > 0) & (hard_depth > voxel_depth_tensor * 1.1) # 有初值。但是空了 missing points
                if _similarity < 0.6 or m1.sum() > (voxel_depth_tensor > 0).sum() * 0.5:
                    flag_global_reconstruct = True
                if m1.sum() > (voxel_depth_tensor > 0).sum() * 0.2:
                    flag_local_reconstruct = True
                del hard_depth, voxel_depth_tensor, m1

                # m2 = (voxel_depth_tensor > 0) & (hard_depth < voxel_depth_tensor * 0.9) # objects occluded
                # mask = m1 | m2

                # if optim_args.lambda_depth_lidar > 0 and lidar_depth is not None:            
                #     depth_error = torch.abs((hard_depth[mask] - voxel_depth_tensor[mask]))
                #     voxel_depth_loss = depth_error.mean()

                #     loss_hard += optim_args.lambda_depth_lidar * voxel_depth_loss
                        

                # patch_range = (min(hard_depth.shape[1], hard_depth.shape[2]) // 20, max(hard_depth.shape[1], hard_depth.shape[2]) // 10) # zyk: to be tuned
                # mono_depth[sky_mask] = mono_depth[~sky_mask].mean() # zyk: check if works?
                # hard_depth[sky_mask] = hard_depth[~sky_mask].mean().detach()

                # loss_l2_dpt = patch_norm_mse_loss(hard_depth[None,...], mono_depth[None,...], randint(patch_range[0], patch_range[1]), optim_args.error_tolerance)
                # loss_hard += 0.1 * loss_l2_dpt

                # loss_global = patch_norm_mse_loss_global(hard_depth[None,...], mono_depth[None,...], randint(patch_range[0], patch_range[1]), optim_args.error_tolerance)
                # loss_hard += 1 * loss_global

                acc = soft_render_pkg['acc']

                bkgd_mask = torch.ones_like(acc, dtype=torch.bool)
                if sky_mask is not None:
                    bkgd_mask = bkgd_mask & ~sky_mask
                if dynamic_mask is not None:
                    bkgd_mask = bkgd_mask & ~torch.any(dynamic_mask != 255, axis=0, keepdim=True)
                k = int(len(acc[bkgd_mask]) * 0.25)
                if k > 0:
                    v = acc[bkgd_mask].kthvalue(k).values

                    if v.item() < 0.7:
                        flag_global_reconstruct = True

                    if v.item() < 0.9:
                        flag_local_reconstruct = True

                gaussians.set_visibility(include_list)
                gaussians.parse_camera(viewpoint_cam)
                obj_render_pkg = gaussians_renderer.render_object(viewpoint_cam, gaussians, parse_camera_again=True)
                obj_acc = obj_render_pkg["acc"]

                actor_mask = torch.any(dynamic_mask != 255, axis=0, keepdim=True)

                if len(obj_acc[actor_mask]) > 400:
                    k = int(len(obj_acc[actor_mask]) * 0.25)
                    v = obj_acc[actor_mask].kthvalue(k).values
                    if v.item() < 0.7:
                        flag_actor_reconstruct = True
                del obj_acc


                # if flag_global_reconstruct or flag_local_reconstruct or flag_actor_reconstruct:
                #     check_history = 10
                #     check_interval = 0
                
                # if check_history > 0:
                #     check_history -= 1
                # else:
                #     check_interval += 1
                #     check_interval = min(check_interval, 19)

                # acc = torch.clamp(acc, min=1e-6, max=1.-1e-6)
                # sky_loss = torch.where(sky_mask, -torch.log(1 - acc), -torch.log(acc)).mean()

                # loss_hard.backward()


                # # Optimizer step
                # if iteration < training_args.iterations:                
                #     gaussians.update_optimizer()




        with torch.no_grad():

######################################################## BACKGROUND CONTINUOUS ###############################################################################
            # if iteration > FULL_STACK_LENGTH * 1 and iteration < FULL_STACK_LENGTH * 2:
            if flag_global_reconstruct:
            # if True and iteration > optim_args.propagated_iteration_begin and iteration < optim_args.propagated_iteration_end and (iteration % optim_args.propagation_interval == 0):
                view_has_voxels = gaussians.background.grape_trellis.get_view_has_voxels()
                # src_idxs = [randidx+itv*cams_per_frame for itv in [-2, -1, 1, 2] if ((itv*cams_per_frame + randidx > 0) and (itv*cams_per_frame + randidx < FULL_STACK_LENGTH))] # 随机選一个視角，與前后2帧作patch matching
                src_idxs = []

                pre = randidx - cams_per_frame
                while pre >= 0 and not view_has_voxels[pre]:
                    pre -= cams_per_frame
                if pre >= 0:
                    src_idxs.append(pre)

                pre -= cams_per_frame
                while pre >= 0 and not view_has_voxels[pre]:
                    pre -= cams_per_frame
                if pre >= 0:
                    src_idxs.append(pre)

                post = randidx + cams_per_frame
                while post < FULL_STACK_LENGTH and not view_has_voxels[post]:
                    post += cams_per_frame
                if post < FULL_STACK_LENGTH:
                    src_idxs.append(post)

                post += cams_per_frame
                while post < FULL_STACK_LENGTH and not view_has_voxels[post]:
                    post += cams_per_frame
                if post < FULL_STACK_LENGTH:
                    src_idxs.append(post)        

                # render_pkg = gaussians_renderer.render(viewpoint_cam, gaussians) #, render_type="hard_depth")
                
                # rendered_depth = render_pkg['depth'][0]
                # rendered_normal = render_pkg['normals']
                # radii = render_pkg["radii"]

                rendered_depth = soft_render_pkg['depth'][0]
                rendered_normal = soft_render_pkg['normals']

                # get the propagated depth
                # depth_propagation_old(randidx, src_idxs, rendered_depth.detach().cpu().numpy(), rendered_normal.detach().cpu().numpy().transpose(1,2,0), viewpoint_full_stack, dataset, vehicle_name=None, patch_size=20)
                propagated_depth, cost, propagated_normal = depth_propagation(randidx, src_idxs, rendered_depth, rendered_normal, viewpoint_full_stack, dataset, vehicle_name=None, patch_size=20)

                # propagated_depth_old, cost_old, normal_old = read_propagted_depth('./cache/propagated_depth')
                # cost = torch.tensor(cost).to(rendered_depth.device)
                # normal = torch.tensor(normal).to(rendered_depth.device)
                # #transform normal to camera coordinate
                # # R_w2c = torch.tensor(viewpoint_cam.R.T).cuda().to(torch.float32)
                # # # R_w2c[:, 1:] *= -1
                # # normal = (R_w2c @ normal.view(-1, 3).permute(1, 0)).view(3, viewpoint_cam.image_height, viewpoint_cam.image_width)                
                propagated_normal = (propagated_normal.view(-1, 3).permute(1, 0)).view(3, viewpoint_cam.image_height, viewpoint_cam.image_width)                
                
                # propagated_depth = torch.tensor(propagated_depth).to(rendered_depth.device)
                valid_mask = propagated_depth != 300

                # calculate the abs rel depth error of the propagated depth and rendered depth & render color error
                abs_rel_error = torch.abs(propagated_depth - rendered_depth) / propagated_depth
                depth_error_max_threshold = 1.0
                depth_error_min_threshold = 0.8
                abs_rel_error_threshold = depth_error_max_threshold - (depth_error_max_threshold - depth_error_min_threshold) * (iteration - optim_args.propagated_iteration_begin) / (optim_args.propagated_iteration_end - optim_args.propagated_iteration_begin)
                # color error
                render_color = soft_render_pkg['rgb']

                color_error = torch.abs(render_color - gt_image)
                color_error = color_error.mean(dim=0).squeeze()
                #for waymo, quantile 0.6; for free dataset, quantile 0.4
                error_mask = (abs_rel_error > abs_rel_error_threshold)
                
                # calculate the geometric consistency
                ref_K = viewpoint_cam.K
                ref_pose = viewpoint_cam.world_view_transform.transpose(0, 1).inverse()
                geometric_counts = None
                # for idx, src_idx in enumerate(src_idxs):
                for idx in range(len(src_idxs)):
                    src_idx = src_idxs[idx]
                    src_viewpoint = viewpoint_full_stack[src_idx]
                    #c2w
                    src_pose = src_viewpoint.world_view_transform.transpose(0, 1).inverse()
                    src_K = src_viewpoint.K

                    src_render_pkg = gaussians_renderer.render(src_viewpoint, gaussians) #, render_type="hard_depth")
                    src_rendered_depth = src_render_pkg['depth'][0]
                    src_rendered_normal = src_render_pkg['normals']
                    del src_render_pkg

                    #get the src_depth first
                    # depth_propagation(src_viewpoint, torch.zeros_like(src_projected_depth).cuda(), viewpoint_stack, src_idxs, opt.dataset, opt.patch_size)
                    src_idxs_for_src = src_idxs[:idx] + src_idxs[idx:] + [randidx]
                    # depth_propagation(src_idx, src_idxs_for_src, src_rendered_depth.detach().cpu().numpy(), src_rendered_normal.detach().cpu().numpy().transpose(1,2,0), viewpoint_full_stack, dataset, vehicle_name=None, patch_size=20)
                    src_depth, src_cost, src_normal = depth_propagation(src_idx, src_idxs_for_src, src_rendered_depth, src_rendered_normal, viewpoint_full_stack, dataset, vehicle_name=None, patch_size=20)
                    del src_cost, src_normal, src_rendered_depth, src_rendered_normal

                    # src_depth, cost, src_normal = read_propagted_depth('./cache/propagated_depth')
                    # src_depth = torch.tensor(src_depth).cuda()
                    mask, depth_reprojected, x2d_src, y2d_src, relative_depth_diff = check_geometric_consistency(propagated_depth.unsqueeze(0), ref_K.unsqueeze(0),
                                                                                                                    ref_pose.unsqueeze(0), src_depth.unsqueeze(0),
                                                                                                                    src_K.unsqueeze(0), src_pose.unsqueeze(0), thre1=2, thre2=0.01)
                    del depth_reprojected, x2d_src, y2d_src, relative_depth_diff, src_depth
                    if geometric_counts is None:
                        geometric_counts = mask.to(torch.uint8)
                    else:
                        geometric_counts += mask.to(torch.uint8)
                    del mask
                    torch.cuda.empty_cache()

                cost = geometric_counts.squeeze() # 这里cost大约代表各视角下共享视野的部分，越高代表被共同观测且匹配成功的视角越多
                # cost_mask = cost >= 2
                cost_mask = cost >= len(src_idxs)*0.5 # 0.75
                
                #set -10 as nan              
                update_mask = (cost_mask & ~torch.any(dynamic_mask != 255, axis=0)).unsqueeze(0).repeat(3, 1, 1)
                # normal[update_mask] = -10
                # viewpoint_cam.guidance['mono_normal'][update_mask] = normal[update_mask] #.cpu()
                mono_normal[update_mask] = propagated_normal[update_mask] #.cpu()

                propagated_mask = valid_mask & ~error_mask & cost_mask # propagation有值 & 由渲染获得的D误差足够大 & 被多个视角共同观测 
                if sky_mask is not None:
                    propagated_mask = propagated_mask & ~sky_mask[0]

                # if obj_bound is not None:
                #     propagated_mask = propagated_mask & ~obj_bound[0]

                if dynamic_mask is not None:
                    propagated_mask = propagated_mask & ~torch.any(dynamic_mask != 255, axis=0)

                if propagated_mask.sum() > 100:
                    K = viewpoint_cam.K
                    cam2target = viewpoint_cam.world_view_transform.transpose(0, 1).inverse()
                    bkgd_count = gaussians.background.get_xyz.shape[0]
                    # if bkgd_count < N_bkgd_init:
                    #     target_count = propagated_mask.sum() // 4 # min(propagated_mask.sum(), 1000)
                    # elif bkgd_count < N_bkgd_init*2:
                    #     target_count = propagated_mask.sum() // 16 # min(propagated_mask.sum() // 4, 800)
                    # elif bkgd_count < N_bkgd_init*4:
                    #     target_count = propagated_mask.sum() // 64 # min(propagated_mask.sum() // 16, 400)
                    # else:
                    #     target_count = propagated_mask.sum() // 256 # min(propagated_mask.sum() // 64, 200)
                    if bkgd_count < 1e6:
                        target_count = propagated_mask.sum()  # min(propagated_mask.sum() // 4, 800)
                    elif bkgd_count < 2e6:
                        target_count = propagated_mask.sum() // 4 # min(propagated_mask.sum() // 16, 400)
                    else:
                        target_count = propagated_mask.sum() // 16 # min(propagated_mask.sum() // 64, 200)


                    gaussians.background.densify_from_depth_propagation(K, cam2target, propagated_depth, propagated_normal, propagated_mask.to(torch.bool), acc, gt_image, init_opacity=0.1, target_count=target_count) 
                

##################################### BACKGROUND DISCIRETE ##########################################################
            if flag_local_reconstruct and not optim_args.skip_view_selection: #  or iteration > FULL_STACK_LENGTH * 5 and iteration < FULL_STACK_LENGTH * 20 and (iteration % optim_args.propagation_interval == 0):

                voxel_depth_value, voxel_depth_source, mask_visible, uvs = gaussians.background.grape_trellis.render_voxel_depth(current_view, img_H, img_W, scaled_K=viewpoint_cam.K.cpu().numpy())
                
                # segment_img = cv2.imread(os.path.join(dataset.source_path, "sam_bkgd_masks/000%.3d_%d.png"%(current_view//3, current_view%3)))[:,:,:3]
                # segment_img = cv2.resize(segment_img, (voxel_depth_value.shape[1], voxel_depth_value.shape[0]), interpolation=cv2.INTER_NEAREST)
                vacancy_exist = False
                if seg_bkgd_mask is not None and sky_mask is not None and dynamic_mask is not None:
                    segment_img = seg_bkgd_mask.cpu().detach().numpy().transpose(1,2,0)
                    no_bkgd_mask = np.logical_or(sky_mask.cpu().detach().numpy()[0], torch.any(dynamic_mask != 255, axis=0).cpu().detach().numpy())

                    vacancy_exist, vacancy_colors = gaussians.background.grape_trellis.if_vacancy_in_ref_view_masks(voxel_depth_value, segment_img, no_bkgd_mask, vacancy_threshold=0.5)
                if vacancy_exist:
                # if not vacancy_exist:
                #     continue

                    # render_pkg = gaussians_renderer.render(viewpoint_cam, gaussians) #, render_type="hard_depth")
                    # rendered_depth = render_pkg['depth'][0]
                    # rendered_normal = render_pkg['normals']

                    rendered_depth = soft_render_pkg['depth'][0]
                    rendered_normal = soft_render_pkg['normals']

                    for c in vacancy_colors:
                        selected_vals = segment_img[uvs[mask_visible, 1], uvs[mask_visible, 0], 0]==c[0]
                        mask_visible_obj = mask_visible.copy()
                        mask_visible_obj[mask_visible] &= selected_vals

                        if mask_visible_obj.sum() < 10:
                            continue

                        ref_src_views, viewset_diversity_score, rootvine_xyz = gaussians.background.grape_trellis.sample_viewset_from_obj_mask(current_view, mask_visible_obj, point_obs_ratio=0.5, N=4)
                        
                        if len(ref_src_views) < 4 or viewset_diversity_score < DIVERSITY_THRES:
                            continue

                        src_idxs = ref_src_views[1:]

                        # depth_propagation(randidx, src_idxs, rendered_depth.detach().cpu().numpy(), rendered_normal.detach().cpu().numpy().transpose(1,2,0), viewpoint_full_stack, dataset, vehicle_name=None, patch_size=20)
                        propagated_depth, cost, propagated_normal = depth_propagation(randidx, src_idxs, rendered_depth, rendered_normal, viewpoint_full_stack, dataset, vehicle_name=None, patch_size=20)

                        # propagated_depth, cost, normal = read_propagted_depth('./cache/propagated_depth')
                        # cost = torch.tensor(cost).to(rendered_depth.device)
                        # normal = torch.tensor(normal).to(rendered_depth.device)
                        # # #transform normal to camera coordinate
                        # # R_w2c = torch.tensor(viewpoint_cam.R.T).cuda().to(torch.float32)
                        # # # R_w2c[:, 1:] *= -1
                        # # normal = (R_w2c @ normal.view(-1, 3).permute(1, 0)).view(3, viewpoint_cam.image_height, viewpoint_cam.image_width)                
                        propagated_normal = (propagated_normal.view(-1, 3).permute(1, 0)).view(3, viewpoint_cam.image_height, viewpoint_cam.image_width)                
                        
                        propagated_depth = torch.tensor(propagated_depth).to(rendered_depth.device)
                        valid_mask = propagated_depth != 300

                        # calculate the abs rel depth error of the propagated depth and rendered depth & render color error
                        render_depth = soft_render_pkg['depth'][0]
                        abs_rel_error = torch.abs(propagated_depth - render_depth) / propagated_depth
                        depth_error_max_threshold = 1.0
                        depth_error_min_threshold = 0.8
                        abs_rel_error_threshold = depth_error_max_threshold - (depth_error_max_threshold - depth_error_min_threshold) * (iteration - optim_args.propagated_iteration_begin) / (optim_args.propagated_iteration_end - optim_args.propagated_iteration_begin)
                        # color error
                        render_color = soft_render_pkg['rgb']

                        color_error = torch.abs(render_color - gt_image)
                        color_error = color_error.mean(dim=0).squeeze()
                        #for waymo, quantile 0.6; for free dataset, quantile 0.4
                        error_mask = (abs_rel_error > abs_rel_error_threshold)
                        
                        # calculate the geometric consistency
                        ref_K = viewpoint_cam.K
                        ref_pose = viewpoint_cam.world_view_transform.transpose(0, 1).inverse()
                        geometric_counts = None

                        for idx in range(len(src_idxs)):
                            src_idx = src_idxs[idx]
                            src_viewpoint = viewpoint_full_stack[src_idx]
                            #c2w
                            src_pose = src_viewpoint.world_view_transform.transpose(0, 1).inverse()
                            src_K = src_viewpoint.K

                            src_render_pkg = gaussians_renderer.render(src_viewpoint, gaussians) #, render_type="hard_depth")
                            src_rendered_depth = src_render_pkg['depth'][0]
                            src_rendered_normal = src_render_pkg['normals']
                            del src_render_pkg

                            #get the src_depth first
                            src_idxs_for_src = src_idxs[:idx] + src_idxs[idx:] + [randidx]
                            # depth_propagation(src_idx, src_idxs_for_src, src_rendered_depth.detach().cpu().numpy(), src_rendered_normal.detach().cpu().numpy().transpose(1,2,0), viewpoint_full_stack, dataset, vehicle_name=None, patch_size=20)
                            src_depth, cost, src_normal = depth_propagation(src_idx, src_idxs_for_src, src_rendered_depth, src_rendered_normal, viewpoint_full_stack, dataset, vehicle_name=None, patch_size=20)
                            del cost, src_normal, src_rendered_depth, src_rendered_normal

                            # src_depth, cost, src_normal = read_propagted_depth('./cache/propagated_depth')
                            # src_depth = torch.tensor(src_depth).cuda()
                            mask, depth_reprojected, x2d_src, y2d_src, relative_depth_diff = check_geometric_consistency(propagated_depth.unsqueeze(0), ref_K.unsqueeze(0),
                                                                                                                            ref_pose.unsqueeze(0), src_depth.unsqueeze(0),
                                                                                                                            src_K.unsqueeze(0), src_pose.unsqueeze(0), thre1=2, thre2=0.01)
                            del depth_reprojected, x2d_src, y2d_src, relative_depth_diff, src_depth
                            if geometric_counts is None:
                                geometric_counts = mask.to(torch.uint8)
                            else:
                                geometric_counts += mask.to(torch.uint8)
                            del mask
                            torch.cuda.empty_cache()

                        cost = geometric_counts.squeeze() # 这里cost大约代表各视角下共享视野的部分，越高代表被共同观测且匹配成功的视角越多
                        # cost_mask = cost >= 2
                        cost_mask = cost >= len(src_idxs)*0.5 #0.75
                        
                        #set -10 as nan     
                        update_mask = (cost_mask & torch.from_numpy(segment_img[:,:,0]==c[0]).cuda()).unsqueeze(0).repeat(3, 1, 1)
                        # normal[~(cost_mask.unsqueeze(0).repeat(3, 1, 1))] = -10
                        mono_normal[update_mask] = propagated_normal[update_mask] #.cpu()
                        
                        propagated_mask = valid_mask & ~error_mask & cost_mask # propagation有值 & 由渲染获得的D误差足够大 & 被多个视角共同观测 
                        if sky_mask is not None:
                            propagated_mask = propagated_mask & ~sky_mask[0]

                        if dynamic_mask is not None:
                            propagated_mask = propagated_mask & torch.all(dynamic_mask == 255, axis=0)
                        
                        obj_mask = torch.from_numpy(np.all(segment_img==c, axis=2)).cuda()
                        propagated_mask = propagated_mask & obj_mask
                        
                        if propagated_mask.sum() > 100:
                            K = viewpoint_cam.K
                            cam2target = viewpoint_cam.world_view_transform.transpose(0, 1).inverse()
                            bkgd_count = gaussians.background.get_xyz.shape[0]
                            # if bkgd_count < N_bkgd_init:
                            #     target_count = propagated_mask.sum() // 4  #min(propagated_mask.sum(), 1000)
                            # elif bkgd_count < N_bkgd_init * 2:
                            #     target_count = propagated_mask.sum() // 16 #min(propagated_mask.sum() // 4, 800)
                            # elif bkgd_count < N_bkgd_init * 4:
                            #     target_count = propagated_mask.sum() // 64 #min(propagated_mask.sum() // 16, 500)
                            # else:
                            #     target_count = propagated_mask.sum() // 256 #min(propagated_mask.sum() // 64, 200)

                            if bkgd_count < 1e6:
                                target_count = propagated_mask.sum()
                            elif bkgd_count < 2e6:
                                target_count = propagated_mask.sum() // 4
                            else:
                                target_count = propagated_mask.sum() // 16

                            gaussians.background.densify_from_depth_propagation(K, cam2target, propagated_depth, propagated_normal, propagated_mask.to(torch.bool), acc, gt_image, init_opacity=0.3, target_count=target_count) 



##################################################### OBJECT #######################################################################################
            if flag_actor_reconstruct and not optim_args.skip_view_selection and dynamic_mask is not None: # or iteration > FULL_STACK_LENGTH * 0 and iteration < FULL_STACK_LENGTH * 20 and (iteration % optim_args.propagation_interval == 0):


                # Step1: render foreground

                for dynamic_key in torch.unique(dynamic_mask):
                    if dynamic_key == 255:
                        continue

                    dynamic_id = dynamic_key.item()
                    obj_name = "obj_%.3d"%dynamic_id
                    if obj_name not in include_list:
                        continue

                    # render_pkg_obj = gaussians_renderer.render(viewpoint_cam, gaussians, exclude_list=["background"], render_type="hard_depth")

                    gaussians.set_visibility(include_list)
                    gaussians.parse_camera(viewpoint_cam)
                    # render_pkg_obj = gaussians_renderer.render_object(viewpoint_cam, gaussians, parse_camera_again=True)
                    # render_color = render_pkg_obj['rgb']
                    # rendered_depth = render_pkg_obj['depth'][0]
                    # rendered_normal = render_pkg_obj['normals']
                    # render_acc = render_pkg_obj['acc']

                    render_color = obj_render_pkg['rgb']
                    rendered_depth = obj_render_pkg['depth'][0]
                    rendered_normal = obj_render_pkg['normals']
                    render_acc = obj_render_pkg['acc']

                    if obj_name not in gaussians.graph_obj_list:
                        continue

                    obj_model = getattr(gaussians, obj_name)
                    if obj_model.random_initialization or obj_model.deformable: # should be enough
                        continue

                    obj_mask = torch.any(dynamic_mask==dynamic_id, axis=0)
                    obj_acc = render_acc[:, obj_mask]
                    if (obj_acc < 0.5).sum() < (obj_acc > 0.9).sum() * 0.1:
                        continue


                    track_id = obj_model.track_id
                    obj_rot = gaussians.actor_pose.get_tracking_rotation(track_id, viewpoint_cam)
                    obj_trans = gaussians.actor_pose.get_tracking_translation(track_id, viewpoint_cam)                
                    ego_pose = viewpoint_cam.ego_pose
                    ego_pose_rot = matrix_to_quaternion(ego_pose[:3, :3].unsqueeze(0)).squeeze(0)
                    obj_rot = quaternion_raw_multiply(ego_pose_rot.unsqueeze(0), obj_rot.unsqueeze(0)).squeeze(0)
                    obj_rots = quaternion_to_matrix(obj_rot)
                    obj_trans = ego_pose[:3, :3] @ obj_trans + ego_pose[:3, 3]

                    voxel_depth_value, voxel_depth_source, mask_visible, uvs = obj_model.grape_trellis.render_voxel_depth(current_view, img_H, img_W, obj_rots, obj_trans, scaled_K=viewpoint_cam.K.cpu().numpy())
                    if mask_visible.sum() < 10:
                        continue
                    # 过低的 point_obs_ratio 可能没有输出。按逻辑而言需要在此处加入更新voxel部分
                    # ref_src_views, viewset_diversity_score, rootvine_xyz = obj_model.grape_trellis.sample_viewset_from_obj_mask(current_view, dynamic_mask==dynamic_id, voxel_depth_value, voxel_depth_source, mask_visible, point_obs_ratio=0.8)
                    ref_src_views, viewset_diversity_score, rootvine_xyz = obj_model.grape_trellis.sample_viewset_from_obj_mask(current_view, mask_visible, point_obs_ratio=0.4, N=4, obj_rots=obj_rots, obj_trans=obj_trans)

                    if len(ref_src_views) <= 4 or viewset_diversity_score < DIVERSITY_THRES:
                        continue

                    # vehicle_name = None
                    vehicle_name = int(obj_name.split("_")[-1])

                    src_idxs = ref_src_views[1:]
                    # get the propagated depth
                    # depth_propagation(randidx, src_idxs, rendered_depth.detach().cpu().numpy(), rendered_normal.detach().cpu().numpy().transpose(1,2,0), viewpoint_full_stack, dataset, vehicle_name=vehicle_name, patch_size=20)
                    propagated_depth, cost, propagated_normal = depth_propagation(randidx, src_idxs, rendered_depth, rendered_normal, viewpoint_full_stack, dataset, vehicle_name=vehicle_name, patch_size=20)
                    if propagated_depth is None:
                        print("no props")
                        continue 

                    # propagated_depth, cost, propagated_normal = read_propagted_depth('./cache/propagated_depth')
                    # cost = torch.tensor(cost).to(rendered_depth.device)
                    # normal = torch.tensor(normal).to(rendered_depth.device)
                    # #transform normal to camera coordinate
                    # R_w2c = torch.tensor(viewpoint_cam.R.T).cuda().to(torch.float32)
                    # # R_w2c[:, 1:] *= -1
                    # normal = (R_w2c @ normal.view(-1, 3).permute(1, 0)).view(3, viewpoint_cam.image_height, viewpoint_cam.image_width)                
                    propagated_normal = (propagated_normal.view(-1, 3).permute(1, 0)).view(3, viewpoint_cam.image_height, viewpoint_cam.image_width)                
                    
                    # propagated_depth = torch.tensor(propagated_depth).to(rendered_depth.device)
                    valid_mask = propagated_depth != 300

                    # calculate the abs rel depth error of the propagated depth and rendered depth & render color error
                    abs_rel_error = torch.abs(propagated_depth - rendered_depth) / propagated_depth
                    depth_error_max_threshold = 1.0
                    depth_error_min_threshold = 0.8
                    abs_rel_error_threshold = depth_error_max_threshold - (depth_error_max_threshold - depth_error_min_threshold) * (iteration - optim_args.propagated_iteration_begin) / (optim_args.propagated_iteration_end - optim_args.propagated_iteration_begin)
                    # color error

                    # color_error = torch.abs(render_color - gt_image)
                    # color_error = color_error.mean(dim=0).squeeze()
                    #for waymo, quantile 0.6; for free dataset, quantile 0.4
                    error_mask = (abs_rel_error > abs_rel_error_threshold)
                    
                    # calculate the geometric consistency
                    ref_K = viewpoint_cam.K
                    ref_pose = viewpoint_cam.world_view_transform.transpose(0, 1).inverse()
                    geometric_counts = None
                    # for idx, src_idx in enumerate(src_idxs):
                    for idx in range(len(src_idxs)):
                        src_idx = src_idxs[idx]
                        src_viewpoint = viewpoint_full_stack[src_idx]
                        #c2w
                        src_pose = src_viewpoint.world_view_transform.transpose(0, 1).inverse()
                        src_K = src_viewpoint.K

                        gaussians.set_visibility(include_list)
                        gaussians.parse_camera(src_viewpoint)

                        # src_render_pkg = gaussians_renderer.render(src_viewpoint, gaussians) #, render_type="hard_depth")
                        src_render_pkg = gaussians_renderer.render_object(src_viewpoint, gaussians, parse_camera_again=False)
                        src_rendered_depth = src_render_pkg['depth'][0]
                        src_rendered_normal = src_render_pkg['normals']
                        del src_render_pkg

                        #get the src_depth first
                        # depth_propagation(src_viewpoint, torch.zeros_like(src_projected_depth).cuda(), viewpoint_stack, src_idxs, opt.dataset, opt.patch_size)
                        src_idxs_for_src = src_idxs[:idx] + src_idxs[idx:] + [randidx]
                        # depth_propagation(src_idx, src_idxs_for_src, src_rendered_depth.detach().cpu().numpy(), src_rendered_normal.detach().cpu().numpy().transpose(1,2,0), viewpoint_full_stack, dataset, vehicle_name=vehicle_name, patch_size=20)
                        src_depth, cost, src_normal = depth_propagation(src_idx, src_idxs_for_src, src_rendered_depth, src_rendered_normal, viewpoint_full_stack, dataset, vehicle_name=vehicle_name, patch_size=20)
                        del cost, src_normal, src_rendered_depth, src_rendered_normal
                        if src_depth is None:
                            print("no props")
                            continue

                        # src_depth, cost, src_normal = read_propagted_depth('./cache/propagated_depth')
                        # src_depth = torch.tensor(src_depth).cuda()
                        mask, depth_reprojected, x2d_src, y2d_src, relative_depth_diff = check_geometric_consistency(propagated_depth.unsqueeze(0), ref_K.unsqueeze(0),
                                                                                                                        ref_pose.unsqueeze(0), src_depth.unsqueeze(0),
                                                                                                                        src_K.unsqueeze(0), src_pose.unsqueeze(0), thre1=2, thre2=0.01)
                        del depth_reprojected, x2d_src, y2d_src, relative_depth_diff, src_depth
                        if geometric_counts is None:
                            geometric_counts = mask.to(torch.uint8)
                        else:
                            geometric_counts += mask.to(torch.uint8)
                        del mask
                        torch.cuda.empty_cache()
                            
                    if geometric_counts is None:
                        continue
                    cost = geometric_counts.squeeze() # 这里cost大约代表各视角下共享视野的部分，越高代表被共同观测且匹配成功的视角越多
                    # cost_mask = cost >= 2
                    cost_mask = cost >= len(src_idxs)*0.5 #0.75
                    
                    #set -10 as nan              
                    # normal[~(cost_mask.unsqueeze(0).repeat(3, 1, 1))] = -10
                    # if viewpoint_cam.guidance['mono_normal'] is None:
                    #     viewpoint_cam.guidance['mono_normal'] = normal.cpu()
                    # else:
                    update_mask = (cost_mask & obj_mask).unsqueeze(0).repeat(3, 1, 1)
                    # update_mask = obj_mask & normal[:,:,0] != -10
                    mono_normal[update_mask] = propagated_normal[update_mask]
                    # viewpoint_cam.guidance['mono_normal'] = mono_normal #.cpu()
                    
                    propagated_mask = valid_mask & ~error_mask & cost_mask # propagation有值 & 由渲染获得的D误差足够大 & 被多个视角共同观测 
                    if sky_mask is not None:
                        propagated_mask = propagated_mask & ~sky_mask[0]

                    # if obj_bound is not None:
                    #     propagated_mask = propagated_mask & ~obj_bound[0]

                    if obj_mask is not None:
                        propagated_mask = propagated_mask & obj_mask

                    if propagated_mask.sum() > 100:
                        K = viewpoint_cam.K
                        cam2target = viewpoint_cam.world_view_transform.transpose(0, 1).inverse()
                        
                        actor_count = obj_model.get_xyz.shape[0]
                        if actor_count < 1000:
                            target_count = min(propagated_mask.sum() // 2, 800)
                        if actor_count < 2000:
                            target_count = min(propagated_mask.sum() // 4, 500)
                        elif actor_count < 4000:
                            target_count = min(propagated_mask.sum() // 8, 200)
                        else:
                            target_count = min(propagated_mask.sum() // 16, 100)

                        if target_count < 10:
                            continue
                        obj_model.densify_from_depth_propagation(K, cam2target, propagated_depth, propagated_normal, propagated_mask.to(torch.bool), render_acc, gt_image, obj_rots, obj_trans, init_opacity=0.3, target_count=target_count) 
                    
            # Free source camera images/guidance loaded during depth propagation
            if _propagation_ran:
                for _vp in viewpoint_full_stack:
                    if _vp is not viewpoint_cam:
                        _vp.unload_image()
                        if hasattr(_vp.guidance, 'unload'):
                            _vp.guidance.unload()
                del soft_render_pkg, image
            torch.cuda.empty_cache()

        # In multi-GPU mode, broadcast updated model params after rank 0
        # runs depth-propagation-based densification.  The flag must be
        # shared so all ranks enter the collective broadcast together.
        if is_distributed():
            _flag = torch.tensor([1 if _propagation_ran else 0], device='cuda')
            dist.broadcast(_flag, src=0)
            if _flag.item():
                # rank 0 ran depth-propagation densify; sync the full state
                # (param data + Adam state + densify accumulators) so other
                # ranks pick up the new tensor shapes.
                broadcast_model_state(gaussians, src=0)
            del _flag


        voxel_depth_value, voxel_depth_source = viewpoint_cam.guidance["bkgd_voxel_depth"]
        voxel_depth_tensor = torch.from_numpy(voxel_depth_value).cuda()

        if iteration > optim_args.hard_depth_start and iteration < optim_args.hard_depth_end and "mono_depth" in viewpoint_cam.guidance:
            with perf_section("hard_depth_step"):
                loss_hard = 0
                with autocast('cuda', enabled=use_amp):
                    hard_render_pkg = gaussians_renderer.render(viewpoint_cam, gaussians, render_type="hard_depth")
                    hard_depth = hard_render_pkg["depth"]

                    patch_range = (min(hard_depth.shape[1], hard_depth.shape[2]) // 20, max(hard_depth.shape[1], hard_depth.shape[2]) // 10) # zyk: to be tuned
                    if sky_mask is not None:
                        mono_depth[sky_mask] = mono_depth[~sky_mask].mean() # zyk: check if works?
                        hard_depth[sky_mask] = hard_depth[~sky_mask].mean().detach()

                    loss_l2_dpt = patch_norm_mse_loss(hard_depth[None,...], mono_depth[None,...], randint(patch_range[0], patch_range[1]), 0.01)
                    loss_hard += 1 * loss_l2_dpt

                    loss_global = patch_norm_mse_loss_global(hard_depth[None,...], mono_depth[None,...], randint(patch_range[0], patch_range[1]), 0.01)
                    loss_hard += 1 * loss_global

                scaler.scale(loss_hard).backward()
                if is_distributed():
                    all_reduce_gradients(gaussians)
                # Optimizer step
                if iteration < training_args.iterations:
                    gaussians.update_optimizer(scaler=scaler if use_amp else None)
                    if use_amp:
                        scaler.update()
                del hard_render_pkg, hard_depth, loss_hard, loss_l2_dpt, loss_global
                torch.cuda.empty_cache()

        with autocast('cuda', enabled=use_amp):
            with perf_section("main_render"):
                soft_render_pkg = gaussians_renderer.render(viewpoint_cam, gaussians)
                image, acc, viewspace_point_tensor, visibility_filter, radii = soft_render_pkg["rgb"], soft_render_pkg['acc'], soft_render_pkg["viewspace_points"], soft_render_pkg["visibility_filter"], soft_render_pkg["radii"]

            scalar_dict = dict()

            # rgb loss
            with perf_section("loss_rgb"):
                Ll1 = l1_loss(image, gt_image, mask=loss_mask)
                scalar_dict['l1_loss'] = Ll1.item()
                loss = (1.0 - optim_args.lambda_dssim) * optim_args.lambda_l1 * Ll1 + optim_args.lambda_dssim * (1.0 - ssim(image, gt_image, mask=loss_mask))

            # shape_pena = (gaussians.get_scaling.max(dim=1).values / gaussians.get_scaling.min(dim=1).values).mean()
            # scale_pena = ((gaussians.get_scaling.max(dim=1, keepdim=True).values)**2).mean()
            # loss_reg = optim_args.lambda_shape_pena*shape_pena + optim_args.lambda_scale_pena*scale_pena
            # loss += loss_reg


            # sky loss
            with perf_section("loss_sky"):
                if optim_args.lambda_sky > 0 and gaussians.include_sky and sky_mask is not None:
                    acc = torch.clamp(acc, min=1e-6, max=1.-1e-6)
                    sky_loss = torch.where(sky_mask, -torch.log(1 - acc), -torch.log(acc)).mean()
                    if len(optim_args.lambda_sky_scale) > 0:
                        sky_loss *= optim_args.lambda_sky_scale[viewpoint_cam.meta['cam']]
                    scalar_dict['sky_loss'] = sky_loss.item()
                    loss += optim_args.lambda_sky * sky_loss

            with perf_section("loss_obj_acc"):
                if optim_args.lambda_reg > 0 and gaussians.include_obj and iteration >= optim_args.densify_until_iter:
                    render_pkg_obj = gaussians_renderer.render_object(viewpoint_cam, gaussians, parse_camera_again=False)
                    image_obj, acc_obj = render_pkg_obj["rgb"], render_pkg_obj['acc']
                    del render_pkg_obj
                    acc_obj = torch.clamp(acc_obj, min=1e-6, max=1.-1e-6)
                    obj_acc_loss = torch.where(torch.any(dynamic_mask != 255, axis=0), # obj_bound,
                        -(acc_obj * torch.log(acc_obj) +  (1. - acc_obj) * torch.log(1. - acc_obj)),
                        -torch.log(1. - acc_obj)).mean()
                    scalar_dict['obj_acc_loss'] = obj_acc_loss.item()
                    loss += optim_args.lambda_reg * obj_acc_loss
                    del image_obj, acc_obj



            # lidar depth loss
            with perf_section("loss_depth"):
                if optim_args.lambda_depth_lidar > 0:
                    if optim_args.use_lidar_depth and lidar_depth is not None:
                        depth_mask = torch.logical_and((lidar_depth > 0.), loss_mask)
                        expected_depth = soft_render_pkg['depth'] / (soft_render_pkg['acc'] + 1e-10)
                        depth_error = torch.abs((expected_depth[depth_mask] - lidar_depth[depth_mask]))
                        depth_error, _ = torch.topk(depth_error, int(0.95 * depth_error.size(0)), largest=False)
                        lidar_depth_loss = depth_error.mean()
                        scalar_dict['lidar_depth_loss'] = lidar_depth_loss.item()
                        loss += optim_args.lambda_depth_lidar * lidar_depth_loss
                        del expected_depth, depth_error, lidar_depth_loss

                    if optim_args.use_voxel_depth:
                        depth_mask = torch.logical_and((voxel_depth_tensor > 0.), loss_mask)
                        expected_depth = soft_render_pkg['depth'] / (soft_render_pkg['acc'] + 1e-10)
                        depth_error = torch.abs((expected_depth[depth_mask] - voxel_depth_tensor[depth_mask[0]]))
                        depth_error, _ = torch.topk(depth_error, int(0.95 * depth_error.size(0)), largest=False)
                        voxel_depth_loss = depth_error.mean()
                        scalar_dict['lidar_depth_loss'] = voxel_depth_loss.item()
                        loss += optim_args.lambda_depth_lidar * voxel_depth_loss
                        del expected_depth, depth_error, voxel_depth_loss


            # color correction loss
            with perf_section("loss_color_correction"):
                if optim_args.lambda_color_correction > 0 and gaussians.use_color_correction:
                    color_correction_reg_loss = gaussians.color_correction.regularization_loss(viewpoint_cam)
                    scalar_dict['color_correction_reg_loss'] = color_correction_reg_loss.item()
                    loss += optim_args.lambda_color_correction * color_correction_reg_loss

            with perf_section("loss_normal"):
                if optim_args.normal_loss:
                    if mono_normal is not None and 'normals' in soft_render_pkg:
                        rendered_normal = soft_render_pkg['normals']
                        normal_gt = mono_normal #
                        if sky_mask is not None: # if viewpoint_cam.sky_mask is not None:
                            filter_mask = sky_mask.to(normal_gt.device).to(torch.bool)
                            normal_gt[(filter_mask.repeat(3, 1, 1))] = -10

                        filter_mask = (normal_gt != -10)[0, :, :].to(torch.bool)
                        l1_normal = torch.abs(rendered_normal - normal_gt).sum(dim=0)[filter_mask].mean()
                        cos_normal = (1. - torch.sum(rendered_normal * normal_gt, dim = 0))[filter_mask].mean()

                        lambda_l1_normal = 0.02
                        lambda_cos_normal = 0.02
                        loss += lambda_l1_normal * l1_normal + lambda_cos_normal * cos_normal

                        viewpoint_cam.guidance['mono_normal'] = mono_normal.cpu()

        scalar_dict['loss'] = loss.item()

        with perf_section("backward"):
            scaler.scale(loss).backward()
            if is_distributed():
                all_reduce_gradients(gaussians)

        iter_end.record()

        is_save_images = True
        if is_save_images and (iteration % 100 == 0) and is_main_process():
            # row0: gt_image, image, depth
            # row1: acc, image_obj, acc_obj
            depth_colored, _ = visualize_depth_numpy(soft_render_pkg['depth'].detach().cpu().numpy().squeeze(0))
            depth_colored = depth_colored[..., [2, 1, 0]] / 255.
            depth_colored = torch.from_numpy(depth_colored).permute(2, 0, 1).float().cuda()
            row0 = torch.cat([gt_image, image, depth_colored], dim=2)
            acc = acc.repeat(3, 1, 1)
            with torch.no_grad():
                render_pkg_obj = gaussians_renderer.render_object(viewpoint_cam, gaussians)
                image_obj, acc_obj = render_pkg_obj["rgb"], render_pkg_obj['acc']
                del render_pkg_obj
            acc_obj = acc_obj.repeat(3, 1, 1)
            # row1 = torch.cat([acc, image_obj, acc_obj], dim=2)
            if mono_depth is None:
                raise RuntimeError(
                    f"mono_depth is None for image={viewpoint_cam._image_path}. "
                    f"Check that mono depth maps exist under <dataroot>/preprocessed/depth/. "
                    f"Run: python script/t4/generate_mono_depth.py --config <your_config.yaml>"
                )
            row1 = torch.cat([voxel_depth_tensor[None,:,:].repeat(3,1,1) / voxel_depth_tensor.max(), image_obj, mono_depth.repeat(3,1,1)], dim=2)
            # row1 = torch.cat([normal_gt/2+0.5, image_obj, soft_render_pkg['normals']/2+0.5], dim=2)
            image_to_show = torch.cat([row0, row1], dim=1)
            image_to_show = torch.clamp(image_to_show, 0.0, 1.0)
            os.makedirs(f"{cfg.model_path}/log_images", exist_ok = True)
            save_img_torch(image_to_show, f"{cfg.model_path}/log_images/{iteration}.jpg")
            print(f"[log_image] iter={iteration} image={viewpoint_cam._image_path} frame={viewpoint_cam.meta.get('frame', '?')} cam={viewpoint_cam.meta.get('cam', '?')} obj_in_graph={getattr(gaussians, 'graph_obj_list', [])}")
            del row0, row1, image_to_show, depth_colored, image_obj, acc_obj
        
        with torch.no_grad():
            
            # Log
            tensor_dict = dict()

            if iteration % 10 == 0:                    
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if np.isnan(ema_loss_for_log):
                    ema_loss_for_log = loss.item()

                ema_psnr_for_log = 0.4 * psnr(image, gt_image, loss_mask).mean().float().item() + 0.6 * ema_psnr_for_log
                if np.isnan(ema_psnr_for_log):
                    ema_psnr_for_log = psnr(image, gt_image, loss_mask).mean().float().item()
                
                progress_bar.set_postfix({"Exp": f"{cfg.task}-{cfg.exp_name}", 
                                          "Loss": f"{ema_loss_for_log:.{7}f},", 
                                          "PSNR": f"{ema_psnr_for_log:.{4}f}",
                                          "SSIM": f"{ssim(image, gt_image):.{4}f}",
                                          "GS":  str(gaussians.background.get_xyz.shape[0])
                                          })
            progress_bar.update(1)
            # if iteration == training_args.iterations:
            #     progress_bar.close()

            # Save ply
            if (iteration in training_args.save_iterations) and is_main_process():
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < optim_args.densify_until_iter:
                with perf_section("densify_stats"):
                    gaussians.set_visibility(include_list=list(set(gaussians.model_name_id.keys()) - set(['sky'])))
                    gaussians.set_max_radii2D(radii, visibility_filter)
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                
                prune_big_points = iteration > optim_args.opacity_reset_interval

                if iteration > optim_args.densify_from_iter:
                    if iteration % optim_args.densification_interval == 0:
                        with perf_section("densify_and_prune"):
                            if is_distributed():
                                # Aggregate densification accumulators so rank 0
                                # makes the decision using the full data.
                                sync_densification_stats(gaussians)
                                if is_main_process():
                                    scalars, tensors = gaussians.densify_and_prune(
                                        max_grad=optim_args.densify_grad_threshold,
                                        min_opacity=optim_args.min_opacity,
                                        prune_big_points=prune_big_points,
                                    )
                                else:
                                    scalars, tensors = {}, {}
                                # Broadcast the new param shapes / data / optimizer
                                # state / accumulators so every rank matches rank 0.
                                broadcast_model_state(gaussians, src=0)
                            else:
                                scalars, tensors = gaussians.densify_and_prune(
                                    max_grad=optim_args.densify_grad_threshold,
                                    min_opacity=optim_args.min_opacity,
                                    prune_big_points=prune_big_points,
                                )

                            scalar_dict.update(scalars)
                            tensor_dict.update(tensors)

            # Reset opacity
            if iteration < optim_args.densify_until_iter:
                if iteration % optim_args.opacity_reset_interval == 0:
                    gaussians.reset_opacity()
                if data_args.white_background and iteration == optim_args.densify_from_iter:
                    gaussians.reset_opacity()

            if is_main_process():
                with perf_section("training_report"):
                    training_report(tb_writer, iteration, scalar_dict, tensor_dict, training_args.test_iterations, scene, gaussians_renderer)
            del scalar_dict, tensor_dict, soft_render_pkg, image, acc, viewspace_point_tensor, visibility_filter, radii, loss

            # Optimizer step
            if iteration < training_args.iterations:
                with perf_section("optimizer_step"):
                    gaussians.update_optimizer(scaler=scaler if use_amp else None)
                if use_amp:
                    scaler.update()
                    if is_distributed():
                        sync_grad_scaler(scaler)

            if (iteration in training_args.checkpoint_iterations) and is_main_process():
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                state_dict = gaussians.save_state_dict(is_final=(iteration == training_args.iterations))
                state_dict['iter'] = iteration
                ckpt_path = os.path.join(cfg.trained_model_dir, f'iteration_{iteration}.pth')
                torch.save(state_dict, ckpt_path)
                del state_dict
                gc.collect()

            # Viewer PLY export is memory-heavy (several GB of CPU RAM spike for
            # multi-million Gaussians). Only run on save_iterations to avoid OOM
            # during routine checkpoints.
            if (iteration in training_args.save_iterations) and is_main_process():
                gaussians.set_visibility(list(set(gaussians.model_name_id.keys())))
                gaussians.parse_camera(camera=viewpoint_cam)

                # Pull each tensor to CPU individually and drop GPU/intermediate
                # references as soon as possible to keep peak RAM low.
                xyz = gaussians.get_xyz.detach().cpu().numpy()
                N = xyz.shape[0]

                f = gaussians.get_features.detach().transpose(1, 2).contiguous()  # [N, 3, sh_degree]
                f_dc = f[..., :1].flatten(start_dim=1).cpu().numpy()
                f_rest = f[..., 1:].flatten(start_dim=1).cpu().numpy()
                del f
                torch.cuda.empty_cache()

                opacities = gaussians.get_opacity.detach().cpu().numpy()
                np.clip(opacities, a_min=1e-6, a_max=1. - 1e-6, out=opacities)
                opacities = inverse_opacity(opacities)

                scale = inverse_scale(gaussians.get_scaling.detach().cpu().numpy())
                rotation = gaussians.get_rotation.detach().cpu().numpy()

                l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
                for i in range(f_dc.shape[1]):
                    l.append('f_dc_{}'.format(i))
                for i in range(f_rest.shape[1]):
                    l.append('f_rest_{}'.format(i))
                l.append('opacity')
                for i in range(scale.shape[1]):
                    l.append('scale_{}'.format(i))
                for i in range(rotation.shape[1]):
                    l.append('rot_{}'.format(i))
                dtype_full = [(attribute, 'f4') for attribute in l]

                # Build the structured array field-by-field. This avoids the
                # Python `list(map(tuple, attributes))` which allocates N tuple
                # objects (hundreds of MB of overhead for N ~ 6M) and also
                # avoids the full np.concatenate copy.
                elements = np.empty(N, dtype=dtype_full)
                elements['x'] = xyz[:, 0]
                elements['y'] = xyz[:, 1]
                elements['z'] = xyz[:, 2]
                elements['nx'] = 0.0
                elements['ny'] = 0.0
                elements['nz'] = 0.0
                del xyz
                for i in range(f_dc.shape[1]):
                    elements['f_dc_{}'.format(i)] = f_dc[:, i]
                del f_dc
                for i in range(f_rest.shape[1]):
                    elements['f_rest_{}'.format(i)] = f_rest[:, i]
                del f_rest
                elements['opacity'] = opacities[:, 0]
                del opacities
                for i in range(scale.shape[1]):
                    elements['scale_{}'.format(i)] = scale[:, i]
                del scale
                for i in range(rotation.shape[1]):
                    elements['rot_{}'.format(i)] = rotation[:, i]
                del rotation
                gc.collect()

                save_dir = os.path.join(cfg.model_path, 'viewer', f'iteration_{iteration}_{current_view:06d}')
                pointcloud_dir = os.path.join(save_dir, 'point_cloud', f'iteration_{iteration}')
                os.makedirs(save_dir, exist_ok=True)
                os.makedirs(pointcloud_dir, exist_ok=True)
                shutil.copyfile(os.path.join(cfg.model_path, 'cameras.json'), os.path.join(save_dir, 'cameras.json'))
                shutil.copyfile(os.path.join(cfg.model_path, 'cfg_args'), os.path.join(save_dir, 'cfg_args'))
                shutil.copyfile(os.path.join(cfg.model_path, 'input.ply'), os.path.join(save_dir, 'input.ply'))

                ply_element = PlyElement.describe(elements, 'vertex')
                PlyData([ply_element]).write(os.path.join(pointcloud_dir, 'point_cloud.ply'))
                del elements, ply_element
                gc.collect()

            # End-of-iteration cleanup: free lazily-loaded data to prevent RAM accumulation.
            # Skip unloading when views are preloaded — otherwise we'd defeat
            # the whole point of the preload by re-reading from disk next iter.
            if not preload_vram:
                if hasattr(viewpoint_cam.guidance, 'unload'):
                    viewpoint_cam.guidance.unload()
                viewpoint_cam.unload_image()

        perf_iter_end()


def prepare_output_and_logger() -> SummaryWriter | None:
    
    # if cfg.model_path == '':
    #     if os.getenv('OAR_JOB_ID'):
    #         unique_str = os.getenv('OAR_JOB_ID')
    #     else:
    #         unique_str = str(uuid.uuid4())
    #     cfg.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(cfg.model_path))

    os.makedirs(cfg.model_path, exist_ok=True)
    os.makedirs(cfg.trained_model_dir, exist_ok=True)
    os.makedirs(cfg.record_dir, exist_ok=True)
    if not cfg.resume:
        os.system('rm -rf {}/*'.format(cfg.record_dir))
        os.system('rm -rf {}/*'.format(cfg.trained_model_dir))

    with open(os.path.join(cfg.model_path, "cfg_args"), 'w') as cfg_log_f:
        viewer_arg = dict()
        viewer_arg['sh_degree'] = cfg.model.gaussian.sh_degree
        viewer_arg['white_background'] = cfg.data.white_background
        viewer_arg['source_path'] = cfg.source_path
        viewer_arg['model_path']= cfg.model_path
        cfg_log_f.write(str(Namespace(**viewer_arg)))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(cfg.record_dir)
    else:
        print("Tensorboard not available: not logging progress")

    if _wandb_active():
        try:
            _wandb.run.summary["model_path"] = cfg.model_path
            _wandb.run.summary["record_dir"] = cfg.record_dir
            _wandb.run.summary["exp_name"] = cfg.exp_name
        except Exception as exc:
            print(f"[wandb] summary init failed: {exc}")
    return tb_writer

def training_report(tb_writer: SummaryWriter | None, iteration: int, scalar_stats: dict[str, float], tensor_stats: dict[str, torch.Tensor], testing_iterations: list[int], scene: Scene, renderer: StreetGaussianRenderer) -> None:
    if tb_writer:
        try:
            for key, value in scalar_stats.items():
                tb_writer.add_scalar('train/' + key, value, iteration)
            for key, value in tensor_stats.items():
                tb_writer.add_histogram('train/' + key, value, iteration)
        except:
            print('Failed to write to tensorboard')

    if _wandb_active() and scalar_stats:
        _wandb_log({f"train/{k}": float(v) for k, v in scalar_stats.items()}, step=iteration)
            
            
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test/test_view', 'cameras' : scene.getTestCameras()},
                              {'name': 'test/train_view', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderer.render(viewpoint, scene.gaussians)["rgb"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    
                    if hasattr(viewpoint, 'original_mask'):
                        mask = viewpoint.original_mask.cuda().bool()
                    else:
                        mask = torch.ones_like(gt_image[0]).bool()
                    l1_test += l1_loss(image, gt_image, mask).mean().double()
                    psnr_test += psnr(image, gt_image, mask).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

                if _wandb_active():
                    metric_prefix = config['name']  # e.g. "test/test_view" or "test/train_view"
                    _wandb_log({
                        f"{metric_prefix}/psnr": float(psnr_test),
                        f"{metric_prefix}/l1_loss": float(l1_test),
                    }, step=iteration)
                    if _wandb.run is not None:
                        # Track best PSNR per split as a run-level summary for sweep ranking.
                        summary_key = f"{metric_prefix}/best_psnr"
                        prev = _wandb.run.summary.get(summary_key)
                        if prev is None or float(psnr_test) > float(prev):
                            _wandb.run.summary[summary_key] = float(psnr_test)
                            _wandb.run.summary[f"{metric_prefix}/best_psnr_iter"] = iteration

        if tb_writer:
            tb_writer.add_histogram("test/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('test/points_total', scene.gaussians.get_xyz.shape[0], iteration)
        if _wandb_active():
            _wandb_log({"test/points_total": int(scene.gaussians.get_xyz.shape[0])}, step=iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # --- distributed setup (no-op when cfg.dist.enabled is False) ---------
    rank, world_size, local_rank = setup_distributed()

    if not is_main_process():
        cfg.train.quiet = True

    if is_main_process():
        print("Optimizing " + cfg.model_path)
        if is_distributed():
            print(f"Distributed training: {world_size} GPUs")

    # Initialize system state (RNG)
    safe_state(cfg.train.quiet)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(cfg.train.detect_anomaly)

    stop_event = threading.Event()
    if is_main_process():
        monitor_thread = threading.Thread(
            target=monitor_resources,
            kwargs={
                "interval": 1.0,
                "log_file": os.path.join(cfg.record_dir, "resource.log"),
                "gpu_id": local_rank,
                "stop_event": stop_event,
                "marker_queue": marker_queue,
            },
            daemon=True,
        )
        monitor_thread.start()
    time_start=time.time()

    training(rank, world_size)

    time_end=time.time()
    if is_main_process():
        stop_event.set()
        monitor_thread.join()
        print('time cost', time_end-time_start,'s')
        print('scene id:', cfg.workspace)
        print("\nTraining complete.")

    cleanup_distributed()