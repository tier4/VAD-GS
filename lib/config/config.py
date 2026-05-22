from .yacs import CfgNode as CN
import argparse
import os
import numpy as np

from lib.utils.cfg_utils import make_cfg

cfg = CN()

cfg.workspace = os.environ['PWD']
cfg.loaded_iter = -1
cfg.ip = '127.0.0.1'
cfg.port = 6009
cfg.data_device = 'cuda'
cfg.mode = 'train' 
cfg.task = 'hello' # task folder name
cfg.exp_name = 'test' # experiment folder name
cfg.gpus = [0] # list of gpus to use 
cfg.debug = False
cfg.resume = True # If set to True, resume training from the last checkpoint.
cfg.to_cuda = False # higher GPU utilization with larger memory required

cfg.source_path = ''
cfg.model_path = ''
cfg.record_dir = None
cfg.resolution = -1
cfg.resolution_scales = [1]

cfg.eval = CN()
cfg.eval.skip_train = False 
cfg.eval.skip_test = False 
cfg.eval.eval_train = False
cfg.eval.eval_test = True
cfg.eval.quiet = False

cfg.train = CN()
cfg.train.debug_from = -1
cfg.train.detect_anomaly = False
cfg.train.test_iterations = [7000, 30000]
cfg.train.save_iterations = [7000, 30000]
cfg.train.iterations = 30000
cfg.train.quiet = False
cfg.train.checkpoint_iterations = [30000]
cfg.train.start_checkpoint = None
cfg.train.importance_sampling = False
cfg.train.preload_vram = False  # if True, preload all train views to VRAM at startup (single-GPU). Distributed always preloads.
cfg.train.log_image_interval = 100  # save a debug composite to <model_path>/log_images every N iters. Raise to reduce disk I/O (esp. for parallel sweep agents).
cfg.train.bg_init_from = ''  # Path to a checkpoint whose `background` state should overwrite the freshly-built BG model after training_setup. Used by the segmented-BG-merge pipeline (see configs/experiments/segmented/finetune_merged.yaml) to fine-tune from a stack of per-segment BG Gaussians. Leave empty for normal flow.
cfg.train.obj_init_from = ''  # Path to a `merged_obj` checkpoint produced by script/experiments/merge_obj_checkpoints.py. The loader matches actors across segments via the seq_id <-> T4 track_id maps and replaces each obj_<full_seq_id>'s freshly-built state with the best-segment's trained state. Leave empty for normal flow.
cfg.train.bg_lidar_prune_dry_run = False  # If True, after bg_init_from loads, vote every train view's lidar_depth against each merged BG Gaussian's center depth, print a conflict-vote histogram, then exit. Read-only — used to pick prune thresholds before enabling hard prune.
cfg.train.bg_lidar_prune_enable = False  # If True, hard-prune merged BG Gaussians whose center sits in front of LiDAR returns in >= bg_lidar_prune_min_conflict_views views (and conflict outnumbers consistent). Runs once after bg_init_from + obj_init_from, before the training loop.
cfg.train.bg_lidar_prune_min_conflict_views = 5  # K_lidar threshold: a Gaussian is pruned via LiDAR signal when it conflicts with LiDAR in at least this many views AND conflict_count > consistent_count. Set 0 to disable the LiDAR signal entirely.
cfg.train.bg_lidar_prune_min_sky_conflict_views = 1  # K_sky threshold: a Gaussian is pruned via sky-mask signal when its center projects onto a sky_mask pixel in at least this many views. Sky belongs to sky_cubemap, never to BG, so K_sky=1 (any one view) is the natural default. Set 0 to disable the sky signal.
cfg.train.bg_lidar_prune_tau_scale_mul = 3.0  # A Gaussian is flagged as "in front of LiDAR" only when (center_depth + tau) < lidar_depth, where tau = tau_scale_mul * max(scaling_xyz) + tau_eps. Larger -> more permissive (Gaussian radius is forgiven).
cfg.train.bg_lidar_prune_tau_eps = 0.05  # Absolute slack (scene-unit meters) added to tau on top of the scale-derived term.
cfg.train.bg_lidar_prune_scale_anomaly_thr = 5.0  # Scale-anomaly threshold: a Gaussian is pruned when its max(scale_xyz) is >= this multiple of the mean max(scale_xyz) of OTHER Gaussians in the same voxel cell. Set 0 to disable the scale signal.
cfg.train.bg_lidar_prune_scale_voxel_size = 0.5  # Voxel edge length (scene-unit meters) used to bin Gaussians for the scale-anomaly check. Larger -> coarser neighborhood definition.
cfg.train.bg_lidar_prune_scale_min_neighbors = 3  # Minimum neighbor Gaussians (excluding self) in the same voxel before the scale-anomaly score is acted on. Avoids flagging Gaussians in very sparse voxels where the mean estimate is noisy.

cfg.optim = CN()
cfg.optim.use_amp = False # If set to True, use Automatic Mixed Precision (AMP) for training.
# learning rate
cfg.optim.position_lr_init = 0.00016 # position_lr_init_{bkgd, obj ...}, similar to the following
cfg.optim.position_lr_final = 0.0000016
cfg.optim.position_lr_delay_mult = 0.01
cfg.optim.position_lr_max_steps = 30000
cfg.optim.feature_lr = 0.0025
cfg.optim.opacity_lr = 0.05
cfg.optim.scaling_lr = 0.005
cfg.optim.rotation_lr = 0.001
# densification and pruning
cfg.optim.percent_dense = 0.01 
cfg.optim.densification_interval = 100
cfg.optim.opacity_reset_interval = 3000
cfg.optim.densify_from_iter = 500
cfg.optim.densify_until_iter = 15000
cfg.optim.densify_grad_threshold = 0.0002 # densify_grad_threshold_{bkgd, obj ...}
cfg.optim.densify_grad_abs_bkgd = False # densification strategy from AbsGS
cfg.optim.densify_grad_abs_obj = False 
cfg.optim.max_screen_size = 20
cfg.optim.min_opacity = 0.005
cfg.optim.percent_big_ws = 0.1
cfg.optim.prune_small_radii = 1  # prune Gaussians with 0 < max_radii2D <= this value. Set 0 to disable.
# loss weight
cfg.optim.lambda_l1 = 1.
cfg.optim.lambda_dssim = 0.2
cfg.optim.lambda_sky = 0.
cfg.optim.lambda_sky_scale = []
cfg.optim.lambda_semantic = 0.
cfg.optim.lambda_reg = 0.
cfg.optim.lambda_depth_lidar = 0.
# Continuous "free-space" LiDAR loss: at each iter, project every BG
# Gaussian center into the current view; if the center sits in front of
# a positive LiDAR return ((d_g + tau) < lidar_depth) it lands in
# observed empty space, so we add a differentiable penalty on those
# Gaussians' opacity to push them away. Uses the same ellipsoid-aware
# tau formula as bg_lidar_prune (tau = tau_scale_mul * extent_along_ray
# + tau_eps). Geometry is detached for the mask — only opacity
# receives gradient, which composes naturally with densify/prune.
cfg.optim.lambda_lidar_freespace = 0.
cfg.optim.lidar_freespace_tau_scale_mul = 3.0
cfg.optim.lidar_freespace_tau_eps = 0.05
cfg.optim.lidar_freespace_start_iter = 0  # delay activation until BG has had time to settle; 0 means active from iter 1.
cfg.optim.lambda_depth_mono = 0.
cfg.optim.lambda_normal_mono = 0.
cfg.optim.lambda_color_correction = 0.
cfg.optim.lambda_pose_correction = 0.
cfg.optim.lambda_scale_flatten = 0.
cfg.optim.lambda_opacity_sparse = 0.
# Foreground (object) shape regularizers — penalize anisotropy
# (max/min scale ratio) and oversized object Gaussians. See
# train.py loss_obj_shape block.
cfg.optim.lambda_shape_pena = 0.
cfg.optim.lambda_scale_pena = 0.
# Same regularizers on the BACKGROUND model. Applied to bkgd Gaussians
# (roads, signs, poles, buildings — everything outside tracked actor
# boxes). Tune independently from the obj weights because BG has orders
# of magnitude more Gaussians and contains legitimately flat geometry
# (road surfaces) that we don't want to crush to spheres — start small.
cfg.optim.lambda_shape_pena_bkgd = 0.
cfg.optim.lambda_scale_pena_bkgd = 0.


cfg.model = CN()
cfg.model.gaussian = CN()
cfg.model.gaussian.sh_degree = 3
cfg.model.gaussian.fourier_dim = 1 # fourier spherical harmonics dimension
cfg.model.gaussian.fourier_scale = 1.
cfg.model.gaussian.flip_prob = 0. # symmetry prior for rigid objects, flip gaussians with this probability during training
cfg.model.gaussian.semantic_mode = 'logits'

cfg.model.nsg = CN()
cfg.model.nsg.include_bkgd = True # include background
cfg.model.nsg.include_obj = True # include object
cfg.model.nsg.include_sky = False # include sky cubemap
cfg.model.nsg.opt_track = True # tracklets optimization
cfg.model.sky = CN()
cfg.model.sky.resolution = 1024
cfg.model.sky.white_background = True


#### Note: We have not fully tested this.
cfg.model.use_color_correction = False # If set to True, learn transformation matrixs for appearance embedding
cfg.model.color_correction = CN() 
cfg.model.color_correction.mode = 'image' # If set to 'image', learn separate embedding for each image. If set to 'sensor', learn a single embedding for all images captured by one camera senosor. 
cfg.model.color_correction.use_mlp = False # If set to True, regress embedding from extrinsic by a mlp. Otherwise, define the embedding explicitly.
cfg.model.color_correction.use_sky = False # If set to True, using spparate embedding for background and sky
# Alternative choice from GOF: https://github.com/autonomousvision/gaussian-opacity-fields/blob/main/scene/appearance_network.py

cfg.model.use_pose_correction = False # If set to True, use pose correction for camera poses. 
cfg.model.pose_correction = CN()
cfg.model.pose_correction.mode = 'image' # If set to 'image', learn separate correction matrix for each image. If set to 'frame', learn a single correction matrix for all images corresponding to the same frame timestamp. 
####

cfg.data = CN()
cfg.data.white_background = False # If set to True, use white background. Should be False when using sky cubemap.
cfg.data.use_colmap_pose = False # If set to True, use colmap to recalibrate camera poses as input (rigid bundle adjustment now).
cfg.data.filter_colmap = False # If set to True, filter out SfM points by camera poses.
cfg.data.box_scale = 1.0 # Scale the bounding box by this factor.
cfg.data.split_test = -1 
cfg.data.shuffle = True
cfg.data.eval = True
cfg.data.type = 'Colmap'
cfg.data.images = 'images'
cfg.data.use_semantic = False
cfg.data.use_seg_bkgd = True
cfg.data.use_mono_depth = False
cfg.data.use_mono_normal = False
cfg.data.use_colmap = True
# data.load_pcd_from: Load the initialization point cloud from a previous experiment without generation.
# data.extent: radius of the scene, we recommend 10 - 20 meters.
# data.sphere_scale: Scale the sphere radius by this factor.
# data.regenerate_pcd: Regenerate the initialization point cloud.

cfg.dist = CN()
cfg.dist.enabled = False  # Set True + torchrun to enable multi-GPU training
cfg.dist.backend = 'nccl'

cfg.render = CN()
cfg.render.convert_SHs_python = False
cfg.render.compute_cov3D_python = False
cfg.render.debug = False
cfg.render.scaling_modifier = 1.0
cfg.render.fps = 24
cfg.render.render_normal = False
cfg.render.save_video = True
cfg.render.save_image = True
cfg.render.coord = 'world' # ['world', 'vehicle']
cfg.render.concat_cameras = []
cfg.viewer = CN()
cfg.viewer.frame_id = 0 # Select the frame_id (start from 0) to save for viewer


parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/default.yaml", type=str)
parser.add_argument("--mode", type=str, default="")
parser.add_argument('--det', type=str, default='')
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)

args = parser.parse_args()
cfg = make_cfg(cfg, args)

