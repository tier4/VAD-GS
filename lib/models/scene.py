from __future__ import annotations

import os
import torch
from lib.datasets.dataset import Dataset
from lib.models.gaussian_model import GaussianModel
from lib.models.street_gaussian_model import StreetGaussianModel
from lib.utils.camera_utils import Camera
from lib.config import cfg
from lib.utils.system_utils import searchForMaxIteration
import numpy as np

class Scene:

    gaussians : GaussianModel | StreetGaussianModel
    dataset: Dataset

    def __init__(self, gaussians: GaussianModel | StreetGaussianModel, dataset: Dataset) -> None:
        self.dataset = dataset
        self.gaussians = gaussians
        
        if cfg.mode == 'train':
            point_cloud = self.dataset.scene_info.point_cloud
            assert point_cloud is not None, "point_cloud is required in train mode"
            scene_raidus = self.dataset.scene_info.metadata['scene_radius']
            print("Creating gaussian model from point cloud")
            # self.gaussians.create_from_pcd(point_cloud, scene_raidus, len(self.dataset.train_cameras[1]) + len(self.dataset.test_cameras[1]))
            self.gaussians.create_from_pcd(point_cloud, scene_raidus, np.array([c.id for c in self.dataset.train_cameras[1]]))
            del point_cloud

            if cfg.get('to_cuda', False):
                print('Moving training cameras to GPU')
                for camera in self.getTrainCameras():
                    camera.set_device('cuda')
                    
        else:
            # First check if there is a point cloud saved and get the iteration to load from
            if not os.path.exists(cfg.point_cloud_dir):
                raise FileNotFoundError(
                    f"Evaluation requires a trained model, but point cloud directory was not found: {cfg.point_cloud_dir}"
                )
            if cfg.loaded_iter == -1:
                self.loaded_iter = searchForMaxIteration(cfg.point_cloud_dir)
            else:
                self.loaded_iter = cfg.loaded_iter
            
            # Load checkpoint if it exists (this loads other parameters like the optimized tracking poses)
            print("Loading checkpoint at iteration {}".format(self.loaded_iter))
            checkpoint_path = os.path.join(cfg.trained_model_dir, f"iteration_{str(self.loaded_iter)}.pth")
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(
                    f"Evaluation requires a trained checkpoint, but it was not found: {checkpoint_path}"
                )
            state_dict = torch.load(checkpoint_path)
            self.gaussians.load_state_dict(state_dict=state_dict)
            
    def save(self, iteration: int) -> None:
        point_cloud_path = os.path.join(cfg.point_cloud_dir, f"iteration_{iteration}", "point_cloud.ply")
        self.gaussians.save_ply(point_cloud_path)

    def getTrainCameras(self, scale: int = 1) -> list[Camera]:
        return self.dataset.train_cameras[scale]

    def getTestCameras(self, scale: int = 1) -> list[Camera]:
        return self.dataset.test_cameras[scale]

    def getNovelViewCameras(self, scale: int = 1) -> list[Camera]:
        return self.dataset.novel_view_cameras[scale]
