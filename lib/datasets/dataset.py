import os
import random
import json
from lib.utils.camera_utils import camera_to_JSON, cameraList_from_camInfos
from lib.config import cfg
from lib.datasets.base_readers import storePly, SceneInfo
from lib.datasets.colmap_readers import readColmapSceneInfo
from lib.datasets.blender_readers import readNerfSyntheticInfo
from lib.datasets.waymo_full_readers import readWaymoFullInfo
from lib.datasets.my_drivestudio_readers import readDriveStudioInfo
from lib.datasets.t4_readers import readT4Info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "Waymo": readWaymoFullInfo,
    "DriveStudio": readDriveStudioInfo,
    "T4": readT4Info,
}

class Dataset():
    def __init__(self):
        self.cfg = cfg.data
        self.model_path = cfg.model_path
        self.source_path = cfg.source_path
        self.images = self.cfg.images

        self.train_cameras = {}
        self.test_cameras = {}

        dataset_type = cfg.data.get('type', "Colmap")
        assert dataset_type in sceneLoadTypeCallbacks.keys(), 'Could not recognize scene type!'
        
        scene_info: SceneInfo = sceneLoadTypeCallbacks[dataset_type](self.source_path, **cfg.data)

        if cfg.mode == 'train':
            ply_path = os.path.join(self.model_path, "input.ply")
            # Skip rewriting when sweep_run.py has symlinked a shared cache copy
            # (deterministic from the dataset, so the rewrite would just duplicate
            # bytes back into the cache and risk parallel-agent write races).
            if os.path.lexists(ply_path):
                print(f'input.ply already present at {ply_path} — skipping write')
            else:
                print(f'Saving input pointcloud to {ply_path}')
                pcd = scene_info.point_cloud
                storePly(ply_path, pcd.points, pcd.colors)

            cams_path = os.path.join(self.model_path, "cameras.json")
            if os.path.lexists(cams_path):
                print(f'cameras.json already present at {cams_path} — skipping write')
            else:
                json_cams = []
                camlist = []
                if scene_info.test_cameras:
                    camlist.extend(scene_info.test_cameras)
                if scene_info.train_cameras:
                    camlist.extend(scene_info.train_cameras)
                for id, cam in enumerate(camlist):
                    json_cams.append(camera_to_JSON(id, cam))

                print(f'Saving input camera to {cams_path}')
                with open(cams_path, 'w') as file:
                    json.dump(json_cams, file)
       
        self.scene_info = scene_info
        
        # if self.cfg.shuffle and cfg.mode == 'train':
        #     random.shuffle(self.scene_info.train_cameras)  # Multi-res consistent random shuffling
        #     random.shuffle(self.scene_info.test_cameras)  # Multi-res consistent random shuffling
        
        for resolution_scale in cfg.resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(self.scene_info.train_cameras, resolution_scale)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(self.scene_info.test_cameras, resolution_scale)
            
        self.scene_info.train_cameras.clear() #################### danger
        self.scene_info.test_cameras.clear()