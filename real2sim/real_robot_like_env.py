import os
from datetime import datetime
from omegaconf import DictConfig

import gym
import isaacgymenvs
from isaacgymenvs.utils.utils import set_seed
from isaacgymenvs.utils.torch_jit_utils import *
import tasks

import torch
import numpy as np
import cv2
import open3d as o3d
import sys
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur
import h5py

class RealRobotLikeEnv:
    def __init__(self, cfg: DictConfig):
        # global rank of the GPU
        global_rank = int(os.getenv("RANK", "0"))
        # sets seed. if seed is -1 will pick a random one
        cfg.seed = set_seed(
            cfg.seed, torch_deterministic=cfg.torch_deterministic, rank=global_rank
        )
        def create_isaacgym_env(**kwargs):
            envs = isaacgymenvs.make(
                cfg.seed,
                cfg.task_name,
                cfg.task.env.numEnvs,
                cfg.sim_device,
                cfg.rl_device,
                cfg.graphics_device_id,
                cfg.headless,
                cfg.multi_gpu,
                cfg.capture_video,
                cfg.force_render,
                cfg,
                **kwargs,
            )
            if cfg.capture_video:
                envs.is_vector_env = True
                envs = gym.wrappers.RecordVideo(
                    envs,
                    f"videos/{cfg.task_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
                    step_trigger=lambda step: step % cfg.capture_video_freq == 0,
                    video_length=cfg.capture_video_len,
                )
            return envs

        self.env = create_isaacgym_env()
        self.num_envs = self.env.num_envs
        self.env.reset_idx(torch.arange(self.num_envs))

        #assert self.num_envs == 1
        assert self.env.use_camera
        self.pinhole_camera_intrinsics = [
            o3d.camera.PinholeCameraIntrinsic(
                self.env.camera_cfg[f'camera_{i}']['width'], self.env.camera_cfg[f'camera_{i}']['height'], 
                self.env.camera_cfg[f'camera_{i}']['intrinsics'][0][0], self.env.camera_cfg[f'camera_{i}']['intrinsics'][1][1], 
                self.env.camera_cfg[f'camera_{i}']['intrinsics'][0][2], self.env.camera_cfg[f'camera_{i}']['intrinsics'][1][2]
            )
            for i in self.env.camera_ids
        ]
        self.camera_intrinsic_matrices = [np.array(self.env.camera_cfg[f'camera_{i}']['intrinsics'], dtype=np.float32)
                                          for i in self.env.camera_ids]
        self.last_pcls = [np.zeros((self.num_envs, self.env.render_cfg['n_pcl_downsample'], 6), dtype=np.float32) for i in self.env.camera_ids]
    
        # load  depth maps
        if "depth" in self.env.render_data_type:
            with h5py.File('assets/nyu_depth_v2_labeled.mat', 'r') as f:
                self.depth_maps = f['rawDepths']
                # convert to torch tensor
                self.depth_maps = torch.from_numpy(np.array(self.depth_maps)).float().to("cuda:0")
                self.depth_maps = torch.transpose(self.depth_maps, 1, 2)
                print("Loaded nyu depth maps:", self.depth_maps.shape)

    def _process_depth(self, depth_image):
        ''' Process the depth image to align with real-world sensors'''
        # Ensure input is at least 3D (batch, height, width)
        if depth_image.ndim == 2:
            depth_image = depth_image.unsqueeze(0)
        # print the largest value and smallest value in the depth image
        device = depth_image.device
        batch_size, height, width = depth_image.shape
        depth = depth_image.clone()
        depth_shape = depth.shape

        # # depth map is about 0 to 10
        # # randomly select batch_size depth maps from the dataset
        alpha = 0.005
        rand_indices = torch.randint(0, self.depth_maps.shape[0], (batch_size,), device=device)
        depth = (1 - alpha) * depth + alpha * self.depth_maps[rand_indices, :depth_shape[1], :depth_shape[2]].to(device)
        # # clip to max depth 1m
        depth = torch.clamp(depth, min=0, max=1.0)
        # # add gaussian blur
        depth = gaussian_blur(depth.unsqueeze(1), kernel_size=5, sigma=5).squeeze(1) #(B, 1, H, W) -> (B, H, W)
        # # add Gaussian noise
        depth = depth + torch.randn_like(depth) * 0.01 #np.random.normal(0, 0.01, depth_image.shape)  
        # # add noise proportional to depth
        depth = depth + torch.randn_like(depth) * (depth * 0.02) #np.random.normal(0, depth_image * 0.02)
        # # simulated 1% invalid pixels
        invalid_mask = torch.rand_like(depth) < 0.01 #np.random.rand(*depth_image.shape) < 0.01
        depth[invalid_mask] = 0
        # # round to mm
        depth = torch.round(depth * 1000) / 1000.0  
        return depth.contiguous() #np.ascontiguousarray(depth, dtype=np.float32)


    def get_real_observations(self, noisy_depth=True):
        obs_dict = {
            "instruction": self.env.instructions, # [num_envs] str 
            "right_arm_qpos": self.env.robot_dof_pos[:, self.env.arm_dof_indices].cpu().numpy(), # [num_envs, 7]
            "right_arm_eef_pose": self.env.rigid_body_states.view(-1, 13)[self.env.eef_idx, 0:7].cpu().numpy(), # [num_envs, 7]
            "right_hand_qpos": self.env.robot_dof_pos[:, self.env.active_hand_dof_indices].cpu().numpy(), # [num_envs, 6]
        }
        for i in range(len(self.env.camera_ids)):
            cam_id = self.env.camera_ids[i]
            rgb_image = self.env.rgb_tensors[i][:, :, :, :3] #.cpu().numpy().astype(np.uint8) # [num_envs, H, W, 3]
            if "depth" in self.env.render_data_type or "pcl" in self.env.render_data_type:
                depth_image = self.env.depth_tensors[i][:, :, :] #.cpu().numpy().astype(np.float32) # [num_envs, H, W]
                if noisy_depth:
                    depth_image = self._process_depth(depth_image)
                depth_image = torch.clamp(depth_image, self.env.save_depth_range[0], self.env.save_depth_range[1])
            
            if "rgb" in self.env.render_data_type:
                #obs_dict[f"camera_{cam_id}.rgb"] = cv2.resize(rgb_image, self.env.render_cfg['resize'], interpolation=cv2.INTER_AREA)
                obs_dict[f"camera_{cam_id}.rgb"] = F.interpolate(
                    rgb_image.permute(0,3,1,2).to(torch.float32), 
                    size=tuple(self.env.render_cfg['resize']), 
                    mode='area'
                ).permute(0,2,3,1).cpu().numpy().astype(np.uint8)
            
            if "depth" in self.env.render_data_type:
                #obs_dict[f"camera_{cam_id}.depth"] = cv2.resize(depth_image, self.env.render_cfg['resize'], interpolation=cv2.INTER_AREA)
                obs_dict[f"camera_{cam_id}.depth"] = F.interpolate(
                    depth_image.unsqueeze(1), 
                    size=tuple(self.env.render_cfg['resize']), 
                    mode='area'
                ).squeeze(1).cpu().numpy().astype(np.float32)

        return obs_dict
    

