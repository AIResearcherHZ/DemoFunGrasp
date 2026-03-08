'''
Used to generate affordance points for functional tasks

Some ways to generate affordance points include:
1. randomly sample points on the object surface
2. Use a gripper model to sample points, use point as affordance point.

'''
import numpy as np
import torch

import torch.nn.functional as F

class functional_generator:
    '''
    Generate functional point and calculate the distance of the affordance
    '''
    def __init__(self, afford_type:str):
        self.afford_type = afford_type

        self.affordance_point = None
        self.style_label = None
        self.pc_voxel = None


    def generate_affordance_points(self, point_cloud:torch,object_pose:torch,if_use_data_afford:bool,afford=None):
        assert len(point_cloud.shape) == 3, "point cloud should be (B,N,6)"
        if if_use_data_afford:
            assert self.afford_type == "score_based", "only score_based affordance can use dataset affordance"
            assert afford is not None, "affordance scores from dataset is required."
        self.num_envs = point_cloud.shape[0]
        self.num_points = point_cloud.shape[1]

        if self.afford_type == 'random':
            # choose one random index per environment
            indices = torch.randint(
                low=0, high=self.num_points, size=(self.num_envs,), device=point_cloud.device
            )  # (num_envs,)
            affordance_points = point_cloud[torch.arange(self.num_envs), indices]  # (num_envs, D)

            self.affordance_point = affordance_points[...,:3] # only xyz

        elif self.afford_type == 'score_based':
            assert point_cloud.shape[-1] == 6, "point cloud should have xyz + norm (6D)"

            # (B,N,3)
            pos = point_cloud[:, :, :3]
            norm = point_cloud[:, :, 3:]

            # 1. height score (0 or 1)
            # obj_pose: (B,3) or (B,) for z
            obj_z = object_pose[:, 2].unsqueeze(1)  # (B,1)
            height_score = (pos[:, :, 2] > obj_z).float()  # (B,N)

            # 2. angle score (normal vs z-axis) (0~1)
            z_axis = torch.tensor([0, 0, 1], dtype=torch.float32, device=point_cloud.device)
            cosine_sim = F.cosine_similarity(norm, z_axis.expand_as(norm), dim=-1)  # (B,N)
            cosine_sim = torch.clamp(cosine_sim, -1.0, 1.0)
            angle = torch.acos(cosine_sim)  # (B,N), in radians

            # interpolate: 0 rad → 1,  pi/3 → 0
            angle_score = torch.clamp(1 - angle / (torch.pi / 2), 0.0, 1.0)  # (B,N)

            # 3. final score
            scores = height_score + angle_score  # (B,N)



            if if_use_data_afford: # filter the points without affordance
                assert afford is not None, "affordance scores from dataset is required."
                assert afford.shape[0] == point_cloud.shape[0], "affordance shape should be (B,N)"
                
                no_afford_mask = (afford.sum(dim=-1)==0)  # (B,)
                afford = afford.squeeze(-1)  # (B,N)
                afford[no_afford_mask,:] = 1.0 # if no affordance score, set all to 1.0

                scores [~afford.bool()] = 0  # (B,N), make points without affordance score to 0

            # handle all-zero scores: fallback to uniform
            probs = torch.where(
                scores.sum(dim=1, keepdim=True) > 0,
                scores / scores.sum(dim=1, keepdim=True),
                torch.ones_like(scores) / scores.shape[1]
            )  # (B,N)

            # import matplotlib
            # matplotlib.use('Agg')  # headless 模式
            # import matplotlib.pyplot as plt

            # # 只可视化第一个 batch
            # pts = point_cloud[0, :, :3].detach().cpu().numpy()
            # p = probs[0].detach().cpu().numpy()

            # # 归一化概率
            # p = (p - p.min()) / (p.max() - p.min() + 1e-8)

            # plt.figure(figsize=(6,6))
            # plt.scatter(pts[:,0], pts[:,1], c=p, s=4)
            # plt.colorbar(label='affordance prob')
            # plt.xlabel('X')
            # plt.ylabel('Y')
            # plt.title('Affordance heatmap (batch 0, XY projection)')
            # plt.tight_layout()
            # plt.savefig('./afford_heatmap.png', dpi=150)
            # plt.close()

            # exit(0)
            # sample one index per environment
            indices = torch.multinomial(probs, num_samples=1).squeeze(1)  # (B,)
            affordance_points = point_cloud[torch.arange(point_cloud.shape[0]), indices]  # (B,6)
            self.affordance_point = affordance_points.clone()
        elif self.afford_type == 'inner_points':
            raise NotImplemented
        else:
            raise NotImplementedError
        
        return self.affordance_point,indices
    
    def get_static_style(self,dict_path):
        # load style dict: npz file
        style_data = np.load(dict_path, allow_pickle=True)
        static_style = torch.tensor(style_data, dtype=torch.float32) # (num_styles, style_dim)
        return static_style
    
    def get_style_labels(self,num_envs,style_list,device):

        # randomly choose a style label from style_list for each env
        indices = torch.randint(
            low=0, high=len(style_list), size=(num_envs,), device=device
        )  # (num_envs,)
        self.style_labels = torch.tensor(style_list, dtype=torch.long, device=device)[indices]  # (num_envs,)
        return self.style_labels
    
