import torch
import torch.nn as nn

from common import POINTS_PER_OBJECT, STATE_DIM


class SimplePointNet(nn.Module):
    def __init__(self, feat_dim=128):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(6, 128, 1), nn.ReLU(), nn.Conv1d(128, 256, 1), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, feat_dim), nn.ReLU())

    def forward(self, pcd):
        mean = pcd.mean(dim=1, keepdim=True)
        x = torch.cat([mean.expand_as(pcd), pcd - mean], dim=2)
        feat = self.conv(x.transpose(2, 1)).transpose(2, 1)
        half = feat.shape[-1] // 2
        g = torch.cat([feat[..., :half].amax(dim=1), feat[..., half:].mean(dim=1)], dim=-1)
        return self.head(g)


class GraspActor(nn.Module):
    def __init__(self, act_dim=13, pc_emb_dim=128, hidden=(1024, 1024, 512, 512)):
        super().__init__()
        self.pointnet = SimplePointNet(pc_emb_dim)
        dims = (STATE_DIM + pc_emb_dim,) + hidden
        layers = []
        for i in range(len(hidden)):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.ELU()]
        self.actor_mean = nn.Sequential(*layers, nn.Linear(hidden[-1], act_dim))

    @torch.no_grad()
    def act_inference(self, obs):
        state = obs[:, :STATE_DIM]
        pc = obs[:, STATE_DIM:].reshape(-1, POINTS_PER_OBJECT, 3)
        return torch.tanh(self.actor_mean(torch.cat([state, self.pointnet(pc)], dim=1)))


CKPT_RENAME = {
    "backbone.backbone.conv_mlp.mlp.layer0.conv.": "pointnet.conv.0.",
    "backbone.backbone.conv_mlp.mlp.layer1.conv.": "pointnet.conv.2.",
    "backbone.backbone.global_mlp.mlp.linear0.": "pointnet.head.0.",
    "backbone.backbone.global_mlp.mlp.linear1.": "pointnet.head.2.",
    "actor_mean.": "actor_mean.",
}


def load_policy(checkpoint, device):
    state = {}
    for k, v in torch.load(checkpoint, map_location="cpu").items():
        for old, new in CKPT_RENAME.items():
            if k.startswith(old):
                state[new + k[len(old):]] = v
                break
    policy = GraspActor()
    policy.load_state_dict(state)
    return policy.to(device).eval()
