import os
import pickle
import argparse

import yaml
import numpy as np
import torch

from common import REPO_ROOT, ASSET_ROOT, MAX_EPISODE_LENGTH
from policy import load_policy
from grasp_env import MujocoGraspEnv


def stage_visual_input(env, style_label):
    print(f"[阶段1/3 视觉输入] 深度相机渲染深度/分割图,反投影融合物体点云并采样可供性点 (style={style_label})")
    obs = env.compute_observation(style_label)
    env.mark_visual_input()
    return obs


def stage_model_inference(policy, obs, device):
    print("[阶段2/3 模型推理] 策略网络单次前向,输出抓取动作参数")
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    return policy.act_inference(obs_t)[0].cpu().numpy().astype(np.float64)


def stage_execute(env, actions, style_qpos):
    print("[阶段3/3 执行] 生成参考轨迹并闭环跟踪")
    env.generate_plan(actions, style_qpos)
    for _ in range(MAX_EPISODE_LENGTH - 1):
        env.control_step(*env.reference_targets())
    return env.check_success()


def run_policy(env, policy, static_style, style_list, rng, device):
    env.begin_run()
    style_label = int(rng.choice(style_list))
    obs = stage_visual_input(env, style_label)
    actions = stage_model_inference(policy, obs, device)
    success, delta_z = stage_execute(env, actions, static_style[style_label][:6])
    print(f"object={env.object_name} style={style_label} success={success} delta_z={delta_z:.3f}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=os.path.join(REPO_ROOT, "checkpoint/inspire_example/model_4000.pt"))
    parser.add_argument("--object-set", default="union_object_dataset/small_debug_set.yaml")
    parser.add_argument("--object", default=None)
    parser.add_argument("--ref-file", default=os.path.join(REPO_ROOT, "tasks/grasp_ref_inspire.pkl"))
    parser.add_argument("--style-dict", default=os.path.join(REPO_ROOT, "dataset_processor/inspire_static_style_cali.npy"))
    parser.add_argument("--style", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_runtime(args):
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    policy = load_policy(args.checkpoint, args.device)
    print(f"Loaded checkpoint: {args.checkpoint}")

    with open(args.ref_file, "rb") as f:
        tracking_reference = {k: np.asarray(v, dtype=np.float64) for k, v in pickle.load(f).items()}
    static_style = np.load(args.style_dict).astype(np.float64)
    style_list = [0, 1, 2, 3] if args.style < 0 else [args.style]

    if args.object is not None:
        obj_name = args.object
    else:
        with open(os.path.join(ASSET_ROOT, args.object_set)) as f:
            obj_name = sorted(n.split(".")[0] for n in yaml.safe_load(f))[0]
    print(f"Object: {obj_name}")

    env = MujocoGraspEnv(obj_name, tracking_reference, rng)
    env.reset()
    return env, policy, static_style, style_list, rng
