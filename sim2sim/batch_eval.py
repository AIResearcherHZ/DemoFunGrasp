import os
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import torch

from common import REPO_ROOT, MAX_EPISODE_LENGTH
from policy import load_policy
from grasp_env import MujocoGraspEnv
import pickle, yaml

OBJECTS = ["014_lemon", "015_peach"]
N_SEED = 16

ref_file = os.path.join(REPO_ROOT, "tasks/grasp_ref_inspire.pkl")
style_file = os.path.join(REPO_ROOT, "dataset_processor/inspire_static_style_cali.npy")
ckpt = os.path.join(REPO_ROOT, "checkpoint/inspire_example/model_4000.pt")
device = "cuda:0" if torch.cuda.is_available() else "cpu"

policy = load_policy(ckpt, device)
with open(ref_file, "rb") as f:
    ref = {k: np.asarray(v, dtype=np.float64) for k, v in pickle.load(f).items()}
static_style = np.load(style_file).astype(np.float64)

for obj in OBJECTS:
    succ = 0
    dzs = []
    for seed in range(N_SEED):
        rng = np.random.default_rng(seed)
        env = MujocoGraspEnv(obj, ref, rng)
        env.reset()
        env.begin_run()
        style_label = int(rng.choice([0, 1, 2, 3]))
        try:
            obs = env.compute_observation(style_label)
        except Exception as e:
            print(f"  {obj} seed={seed} PERCEPT-SKIP {e}")
            continue
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        actions = policy.act_inference(obs_t)[0].cpu().numpy().astype(np.float64)
        env.generate_plan(actions, static_style[style_label][:6])
        for _ in range(MAX_EPISODE_LENGTH - 1):
            env.control_step(*env.reference_targets())
        ok, dz = env.check_success()
        succ += int(ok)
        dzs.append(dz)
        print(f"  {obj} seed={seed} style={style_label} succ={ok} dz={dz:+.3f}")
    print(f"== {obj}: {succ}/{N_SEED} succ, dz mean={np.mean(dzs):+.3f} min={np.min(dzs):+.3f}")
