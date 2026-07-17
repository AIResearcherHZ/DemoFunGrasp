import os
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import mujoco

from common import MAX_EPISODE_LENGTH, T_REF_START_LIFTING
from runtime import parse_args, load_runtime


def object_contacts(env):
    m, d = env.model, env.data
    obj_gid = m.geom("object_geom").id
    total_fn = 0.0
    n = 0
    bodies = {}
    f6 = np.zeros(6)
    for i in range(d.ncon):
        c = d.contact[i]
        if obj_gid in (c.geom1, c.geom2):
            mujoco.mj_contactForce(m, d, i, f6)
            fn = abs(f6[0])
            total_fn += fn
            n += 1
            other = c.geom2 if c.geom1 == obj_gid else c.geom1
            bn = m.body(m.geom_bodyid[other]).name
            bodies[bn] = bodies.get(bn, 0.0) + fn
    return n, total_fn, bodies


def main():
    args = parse_args()
    args.style = 0
    env, policy, static_style, style_list, rng = load_runtime(args)
    import torch
    env.begin_run()
    obs = env.compute_observation(0)
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=args.device).unsqueeze(0)
    actions = policy.act_inference(obs_t)[0].cpu().numpy().astype(np.float64)
    env.generate_plan(actions, static_style[0][:6])
    print(f"reaching_timesteps={env.reaching_timesteps} lift_start_progress={env.reaching_timesteps + T_REF_START_LIFTING}")
    z0 = env.object_pose()[2]
    for step in range(MAX_EPISODE_LENGTH - 1):
        env.control_step(*env.reference_targets())
        obj = env.object_pose()
        vz = env.data.qvel[env.model.joint("object_free").dofadr[0] + 2]
        nc, fn, bodies = object_contacts(env)
        act = env.cur_targets[env.active_hand_idx]
        qpos = env.data.qpos[env.joint_qadr[env.active_hand_idx]]
        ferr = np.abs(act - qpos).max()
        tag = "LIFT" if env.progress >= env.reaching_timesteps + T_REF_START_LIFTING else "    "
        print(f"[{step:2d}]{tag} z={obj[2]:.3f} dz={obj[2]-z0:+.3f} vz={vz:+.2f} "
              f"ncon={nc} Fn={fn:5.1f} fingErr={ferr:.3f} contacts={ {k: round(v,1) for k,v in bodies.items()} }")
    print("success:", env.check_success())


if __name__ == "__main__":
    main()
