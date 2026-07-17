import argparse
import os
import pickle
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "sim2sim"))

import mujoco
import mujoco.viewer

from common import (SIM_DT, DECIMATION, PHYSICS_SUBSTEPS, TABLE_HEIGHT,
                    wxyz2xyzw, quat_diff_rad, slerp, orientation_error)
from scene import (_prepare_robot_urdf, _add_visual_geoms, _configure_robot_bodies,
                   _add_scene, _add_camera, _add_object, _add_mocap,
                   OFFSCREEN_WIDTH, OFFSCREEN_HEIGHT, CAMERA_NAME)

HANDS = {
    "inspire": {
        "yaml": "tasks/hand/fr3_inspire_tac.yaml",
        "ref": "tasks/grasp_ref_inspire.pkl",
        "active_hand": ["right_index_1_joint", "right_little_1_joint", "right_middle_1_joint",
                        "right_ring_1_joint", "right_thumb_1_joint", "right_thumb_2_joint"],
    },
    "shadow": {
        "yaml": "tasks/hand/shadow_simple.yaml",
        "ref": "tasks/grasp_ref_shadow.pkl",
        "active_hand": ["rh_FFJ4", "rh_FFJ3", "rh_FFJ2", "rh_LFJ5", "rh_LFJ4", "rh_LFJ3", "rh_LFJ2",
                        "rh_MFJ4", "rh_MFJ3", "rh_MFJ2", "rh_RFJ4", "rh_RFJ3", "rh_RFJ2",
                        "rh_THJ5", "rh_THJ4", "rh_THJ3", "rh_THJ2", "rh_THJ1"],
    },
}


def movable_joints(urdf):
    root = ET.parse(urdf).getroot()
    return [j.get("name") for j in root.iter("joint")
            if j.get("type") in ("revolute", "continuous", "prismatic")]


def build_model(robot_urdf, object_urdf, arm_joints, hand_joints):
    tmpdir = tempfile.mkdtemp(prefix="viz_demo_")
    try:
        visuals, joint_effort, tmp_urdf = _prepare_robot_urdf(robot_urdf, tmpdir)
        tree = ET.parse(tmp_urdf)
        compiler = tree.getroot().find("mujoco/compiler")
        compiler.set("boundmass", "0.001")
        compiler.set("boundinertia", "0.0001")
        tree.write(tmp_urdf)
        spec = mujoco.MjSpec()
        spec.from_file(tmp_urdf)
        spec.strippath = False
        _add_visual_geoms(spec, visuals, robot_urdf, tmpdir)
        _configure_robot_bodies(spec)
        _add_scene(spec)
        _add_camera(spec)
        _add_object(spec, object_urdf)
        _add_mocap(spec)
        for jname in arm_joints + hand_joints:
            act = spec.add_actuator()
            act.name = jname
            act.target = jname
            act.trntype = mujoco.mjtTrn.mjTRN_JOINT
            act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
            act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
            kp, kv = (16000.0, 600.0) if jname in arm_joints else (600.0, 20.0)
            act.gainprm[0] = kp
            act.biasprm[0], act.biasprm[1], act.biasprm[2] = 0.0, -kp, -kv
            eff = joint_effort.get(jname, 50.0)
            act.forcerange = [-eff, eff]
        model = spec.compile()
    finally:
        shutil.rmtree(tmpdir)

    model.vis.global_.offwidth = OFFSCREEN_WIDTH
    model.vis.global_.offheight = OFFSCREEN_HEIGHT
    model.opt.timestep = SIM_DT / PHYSICS_SUBSTEPS
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model.opt.gravity[:] = [0, 0, -9.81]
    model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    model.opt.impratio = 100.0
    model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_MULTICCD

    dofs = np.array([model.joint(n).dofadr[0] for n in arm_joints + hand_joints])
    model.dof_armature[dofs] = 0.001
    model.dof_frictionloss[dofs] = 0.01
    hand_jids = np.array([model.joint(n).id for n in hand_joints])
    hand_dofs = np.array([model.joint(n).dofadr[0] for n in hand_joints])
    model.jnt_solref[hand_jids] = [0.002, 1.0]
    model.jnt_solimp[hand_jids, :3] = [0.99, 0.999, 0.0005]
    model.dof_damping[hand_dofs] = 1.0
    return model


class ReplayEnv:
    def __init__(self, hand_key, ref, object_name):
        cfg = yaml.safe_load(open(os.path.join(REPO_ROOT, HANDS[hand_key]["yaml"])))
        robot_urdf = os.path.join(REPO_ROOT, "assets", cfg["robotAssetFile"])
        object_urdf = os.path.join(REPO_ROOT, "assets", "union_object_dataset", "urdf", object_name + ".urdf")

        self.arm_joints = list(cfg["arm_dof_names"])
        passive = cfg.get("passive_joints") or {}
        all_hand = [j for j in movable_joints(robot_urdf) if j not in self.arm_joints]
        self.hand_joints = all_hand
        self.active_hand = HANDS[hand_key]["active_hand"]
        assert len(self.active_hand) == ref["hand_qpos"].shape[1], \
            f"active hand dof {len(self.active_hand)} != pkl hand_qpos {ref['hand_qpos'].shape[1]}"

        self.model = build_model(robot_urdf, object_urdf, self.arm_joints, self.hand_joints)
        self.data = mujoco.MjData(self.model)
        m = self.model

        self.all_joints = self.arm_joints + self.hand_joints
        self.act_ids = np.array([m.actuator(n).id for n in self.all_joints])
        self.joint_qadr = np.array([m.joint(n).qposadr[0] for n in self.all_joints])
        self.arm_qadr = np.array([m.joint(n).qposadr[0] for n in self.arm_joints])
        self.arm_dofadr = np.array([m.joint(n).dofadr[0] for n in self.arm_joints])
        name2idx = {n: i for i, n in enumerate(self.all_joints)}
        self.active_hand_idx = np.array([name2idx[n] for n in self.active_hand])
        self.mimic_child_idx = np.array([name2idx[c] for c in passive])
        self.mimic_parent_idx = np.array([name2idx[passive[c]["mimic"]] for c in passive])
        self.mimic_mult = np.array([passive[c]["multiplier"] for c in passive], dtype=np.float64)
        self.hand_lower = np.array([m.joint(n).range[0] for n in self.active_hand])
        self.hand_upper = np.array([m.joint(n).range[1] for n in self.active_hand])

        self.eef_bid = m.body(cfg["eef_link"]).id
        self.palm_bid = m.body(cfg["palm_link"]).id
        self.palm_offset = np.array(cfg["palm_offset"], dtype=np.float64)
        self.ft_bids = np.array([m.body(n).id for n in cfg["fingertips_link"]])
        self.obj_qadr = m.joint("object_free").qposadr[0]
        self.arm_home = np.array(cfg["default_dof_pos"][:len(self.arm_joints)], dtype=np.float64)

        self.ref = ref
        self.T_ref = ref["wrist_initobj_pos"].shape[0]
        self.prev_targets = np.zeros(len(self.all_joints))
        self.cur_targets = np.zeros(len(self.all_joints))
        self.jac = np.zeros((6, m.nv))
        self.viewer = None
        self.object_name = object_name

    def eef_pose(self):
        return np.concatenate([self.data.xpos[self.eef_bid], wxyz2xyzw(self.data.xquat[self.eef_bid])])

    def object_pose(self):
        q = self.data.qpos[self.obj_qadr: self.obj_qadr + 7]
        return np.concatenate([q[:3], wxyz2xyzw(q[3:7])])

    def reset(self, obj_pos, obj_yaw):
        home = np.zeros(len(self.all_joints))
        home[:len(self.arm_joints)] = self.arm_home
        self.data.qpos[self.joint_qadr] = home
        self.prev_targets[:] = home
        self.cur_targets[:] = home
        self.data.ctrl[self.act_ids] = home
        a = self.obj_qadr
        self.data.qpos[a: a + 3] = obj_pos
        self.data.qpos[a + 3: a + 7] = [np.cos(obj_yaw / 2), 0, 0, np.sin(obj_yaw / 2)]
        self.data.qvel[:] = 0
        mujoco.mj_step(self.model, self.data, nstep=int(2.0 / self.model.opt.timestep))
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)

    def build_plan(self):
        ref = self.ref
        self.obj_pos = self.object_pose()[:3]
        wrist_pose = self.eef_pose()
        target = np.concatenate([ref["wrist_initobj_pos"][0] + self.obj_pos, ref["wrist_quat"][0]])
        n_trans = int(np.ceil(np.linalg.norm(target[:3] - wrist_pose[:3]) / 0.04))
        n_rot = int(np.ceil(quat_diff_rad(wrist_pose[3:7], target[3:7]) / 0.1))
        n = max(n_trans, n_rot, 1)
        t = np.arange(n + 1)[:, None] / n
        plan_p = wrist_pose[:3] + t * (target[:3] - wrist_pose[:3])
        plan_q = slerp(wrist_pose[3:7], target[3:7], t)
        self.reaching_plan = np.concatenate([plan_p, plan_q], axis=-1)[1:]
        self.reaching_timesteps = n - 1
        self.progress = 0

    def reference_targets(self):
        t = self.progress
        track_t = int(np.clip(t - self.reaching_timesteps, 0, self.T_ref - 1))
        if t < self.reaching_timesteps:
            wrist_target = self.reaching_plan[min(t, len(self.reaching_plan) - 1)]
        else:
            wrist_target = np.concatenate([self.ref["wrist_initobj_pos"][track_t] + self.obj_pos,
                                           self.ref["wrist_quat"][track_t]])
        hand_target = np.clip(self.ref["hand_qpos"][track_t], self.hand_lower, self.hand_upper)
        return wrist_target, hand_target

    def arm_ik(self, wrist_pose_target):
        eef = self.eef_pose()
        pos_err = wrist_pose_target[:3] - eef[:3]
        orn_err = orientation_error(wrist_pose_target[3:7], eef[3:7])
        mujoco.mj_jacBody(self.model, self.data, self.jac[:3], self.jac[3:], self.eef_bid)
        J = self.jac[:, self.arm_dofadr]
        dpose = np.concatenate([pos_err, orn_err])
        lmbda = np.eye(6) * (0.1 ** 2)
        return J.T @ np.linalg.solve(J @ J.T + lmbda, dpose)

    def control_step(self, wrist_pose_target, hand_qpos_target):
        self.cur_targets[self.active_hand_idx] = hand_qpos_target
        if len(self.mimic_child_idx) > 0:
            self.cur_targets[self.mimic_child_idx] = self.cur_targets[self.mimic_parent_idx] * self.mimic_mult
        self.cur_targets[:len(self.arm_joints)] = self.data.qpos[self.arm_qadr] + self.arm_ik(wrist_pose_target)
        for k in range(DECIMATION):
            self.data.ctrl[self.act_ids] = self.prev_targets + (k + 1) / DECIMATION * (self.cur_targets - self.prev_targets)
            mujoco.mj_step(self.model, self.data, nstep=PHYSICS_SUBSTEPS)
            if self.viewer is not None:
                self.viewer.sync()
        self.prev_targets[:] = self.cur_targets
        self.progress += 1

    def check_success(self):
        obj = self.object_pose()[:3]
        delta_z = obj[2] - self.obj_pos[2]
        ft = sum(min(np.linalg.norm(self.data.xpos[b] - obj), 3.0) for b in self.ft_bids)
        palm = self.data.xpos[self.palm_bid] + Rotation.from_quat(
            wxyz2xyzw(self.data.xquat[self.palm_bid])).apply(self.palm_offset)
        palm_d = min(np.linalg.norm(obj - palm), 0.5)
        flag = (ft <= 0.12 * len(self.ft_bids)) or (palm_d <= 0.15)
        return bool(flag and delta_z > 0.1), delta_z


def replay_once(env):
    env.build_plan()
    for _ in range(env.reaching_timesteps + env.T_ref):
        if env.viewer is not None and not env.viewer.is_running():
            return
        env.control_step(*env.reference_targets())
    ok, dz = env.check_success()
    print(f"object={env.object_name}  T_ref={env.T_ref}  reach={env.reaching_timesteps}  success={ok}  delta_z={dz:.3f}")


def load_ref(path):
    with open(path, "rb") as f:
        return {k: np.asarray(v, dtype=np.float64) for k, v in pickle.load(f).items()}


def resolve_hand(ref, hand):
    if hand is not None:
        return hand
    ndof = ref["hand_qpos"].shape[1]
    matches = [k for k, v in HANDS.items() if len(v["active_hand"]) == ndof]
    if len(matches) != 1:
        raise ValueError(f"cannot infer hand from hand_qpos dim {ndof}; pass --hand explicitly (choices: {list(HANDS)})")
    return matches[0]


def parse_args():
    p = argparse.ArgumentParser(description="Replay a reference demonstration pkl in the MuJoCo viewer")
    p.add_argument("pkl", nargs="?", default=None, help="path to grasp_ref_*.pkl (default: inspire ref)")
    p.add_argument("--hand", choices=list(HANDS), default=None, help="override; inferred from pkl hand_qpos dim if omitted")
    p.add_argument("--object", default="011_banana")
    p.add_argument("--obj-pos", type=float, nargs=3, default=[0.55, 0.0, 0.13])
    p.add_argument("--obj-yaw", type=float, default=0.0)
    p.add_argument("--auto", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    ref_path = args.pkl or os.path.join(REPO_ROOT, HANDS["inspire"]["ref"])
    ref = load_ref(ref_path)
    hand = resolve_hand(ref, args.hand)
    print(f"[pkl={os.path.basename(ref_path)}  hand={hand}]")
    env = ReplayEnv(hand, ref, args.object)
    env.reset(args.obj_pos, args.obj_yaw)

    trigger = {"run": args.auto, "reset": False}

    def key_callback(keycode):
        if keycode == 32:
            trigger["run"] = True
        elif keycode == 259:
            trigger["reset"] = True

    env.viewer = mujoco.viewer.launch_passive(env.model, env.data, key_callback=key_callback)
    print(f"[hand={hand}] 空格: 回放示教   退格: 复位   (纯回放, 零编辑)")

    while env.viewer.is_running():
        if trigger["reset"]:
            trigger["reset"] = False
            env.reset(args.obj_pos, args.obj_yaw)
            env.viewer.sync()
        elif trigger["run"]:
            trigger["run"] = False
            replay_once(env)
        else:
            mujoco.mj_forward(env.model, env.data)
            env.viewer.sync()


if __name__ == "__main__":
    main()
