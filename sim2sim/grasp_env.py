import os
import time

import numpy as np
import mujoco
from scipy.spatial.transform import Rotation

from common import (ASSET_ROOT, ARM_JOINTS, HAND_JOINTS, ACTIVE_HAND_JOINTS,
                    MIMIC_JOINTS, SIM_DT, DECIMATION, PHYSICS_SUBSTEPS, T_REF_START_LIFTING,
                    TABLE_HEIGHT, POINTS_PER_OBJECT,
                    wxyz2xyzw, quat_diff_rad, slerp, orientation_error, act_scale)
from scene import build_model, CAMERA_NAME
from perception import CameraCloudSensor, estimate_center


class MujocoGraspEnv:
    def __init__(self, object_name, tracking_reference, rng):
        self.object_name = object_name
        self.rng = rng
        object_urdf = os.path.join(ASSET_ROOT, "union_object_dataset", "urdf", object_name + ".urdf")
        self.model = build_model(object_urdf)
        self.data = mujoco.MjData(self.model)
        self.sensor = CameraCloudSensor(self.model, CAMERA_NAME, "object_geom", POINTS_PER_OBJECT, rng)

        m = self.model
        self.arm_qadr = np.array([m.joint(n).qposadr[0] for n in ARM_JOINTS])
        self.arm_dofadr = np.array([m.joint(n).dofadr[0] for n in ARM_JOINTS])
        self.all_joint_names = ARM_JOINTS + HAND_JOINTS
        self.joint_qadr = np.array([m.joint(n).qposadr[0] for n in self.all_joint_names])
        self.joint_lower = np.array([m.joint(n).range[0] for n in self.all_joint_names])
        self.joint_upper = np.array([m.joint(n).range[1] for n in self.all_joint_names])
        name2idx = {n: i for i, n in enumerate(self.all_joint_names)}
        self.arm_idx = np.array([name2idx[n] for n in ARM_JOINTS])
        self.active_hand_idx = np.array([name2idx[n] for n in ACTIVE_HAND_JOINTS])
        self.mimic_child_idx = np.array([name2idx[c] for c in MIMIC_JOINTS])
        self.mimic_parent_idx = np.array([name2idx[MIMIC_JOINTS[c][0]] for c in MIMIC_JOINTS])
        self.mimic_mult = np.array([MIMIC_JOINTS[c][1] for c in MIMIC_JOINTS])
        self.act_ids = np.array([m.actuator(n).id for n in self.all_joint_names])
        self.eef_bid = m.body("fr3_link8").id
        self.palm_bid = m.body("base_link").id
        self.ft_bids = np.array([m.body(n).id for n in
                                 ["right_thumb_4", "right_index_2", "right_middle_2", "right_ring_2", "right_little_2"]])
        self.obj_qadr = m.joint("object_free").qposadr[0]
        self.active_hand_lower = self.joint_lower[self.active_hand_idx]
        self.active_hand_upper = self.joint_upper[self.active_hand_idx]

        self.tracking_reference = tracking_reference
        self.T_ref = tracking_reference["wrist_initobj_pos"].shape[0]
        self.prev_targets = np.zeros(len(self.all_joint_names))
        self.cur_targets = np.zeros(len(self.all_joint_names))
        self.jac = np.zeros((6, m.nv))
        self.viewer = None
        self.cam_viewer = None

    def eef_pose(self):
        return np.concatenate([self.data.xpos[self.eef_bid], wxyz2xyzw(self.data.xquat[self.eef_bid])])

    def object_pose(self):
        q = self.data.qpos[self.obj_qadr: self.obj_qadr + 7]
        return np.concatenate([q[:3], wxyz2xyzw(q[3:7])])

    def reset(self):
        default_qpos = np.array([0, 0, 0, -1.6, 0, 1.6, 0] + [0.0] * 12, dtype=np.float64)
        self.data.qpos[self.joint_qadr] = default_qpos
        self.prev_targets[:] = default_qpos
        self.cur_targets[:] = default_qpos
        self.data.ctrl[self.act_ids] = default_qpos

        xy = self.rng.uniform([0.45, -0.2], [0.65, 0.2])
        yaw = self.rng.uniform(0, 2 * np.pi)
        self.data.qpos[self.obj_qadr: self.obj_qadr + 3] = [xy[0], xy[1], 0.11 + TABLE_HEIGHT]
        self.data.qpos[self.obj_qadr + 3: self.obj_qadr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        self.data.qvel[:] = 0

        mujoco.mj_step(self.model, self.data, nstep=int(2.0 / self.model.opt.timestep))
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)

    def begin_run(self):
        self.object_init_pose = self.object_pose()
        self.progress = 0

    def sample_affordance(self, pcl_w):
        obj_z = self.object_est_pose[2]
        height_score = (pcl_w[:, 2] > obj_z).astype(np.float64)
        nrm = pcl_w[:, 3:6] / (np.linalg.norm(pcl_w[:, 3:6], axis=-1, keepdims=True) + 1e-12)
        angle = np.arccos(np.clip(nrm[:, 2], -1, 1))
        angle_score = np.clip(1 - angle / (np.pi / 2), 0, 1)
        scores = height_score + angle_score
        probs = scores / scores.sum() if scores.sum() > 0 else np.full(len(scores), 1 / len(scores))
        idx = self.rng.choice(len(probs), p=probs)
        self.afford_scores = scores
        self.afford_pos = pcl_w[idx, :3].copy()
        return self.afford_pos

    def compute_observation(self, style_label):
        pcl_w = self.sensor.capture(self.data)
        self.last_pcl_w = pcl_w
        self.object_est_pose = np.concatenate([estimate_center(pcl_w), [0.0, 0.0, 0.0, 1.0]])
        afford = self.sample_affordance(pcl_w)
        obs = np.concatenate([self.eef_pose(), self.object_est_pose, afford,
                              np.eye(4)[style_label], pcl_w[:, :3].reshape(-1)])
        return np.clip(obs, -5.0, 5.0)

    def _hand_plan(self, actions, style_qpos):
        delta_param = act_scale(actions[6:12], -0.05, 0.05)
        scale_param = act_scale(actions[12], 0.1, 1.9)
        style_hand_qpos = np.clip(style_qpos * scale_param + delta_param,
                                  self.active_hand_lower, self.active_hand_upper)
        hand_seq = self.tracking_reference["hand_qpos"]
        t0 = hand_seq[0]
        fraction = (style_hand_qpos - t0) / (hand_seq[T_REF_START_LIFTING - 1] - t0 + 1e-6)
        hand_qpos = np.empty_like(hand_seq)
        hand_qpos[:T_REF_START_LIFTING - 1] = t0 + (hand_seq[:T_REF_START_LIFTING - 1] - t0) * fraction[None]
        hand_qpos[T_REF_START_LIFTING - 1:] = style_hand_qpos
        np.clip(hand_qpos, self.active_hand_lower, self.active_hand_upper, out=hand_qpos)
        return hand_qpos

    def _generate_plan(self, rot, offset, actions, style_qpos):
        ref = self.tracking_reference
        wrist_quat = (rot * Rotation.from_quat(ref["wrist_quat"])).as_quat()
        wrist_pos = rot.apply(ref["wrist_initobj_pos"]) + offset
        wrist_pos[T_REF_START_LIFTING:] = (
            ref["wrist_initobj_pos"][T_REF_START_LIFTING:]
            - ref["wrist_initobj_pos"][T_REF_START_LIFTING - 1]
            + wrist_pos[T_REF_START_LIFTING - 1]
        )
        self.current_ref = {"wrist_initobj_pos": wrist_pos, "wrist_quat": wrist_quat,
                            "hand_qpos": self._hand_plan(actions, style_qpos)}

        wrist_pose = self.eef_pose()
        target = np.concatenate([wrist_pos[0] + self.object_est_pose[:3], wrist_quat[0]])
        n_trans = int(np.ceil(np.linalg.norm(target[:3] - wrist_pose[:3]) / 0.04))
        n_rot = int(np.ceil(quat_diff_rad(wrist_pose[3:7], target[3:7]) / 0.1))
        n = max(n_trans, n_rot, 1)
        t = np.arange(n + 1)[:, None] / n
        plan_p = wrist_pose[:3] + t * (target[:3] - wrist_pose[:3])
        plan_q = slerp(wrist_pose[3:7], target[3:7], t)
        self.reaching_plan = np.concatenate([plan_p, plan_q], axis=-1)[1:]
        self.reaching_timesteps = n - 1

    def generate_plan(self, actions, style_qpos):
        self._generate_plan(Rotation.from_euler("xyz", actions[3:6] * 1.57),
                            actions[0:3] * 0.05, actions, style_qpos)

    def generate_directed_plan(self, actions, style_qpos):
        ref = self.tracking_reference
        eef = self.eef_pose()
        u = eef[:3] - self.object_est_pose[:3]
        u = u / np.linalg.norm(u)
        ref_dir = ref["wrist_initobj_pos"][T_REF_START_LIFTING - 1]
        v = ref_dir / np.linalg.norm(ref_dir)
        axis = np.cross(v, u)
        s = np.linalg.norm(axis)
        c = float(np.dot(v, u))
        if s < 1e-8:
            if c > 0:
                r_min = Rotation.identity()
            else:
                perp = np.cross(v, [1.0, 0.0, 0.0])
                if np.linalg.norm(perp) < 1e-8:
                    perp = np.cross(v, [0.0, 1.0, 0.0])
                r_min = Rotation.from_rotvec(np.pi * perp / np.linalg.norm(perp))
        else:
            r_min = Rotation.from_rotvec(axis / s * np.arctan2(s, c))
        ref_close_rot = r_min * Rotation.from_quat(ref["wrist_quat"][T_REF_START_LIFTING - 1])
        residual = (Rotation.from_quat(eef[3:7]) * ref_close_rot.inv()).as_quat()
        phi = 2.0 * np.arctan2(float(np.dot(residual[:3], u)), float(residual[3]))
        rot = Rotation.from_rotvec(phi * u) * r_min
        self._generate_plan(rot, np.zeros(3), actions, style_qpos)

    def reference_targets(self):
        t = self.progress
        track_t = int(np.clip(t - self.reaching_timesteps, 0, self.T_ref - 1))
        if t < self.reaching_timesteps:
            wrist_target = self.reaching_plan[min(t, len(self.reaching_plan) - 1)]
        else:
            wrist_target = np.concatenate([
                self.current_ref["wrist_initobj_pos"][track_t] + self.object_est_pose[:3],
                self.current_ref["wrist_quat"][track_t]])
        hand_target = self.current_ref["hand_qpos"][track_t]
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
        self.cur_targets[self.mimic_child_idx] = self.cur_targets[self.mimic_parent_idx] * self.mimic_mult
        self.cur_targets[self.arm_idx] = self.data.qpos[self.arm_qadr] + self.arm_ik(wrist_pose_target)

        for k in range(DECIMATION):
            frame_start = time.perf_counter()
            self.data.ctrl[self.act_ids] = self.prev_targets + (k + 1) / DECIMATION * (self.cur_targets - self.prev_targets)
            mujoco.mj_step(self.model, self.data, nstep=PHYSICS_SUBSTEPS)
            if self.cam_viewer is not None:
                self.cam_viewer.refresh(self.data)
            if self.viewer is not None:
                self.viewer.sync()
                remain = SIM_DT - (time.perf_counter() - frame_start)
                if remain > 0:
                    time.sleep(remain)
        self.prev_targets[:] = self.cur_targets
        self.progress += 1

    def check_success(self):
        obj_pos = self.object_pose()[:3]
        delta_z = obj_pos[2] - self.object_init_pose[2]
        ft_dist = sum(min(np.linalg.norm(self.data.xpos[b] - obj_pos), 3.0) for b in self.ft_bids)
        palm_center = self.data.xpos[self.palm_bid] + Rotation.from_quat(wxyz2xyzw(self.data.xquat[self.palm_bid])).apply(np.array([0.0, 0.0, 0.05]))
        palm_dist = min(np.linalg.norm(obj_pos - palm_center), 0.5)
        flag = (ft_dist <= 0.6) or (palm_dist <= 0.15)
        return bool(flag and delta_z > 0.1), delta_z

    def clear_markers(self):
        if self.viewer is None:
            return
        self.viewer.user_scn.ngeom = 0
        self.viewer.sync()

    def mark_visual_input(self):
        if self.viewer is None:
            return
        scn = self.viewer.user_scn
        eye = np.eye(3).flatten()
        t = self.afford_scores / (self.afford_scores.max() + 1e-12)
        colors = np.zeros((len(t), 4), dtype=np.float32)
        colors[:, 0] = t
        colors[:, 2] = 1.0 - t
        colors[:, 3] = 1.0
        for i, (p, c) in enumerate(zip(self.last_pcl_w[:, :3], colors)):
            mujoco.mjv_initGeom(scn.geoms[i], mujoco.mjtGeom.mjGEOM_SPHERE,
                                np.array([0.004, 0, 0]), p, eye, c)
        mujoco.mjv_initGeom(scn.geoms[len(t)], mujoco.mjtGeom.mjGEOM_SPHERE,
                            np.array([0.012, 0, 0]), self.afford_pos, eye,
                            np.array([0, 1, 0, 1], dtype=np.float32))
        scn.ngeom = len(t) + 1
        self.viewer.sync()
