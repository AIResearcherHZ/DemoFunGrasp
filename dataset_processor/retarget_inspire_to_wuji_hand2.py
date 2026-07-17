import os
import pickle

import numpy as np
import mujoco
from scipy.optimize import least_squares

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INSPIRE_XML = os.path.join(REPO_ROOT, "assets/inspire_tac/mjcf/scene.xml")
WUJI_XML = os.path.join(REPO_ROOT, "assets/wuji_hand2/mjcf/scene.xml")
REF_IN = os.path.join(REPO_ROOT, "tasks/grasp_ref_inspire.pkl")
STYLE_IN = os.path.join(REPO_ROOT, "dataset_processor/inspire_static_style_cali.npy")
REF_OUT = os.path.join(REPO_ROOT, "tasks/grasp_ref_wuji_hand2.pkl")
STYLE_OUT = os.path.join(REPO_ROOT, "dataset_processor/wuji_hand2_style.npy")

INSPIRE_ACTIVE = ["right_index_1_joint", "right_little_1_joint", "right_middle_1_joint",
                  "right_ring_1_joint", "right_thumb_1_joint", "right_thumb_2_joint"]
INSPIRE_MIMIC = {
    "right_thumb_3_joint": ("right_thumb_2_joint", 0.6),
    "right_thumb_4_joint": ("right_thumb_2_joint", 0.8),
    "right_index_2_joint": ("right_index_1_joint", 1.05),
    "right_middle_2_joint": ("right_middle_1_joint", 1.05),
    "right_ring_2_joint": ("right_ring_1_joint", 1.05),
    "right_little_2_joint": ("right_little_1_joint", 1.18),
}
INSPIRE_TIPS = ["right_thumb_tip", "right_index_tip", "right_middle_tip",
                "right_ring_tip", "right_little_tip"]
INSPIRE_MIDS = ["right_thumb_4", "right_index_2", "right_middle_2",
                "right_ring_2", "right_little_2"]

WUJI_ACTIVE = [
    "r_index_finger_mcp_flex", "r_index_finger_mcp_abd", "r_index_finger_pip", "r_index_finger_dip",
    "r_middle_finger_mcp_flex", "r_middle_finger_mcp_abd", "r_middle_finger_pip", "r_middle_finger_dip",
    "r_pinky_mcp_flex", "r_pinky_mcp_abd", "r_pinky_pip", "r_pinky_dip",
    "r_ring_finger_mcp_flex", "r_ring_finger_mcp_abd", "r_ring_finger_pip", "r_ring_finger_dip",
    "r_thumb_cmc_flex", "r_thumb_cmc_abd", "r_thumb_mcp", "r_thumb_ip",
]
WUJI_TIP_SITES = ["r_thumb_tip", "r_index_finger_tip", "r_middle_finger_tip",
                  "r_ring_finger_tip", "r_pinky_tip"]
WUJI_MIDS = ["r_thumb_distal", "r_index_finger_distal", "r_middle_finger_distal",
             "r_ring_finger_distal", "r_pinky_distal"]

W_TIP = 1.0
W_MID = 0.3
W_REG = 0.01
W_SMOOTH = 0.02
W_DREG = 0.02
D_MAX = 0.08
T_GRASP = 12
DEEPEN = 1.5


class HandModel:
    def __init__(self, xml, joint_names):
        self.model = mujoco.MjModel.from_xml_path(xml)
        self.data = mujoco.MjData(self.model)
        self.qadr = np.array([self.model.joint(n).qposadr[0] for n in joint_names])
        self.lower = np.array([self.model.joint(n).range[0] for n in joint_names])
        self.upper = np.array([self.model.joint(n).range[1] for n in joint_names])
        self.wrist_bid = self.model.body("wrist").id

    def to_wrist_frame(self, pts_world):
        p0 = self.data.xpos[self.wrist_bid]
        R = self.data.xmat[self.wrist_bid].reshape(3, 3)
        return (pts_world - p0) @ R


class InspireFK(HandModel):
    def __init__(self):
        super().__init__(INSPIRE_XML, INSPIRE_ACTIVE)
        m = self.model
        self.mimic_qadr = np.array([m.joint(c).qposadr[0] for c in INSPIRE_MIMIC])
        self.mimic_parent = np.array([INSPIRE_ACTIVE.index(INSPIRE_MIMIC[c][0]) for c in INSPIRE_MIMIC])
        self.mimic_mult = np.array([INSPIRE_MIMIC[c][1] for c in INSPIRE_MIMIC])
        self.tip_bids = np.array([m.body(n).id for n in INSPIRE_TIPS])
        self.mid_bids = np.array([m.body(n).id for n in INSPIRE_MIDS])

    def keypoints(self, q6):
        self.data.qpos[self.qadr] = q6
        self.data.qpos[self.mimic_qadr] = q6[self.mimic_parent] * self.mimic_mult
        mujoco.mj_kinematics(self.model, self.data)
        tips = self.to_wrist_frame(self.data.xpos[self.tip_bids])
        mids = self.to_wrist_frame(self.data.xpos[self.mid_bids])
        return tips, mids


class WujiIK(HandModel):
    def __init__(self):
        super().__init__(WUJI_XML, WUJI_ACTIVE)
        m = self.model
        self.tip_sids = np.array([m.site(n).id for n in WUJI_TIP_SITES])
        self.mid_bids = np.array([m.body(n).id for n in WUJI_MIDS])

    def keypoints(self, q20):
        self.data.qpos[self.qadr] = q20
        mujoco.mj_kinematics(self.model, self.data)
        tips = self.to_wrist_frame(self.data.site_xpos[self.tip_sids])
        mids = self.to_wrist_frame(self.data.xpos[self.mid_bids])
        return tips, mids

    def solve(self, tips_t, mids_t, x0, q_prev=None):
        def residual(q):
            tips, mids = self.keypoints(q)
            r = [W_TIP * (tips - tips_t).ravel(), W_MID * (mids - mids_t).ravel(), W_REG * q]
            if q_prev is not None:
                r.append(W_SMOOTH * (q - q_prev))
            return np.concatenate(r)

        sol = least_squares(residual, np.clip(x0, self.lower, self.upper),
                            bounds=(self.lower, self.upper), method="trf", xtol=1e-10, ftol=1e-10)
        tips, _ = self.keypoints(sol.x)
        err = np.linalg.norm(tips - tips_t, axis=-1)
        return sol.x, err

    def solve_with_offset(self, tips_t, mids_t, x0):
        def residual(x):
            q, d = x[:20], x[20:]
            tips, mids = self.keypoints(q)
            return np.concatenate([W_TIP * (tips - (tips_t - d)).ravel(),
                                   W_MID * (mids - (mids_t - d)).ravel(),
                                   W_REG * q, W_DREG * d])

        lb = np.concatenate([self.lower, -D_MAX * np.ones(3)])
        ub = np.concatenate([self.upper, D_MAX * np.ones(3)])
        sol = least_squares(residual, np.clip(np.concatenate([x0, np.zeros(3)]), lb, ub),
                            bounds=(lb, ub), method="trf", xtol=1e-10, ftol=1e-10)
        return sol.x[:20], sol.x[20:]

    def wrist_from_eef_rot(self):
        mujoco.mj_kinematics(self.model, self.data)
        R8 = self.data.xmat[self.model.body("fr3_link8").id].reshape(3, 3)
        Rw = self.data.xmat[self.wrist_bid].reshape(3, 3)
        return R8.T @ Rw


def main():
    from scipy.spatial.transform import Rotation

    insp = InspireFK()
    wuji = WujiIK()

    with open(REF_IN, "rb") as f:
        ref = pickle.load(f)
    hand6 = np.asarray(ref["hand_qpos"], dtype=np.float64)
    T = hand6.shape[0]

    tips_g, mids_g = insp.keypoints(hand6[T_GRASP])
    q_g, d = wuji.solve_with_offset(tips_g, mids_g, x0=np.zeros(20))
    print("腕系补偿偏移 d(mm) =", np.round(d * 1e3, 1))

    hand20 = np.zeros((T, 20))
    q = np.zeros(20)
    print("=== 参考轨迹 retarget (%d 帧, 目标已按 d 平移) ===" % T)
    for t in range(T):
        tips_t, mids_t = insp.keypoints(hand6[t])
        q, err = wuji.solve(tips_t - d, mids_t - d, x0=(q_g if t == 0 else q),
                            q_prev=(hand20[t - 1] if t > 0 else None))
        hand20[t] = q
        print("t=%02d tip_err(mm): mean=%.1f max=%.1f  [thumb=%.1f index=%.1f middle=%.1f ring=%.1f pinky=%.1f]"
              % (t, err.mean() * 1e3, err.max() * 1e3, *(err * 1e3)))

    q0 = hand20[0].copy()
    qg = hand20[T_GRASP].copy()
    qg_deep = np.clip(qg * DEEPEN, wuji.lower, wuji.upper)
    frac = (qg_deep - q0) / (qg - q0 + 1e-6)
    hand20[:T_GRASP] = q0 + (hand20[:T_GRASP] - q0) * frac[None]
    hand20[T_GRASP:] = qg_deep
    hand20 = np.clip(hand20, wuji.lower, wuji.upper)

    R_rel = wuji.wrist_from_eef_rot()
    d_eef = R_rel @ d
    wrist_pos = np.asarray(ref["wrist_initobj_pos"], dtype=np.float64)
    wrist_quat = np.asarray(ref["wrist_quat"], dtype=np.float64)
    wrist_pos_shifted = wrist_pos + Rotation.from_quat(wrist_quat).apply(d_eef)

    out = {"wrist_initobj_pos": wrist_pos_shifted.astype(np.float32),
           "wrist_quat": wrist_quat.astype(np.float32),
           "hand_qpos": hand20.astype(np.float32),
           "obj_initobj_pos": np.asarray(ref["obj_initobj_pos"], dtype=np.float32)}
    with open(REF_OUT, "wb") as f:
        pickle.dump(out, f)
    print("wrote", REF_OUT)

    style_in = np.load(STYLE_IN)
    n_style = style_in.shape[0]
    style_out = np.zeros((n_style, 25))
    print("=== style retarget (%d 个) ===" % n_style)
    for s in range(n_style):
        tips_t, mids_t = insp.keypoints(style_in[s, :6])
        q, err = wuji.solve(tips_t - d, mids_t - d, x0=hand20[T_GRASP])
        style_out[s, :20] = np.clip(q * DEEPEN, wuji.lower, wuji.upper)
        style_out[s, 20:] = style_in[s, 6:]
        print("style=%d tip_err(mm): mean=%.1f max=%.1f" % (s, err.mean() * 1e3, err.max() * 1e3))
    np.save(STYLE_OUT, style_out)
    print("wrote", STYLE_OUT)


if __name__ == "__main__":
    main()
