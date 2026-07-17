import os

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSET_ROOT = os.path.join(REPO_ROOT, "assets")

ARM_JOINTS = ["fr3_joint1", "fr3_joint2", "fr3_joint3", "fr3_joint4", "fr3_joint5", "fr3_joint6", "fr3_joint7"]
HAND_JOINTS = [
    "right_index_1_joint", "right_index_2_joint",
    "right_little_1_joint", "right_little_2_joint",
    "right_middle_1_joint", "right_middle_2_joint",
    "right_ring_1_joint", "right_ring_2_joint",
    "right_thumb_1_joint", "right_thumb_2_joint", "right_thumb_3_joint", "right_thumb_4_joint",
]
ACTIVE_HAND_JOINTS = ["right_index_1_joint", "right_little_1_joint", "right_middle_1_joint",
                      "right_ring_1_joint", "right_thumb_1_joint", "right_thumb_2_joint"]
MIMIC_JOINTS = {
    "right_thumb_3_joint": ("right_thumb_2_joint", 0.6),
    "right_thumb_4_joint": ("right_thumb_2_joint", 0.8),
    "right_index_2_joint": ("right_index_1_joint", 1.05),
    "right_middle_2_joint": ("right_middle_1_joint", 1.05),
    "right_ring_2_joint": ("right_ring_1_joint", 1.05),
    "right_little_2_joint": ("right_little_1_joint", 1.18),
}

SIM_DT = 0.01667
DECIMATION = 20
PHYSICS_SUBSTEPS = 64
MAX_EPISODE_LENGTH = 40
T_REF_START_LIFTING = 13
TABLE_HEIGHT = 0.018
POINTS_PER_OBJECT = 512
STATE_DIM = 21


def wxyz2xyzw(q):
    return np.roll(q, -1, axis=-1)


def xyzw2wxyz(q):
    return np.roll(q, 1, axis=-1)


def quat_diff_rad(a, b):
    return (Rotation.from_quat(a).inv() * Rotation.from_quat(b)).magnitude()


def slerp(q1, q2, t):
    return Slerp([0.0, 1.0], Rotation.from_quat([q1, q2]))(np.ravel(t)).as_quat()


def orientation_error(desired, current):
    q = (Rotation.from_quat(desired) * Rotation.from_quat(current).inv()).as_quat()
    return q[..., :3] * np.sign(q[..., 3:4])


def act_scale(x, lower, upper):
    return 0.5 * (upper + lower) + 0.5 * x * (upper - lower)
