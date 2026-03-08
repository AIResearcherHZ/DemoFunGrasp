import torch
import numpy as np
import os, sys
from isaacgymenvs.utils.torch_jit_utils import *

'''
action: (N, 7+6) arm eef pose & hand qpos
'''

def process_action_vanilla(action, **kwargs):
    return action

# use next qpos as action
# for hand dofs, average the values of each tendon group to represent active dof qpos
def process_action_nextstate(action, env, next_eef_pose, next_robot_dof_pos, **kwargs):
    qpos = next_robot_dof_pos.clone()
    weight = torch.zeros_like(qpos)
    weight[:, env.active_hand_dof_indices] = 1.0

    if env.have_passive_joints:
        for idx, p_idx, mul in zip(env.passive_hand_dof_indices, env.mimic_parent_dof_indices, env.mimic_multipliers):
            qpos[:, p_idx] += qpos[:, idx]
            weight[:, p_idx] += mul

    next_hand_qpos = qpos[:, env.active_hand_dof_indices] / weight[:, env.active_hand_dof_indices]

    action = torch.cat([
        next_eef_pose,
        unscale(
            next_hand_qpos,
            env.robot_dof_lower_limits[env.active_hand_dof_indices],
            env.robot_dof_upper_limits[env.active_hand_dof_indices]
        )
    ], dim=-1)
    return action

# only right_thumb_1_joint use next state to avoid collision, others use action
def process_action_thumb1state(action, env, next_eef_pose, next_robot_dof_pos, **kwargs):
    next_hand_qpos_unscaled = unscale(
        next_robot_dof_pos[:, env.active_hand_dof_indices],
        env.robot_dof_lower_limits[env.active_hand_dof_indices],
        env.robot_dof_upper_limits[env.active_hand_dof_indices]
    )
    assert env.active_hand_dof_names[-2] == 'right_thumb_1_joint'
    action = action.clone()
    action[:, -2] = next_hand_qpos_unscaled[:, -2]
    return action

# arm and right_thumb_1_joint use next state to avoid collision, others use action
def process_action_armthumb1state(action, env, next_eef_pose, next_robot_dof_pos, **kwargs):
    next_hand_qpos_unscaled = unscale(
        next_robot_dof_pos[:, env.active_hand_dof_indices],
        env.robot_dof_lower_limits[env.active_hand_dof_indices],
        env.robot_dof_upper_limits[env.active_hand_dof_indices]
    )
    assert env.active_hand_dof_names[-2] == 'right_thumb_1_joint'
    action = action.clone()
    action[:, 0:7] = next_eef_pose
    action[:, -2] = next_hand_qpos_unscaled[:, -2]
    return action

PROCESS_ACTION_FUNCS = {
    "vanilla": process_action_vanilla,
    "nextstate": process_action_nextstate,
    "thumb1state": process_action_thumb1state,
    "armthumb1state": process_action_armthumb1state,
}
