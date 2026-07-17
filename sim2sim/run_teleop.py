import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import mujoco.viewer

from camera_view import CameraViewer
from common import wxyz2xyzw, xyzw2wxyz
from perception import PerceptionError
from runtime import parse_args, load_runtime, stage_visual_input, stage_model_inference

HELP_TEXT = """
================ 遥操作说明 ================
  拖动橙色方块控制 EEF:
    双击选中橙色方块,
    Ctrl + 左键拖动  -> 平移
    Ctrl + 右键拖动  -> 旋转
  空格 : 沿手所在方位自主接近并抓取;完成后再按恢复遥操作
  退格 : 随机化重置物体与机器人
===========================================
"""


def directed_grasp(env, policy, static_style, style_list, rng, device):
    env.begin_run()
    style_label = int(rng.choice(style_list))
    obs = stage_visual_input(env, style_label)
    actions = stage_model_inference(policy, obs, device)
    print("[阶段3/3 执行] 沿遥操作指定方向自主接近-合拢-抬升")
    env.generate_directed_plan(actions, static_style[style_label][:6])
    for _ in range(env.reaching_timesteps + env.T_ref):
        env.control_step(*env.reference_targets())
    success, delta_z = env.check_success()
    print(f"object={env.object_name} style={style_label} success={success} delta_z={delta_z:.3f}")


def main():
    args = parse_args()
    env, policy, static_style, style_list, rng = load_runtime(args)

    mocap_id = env.model.body("eef_target").mocapid[0]

    def sync_mocap_to_eef():
        pose = env.eef_pose()
        env.data.mocap_pos[mocap_id] = pose[:3]
        env.data.mocap_quat[mocap_id] = xyzw2wxyz(pose[3:7])

    def mocap_pose():
        return np.concatenate([env.data.mocap_pos[mocap_id], wxyz2xyzw(env.data.mocap_quat[mocap_id])])

    pending = {"space": False, "reset": False}

    def key_callback(keycode):
        if keycode == 32:
            pending["space"] = True
        elif keycode == 259:
            pending["reset"] = True

    env.viewer = mujoco.viewer.launch_passive(env.model, env.data, key_callback=key_callback)
    env.cam_viewer = CameraViewer(env.model)
    env.begin_run()
    sync_mocap_to_eef()
    print(HELP_TEXT)
    print("[遥操作模式] 拖动橙色方块控制 EEF,按空格触发自适应抓取")

    mode = "teleop"
    hand_hold = env.prev_targets[env.active_hand_idx].copy()

    while env.viewer.is_running():
        if pending["reset"]:
            pending["reset"] = False
            env.clear_markers()
            env.reset()
            env.begin_run()
            sync_mocap_to_eef()
            hand_hold = env.prev_targets[env.active_hand_idx].copy()
            mode = "teleop"
            print("[重置] 物体位置与朝向已随机化,恢复遥操作模式")

        if pending["space"]:
            pending["space"] = False
            if mode == "teleop":
                print("[定向抓取] 手的方位即接近方向,开始自主抓取")
                try:
                    directed_grasp(env, policy, static_style, style_list, rng, args.device)
                except PerceptionError as e:
                    print(f"[定向抓取取消] {e},保持遥操作模式")
                else:
                    hand_hold = env.prev_targets[env.active_hand_idx].copy()
                    mode = "hold"
                    print("[保持模式] 按空格恢复 EEF 遥操作")
            else:
                mode = "teleop"
                print("[遥操作模式] 拖动橙色方块控制 EEF,按空格触发自适应抓取")

        if mode == "hold":
            sync_mocap_to_eef()
        env.control_step(mocap_pose(), hand_hold)


if __name__ == "__main__":
    main()
