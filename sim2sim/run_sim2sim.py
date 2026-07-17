import os

os.environ.setdefault("MUJOCO_GL", "egl")

import glfw

glfw.init_hint(glfw.PLATFORM, glfw.PLATFORM_X11)
glfw.init()

import time

import mujoco.viewer

from camera_view import CameraViewer
from runtime import parse_args, load_runtime, run_policy


def main():
    args = parse_args()
    env, policy, static_style, style_list, rng = load_runtime(args)

    trigger = {"run": False, "reset": False}

    def key_callback(keycode):
        if keycode == 32:
            trigger["run"] = True
        elif keycode == 259:
            trigger["reset"] = True

    env.viewer = mujoco.viewer.launch_passive(env.model, env.data, key_callback=key_callback)
    env.cam_viewer = CameraViewer(env.model)
    print("空格: 运行策略  退格: 随机化重置")

    while env.viewer.is_running():
        env.cam_viewer.refresh(env.data)
        if trigger["reset"]:
            trigger["reset"] = False
            env.clear_markers()
            env.reset()
            print("[重置] 物体位置与朝向已随机化,空格运行策略")
        elif trigger["run"]:
            trigger["run"] = False
            run_policy(env, policy, static_style, style_list, rng, args.device)
            print("空格: 再次运行  退格: 随机化重置")
        else:
            env.viewer.sync()
            time.sleep(0.05)


if __name__ == "__main__":
    main()
