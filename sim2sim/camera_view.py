import atexit

import cv2
import mujoco

from scene import CAMERA_NAME


class CameraViewer:
    def __init__(self, model, camera_name=CAMERA_NAME, height=480, width=640, refresh_every=4):
        self.cam_id = model.camera(camera_name).id
        self.refresh_every = refresh_every
        self.count = -1
        self.renderer = mujoco.Renderer(model, height, width)
        atexit.register(self.renderer.close)
        self.scene_option = mujoco.MjvOption()
        self.scene_option.geomgroup[:] = 0
        self.scene_option.geomgroup[0] = 1
        self.scene_option.geomgroup[2] = 1

    def _compose(self, data):
        self.renderer.update_scene(data, camera=self.cam_id, scene_option=self.scene_option)
        return self.renderer.render()

    def refresh(self, data):
        self.count += 1
        if self.count % self.refresh_every:
            return
        cv2.imshow("global_cam", self._compose(data)[:, :, ::-1])
        cv2.waitKey(1)
