import atexit

import numpy as np
import mujoco

from common import TABLE_HEIGHT


class PerceptionError(RuntimeError):
    pass


def estimate_center(pcl_w, cell=0.01):
    xy0 = pcl_w[:, :2].min(axis=0)
    ij = np.floor((pcl_w[:, :2] - xy0) / cell).astype(np.int64)
    key = ij[:, 0] * 4096 + ij[:, 1]
    uniq, inv = np.unique(key, return_inverse=True)
    z_top = np.full(len(uniq), -np.inf)
    np.maximum.at(z_top, inv, pcl_w[:, 2])
    w = np.clip(z_top - TABLE_HEIGHT, 1e-6, None)
    cells_xy = xy0 + (np.stack([uniq // 4096, uniq % 4096], axis=1) + 0.5) * cell
    est_xy = (w[:, None] * cells_xy).sum(axis=0) / w.sum()
    est_z = min(pcl_w[:, 2].mean(), 0.5 * (pcl_w[:, 2].max() + TABLE_HEIGHT))
    return np.array([est_xy[0], est_xy[1], est_z])


def farthest_point_sample(points, k, start):
    idx = np.empty(k, dtype=np.int64)
    dist = np.full(len(points), np.inf, dtype=np.float32)
    cur = start
    for i in range(k):
        idx[i] = cur
        dist = np.minimum(dist, ((points - points[cur]) ** 2).sum(axis=1))
        cur = int(dist.argmax())
    return idx


class CameraCloudSensor:
    MAX_RAW_POINTS = 8192

    def __init__(self, model, camera_name, target_geom, num_points, rng, height=720, width=960):
        self.cam_id = model.camera(camera_name).id
        self.geom_id = model.geom(target_geom).id
        self.num_points = num_points
        self.rng = rng
        self.height, self.width = height, width
        self.focal = 0.5 * height / np.tan(np.radians(model.cam_fovy[self.cam_id]) / 2)
        self.renderer = mujoco.Renderer(model, height, width)
        atexit.register(self.renderer.close)
        self.scene_option = mujoco.MjvOption()
        self.scene_option.geomgroup[:] = 0
        self.scene_option.geomgroup[0] = 1
        self.scene_option.geomgroup[2] = 1

    def capture(self, data):
        self.renderer.update_scene(data, camera=self.cam_id, scene_option=self.scene_option)
        self.renderer.enable_segmentation_rendering()
        seg = self.renderer.render()
        mask = (seg[:, :, 0] == self.geom_id) & (seg[:, :, 1] == mujoco.mjtObj.mjOBJ_GEOM)
        if not mask.any():
            raise PerceptionError("相机视野中未观测到物体像素,无法构建点云")
        self.renderer.enable_depth_rendering()
        depth = self.renderer.render()
        self.renderer.disable_depth_rendering()

        rows = np.flatnonzero(mask.any(axis=1))
        cols = np.flatnonzero(mask.any(axis=0))
        r0, r1 = max(rows[0] - 1, 0), min(rows[-1] + 2, self.height)
        c0, c1 = max(cols[0] - 1, 0), min(cols[-1] + 2, self.width)
        if r1 - r0 < 3 or c1 - c0 < 3:
            raise PerceptionError("物体可见区域过小,无法估计点云法向")
        m = mask[r0:r1, c0:c1]
        d = depth[r0:r1, c0:c1]
        u = (np.arange(c0, c1, dtype=np.float32) + 0.5 - 0.5 * self.width) / self.focal
        v = (0.5 * self.height - np.arange(r0, r1, dtype=np.float32) - 0.5) / self.focal
        pts_cam = np.stack([d * u[None, :], d * v[:, None], -d], axis=-1)
        rot = data.cam_xmat[self.cam_id].reshape(3, 3).astype(np.float32)
        cam_pos = data.cam_xpos[self.cam_id].astype(np.float32)
        pts = pts_cam @ rot.T + cam_pos

        core = m[1:-1, 1:-1] & m[:-2, 1:-1] & m[2:, 1:-1] & m[1:-1, :-2] & m[1:-1, 2:]
        du = pts[1:-1, 2:] - pts[1:-1, :-2]
        dv = pts[2:, 1:-1] - pts[:-2, 1:-1]
        nrm = np.cross(du, dv)
        length = np.linalg.norm(nrm, axis=-1)
        core &= length > 1e-9
        if not core.any():
            raise PerceptionError("物体可见区域过小,无法估计点云法向")
        nrm /= np.maximum(length, 1e-12)[..., None]
        inner = pts[1:-1, 1:-1]
        flip = np.where(np.sum(nrm * (cam_pos - inner), axis=-1, keepdims=True) < 0, -1.0, 1.0)
        pts, nrm = inner[core], (nrm * flip)[core]

        if len(pts) > self.MAX_RAW_POINTS:
            keep = self.rng.choice(len(pts), self.MAX_RAW_POINTS, replace=False)
            pts, nrm = pts[keep], nrm[keep]
        if len(pts) >= self.num_points:
            idx = farthest_point_sample(pts, self.num_points, int(self.rng.integers(len(pts))))
        else:
            idx = np.resize(np.arange(len(pts)), self.num_points)
        return np.concatenate([pts[idx], nrm[idx]], axis=1).astype(np.float64)
