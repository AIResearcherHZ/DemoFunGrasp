import os
import xml.etree.ElementTree as ET

import mujoco

from common import ASSET_ROOT

CAMERA_NAME = "global_cam"
SCENE_XML = os.path.join(ASSET_ROOT, "inspire_tac", "mjcf", "scene.xml")


def _attr_vec(el, name, default):
    text = el.get(name) if el is not None else None
    return [float(v) for v in (text or default).split()]


def build_model(object_urdf):
    spec = mujoco.MjSpec()
    spec.from_file(SCENE_XML)

    obj_tree = ET.parse(object_urdf).getroot()
    obj_mesh_el = obj_tree.find(".//collision/geometry/mesh")
    obj_mesh = spec.add_mesh()
    obj_mesh.name = "object_mesh"
    obj_mesh.file = os.path.abspath(os.path.join(os.path.dirname(object_urdf), obj_mesh_el.get("filename")))
    obj_mesh.scale = _attr_vec(obj_mesh_el, "scale", "1 1 1")

    obj_body = spec.worldbody.add_body()
    obj_body.name = "object"
    obj_body.pos = [0.55, 0, 0.2]
    joint = obj_body.add_joint()
    joint.name = "object_free"
    joint.type = mujoco.mjtJoint.mjJNT_FREE
    g = obj_body.add_geom()
    g.name = "object_geom"
    g.type = mujoco.mjtGeom.mjGEOM_MESH
    g.meshname = "object_mesh"
    g.friction = [2.0, 0.5, 0.05]
    g.condim = 6
    g.solref = [0.005, 1.0]
    g.margin = 0.002
    g.contype, g.conaffinity = 1, 1
    g.rgba = [0.2, 0.7, 0.3, 1.0]

    return spec.compile()
