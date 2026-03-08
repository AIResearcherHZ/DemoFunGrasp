import numpy as np
import os, sys
import yaml
import shutil

def single_mesh_to_urdf(template, mesh_file, output_dir):
    with open(template, "r") as f:
        urdf = f.read()
    urdf = urdf.replace("template.obj", mesh_file)
    with open(os.path.join(output_dir, mesh_file.split('.')[0] + ".urdf"), "w") as f:
        f.write(urdf)

def load(dataset_name):
    object_asset_dir = os.path.join("../assets", dataset_name, "meshes")
    urdf_name_list = []
    for fn in os.listdir(object_asset_dir):
        obj_name = fn.split(".")[0]
        single_mesh_to_urdf("../assets/template.urdf",
                            fn,
                            os.path.join('../assets', dataset_name, "urdf"))
        urdf_name_list.append(obj_name + ".urdf")

    urdf_name_list = sorted(urdf_name_list)
    with open(os.path.join('../assets', dataset_name, 'urdf_list.yaml'), 'w') as f:
        yaml.dump(urdf_name_list, f, default_flow_style=False, sort_keys=False)

if __name__ =="__main__":
    dataset_name = "AffordPoseObj"
    load(dataset_name)