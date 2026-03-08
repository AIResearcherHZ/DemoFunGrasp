import numpy as np
import os, sys, trimesh

def get_pointcloud_from_mesh(mesh_dir, filename, num_sample=4096):
    all_points = []
    mesh = trimesh.load_mesh(os.path.join(mesh_dir, filename))
    points = mesh.sample(num_sample)
    return points

def visualize_pcl(fn):
    import open3d as o3d
    pc = np.load(fn)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    o3d.visualization.draw_geometries([pcd])

if __name__ =="__main__":
    visualize_pcl('./link0.pcl.npy')

    # fns = os.listdir('./')
    # for fn in fns:
    #     if fn.endswith('.obj') and fn.startswith('link'):
    #         print(fn)
    #         pc = get_pointcloud_from_mesh('./', fn)
    #         np.save(os.path.join('./', fn.split('.')[0]+'.pcl.npy'), pc)