import isaacgym
from tasks.utils import get_pointcloud_from_mesh
import numpy as np
import trimesh
import os, sys
from sklearn.neighbors import NearestNeighbors


def get_norm_vector_from_pc(pc,k=20):
    '''
    pc: (N,3) xyz

    return pc_w_norm: (N,6) xyz + norm vector
    '''
    N = pc.shape[0]
    normals = np.zeros((N, 3))

    # build kNN graph
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(pc)
    _, indices = nbrs.kneighbors(pc)

    for i in range(N):
        neighbors = pc[indices[i, 1:]]  # exclude the point itself
        p = pc[i]

        # PCA on neighbors
        cov = np.cov((neighbors - neighbors.mean(axis=0)).T)
        eigvals, eigvecs = np.linalg.eigh(cov)

        # smallest eigenvector is the normal
        normal = eigvecs[:, np.argmin(eigvals)]

        # orient normal to point outward (away from origin)
        if np.dot(normal, p) < 0:
            normal = -normal

        normals[i] = normal

    return np.hstack([pc, normals])


def voxelize_and_sample(mesh_path, voxel_size=0.01, n_points=1024):
    mesh = trimesh.load(mesh_path, force='mesh')
    # 体素化
    voxel = mesh.voxelized(pitch=voxel_size)
    # 获取体素中心点
    centers = voxel.points
    # 随机挑选 n_points 个体素中心作为候选点
    if centers.shape[0] > n_points:
        idx = np.random.choice(len(centers), n_points, replace=False)
        centers = centers[idx]
    return centers
    



if __name__ =="__main__":
    dataset_name = "union_object_dataset"
    root_name = "../assets"



    mesh_dir = f'{root_name}/{dataset_name}/meshes'
    save_dir = f'{root_name}/{dataset_name}/pointclouds'
    save_dir_voxel = f'{root_name}/{dataset_name}/pointclouds_voxel'

    n_points = 512

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for fn in os.listdir(mesh_dir):
        pc = get_pointcloud_from_mesh(mesh_dir, fn, num_sample=n_points)
        pc_w_norm = get_norm_vector_from_pc(pc)
        pc = pc_w_norm.astype(np.float32)

        assert pc.shape == (n_points, 6)
        np.save(os.path.join(save_dir, fn.split('.')[0]+'.npy'), pc)
        print(f"Processed {fn}, got {pc.shape} point cloud.")



    # get and save bounding box for each mesh
    # save as numpy dictionary
    bbox_dict = {}

    for fn in os.listdir(mesh_dir):
        mesh_path = os.path.join(mesh_dir, fn)
        mesh = trimesh.load(mesh_path, force='mesh')
        obb = mesh.bounding_box_oriented
        bbox = obb.extents  # <-- correct property name
        print(f"Mesh: {fn}, Bounding Box Extents: {bbox}")
        bbox_dict[fn.split('.')[0]] = bbox.astype(np.float32)

    path = os.path.join(f'{root_name}/{dataset_name}', 'bbox_dict.npy')
    np.save(path, bbox_dict)