import trimesh

fn = "link1.obj"

mesh = trimesh.load(fn)

vertices = mesh.vertices
modified_vertices = vertices.copy()
modified_vertices[:, 1] = -vertices[:, 2]  # y_new = -z_old
modified_vertices[:, 2] = vertices[:, 1]   # z_new = y_old
mesh.vertices = modified_vertices
mesh.export('../'+fn)


# scene = trimesh.load(fn)
# for name, mesh in scene.geometry.items():
#     vertices = mesh.vertices.copy()
#     modified_vertices = vertices.copy()
#     modified_vertices[:, 1] = -vertices[:, 2]  # y_new = -z_old
#     modified_vertices[:, 2] = vertices[:, 1]   # z_new = y_old
#     mesh.vertices = modified_vertices

# 导出整个 Scene（保持多网格结构）
# scene.export("../"+fn)
