import trimesh

def convert_y_up_to_z_up(input_file, output_file):
    mesh = trimesh.load(input_file)
    #rotation = trimesh.transformations.rotation_matrix(1.5708, [1, 0, 0])  # -90度绕X轴
    #mesh.apply_transform(rotation)
    mesh.export(output_file, file_type='obj')  # 改为导出OBJ

# 示例用法
for i in range(8):
    convert_y_up_to_z_up(f"link{i}.dae", f"link{i}_from_dae.obj")
