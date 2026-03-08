CUDA_LAUNCH_BLOCKING=1 python to_vision.py task=grasp num_envs=15 \
task=grasp train=PPOOneStep test=True \
hand=fr3_inspire_tac \
+debug=collect_dataset \
task.env.randomizeGraspPose=False \
task.env.observationType="eefpose+objinitpose+objpcl+affordance+style" \
task.env.resetDofPosRandomInterval=0.2 \
task.func.if_use_qpos_scale=True task.func.if_use_qpos_delta=True \
task.func.scale_limit="[0.1,1.9]" \
task.env.asset.multiObjectList="union_object_dataset/small_debug_set.yaml" \
task.env.render.enable=True  task.env.render.camera_ids="[1]" task.env.render.data_type="rgb" \
task.env.render.randomize=True \
task.env.render.randomization_params.camera_pos="[-0.02,0.02]" task.env.render.randomization_params.camera_quat="[-0.02,0.02]" \
task.env.render.randomization_params.table_xyz="[0.05,0.02,0]" \
task.func.style_dict_path="./dataset_processor/inspire_static_style_cali.npy" task.func.num_style_obs=4 \
task.func.style_list="[0,1,2,3]" \
task.func.pcl_with_affordance=False task.env.enableRobotTableCollision=True \
checkpoint='./checkpoint/inspire_example/model_4000.pt' \
+vis_rgb=False \

# deploy policy and collect vision data