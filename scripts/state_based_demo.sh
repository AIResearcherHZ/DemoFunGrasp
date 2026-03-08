CUDA_LAUNCH_BLOCKING=1 python to_vision.py task=grasp num_envs=1 \
task=grasp train=PPOOneStep test=True \
+debug=control_state_based_policy \
hand=fr3_inspire_tac \
task.env.randomizeGraspPose=False \
task.env.observationType="eefpose+objinitpose+objpcl+affordance+style" \
task.env.resetDofPosRandomInterval=0.01 \
task.func.if_use_qpos_scale=True task.func.if_use_qpos_delta=True \
task.func.scale_limit="[0.1,1.9]" \
task.env.asset.multiObjectList="union_object_dataset/small_debug_set.yaml" \
task.env.render.enable=True  task.env.render.camera_ids="[1]" task.env.render.data_type="rgb" \
task.env.render.randomize=True \
task.env.resetPositionRange="[[0.4, 0.6], [-0.05, 0.05], [0.1, 0.12]]" \
task.func.style_dict_path="./dataset_processor/inspire_static_style_cali.npy" task.func.num_style_obs=4 \
task.func.style_list="[0,1,2,3]" \
task.func.pcl_with_affordance=False task.env.enableRobotTableCollision=True \
checkpoint='./checkpoint/inspire_example/model_4000.pt' \
+vis_rgb=True \

# deploy policy and collect vision data