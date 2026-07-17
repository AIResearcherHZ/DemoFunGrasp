CUDA_VISIBLE_DEVICES=0 python -u run_demofungrasp.py \
task=grasp \
train=PPOOneStep \
hand=wuji_hand2 \
num_envs=1000 \
headless=True \
task.env.trackingReferenceFile=tasks/grasp_ref_wuji_hand2.pkl \
task.env.asset.multiObjectList="union_object_dataset/small_debug_set.yaml" \
task.env.resetDofPosRandomInterval=0.2 \
task.env.observationType="eefpose+objinitpose+objpcl+affordance+style" \
task.func.use_affordance_reward=True task.func.affordance_reward_clip_dist=4 task.func.affordance_reward_scale=2 \
task.func.if_use_close_reward=True task.func.close_reward_threshold=0.03 task.func.close_reward_scale=0.5 \
task.func.if_use_qpos_reward=True task.func.scale_limit="[0.1,1.9]" task.func.qpos_delta_scale="[-0.05,0.05]" task.func.qpos_reward_scale=0.1 \
task.func.if_use_qpos_scale=True task.func.if_use_qpos_delta=True task.env.randomizeGraspPose=False \
task.func.style_list="[0,1,2,3]" \
task.func.style_dict_path="./dataset_processor/wuji_hand2_style.npy" task.func.num_style_obs=4 \
task.func.metric="succ_rate+afford_dist+style_accuracy" \
task.func.pcl_with_affordance=False task.env.enableRobotTableCollision=True \
if_visualize=False \
+run_name=wuji_hand2_train train.params.log_dir="./inspire_exp_results/" \
