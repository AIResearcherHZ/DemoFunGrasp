CUDA_VISIBLE_DEVICES=0 python -u run_demofungrasp.py \
task=grasp \
train=PPOOneStep \
hand=shadow_simple \
num_envs=5 \
headless=False \
task.env.trackingReferenceFile=tasks/grasp_ref_shadow.pkl \
task.env.asset.multiObjectList="union_object_dataset/small_debug_set.yaml" \
task.env.resetDofPosRandomInterval=0.2 \
task.env.observationType="eefpose+objinitpose+objpcl+affordance+style" \
task.func.use_affordance_reward=True task.func.affordance_reward_clip_dist=0.05 task.func.affordance_reward_scale=2 \
task.func.if_use_close_reward=True task.func.close_reward_scale=0.05 task.func.close_reward_threshold=0.03 \
task.func.if_use_qpos_scale=True task.func.if_use_qpos_delta=True task.env.randomizeGraspPose=False \
task.func.if_use_qpos_reward=True task.func.qpos_reward_scale=0.3 task.func.scale_limit="[0.1,1.9]" task.func.qpos_delta_scale="[-0.2,0.2]" \
task.func.style_dict_path="./dataset_processor/shadow_style.npy" \
task.func.style_list="[0,1,2,3,4,5,6,7,8]" task.func.num_style_obs=9 \
task.func.metric="succ_rate+afford_dist+style_accuracy" \
task.func.pcl_with_affordance=False \
task.env.enableRobotTableCollision=True \
task.func.if_record_contact_pcl_idx=True task.func.if_record_qpos=False \
test=True task.func.use_best_label=False \
if_visualize=True \
+run_name=afford_clip_5cm train.params.log_dir="./inspire_exp_results/" \
checkpoint="./checkpoint/shadow_example/model_8000.pt" \

