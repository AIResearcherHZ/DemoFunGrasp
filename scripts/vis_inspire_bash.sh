VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
CUDA_VISIBLE_DEVICES=0 python -u run_demofungrasp.py \
task=grasp \
train=PPOOneStep \
hand=fr3_inspire_tac \
num_envs=16 \
headless=False \
task.env.trackingReferenceFile=tasks/grasp_ref_inspire.pkl \
task.env.asset.multiObjectList="union_object_dataset/small_debug_set.yaml" \
task.env.resetDofPosRandomInterval=0.2 \
task.env.observationType="eefpose+objinitpose+objpcl+affordance+style" \
task.func.if_use_qpos_scale=True task.func.if_use_qpos_delta=True task.env.randomizeGraspPose=False \
task.func.scale_limit="[0.1,1.9]" task.func.qpos_delta_scale="[-0.05,0.05]" \
task.func.style_list="[0,1,2,3]" \
task.func.style_dict_path="./dataset_processor/inspire_static_style_cali.npy" task.func.num_style_obs=4 \
task.func.pcl_with_affordance=False task.env.enableRobotTableCollision=True \
if_visualize=True \
+run_name=debug_vis train.params.log_dir="./inspire_exp_results/" \
test=True task.func.use_best_label=False \
checkpoint="/home/xhz/DemoFunGrasp/checkpoint/inspire_example/model_4000.pt" \
