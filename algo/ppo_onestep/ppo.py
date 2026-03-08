import os
import time
from gym.spaces import Space

import statistics
from collections import deque
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import copy
from .storage import RolloutStorage
from omegaconf import OmegaConf
from termcolor import cprint
from isaacgymenvs.utils.torch_jit_utils import *
import sys
import os

# Add the parent directory to the Python path for imports
# parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
# if parent_dir not in sys.path:
#     sys.path.append(parent_dir)


class PPO:
    def __init__(self, vec_env, actor_critic_class, train_param, log_dir="run", apply_reset=False, action_dim=6,cfg=None):
        # other parameters
        self.is_testing_all_objects = train_param.get("is_testing_all_objects", False)
        self.times_testing_all_objects = train_param.get("times_testing_all_objects", 200)
        self.plan = train_param.get("plan", False)

        self.cfg = cfg
        self.task_cfg = cfg.task

        # PPO parameters
        self.clip_param = train_param["cliprange"]
        self.num_learning_epochs = train_param["noptepochs"]
        self.num_mini_batches = train_param["nminibatches"]
        self.num_learning_iterations = train_param["max_iterations"]
        self.num_transitions_per_env = train_param["nsteps"]
        self.value_loss_coef = train_param.get("value_loss_coef", 2.0)
        self.entropy_coef = train_param["ent_coef"]
        self.gamma = train_param["gamma"]
        self.lam = train_param["lam"]
        self.max_grad_norm = train_param.get("max_grad_norm", 2.0)
        self.use_clipped_value_loss = train_param.get("use_clipped_value_loss", False)
        self.init_noise_std = train_param.get("init_noise_std", 0.3)

        self.model_cfg = train_param.policy
        self.sampler = train_param.get("sampler", "sequential")
        self.is_vision = train_param.is_vision

        self

        if not isinstance(vec_env.observation_space, Space) or not isinstance(vec_env.state_space, Space) or not isinstance(vec_env.action_space, Space):
            raise TypeError("vec_env.observation_space, vec_env.state_space and vec_env.action_space must be gym Spaces")

        self.observation_space = vec_env.observation_space
        #self.action_space = vec_env.action_space
        self.state_space = vec_env.state_space
        self.device = vec_env.device

        # for DemoGrasp training
        self.action_space = torch.zeros(action_dim, device=self.device)
        self.asymmetric = vec_env.num_states > 0
        assert self.asymmetric==False, "only symmetric actor-critic is supported for DemoGrasp"
        self.desired_kl = train_param.get("desired_kl", None)
        self.schedule = train_param.get("schedule", "fixed")
        self.step_size = train_param["optim_stepsize"]

        # PPO components
        self.vec_env = vec_env
        # print("if use : ",self.asymmetric)
        self.actor_critic = actor_critic_class(
            self.observation_space.shape,
            self.state_space.shape,
            self.action_space.shape,
            self.init_noise_std,
            self.model_cfg,
            asymmetric=self.asymmetric,
            use_pcl=self.is_vision,
        )
        self.actor_critic.to(self.device)
        self.storage = RolloutStorage(
            self.vec_env.num_envs,
            self.num_transitions_per_env,
            self.observation_space.shape,
            self.state_space.shape,
            self.action_space.shape,
            self.device,
            self.sampler,
        )
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.step_size)
        print(self.actor_critic)

        # Log
        self.save_interval = train_param["save_interval"]
        self.log_dir = log_dir
        self.print_log = train_param["print_log"]
        self.tot_timesteps = 0
        self.tot_time = 0
        self.is_testing = train_param["test"]
        self.current_learning_iteration = 0
        if not self.is_testing:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

        self.apply_reset = apply_reset

    def load(self, path, is_testing=False):
        self.actor_critic.load_state_dict(torch.load(path, map_location=self.device))
        if is_testing:
            self.actor_critic.eval()
        else: # continue training, need to extract the learning iteration from the checkpoint name
            # # extract the xxx/model_iteration_num.pt
            path = path.split("/")[-1]
            self.current_learning_iteration = int(path.split("_")[-1].split(".")[0])
            self.actor_critic.train()

    def save(self, path):
        torch.save(self.actor_critic.state_dict(), path)
        # delete older checkpoints
        dir_path = os.path.dirname(path)
        files = os.listdir(dir_path)
        # model_files = [f for f in files if f.startswith("model_") and f.endswith(".pt")]
        # if len(model_files) > 5:
        #     model_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
        #     os.remove(os.path.join(dir_path, model_files[0]))

    def test(self, ckpt_path):
        self.load(ckpt_path, is_testing=True)

    def run(self):
        num_learning_iterations = self.num_learning_iterations
        metric = self.task_cfg.func.metric
        if ('style' in self.task_cfg.env.observationType and self.task_cfg.env.randomizeGraspPose == True):
            cprint('The policy is trained with hand qpos randomly! But has style obs.','red')


        # Testing mode
        if self.is_testing and not self.cfg.if_visualize:
            # make sure successfully load the checking point
            record_best_label = self.task_cfg.func.use_best_label
            if record_best_label:
                cprint("Using the best style label among different style labels for each object during testing.","yellow")

            base_dir = '/'.join(self.cfg.checkpoint.split('/')[:-1])
            print(self.cfg.checkpoint,"  ",base_dir)
            with open(os.path.join(base_dir, "config.json"), "r") as f:
                cfg_dict = json.load(f)
            ckpt_cfg = OmegaConf.create(cfg_dict)
            cprint(f"[Test] Testing the checkpoint {self.cfg.checkpoint}!",'green')
            cprint(f'[Test] Policy obs is : {ckpt_cfg.task.env.observationType}','green')
            cprint(f'[Test] Current obs is : {self.task_cfg.env.observationType}, obs must match.','green')
            print(f'The policy trained on {ckpt_cfg.task.env.asset.multiObjectList}, Now test on {self.task_cfg.env.asset.multiObjectList}.')
            if self.cfg.task.func.pcl_with_affordance == True:
                cprint(f'Test objects pcl has affordance annotation. Make sure obj pcl has 7dim.','green')
            cprint(f'[Test] This time test metrics are : {metric}','yellow')
            cprint(f'[Test] Sample style label from {self.vec_env.style_list}','yellow')
            # if policy is trained without style, each iteration is same
            

            best_style_label = None
            if record_best_label: # record the best style label for each object
                num_obj = len(self.vec_env.object_names)
                if self.vec_env.num_envs % num_obj !=0:
                    repeat_num = self.vec_env.num_envs//num_obj+1
                else:
                    repeat_num = self.vec_env.num_envs//num_obj
                obj_indices = torch.arange(num_obj).repeat(repeat_num)[:self.vec_env.num_envs].to(self.device) # (num_envs,)
                
                succ_recorder = torch.zeros(num_obj, device=self.device)
                style_recorder = torch.zeros(num_obj, device=self.device)
                for style in self.task_cfg.func.style_list: # run 5 times
                    # Rollout
                
                    for _ in range(self.num_transitions_per_env):
                        # Compute the action
                        metric_info = ""

                        # set style label manually
                        self.vec_env.manually_set_style_labels = torch.tensor([style]*self.vec_env.num_envs, device=self.device)
                        current_obs = self.vec_env.reset_idx(torch.arange(self.vec_env.num_envs))["obs"]
                        self.style_labels = self.vec_env.style_labels

                        current_states = self.vec_env.get_state()
                        actions, actions_log_prob, values, mu, sigma= self.actor_critic(current_obs, current_states) #(num_envs, 6)
                        self.vec_env.generate_reaching_plan_idx(torch.arange(self.vec_env.num_envs), actions=actions)
                        for t in range(self.vec_env.max_episode_length):
                            env_action = self.vec_env.compute_reference_actions()
                            _,_,_,_ = self.vec_env.step(env_action)
                            if (t == self.vec_env.max_episode_length - 2):
                                successes  = self.vec_env.successes.clone().to(self.device)
                                print(f"[Test] style {style} success rate: {successes.sum()/successes.shape[0]:.4f}")
                                # get the mean success rate for each object
                                mean_succ_each_object = torch.zeros(num_obj, device=self.device)
                                for obj_idx in range(num_obj):
                                    idx = (obj_indices==obj_idx)
                                    if idx.sum()>0:
                                        mean_succ_each_object[obj_idx] = successes[idx].float().mean()
                                mask = (mean_succ_each_object > succ_recorder).bool()
                                succ_recorder[mask] = mean_succ_each_object[mask] # update success rate
                                style_recorder[mask] = style # update style label if current style gives higher success
            
                # get the best style label for each environment
                best_style_label = torch.zeros(self.vec_env.num_envs, device=self.device)
                print(torch.unique(style_recorder), " unique best style labels found among ", num_obj, " objects.")
                for obj_idx in range(num_obj):
                    idx = (obj_indices==obj_idx)
                    best_style_label[idx] = style_recorder[obj_idx]

            metric_dict = {} # get mean metric value
            total_test_times = 5
            
            if record_best_label:
                # count the number if the best style in each unique style
                unique_styles, counts = torch.unique(best_style_label, return_counts=True)
                for i in range(len(unique_styles)):
                    cprint(f"[Test] Best style label {unique_styles[i].item()} selected for {counts[i].item()} times.","yellow")


            for test_it in range(total_test_times): # run 5 times
                # Rollout
                for _ in range(self.num_transitions_per_env):
                    # Compute the action
                    metric_info = ""
                    if record_best_label and (best_style_label is not None):
                        # set style label manually
                        self.vec_env.manually_set_style_labels = best_style_label
                    current_obs = self.vec_env.reset_idx(torch.arange(self.vec_env.num_envs))["obs"]
                    
                    self.style_labels = self.vec_env.style_labels
                    current_states = self.vec_env.get_state()
                    actions, actions_log_prob, values, mu, sigma= self.actor_critic(current_obs, current_states) #(num_envs, 6)
                    self.vec_env.generate_reaching_plan_idx(torch.arange(self.vec_env.num_envs), actions=actions)
                    for t in range(self.vec_env.max_episode_length):
                        env_action = self.vec_env.compute_reference_actions()
                        _,_,_,_ = self.vec_env.step(env_action)
                        if (t == self.vec_env.max_episode_length - 2):
                            successes  = self.vec_env.successes.clone().to(self.device)
                            if 'succ_rate' in metric:
                                GSR = successes.sum()/successes.shape[0]
                                metric_info+= f" success rate:{GSR:.4f}."
                                if "GSR" not in metric_dict.keys():
                                    metric_dict["GSR"] = GSR
                                else:
                                    metric_dict["GSR"] += GSR
                            # succ style accuracy
                            if "style_accuracy" in metric:
                                most_similar_style = self.vec_env.find_similar_style_label()
                                style_accuracy = (most_similar_style[successes.bool()]==self.style_labels[successes.bool()]).float().mean()
                                metric_info +=  f" style_acc:{style_accuracy:.4f}"

                                if "style_accuracy" not in metric_dict.keys():
                                    metric_dict["style_accuracy"] = style_accuracy
                                else:
                                    metric_dict["style_accuracy"] += style_accuracy
                            if "qpos_dist" in metric:
                                qpos_dist = torch.norm(self.vec_env.style_hand_qpos- \
                                            self.vec_env.static_style[self.vec_env.style_labels][:,:self.vec_env.num_active_hand_dofs], \
                                            dim=-1)
                                metric_info += f" qpos_dist:{qpos_dist.mean():.4f}"
                                if "qpos_dist" not in metric_dict.keys():
                                    metric_dict["qpos_dist"] = qpos_dist.mean()
                                else:
                                    metric_dict["qpos_dist"] += qpos_dist.mean()
                            

                            # succ affordance distance
                            if "afford_dist" in metric:
                                mean_dist = self.vec_env.calcu_affordance_dist().clone().to(self.device)
                                succ_mean_dist = mean_dist[successes.bool()].mean()
                                metric_info += f" succ_mean_dist: {succ_mean_dist:.4f}"
                                if "succ_mean_dist" not in metric_dict.keys():
                                    metric_dict["succ_mean_dist"] = succ_mean_dist
                                else:
                                    metric_dict["succ_mean_dist"] += succ_mean_dist

                            if 'style_div' in metric:
                                from scipy.spatial.distance import pdist
                                qpos_array = self.vec_env.robot_dof_pos[:, self.vec_env.active_hand_dof_indices].clone().detach().cpu().numpy() # (B, dof)
                                dists = pdist(qpos_array[successes.bool().cpu().numpy()], metric='euclidean')
                                style_div = dists.mean()
                                metric_info += f" style_div:{style_div:.4f}"
                                if "style_div" not in metric_dict.keys():
                                    metric_dict["style_div"] = style_div
                                else:
                                    metric_dict["style_div"] += style_div
                            
                            # if 'store_succ_dist' in metric:
                            #     # store the succ affordance distance for later analysis, only store first success 1000 envs
                            #     mean_dist = self.vec_env.calcu_affordance_dist().clone().to(self.device)
                            #     success_dist = mean_dist[successes.bool().cpu().numpy()].clone().detach().cpu().numpy()
                            #     # randomly sample 1000 envs
                            #     if len(success_dist) > 1000:
                            #         success_dist = success_dist[np.random.choice(len(success_dist), 1000, replace=False)]
                            #     save_path = os.path.join("./save_data", "wo_obj_clip_success_dist.npy")
                            #     np.save(save_path, success_dist)
                            #     cprint("success_dist saved to {}".format(save_path), "green")
                            #     exit(0)

                            # succ contact distance
                            if "contact_similarity" in metric:
                                mean_contact_similar = self.vec_env.calcu_contact_similarity(succ = successes).clone().to(self.device)
                                succ_mean_contact_similar = mean_contact_similar[successes.bool()].float().mean()
                                metric_info += f" succ_mean_contact_similar: {succ_mean_contact_similar:.4f}"
                                if "succ_mean_contact_similar" not in metric_dict.keys():
                                    metric_dict["succ_mean_contact_similar"] = succ_mean_contact_similar
                                else:
                                    metric_dict["succ_mean_contact_similar"] += succ_mean_contact_similar
                            
                            if "filtered_env_ratio" in metric:
                                # calculate success rate after filtering out those with large qpos dist
                                filtered_succ = mean_dist[successes.bool()] < 0.04
                                filtered_succ_rate = filtered_succ.float().mean()
                                metric_info += f" filtered_succ_rate: {filtered_succ_rate:.4f}"
                                if "filtered_succ_rate" not in metric_dict.keys():
                                    metric_dict["filtered_env_ratio"] = filtered_succ_rate
                                else:
                                    metric_dict["filtered_env_ratio"] += filtered_succ_rate


                            break
                    # Record the transition
                
                cprint(f'[Test] test iteration {test_it}: {metric_info}.')
            for key in metric_dict.keys():
                print(f"{key} is {metric_dict[key]/total_test_times:.4f}")

        
        elif self.is_testing and self.cfg.if_visualize:
            # make sure successfully load the checking point
            if self.cfg.headless:
                cprint("Headless mode, cannot visualize.","red")

            iter_list = copy.deepcopy(self.vec_env.style_list)
            # if policy is trained without style, each iteration is same
            for it in range(len(iter_list)):
                # Rollout
                for _ in range(self.num_transitions_per_env):
                    # Compute the action
                    metric_info = ""
                    

                    self.vec_env.style_list = [iter_list[it]]
                    current_obs = self.vec_env.reset_idx(torch.arange(self.vec_env.num_envs))["obs"]
                    if self.cfg.headless == False:
                        affordance_points = self.vec_env.transformed_pcl[torch.arange(self.vec_env.num_envs),self.vec_env.afford_idx,:3]
                        for i in range(self.vec_env.num_envs):
                            self.vec_env.draw_sphere(pos=affordance_points[i, :3], radius=0.01, color=(1, 0., 0.), env_id=i)
                    # use specific style label for visualization
                    self.style_labels = self.vec_env.style_labels
                    # check if all style labels are the same
                    # unique_values = torch.unique(self.style_labels)
                    current_states = self.vec_env.get_state()
                    actions, actions_log_prob, values, mu, sigma= self.actor_critic(current_obs, current_states) #(num_envs, 6)
                    self.vec_env.generate_reaching_plan_idx(torch.arange(self.vec_env.num_envs), actions=actions)
                    for t in range(self.vec_env.max_episode_length):
                        env_action = self.vec_env.compute_reference_actions()
                        obs, reward, reset, extras = self.vec_env.step(env_action)



                        if (t == self.vec_env.max_episode_length - 2):
                            successes  = self.vec_env.successes.clone().to(self.device)


                            # measure different metrics
                            # success rate
                            if 'succ_rate' in metric:
                                metric_info+= f" success rate:{successes.sum()/successes.shape[0]:.4f}."


                            # succ style accuracy
                            if "style_accuracy" in metric:
                                assert 'style' in self.task_cfg.env.observationType, "style observation should be used for style accuracy calculation"
                                most_similar_style = self.vec_env.find_similar_style_label()
                                style_accuracy = (most_similar_style[successes.bool()]==self.style_labels[successes.bool()]).float().mean()
                                metric_info +=  f" style_acc:{style_accuracy:.4f}"

                            # succ affordance distance
                            if "afford_dist" in metric:
                                assert 'affordance' in self.task_cfg.env.observationType, "affordance observation should be used for affordance distance calculation"
                                mean_dist = self.vec_env.calcu_affordance_dist().clone().to(self.device)
                                # mean_dist is (B,) only calculate on successful envs
                                metric_info += f" succ_mean_dist: {mean_dist[successes.bool()].mean():.4f}"

                            # succ contact distance
                            if "contact_similarity" in metric:
                                mean_contact_similar = self.vec_env.calcu_contact_similarity(succ = successes).clone().to(self.device)
                                metric_info += f" succ_mean_contact_similar: {mean_contact_similar[successes.bool()].float().mean():.4f}"

                            break
                    # Record the transition
                    cprint(f"[style {iter_list[it]}]. {metric_info}","blue")
                    print("-"*50)
                if self.cfg.headless == False:
                    for i in range(self.vec_env.num_envs):
                        self.vec_env.draw_sphere(pos=affordance_points[i, :3], radius=0.01, color=(0, 1, 0.), env_id=i)


        # Training mode
        else:
            rewbuffer = deque(maxlen=self.vec_env.num_envs) #100)
            lenbuffer = deque(maxlen=self.vec_env.num_envs) #100)
            cur_reward_sum = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)
            cur_episode_length = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)

            reward_sum = []
            episode_length = []
            log_root = self.cfg.train.params.log_dir
            log_txt_path = os.path.join(log_root,self.cfg['run_name'],f"{self.cfg['run_name']}_log.txt")
            print(log_txt_path)

            for it in range(self.current_learning_iteration, num_learning_iterations):
                start = time.time()
                
                ep_infos = []

                # Rollout
                for _ in range(self.num_transitions_per_env):
                    # Compute the action
                    print_info = ""
                    metric_info = ""

                    current_obs = self.vec_env.reset_idx(torch.arange(self.vec_env.num_envs))["obs"]
                    if "style" in self.task_cfg.env.observationType or 'style' in metric:
                        self.style_labels = self.vec_env.style_labels


                    current_states = self.vec_env.get_state()
                    actions, actions_log_prob, values, mu, sigma= self.actor_critic(current_obs, current_states) #(num_envs, 6)
                    self.vec_env.generate_reaching_plan_idx(torch.arange(self.vec_env.num_envs), actions=actions)
                    dones = torch.ones(self.vec_env.num_envs, device=self.device) # to make sure env reset after each step
                    dist_tensor = torch.zeros((self.vec_env.num_envs,self.vec_env.max_episode_length-1), device=self.device)


                    for t in range(self.vec_env.max_episode_length):
                        env_action = self.vec_env.compute_reference_actions()
                        obs, reward, reset, extras = self.vec_env.step(env_action)
                        if self.task_cfg.func.if_use_close_reward:
                            # calculate distance each step
                            mean_dist = self.vec_env.calcu_affordance_dist().clone().to(self.device)
                            dist_tensor[:,t]=mean_dist

                        if (t == self.vec_env.max_episode_length - 2):
                            successes  = self.vec_env.successes.clone().to(self.device)
                            rews = successes.clone().to(self.device) 

                            '''calculate reward and metrics'''
                            # success rate
                            if 'succ_rate' in metric:
                                metric_info+= f" success rate:{successes.sum()/successes.shape[0]:.4f}."
                                print_info += f" succ_reward:{successes.sum()/successes.shape[0]:.4f}"

                            if "style" in self.task_cfg.env.observationType or 'style' in metric: # style-conditioned affordance
                                # most_similar_style
                                most_similar_style = self.vec_env.find_similar_style_label()
                                # print(self.style_labels[:10])
                                # print(most_similar_style[:10])
                                style_accuracy = (most_similar_style[successes.bool()] ==self.style_labels[successes.bool()]).float().mean()
                                if 'style_accuracy' in metric:
                                    metric_info +=  f" style_acc:{style_accuracy:.4f}"
                                if self.task_cfg.func.if_use_qpos_reward: # give qpos scale penalty
                                    assert self.task_cfg.func.if_use_qpos_scale or self.task_cfg.func.if_use_qpos_delta, "if_use_qpos_scale should be True if using qpos_scale_reward"

                                    qpos_reward = self.vec_env.calcu_qpos_rewards(succ=successes).clone().to(self.device)
                                    rews += qpos_reward
                                    print_info += f" qpos_reward: {qpos_reward.mean():.4f}"

                            if "qpos_scale" in metric:
                                assert self.task_cfg.func.if_use_qpos_scale, "if_use_qpos_scale should be True if using qpos_scale metric"
                                qpos_scale = self.vec_env.scale_param # (num_envs,)
                                mean_qpos_scale = qpos_scale.mean()
                                metric_info += f" mean_qpos_scale:{mean_qpos_scale:.4f}"

                            # affordance distance
                            if "affordance" in self.task_cfg.env.observationType or 'afford_dist' in metric:
                                mean_dist = self.vec_env.calcu_affordance_dist().clone().to(self.device)                                  
                                # mean_dist is (B,) only calculate on successful envs
                                if 'afford_dist' in metric:
                                    metric_info += f" succ_mean_dist:{mean_dist[successes.bool()].mean():.4f}"

                                if self.task_cfg.func.use_affordance_reward: # success and in the end distance < clip
                                    afford_reward = self.vec_env.calcu_affordance_rewards(mean_dists=mean_dist, succ=successes).clone().to(self.device)
                                    rews += afford_reward
                                    print_info += f" afford_reward:{afford_reward.mean():.4f}"
                                    # print_info +=f", succ_afford_reward: {afford_reward[successes.bool()].mean():.6f}"
                                
                                if self.task_cfg.func.if_use_close_reward: # min distance during the reaching phase < threshold
                                    # find the cloest distance during the reaching phase
                                    min_dist = dist_tensor[:,:t].min(dim=-1) # (num_envs, )
                                    close_reward = self.vec_env.calcu_close_rewards(min_dists=min_dist.values, succ=successes,type=self.task_cfg.func.close_type).clone().to(self.device)
                                    rews += close_reward
                                    print_info += f" close_reward: {close_reward.mean():.4f}"
                            
                            if self.task_cfg.func.if_use_contact_reward:
                                mean_contact_similarity = self.vec_env.calcu_contact_similarity(succ = successes).clone().to(self.device).float() # (B,)
                                contact_reward = self.vec_env.calcu_contact_rewards(contact_similarity=mean_contact_similarity,succ=successes).clone().to(self.device)
                                rews += contact_reward
                                print_info += f" contact_reward:{contact_reward.mean():.4f}"
                            if 'contact_similarity' in metric:
                                mean_contact_similarity = self.vec_env.calcu_contact_similarity(successes).clone().to(self.device).float()
                                metric_info += f" succ_mean_contact_similarity:{mean_contact_similarity[successes.bool()].mean():.4f}"

                            break
                    # Record the transition

                    cprint(f"[{self.cfg['run_name']}]{metric_info}","blue")
                    cprint(f"{print_info}","green")

                    if it%5==0:
                        with open(log_txt_path, "a", encoding="utf-8") as f:
                            f.write(f"[{self.cfg['run_name']}]{metric_info}\n")
                            f.write(f"iter {it}--{print_info}\n")
                    # print the success rate of each style label
                    if "style" in self.task_cfg.env.observationType:
                        for style in self.vec_env.style_list:
                            idx = (self.style_labels==style) # bool index
                            if idx.sum()>0:
                                style_succ = successes[idx].sum()/idx.sum()
                                cprint(f"style {style} success rate: {style_succ:.4f}. Mean success dist: {mean_dist[successes.bool()&idx].mean():.4f}", "yellow")

                    self.storage.add_transitions(current_obs, current_states, actions, rews, dones, values, actions_log_prob, mu, sigma)

                    if self.print_log:
                        cur_reward_sum[:] += rews
                        cur_episode_length[:] += 1

                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        reward_sum.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        episode_length.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                if self.print_log:
                    rewbuffer.extend(reward_sum)
                    lenbuffer.extend(episode_length)


                stop = time.time()
                collection_time = stop - start

                mean_trajectory_length, mean_reward = self.storage.get_statistics()

                # Learning step
                start = time.time()
                self.storage.compute_returns(None, self.gamma, self.lam)
                mean_value_loss, mean_surrogate_loss = self.update()
                self.storage.clear()
                stop = time.time()
                learn_time = stop - start

                if self.print_log:
                    self.log(locals())

                if it % self.save_interval == 0 and it!=0:
                    self.save(os.path.join(self.log_dir, "model_{}.pt".format(it)))

                ep_infos.clear()

            self.save(os.path.join(self.log_dir, "model_{}.pt".format(num_learning_iterations)))

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_transitions_per_env * self.vec_env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        ep_string = f""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    try:
                        infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                    except:
                        breakpoint()
                value = torch.mean(infotensor)
                self.writer.add_scalar("Episode/" + key, value, locs["it"])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.actor_critic.log_std.exp().mean()

        self.writer.add_scalar("Loss/value_function", locs["mean_value_loss"], locs["it"])
        self.writer.add_scalar("Loss/surrogate", locs["mean_surrogate_loss"], locs["it"])
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_reward/time", statistics.mean(locs["rewbuffer"]), self.tot_time)
            self.writer.add_scalar("Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), self.tot_time)

        self.writer.add_scalar("Train/mean_reward/step", locs["mean_reward"], locs["it"])
        self.writer.add_scalar("Train/mean_episode_length/episode", locs["mean_trajectory_length"], locs["it"])

        fps = int(self.num_transitions_per_env * self.vec_env.num_envs / (locs["collection_time"] + locs["learn_time"]))

        str = f" \033[1m Learning iteration {locs['it']}/{locs['num_learning_iterations']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                # f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
                # f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                # f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n"""
            )
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                # f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n"""
            )

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (locs['num_learning_iterations'] - locs['it']):.1f}s\n"""
        )
        print(log_string)

    def update(self):
        
        mean_value_loss = 0
        mean_surrogate_loss = 0

        batch = self.storage.mini_batch_generator(self.num_mini_batches)
        for epoch in range(self.num_learning_epochs):
            for indices in batch:
                # print(indices)
                obs_batch = self.storage.observations.view(-1, *self.storage.observations.size()[2:])[indices]
                if self.asymmetric:
                    states_batch = self.storage.states.view(-1, *self.storage.states.size()[2:])[indices]
                else:
                    states_batch = None
                
                actions_batch = self.storage.actions.view(-1, self.storage.actions.size(-1))[indices]
                target_values_batch = self.storage.values.view(-1, 1)[indices]
                returns_batch = self.storage.returns.view(-1, 1)[indices]
                old_actions_log_prob_batch = self.storage.actions_log_prob.view(-1, 1)[indices]
                advantages_batch = self.storage.advantages.view(-1, 1)[indices]
                old_mu_batch = self.storage.mu.view(-1, self.storage.actions.size(-1))[indices]
                old_sigma_batch = self.storage.sigma.view(-1, self.storage.actions.size(-1))[indices]

                (
                    actions_log_prob_batch,
                    entropy_batch,
                    value_batch,
                    mu_batch,
                    sigma_batch,
                ) = self.actor_critic.evaluate(obs_batch, states_batch, actions_batch)

                # KL
                if self.desired_kl != None and self.schedule == "adaptive":
                    kl = torch.sum(sigma_batch - old_sigma_batch + (torch.square(old_sigma_batch.exp()) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch.exp())) - 0.5, axis=-1)
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.step_size = max(1e-5, self.step_size / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.step_size = min(1e-2, self.step_size * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.step_size

                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                # for name, param in self.actor_critic.actor.named_parameters():
                #     print(name, param.grad)

                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates

        return mean_value_loss, mean_surrogate_loss

def simple_misclassification_analysis(gt_labels, pred_labels):
    wrong_mask = (gt_labels != pred_labels)
    wrong_gt = gt_labels[wrong_mask]
    wrong_pred = pred_labels[wrong_mask]
    
    # 组合真实标签和预测标签
    pairs = torch.stack([wrong_gt, wrong_pred], dim=1)
    
    # 统计每种(真实标签, 预测标签)组合的数量
    unique_pairs, counts = torch.unique(pairs, dim=0, return_counts=True)
    
    print("\n=== 错误分类对统计 ===")
    for (gt, pred), count in zip(unique_pairs, counts):
        print(f"真实 {gt.item()} → 预测 {pred.item()}: {count.item()}次")
    print("====================\n")