import os
import json
import hydra
from datetime import datetime
from omegaconf import DictConfig, OmegaConf

import gym
from isaacgym import gymapi
from isaacgym import gymutil
import isaacgymenvs
from isaacgymenvs.utils.utils import set_np_formatting, set_seed
from isaacgymenvs.utils.torch_jit_utils import *
import tasks
import torch
from termcolor import cprint

def build_runner(cfg, env):
    train_param = cfg.train.params
    is_testing = cfg.test  # train_param["test"]
    ckpt_path = cfg.checkpoint

    if not is_testing or not cfg.if_visualize: # create log dir for training or non-visualization testing
        time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_name = f"{cfg.task_name}_{time_str}"
        if "run_name" in cfg:
            run_name = cfg.run_name
        log_dir = os.path.join(train_param.log_dir, run_name)
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "config.json"), "w") as f:
            json.dump(OmegaConf.to_container(cfg), f, indent=4)
    else:
        log_dir = None
    if train_param.name == "ppo_onestep":
        # DemoGrasp: train one-step planner
        assert env.randomize_tracking_reference
        act_dim = 6
        if env.randomize_grasp_pose or ('style' in env.obs_type and env.if_use_qpos_delta):
            act_dim += env.num_active_hand_dofs
        if 'style' in env.obs_type and env.if_use_qpos_scale:
            act_dim += 1

        cprint("action dim is : {}".format(act_dim),"green")
        from algo import ppo_onestep
        runner = ppo_onestep.PPO(
            vec_env=env,
            actor_critic_class=ppo_onestep.ActorCritic,
            train_param=train_param,
            log_dir=log_dir,
            apply_reset=False,
            action_dim=act_dim,
            cfg=cfg,
        )
    else:
        raise ValueError("Unrecognized algorithm!")

    if is_testing and ckpt_path != "":
        print(f"Loading model from {ckpt_path}")
        runner.test(ckpt_path)
    elif ckpt_path != "" and cfg.if_visualize:
        print("Visualizing trained policy")
        runner.load(ckpt_path)
    elif ckpt_path != "":
        print(f"\nWarning: load pre-trained policy. Loading model from {ckpt_path}\n")
        print("Continue training with loaded model.")
        runner.load(ckpt_path)

    return runner


@hydra.main(version_base="1.3", config_path="./tasks", config_name="config")
def main(cfg: DictConfig) -> None:
    # set numpy formatting for printing only
    set_np_formatting()

    # global rank of the GPU
    global_rank = int(os.getenv("RANK", "0"))

    # sets seed. if seed is -1 will pick a random one
    cfg.seed = set_seed(
        cfg.seed, torch_deterministic=cfg.torch_deterministic, rank=global_rank
    )

    def create_isaacgym_env(**kwargs):
        envs = isaacgymenvs.make(
            cfg.seed,
            cfg.task_name,
            cfg.task.env.numEnvs,
            cfg.sim_device,
            cfg.rl_device,
            cfg.graphics_device_id,
            cfg.headless,
            cfg.multi_gpu,
            cfg.capture_video,
            cfg.force_render,
            cfg,
            **kwargs,
        )
        if cfg.capture_video:
            envs.is_vector_env = True
            envs = gym.wrappers.RecordVideo(
                envs,
                f"videos/{cfg.task_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
                step_trigger=lambda step: step % cfg.capture_video_freq == 0,
                video_length=cfg.capture_video_len,
            )
        return envs

    env = create_isaacgym_env()
    # env.reset_idx(torch.arange(env.num_envs))
    
    # debug the environment
    if "debug" in cfg: 
        if cfg["debug"] == "replay_demo":
            cprint("Replaying demonstration data...", "green")
            for k in range(5):
                env.reset_idx(torch.arange(env.num_envs))
                env.generate_reaching_plan_idx(torch.arange(env.num_envs), None)
                for t in range(env.max_episode_length):
                    env_action = env.compute_reference_actions()
                    obs, reward, reset, extras = env.step(env_action)
                    if (t == env.max_episode_length - 2):
                        successes  = env.successes.clone().to(env.device)
                        break
                print("Final success rate:", successes.float().mean().item())

        
        elif cfg["debug"] == "test_motion_plan":
            # sim_state = env.encode_init_state()
            # print("Sim state shape:", sim_state.shape)
            # env.decode_and_set_init_state(sim_state)

            #print("Isaac hand dof limits:", env.robot_dof_upper_limits[env.active_hand_dof_indices],
            #       env.robot_dof_lower_limits[env.active_hand_dof_indices])

            # debug object pcl
            if env.enable_pcl:
                from tasks.utils import vis_pointcloud_realtime
                import queue
                pcl_queue = queue.Queue(1)
                vis_pointcloud_realtime(pcl_queue, coord_len=1)

            for t in range(100000):
                #print(env.progress_buf)
                action = env.compute_reference_actions()
                #print(action)
                obs, reward, reset, extras = env.step(action)

                if env.enable_pcl:
                    pcl = env.transformed_pcl[0].cpu().numpy()
                    if pcl_queue.full():
                        pcl_queue.get()
                    pcl_queue.put(pcl)
                
                if (t+1)%env.max_episode_length==0:
                    # success per object
                    if hasattr(env, "object_names"):
                        success_per_object = {}
                        N = len(env.object_names)
                        for i, obj_name in enumerate(env.object_names):
                            if isinstance(obj_name, tuple):
                                obj_name = f'{obj_name[0][:min(len(obj_name[0]),10)]}_{obj_name[1]:.3f}'
                            success_per_object[obj_name] = \
                                (env.successes[i::N].sum()/env.reset_buf[i::N].sum()).cpu().numpy().item()
                        success_per_object = sorted(success_per_object.items(), key=lambda item: item[1], reverse=True)
                        print("Successes per object:", success_per_object)
                        from tasks.utils import vis_success
                        vis_success(success_per_object, "successes.png", "Successes per Object")
                    print("success rate:", env.current_successes.mean())
                
                # reset when done
                env_ids = env.reset_buf.nonzero(as_tuple=False).squeeze(-1)
                if len(env_ids) > 0:
                    env.reset_idx(env_ids)
        
        elif cfg["debug"] == "collect_dataset":
            NUM_ROUNDS = 5 # total rounds to save
            NUM_TRIALS_PER_ROUND = 1 # in each round, try K episodes, select the best one to save
            FIX_INIT_STATE_WITHIN_ROUND = False # keep the same initial state for each trial within a round
            SAVE_ALL_ENV_DATA_AND_INIT_STATE = False # save all env data (N rounds * num_envs) and initial sim state for future analysis and visual rendering 
            PLAY_POLICY = True # play the trained ppo-one-step policy for data collection

            if SAVE_ALL_ENV_DATA_AND_INIT_STATE:
                assert FIX_INIT_STATE_WITHIN_ROUND

            import pickle
            
            dataset = {"obs": [], "act": [], "init_state": []}

            if PLAY_POLICY:
                runner = build_runner(cfg, env)
                policy = runner.actor_critic
                policy.eval()
            
            base_dir = '/'.join(cfg.checkpoint.split('/')[:-1])
            with open(os.path.join(base_dir, "config.json"), "r") as f:
                cfg_dict = json.load(f)
            ckpt_cfg = OmegaConf.create(cfg_dict)
            if ckpt_cfg.task.env.observationType != cfg.task.env.observationType:
                assert False, "Observation type in checkpoint config and current config do not match!"
            
            cprint(f"[Collect step 1] Loaded checkpoint {cfg.checkpoint} for data collection.", "green")
            if 'affordance' in cfg.task.env.observationType:
                cprint("[Collect step 1] Affordance observation is used for data collection.", "green")
            if 'style' in cfg.task.env.observationType:
                cprint("[Collect step 1] Style observation is used for data collection.", "green")


            if 'style' in cfg.task.env.observationType or "affordance" in cfg.task.env.observationType:
                dataset_path = './dataset_func_vision_1'
            else:
                dataset_path = './dataset_base_vision_1'

            if not os.path.exists(dataset_path):
                os.makedirs(dataset_path)
            dataset_path = os.path.join(dataset_path, f'{ckpt_cfg.run_name}.pkl')


            for episode in range(NUM_ROUNDS):
                # roll and save K trials
                obses_trials, acts_trials, scores_trials, successes_trials = [], [], [], []
                if FIX_INIT_STATE_WITHIN_ROUND:
                    env.reset_idx(torch.arange(env.num_envs))
                    init_state = env.encode_init_state()

                for trial in range(NUM_TRIALS_PER_ROUND):
                    obses, acts = [], []
                    #input("reset")
                    env.reset_idx(torch.arange(env.num_envs))
                    if FIX_INIT_STATE_WITHIN_ROUND:
                        #input("decode and set init state")
                        env.decode_and_set_init_state(init_state)
                    if PLAY_POLICY:
                        obs = env.obs_dict["obs"].clone()
                        with torch.no_grad():
                            plan = policy(obs, inference=True)
                            env.generate_reaching_plan_idx(torch.arange(env.num_envs), actions=plan)
                    

                    needed_obs = "handdof+eefpose+objpose"
                    obs_buf_size = len(cfg.hand.hardware_active_hand_dof_names)+7+7
                    if 'affordance' in cfg.task.env.observationType:
                        needed_obs += "+affordance"
                        obs_buf_size += 3
                    if 'style' in cfg.task.env.observationType:
                        env_style = env.style_labels
                        

                    for t in range(env.max_episode_length):
                        obs_buf = torch.zeros((env.num_envs, obs_buf_size), dtype=torch.float32, device=env.device)
                        env.compute_required_observations(obs_buf, needed_obs, obs_buf_size)
                        obs = torch.clamp(obs_buf, -env.clip_obs, env.clip_obs) #env.obs_dict["obs"].clone()
                        obs = torch.cat([obs, env_style[:,None]], dim=-1)
                        action = env.compute_reference_actions()
                        obses.append(obs)
                        acts.append(action.clone())
                        _, _, reset, extras = env.step(action)
                    # all elements in reset are 1
                    assert reset.all()
                    success = extras["current_successes"] > 0.5 # (num_envs,)
                    obses = torch.stack(obses, dim=0).permute(1, 0, 2) # (num_envs, T, obs_dim)
                    acts = torch.stack(acts, dim=0).permute(1, 0, 2) # (num_envs, T, act_dim)
                    obses_trials.append(obses)
                    acts_trials.append(acts)
                    successes_trials.append(success)
                    scores_trials.append(success.float()) # currently use only success to select
                    print(f"Round {episode} Trial {trial}: success rate {success.float().mean().item()}")
                
                # select the best & success episode among the K trials for each env
                obses_trials = torch.stack(obses_trials, dim=0) # (K, num_envs, T, obs_dim)
                acts_trials = torch.stack(acts_trials, dim=0) # (K, num_envs, T, act_dim)
                scores_trials = torch.stack(scores_trials, dim=0) # (K, num_envs)
                successes_trials = torch.stack(successes_trials, dim=0) # (K, num_envs)
                best_trial_indices = torch.argmax(scores_trials, dim=0) # (num_envs,)
                is_best_trial_success = successes_trials[best_trial_indices, torch.arange(env.num_envs)] # (num_envs,)
                if not SAVE_ALL_ENV_DATA_AND_INIT_STATE:
                    # in this mode, save only successful episodes
                    obses = obses_trials[best_trial_indices, torch.arange(env.num_envs)][is_best_trial_success] # (num_success, T, obs_dim)
                    acts = acts_trials[best_trial_indices, torch.arange(env.num_envs)][is_best_trial_success] # (num_success, T, act_dim)
                else:
                    # in this mode, save all episodes with best score among K trials
                    obses = obses_trials[best_trial_indices, torch.arange(env.num_envs)] # (num_envs, T, obs_dim)
                    acts = acts_trials[best_trial_indices, torch.arange(env.num_envs)] # (num_envs, T, act_dim)
                    dataset["init_state"].append(init_state.cpu()) # (num_envs, state_dim)
                dataset["obs"].append(obses)
                dataset["act"].append(acts)
                print(f"Round {episode}: saved {obses.shape[0]} trajs among {env.num_envs} envs, success rate {is_best_trial_success.float().mean().item()}")

            dataset["obs"] = torch.cat(dataset["obs"], dim=0)
            dataset["act"] = torch.cat(dataset["act"], dim=0)
            cprint(f"[Collect step 1] Dataset obs shape is {dataset['obs'].shape}; action shape is {dataset['act'].shape}.", "green")
            if SAVE_ALL_ENV_DATA_AND_INIT_STATE: 
                dataset["init_state"] = torch.stack(dataset["init_state"], dim=0) # (n_rounds, num_envs, state_dim)
                print("Init state shape:", dataset["init_state"].shape)
            with open(dataset_path, "wb") as f:
                pickle.dump(dataset, f)

            
            print("Dataset saved to", dataset_path)


        elif cfg["debug"] == "replay_dataset":
            import pickle
            dataset_path = 'dataset/ref_duck_randrotz_debugset.pkl'
            with open(dataset_path, "rb") as f:
                dataset = pickle.load(f)
            dataset["obs"] = dataset["obs"]#[28000:]
            dataset["act"] = dataset["act"]#[28000:]
            print("Dataset loaded:", dataset["obs"].shape, dataset["act"].shape)
            num_trajs = dataset["obs"].shape[0] // env.num_envs
            traj_length = dataset["obs"].shape[1]
            
            if "init_state" in dataset and len(dataset["init_state"])>0:
                init_states = dataset["init_state"]
                assert init_states.shape[0] == num_trajs
            else:
                init_states = None

            for episode in range(num_trajs):
                env.reset_idx(torch.arange(env.num_envs))
                if init_states is not None:
                    env.decode_and_set_init_state(init_states[episode])
                for t in range(traj_length):
                    obs_data = dataset["obs"][episode*env.num_envs: (episode+1)*env.num_envs, t].to(env.device)
                    act_data = dataset["act"][episode*env.num_envs: (episode+1)*env.num_envs, t].to(env.device)
                    obs, _, _, _ = env.step(act_data)
                    print("Obs MSE:", (obs["obs"] - obs_data).pow(2).mean().item())
                print("Success rate:", env.successes.float().mean().item())

        else:
            for t in range(100000):
                action = env.no_op_action
                env.step(action)
    
    else:
        runner = build_runner(cfg, env)
        runner.run()

if __name__ == "__main__":
    main()
