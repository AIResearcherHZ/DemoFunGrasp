import hydra, os, sys
from omegaconf import DictConfig

import gym
from isaacgym import gymapi
from isaacgym import gymutil
import isaacgymenvs
from isaacgymenvs.utils.utils import set_np_formatting, set_seed
from isaacgymenvs.utils.torch_jit_utils import *
import cv2
from real2sim.real_robot_like_env import RealRobotLikeEnv
from tqdm import trange

from termcolor import cprint
import numpy as np
import torch
from datetime import datetime
import json
from omegaconf import OmegaConf



gym = gymapi.acquire_gym()

def quaternion_to_rotation_matrix(q):
    x = q[:, 0]; y = q[:, 1]; zq = q[:, 2]; w = q[:, 3]
    xx = x*x; yy = y*y; zz = zq*zq
    xy = x*y; xz = x*zq; yz = y*zq
    wx = w*x; wy = w*y; wz = w*zq
    R_c2w = torch.stack([
        torch.stack([1 - 2*(yy + zz),     2*(xy - wz),       2*(xz + wy)], dim=-1),
        torch.stack([    2*(xy + wz), 1 - 2*(xx + zz),       2*(yz - wx)], dim=-1),
        torch.stack([    2*(xz - wy),     2*(yz + wx),   1 - 2*(xx + yy)], dim=-1),
    ], dim=-2)
    return R_c2w  # [B,3,3]

def create_camera(gym, env, width, height, pos, quat, name="cam"):
    cam_props = gymapi.CameraProperties()
    cam_props.width = width
    cam_props.height = height
    cam_props.enable_tensors = False   # True 时会直出 GPU tensor（有时会导致 segfault，见注意）
    cam_props.use_collision_geometry = False
    print("cam_props:", "width",cam_props.width,"height", cam_props.height)
    cam = gym.create_camera_sensor(env, cam_props)
    # 设定变换（position 和朝向 target）
    cam_pos = gymapi.Vec3(*pos)
    cam_quat = gymapi.Quat(*quat)
    tr = gymapi.Transform(cam_pos, cam_quat)
    gym.set_camera_transform(cam, env, tr)
    return cam


def project_point_to_image(points_w, camera_pose, intrinsics, device,
                           resize=None, eps=1e-6, debug=False):
    """
    Project world points to image pixel coordinates and scale to match resized RGB.

    Args:
        points_w: [B,3] world points (one per env)
        camera_pose: [B,7] or [7] -> [tx,ty,tz, qx,qy,qz,qw] (xyzw), camera->world
        intrinsics: [3,3] or [B,3,3] (pixel units for original render size)
        device: torch device
        resize: None or (H_t, W_t) or (H_t, W_t, C) or torch tensor; target rgb size used in F.interpolate
        eps: small value for numerical stability
        debug: prints diagnostics when True

    Returns:
        uv: torch.Tensor [B,2]  # pixel coords aligned with resized image
        z_cam: torch.Tensor [B] # depth in camera frame (positive = in front)
    """
    points_w = torch.as_tensor(points_w, device=device).float()   # [B,3]
    camera_pose = torch.as_tensor(camera_pose, device=device).float()
    intrinsics = torch.as_tensor(intrinsics, device=device).float()

    if points_w.ndim != 2 or points_w.size(-1) != 3:
        raise ValueError("points_w must be [B,3]")

    B = points_w.shape[0]
    # normalize shapes
    if camera_pose.ndim == 1 and camera_pose.numel() == 7:
        camera_pose = camera_pose.unsqueeze(0)
    if camera_pose.size(0) != B:
        if camera_pose.size(0) == 1:
            camera_pose = camera_pose.expand(B, -1)
        else:
            raise ValueError("camera_pose batch dim mismatch")

    if intrinsics.ndim == 2 and intrinsics.shape == (3,3):
        intrinsics = intrinsics.unsqueeze(0)
    if intrinsics.size(0) != B:
        if intrinsics.size(0) == 1:
            intrinsics = intrinsics.expand(B, -1, -1)
        else:
            raise ValueError("intrinsics batch dim mismatch")

    t = camera_pose[:, :3]   # [B,3]
    q = camera_pose[:, 3:7]  # [B,4] xyzw

    # quaternion -> rotation (camera->world).
    R_c2w = quaternion_to_rotation_matrix(q)
    # world -> camera rotation
    R_w2c = R_c2w.transpose(-1, -2)  # [B,3,3]

    # transform: p_cam = R_w2c @ (p_w - t)
    rel = (points_w - t).unsqueeze(-1)        # [B,3,1]
    p_cam = torch.bmm(R_w2c, rel).squeeze(-1) # [B,3]
    x_cam = p_cam[:,0]; y_cam = p_cam[:,1]; z_cam = p_cam[:,2]

    # avoid division by zero
    z_safe = z_cam.clamp(min=eps)
    xn = x_cam / z_safe
    yn = y_cam / z_safe

    pts_norm = torch.stack([xn, yn, torch.ones_like(xn)], dim=-1).unsqueeze(-1)  # [B,3,1]
    uv_h = torch.bmm(intrinsics, pts_norm).squeeze(-1)  # [B,3]
    uv = uv_h[:, :2]  # [B,2]  <-- these are in original intrinsics pixel units

    # mask points behind camera
    behind = z_cam <= eps
    if behind.any():
        uv = uv.clone()
        uv[behind] = float('nan')

    # ---------- scale uv to match resize ----------
    do_scale = False
    if resize is not None:
        # extract H_t,W_t
        if torch.is_tensor(resize):
            rlist = resize.cpu().tolist()
            H_t = int(rlist[0]); W_t = int(rlist[1])
        elif isinstance(resize, (tuple, list)):
            H_t = int(resize[0]); W_t = int(resize[1])
        else:
            raise ValueError("resize must be (H,W,...) or torch tensor")

        # compute per-batch original W_orig,H_orig from intrinsics (heuristic)
        cx = intrinsics[:, 0, 2]  # [B]
        cy = intrinsics[:, 1, 2]  # [B]
        W_orig = (2.0 * cx).clamp(min=1.0)
        H_orig = (2.0 * cy).clamp(min=1.0)

        sx = (float(W_t) / W_orig).unsqueeze(-1)  # [B,1]
        sy = (float(H_t) / H_orig).unsqueeze(-1)  # [B,1]

        # apply scaling: u' = u * sx, v' = v * sy
        # broadcasting: uv [B,2], sx/sy [B,1]
        u = uv[:, 0:1] * sx
        v = uv[:, 1:2] * sy
        uv = torch.cat([u, v], dim=1)
        do_scale = True

    return uv, z_cam

def project_from_image_to_obj_pcl(afford_xy, camera_pose, intrinsics, obj_pcl, device):
    all_pcl_xy = torch.zeros((obj_pcl.shape[0], obj_pcl.shape[1], 2), device=device)  # [B, N, 2]
    threshold = 15.0  # pixels
    for i in range(obj_pcl.shape[1]):
        pcl_img_coords, pcl_cam_z = project_point_to_image(
            obj_pcl[:,i,:], camera_pose, intrinsics, device
        )  # [B, N, 2], [B, N]
        all_pcl_xy[:,i,:] = pcl_img_coords  # [B, N, 2]
    
    # filter the distances< threshold
    dists = torch.norm(
        all_pcl_xy - afford_xy.unsqueeze(1), dim=-1
    )  # [B, N]
    mask = (dists < threshold)  # [B, N]

    camera_position = camera_pose[:, :3]  # [B, 3]
    dists_to_camera = torch.norm(
        obj_pcl - camera_position.unsqueeze(1), dim=-1
    )  # [B, N]
    dists_to_camera[~mask] = 1e6  # set large distance for filtered points
    min_indices = torch.argmin(dists_to_camera, dim=-1)  # [B]
    closest_pcl = obj_pcl[
        torch.arange(obj_pcl.size(0)), min_indices
    ]  # [B, 3]
    return closest_pcl


def build_runner(cfg, env):
    train_param = cfg.train.params
    is_testing = cfg.test  # train_param["test"]
    ckpt_path = cfg.checkpoint

    if not is_testing and not cfg.if_visualize: # create log dir for training or non-visualization testing
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
    
    
    if cfg["debug"] == "collect_dataset":
        '''
        This is only an examply function to deploy the state based policy and collect vision data. You should modify this function to fit your specific policy and observation format. 
        The main point is to show how to get the visual observations, process them into the affordance representation, and save them together with the state and action data.
        '''
        # raise NotImplementedError("This is a data collection function for visual dataset. You should modify it according to your specific policy and observation format.")
        env = RealRobotLikeEnv(cfg)

        NUM_ROUNDS = 20 # total rounds to save data
        runner = build_runner(cfg, env.env) # use grasp env for runner
        policy = runner.actor_critic
        policy.eval()
        
        base_dir = '/'.join(cfg.checkpoint.split('/')[:-1])
        with open(os.path.join(base_dir, "config.json"), "r") as f:
            cfg_dict = json.load(f)
        ckpt_cfg = OmegaConf.create(cfg_dict)
        if ckpt_cfg.task.env.observationType != cfg.task.env.observationType:
            assert False, "Observation type in checkpoint config and current config do not match!"
        
        cprint(f"[Collect Vision Data] Loaded checkpoint {cfg.checkpoint} for data collection.", "green")
        assert 'affordance' in cfg.task.env.observationType
        cprint("[Collect Vision Data] Affordance observation is used for data collection.", "green")
        assert 'style' in cfg.task.env.observationType
        cprint("[Collect Vision Data] Style observation is used for data collection.", "green")

        assert len(cfg['task']['env']['render']['camera_ids']) == 1, "Only support single camera for fundemograsp dataset collection."
        OUTPUT_PATH = ckpt_cfg.run_name + '_vision_data'


        style_dim = cfg['hand']['num_obs_dict']['style']


        from lerobot_dataset.dataset import LerobotDatasetWriter
        dataset_writer = LerobotDatasetWriter(
            output_path=OUTPUT_PATH,
            camera_ids=env.env.camera_ids,
            data_type=env.env.render_cfg["data_type"],
            action_dim=env.env.num_actions,
            state_dim=env.env.num_actions + 2 + style_dim, # add affordance dim and hand style
            image_shape=(*env.env.render_cfg["resize"], 3),
            depth_shape=tuple(env.env.render_cfg["resize"]),
        )


        n_saved_episode = 0

        for round in range(NUM_ROUNDS):
            ### rollout this round
            obs=env.env.reset_idx(torch.arange(env.env.num_envs))['obs']
            episode_data_buffer = []

            camera_pose = env.env.root_state_tensor[env.env.camera_pad_indices[0][torch.arange(env.env.num_envs)], 0:7] # [num_envs, 7]
            afford_xy,z_cam = project_point_to_image(env.env.transformed_pcl[torch.arange(env.env.num_envs),env.env.afford_idx,:3],
                                               camera_pose,env.camera_intrinsic_matrices,
                                               env.env.device,
                                               tuple(env.env.render_cfg['resize'])) # [num_envs, num_afford_points, 2]

            with torch.no_grad():
                plan = policy(obs, inference=True)
                env.env.generate_reaching_plan_idx(torch.arange(env.env.num_envs), actions=plan)

            # draw affordance points
            if cfg['vis_rgb'] and cfg.headless == False:
                affordance_points = env.env.transformed_pcl[torch.arange(env.env.num_envs),env.env.afford_idx,:3]
                for i in range(env.env.num_envs):
                    env.env.draw_sphere(pos=affordance_points[i, :3], radius=0.01, color=(0, 1., 0.), env_id=i)
            # print("afford_xy is : ",afford_xy)
            env.env.speed_up = True # speed up the sim by skipping pcl calculation
            for t in range(env.env.max_episode_length):
                obs_dict = env.get_real_observations()
                action = env.env.compute_reference_actions()
                #print(action.shape)
                env.env.step(action)            
                data_dict = {
                    'instruction': obs_dict["instruction"],
                    'action': action.cpu().numpy(),
                    'right_arm_qpos': obs_dict["right_arm_qpos"],
                    'right_arm_eef_pose': obs_dict["right_arm_eef_pose"],
                    'right_hand_qpos': obs_dict["right_hand_qpos"],
                    'afford_xy': afford_xy.cpu().numpy(),
                    'style': env.env.style_onehot_envs.cpu().numpy(),
                }
                for cam_id in env.env.camera_ids:
                    if "rgb" in env.env.render_cfg["data_type"]:
                        data_dict[f"camera_{cam_id}.rgb"] = obs_dict[f"camera_{cam_id}.rgb"]
                    if "depth" in env.env.render_cfg["data_type"]:
                        data_dict[f"camera_{cam_id}.depth"] = obs_dict[f"camera_{cam_id}.depth"]
                    if "pcl" in env.env.render_cfg["data_type"]:
                        data_dict[f"camera_{cam_id}.pcl"] = obs_dict[f"camera_{cam_id}.pcl"]




                if cfg['vis_rgb'] and cfg.headless == False:
                    env_idx = 0

                    # visualize afford_xy on the image
                    # only visualize the environment 0
                    for cam_id in env.env.camera_ids:
                        rgb = obs_dict[f"camera_{cam_id}.rgb"][env_idx]
                        u = int(afford_xy[env_idx,0].item())
                        v = int(afford_xy[env_idx,1].item())
                        if u >=0 and u < rgb.shape[1] and v >=0 and v < rgb.shape[0]:
                            cv2.circle(rgb, (u,v), 3, (255,0,0), -1)
                        cv2.imshow(f"camera_{cam_id}_rgb", rgb[...,::-1])
                    cv2.waitKey(1)
                episode_data_buffer.append(data_dict)

            env.env.speed_up = False
            ### save successful episodes
            successes = env.env.successes
            #print(successes, [[k,v.shape if hasattr(v,'shape') else v] for k,v in episode_data_buffer[0].items()])
            for env_id in trange(env.num_envs):
                if successes[env_id] > 0.5:
                    for t, d in enumerate(episode_data_buffer):
                        episode_end = (t == len(episode_data_buffer) - 1)
                        dataset_writer.append_step(
                            {k: v[env_id:env_id+1] for k,v in d.items()}, 
                            episode_end = episode_end
                        )
                    n_saved_episode += 1
            print(f"Round {round}, success rate {successes.mean().item()}, total saved episodes {n_saved_episode}")

        dataset_writer.close()
    
    elif cfg["debug"] == "control_state_based_policy":
        env = RealRobotLikeEnv(cfg)
        '''
        Deploy the policy to collect fundemograsp-style dataset with affordance and style labels.
        '''
        if cfg.task.demo.enable:
            cams = []
            width = cfg.task.demo.width
            height = cfg.task.demo.height
            cam_1_pos = cfg.task.demo.camera_1.pos
            cam_1_quat = cfg.task.demo.camera_1.quat
            cam_1 = create_camera(env.env.gym, env.env.env_ptr_list[0], width=width, height=height, pos=cam_1_pos, quat=cam_1_quat, name="cam_1")
            cams.append(cam_1)
            cam_2_pos = cfg.task.demo.camera_2.pos
            cam_2_quat = cfg.task.demo.camera_2.quat
            cam_2 = create_camera(env.env.gym, env.env.env_ptr_list[0], width=width, height=height, pos=cam_2_pos, quat=cam_2_quat, name="cam_2")
            cams.append(cam_2)


        NUM_ROUNDS = 1 # total rounds to save data
        runner = build_runner(cfg, env.env) # use grasp env for runner
        policy = runner.actor_critic
        policy.eval()
        
        base_dir = '/'.join(cfg.checkpoint.split('/')[:-1])
        with open(os.path.join(base_dir, "config.json"), "r") as f:
            cfg_dict = json.load(f)
        ckpt_cfg = OmegaConf.create(cfg_dict)
        if ckpt_cfg.task.env.observationType != cfg.task.env.observationType:
            assert False, "Observation type in checkpoint config and current config do not match!"
        
        cprint(f"[Control state based policy] Loaded checkpoint {cfg.checkpoint} for test.", "green")
        assert 'affordance' in cfg.task.env.observationType
        cprint("[Control state based policy] Affordance observation is used.", "green")
        assert 'style' in cfg.task.env.observationType
        cprint("[Control state based policy] Style observation is used.", "green")


        style_dim = cfg['hand']['num_obs_dict']['style']

        obj_name = env.env.object_names[0]
        this_run_dir = os.path.join(cfg.task.demo.root_path,"state_based_any_style" ,obj_name)
        os.makedirs(this_run_dir, exist_ok=True)
        style_list = cfg['task']['func']['style_list']
        for s in style_list:
            cprint(f"=== Starting style {s} ===", "yellow")
            desired_style = s  # change this to try different styles
            for round in range(NUM_ROUNDS):
                output_dir = os.path.join(this_run_dir,f"style_{desired_style}_round_{round}")
                os.makedirs(output_dir, exist_ok=True)
                ### rollout this round
                
                env.env.manually_set_style_labels = torch.tensor([desired_style]*env.env.num_envs, device=env.env.device)

                obs=env.env.reset_idx(torch.arange(env.env.num_envs))['obs']
                episode_data_buffer = []
                mouse_event = {"clicked": False, "x": None, "y": None, "button": None}
                image_container = {"img": None}  # 将存放用于显示的 BGR 图像（会在点击时被修改）

                win_name = "camera_rgb"
                cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

                def on_mouse(event, x, y, flags, param):
                    # 左键点击：记录坐标并在 image_container['img'] 上画圈
                    if event == cv2.EVENT_LBUTTONDOWN:
                        mouse_event["clicked"] = True
                        mouse_event["x"] = x
                        mouse_event["y"] = y
                        mouse_event["button"] = "L"
                        if image_container["img"] is not None:
                            # 画一个半径为5的圆（BGR color=(0,0,255) 红色），厚度=2
                            cv2.circle(image_container["img"], (x, y), 5, (0, 255, 0), 2)

                # 绑定回调
                cv2.setMouseCallback(win_name, on_mouse)
                obs_dict = env.get_real_observations()
                eg_rgb = obs_dict[f"camera_{env.env.camera_ids[0]}.rgb"][0]  # H,W,3 RGB
                eg_rgb = eg_rgb.astype(np.uint8)
                eg_bgr = eg_rgb[..., ::-1].copy()
                image_container["img"] = eg_bgr
                h, w = eg_bgr.shape[:2]
                cv2.resizeWindow(win_name, w, h)
                cv2.imshow(win_name, image_container["img"])
                cv2.waitKey(1)

                # 阻塞循环：等待点击或按键
                print("等待你在窗口中点击一个点（或按 q 继续 / ESC 退出）...")
                while True:
                    k = cv2.waitKey(50) & 0xFF  # 50ms 轮询，保持窗口响应
                    if mouse_event["clicked"]:
                        break
                    if k == ord('q'):
                        print("按键 q：继续仿真（未点击）")
                        break
                    if k == 27:
                        print("按 ESC：退出程序")
                        break
                print(f"最终选择的像素坐标: ({mouse_event['x']}, {mouse_event['y']})")
                afford_xy = np.array([[mouse_event['x'], mouse_event['y']]], dtype=np.float32)
                cv2.imshow(win_name, image_container["img"])
                cv2.waitKey(50)
                cv2.destroyAllWindows()
                filename = f"humman_affordance.png"
                filepath = os.path.join(output_dir, filename)
                cv2.imwrite(filepath, image_container["img"])

                afford_xyz_tensor = project_from_image_to_obj_pcl(
                    torch.as_tensor(afford_xy, device=env.env.device),
                    env.env.root_state_tensor[env.env.camera_pad_indices[0][torch.arange(env.env.num_envs)], 0:7],
                    env.camera_intrinsic_matrices,
                    env.env.transformed_pcl[:, : , :3].clone(),
                    env.env.device
                ) # [num_envs, 3] 
                # print("pcl affordance content",env.env.transformed_pcl[torch.arange(env.env.num_envs),env.env.afford_idx,:3][:,:3])
                print("****")

                obs[:,14:17] = afford_xyz_tensor.clone() # replace affordance xy with affordance xyz


                with torch.no_grad():
                    plan = policy(obs, inference=True)
                    env.env.generate_reaching_plan_idx(torch.arange(env.env.num_envs), actions=plan)
                print("Affordance xy",afford_xyz_tensor[0])
                # draw affordance points
                # print("Affordance 3D point in world frame: ", afford_xyz_tensor)
                if cfg['vis_rgb'] and cfg.headless == False:
                    for i in range(env.env.num_envs):
                        env.env.draw_sphere(pos=afford_xyz_tensor[i], radius=0.01, color=(0, 1., 0.), env_id=i)
                
                writers = {}
                fps = 10
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                if cfg.task.demo.enable:
                    for name in range(len(cams)):
                        out_path = os.path.join(output_dir, f"{name}.mp4")
                        writers[name] = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
                        # print("writer opened:", writers[name].isOpened())

                env.env.speed_up = True # speed up the sim by skipping pcl calculation
                for t in range(env.env.max_episode_length):

                    if cfg.task.demo.enable:
                        for name in range(len(cams)):
                            # IMAGE_COLOR returns HxWx4 BGRA uint8 in many Isaac Gym versions
                            img = gym.get_camera_image(env.env.sim, env.env.env_ptr_list[0], cams[name], gymapi.IMAGE_COLOR)
                            if img is None:
                                print(f"[warn] camera {name} returned None at step {i}")
                                continue
                            # Depending on API, img may be a 1D bytes buffer - reshape:
                            img_np = np.frombuffer(img, dtype=np.uint8)
                            img_np = img_np.reshape((height,width , 4))
                            # cv2 expects BGR; convert back
                            bgr = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
                            writers[name].write(bgr)

                        # 获取图像数据
                    action = env.env.compute_reference_actions()
                    #print(action.shape)
                    env.env.step(action)            
                env.env.speed_up = False
                if cfg.task.demo.enable:
                    for w in writers.values():
                        print("Releasing writer:", w.isOpened())
                        w.release()
            ### save successful episodes


    elif cfg["debug"]=="test_visual_policy":
        '''
        This is only an example test function to test the visual observation and policy deployment.
        You should modify this function to fit your specific policy and observation format. The main point is to show how to get the visual observations, process them into the affordance representation, and feed them into the policy for action inference.
        '''
        raise NotImplementedError("This is a test function for visual policy deployment. You should modify it according to your specific policy and observation format.")
        

if __name__ == "__main__":
    main()
