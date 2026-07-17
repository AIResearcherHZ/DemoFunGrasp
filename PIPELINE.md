# DemoGrasp / DemoFunGrasp 完整 Pipeline 技术文档

本文件整理 `DemoGrasp`(ICLR 2026,通用灵巧抓取)与 `DemoFunGrasp`(CVPR 2026,通用功能性抓取)两个代码库的端到端流程。所有结论基于代码,带 `文件:行号` 锚点与确切数值。

- DemoGrasp 路径:`/home/xhz/DemoGrasp`
- DemoFunGrasp 路径:`/home/xhz/DemoFunGrasp`

---

## 0. 两个仓库的关系

| | DemoGrasp | DemoFunGrasp |
|---|---|---|
| 论文 | Universal Dexterous Grasping from a **Single** Demonstration (ICLR 2026) | Universal Dexterous **Functional** Grasping via Demonstration-Editing RL (CVPR 2026) |
| 核心问题 | 把物体抓起来(能否抓稳) | 抓起来 **且** 按指定功能/风格抓在正确位置(afford + style) |
| 算法框架 | 单条示教编辑 + one-step PPO | 同框架,**扩展** afford/style 条件、更大动作空间、多路 shaping 奖励 |
| 奖励 | 纯二值成功 `reward_binary` | 二值成功 `reward_resdex` + 可选 afford/close/contact/qpos shaping |
| 观测 | `eefpose+objinitpose+objpcl` | 追加 `affordance+style`,并有更全的状态项 |
| 手 embodiment | 7 种(inspire/shadow/allegro/svh/panda/dclaw/fr3_shadow) | 主要 inspire、shadow |
| 部署 | IsaacGym 内回放 + 真机数据集 | 新增 `sim2sim/` MuJoCo 部署 + 遥操作定向抓取 |

DemoFunGrasp 在 DemoGrasp 之上做功能性扩展,主干(one-step MDP + 示教编辑 + 脚本控制器 + PPO)完全一致。下文先讲共同主干,再分别讲各自差异。

---

## 1. 核心思想:示教编辑式 one-step RL

把整段抓取压成 **一次决策(one-step MDP / contextual bandit)**:

1. 加载 **一条** 人类示教参考轨迹(每种手一条,离线预制)。
2. 策略看观测,输出 **一个** 动作 = 「如何编辑这条示教」(手腕位姿变换 + 手型编辑)。
3. 用该动作把参考轨迹整体编辑一遍。
4. 一个 **脚本控制器**(非学习)把编辑后的轨迹在仿真里跑完:接近 → 跟踪(抓取)→ 抬升。
5. episode 结束判定「有没有抓起来」,把这个成功信号当作那一个动作的奖励回传 PPO。

策略在一个 episode 内只决策一次,后续每步的底层控制都是脚本算出来的。因此 `nsteps=1`(`tasks/train/PPOOneStep.yaml`)。

---

## 2. 人类示教数据(reference demo)

### 2.1 存储形态

每种手一条 pkl,直接随仓库附带。**两个仓库都没有采集/生成脚本**,pkl 是离线制作好的产物。

DemoGrasp `tasks/grasp_ref_*.pkl` 共 6 条:

| 文件 | 帧数 T | hand_qpos 维度 |
|---|---|---|
| grasp_ref_panda_gripper.pkl | 19 | 1 |
| grasp_ref_inspire.pkl | 22 | 6 |
| grasp_ref_dclaw_gripper.pkl | 19 | 9 |
| grasp_ref_svh.pkl | 19 | 9 |
| grasp_ref_allegro.pkl | 19 | 16 |
| grasp_ref_shadow.pkl | 19 | 18 |

DemoFunGrasp `tasks/grasp_ref_inspire.pkl`(22 帧,6 dof)、`grasp_ref_shadow.pkl`(19 帧,18 dof)。

### 2.2 字典结构(4 个键)

- `wrist_initobj_pos` (T,3):手腕位置,表达在 **初始物体坐标系**(相对 t=0 物体)
- `wrist_quat` (T,4):手腕姿态
- `hand_qpos` (T,ndof):**已重定向到机器人手** 的关节角序列
- `obj_initobj_pos` (T,3):物体位置(初始物体系,记录抬升阶段物体运动)

加载:`tasks/grasp.py:633`(FunGrasp)。加载后 `unsqueeze(0).repeat(num_envs,1,1)` 广播到所有并行环境。

### 2.3 关键点

- 所有位置存在 **物体初始坐标系**,才能平移旋转后贴到任意物体/摆放 → 一条 demo 通用化的前提。
- `hand_qpos` 维度精确等于每种手主动自由度 → 说明采集链路含 **retarget(人手→机器人手)**,pkl 里已经是机器人关节空间。
- 分段:`trackingReferenceLiftTimestep`(inspire=13,其余多为 11)把轨迹切成「接近+合拢(0..lift-1)」与「抓取+抬升(lift..T-1)」。
- **原始"一次成功抓取"的采集设备/流程未开源**,论文与主页也未披露具体模态(遥操作/MANO 重定向/运动规划均有可能)。

### 2.4 校验方式

- FunGrasp:`scripts/replay_ref_demo.sh` → `+debug=replay_demo`(`run_demofungrasp.py:113`)直接回放,看示例物体成功率。
- DemoGrasp:`+debug=test_demo_replay`(`run_rl_grasp.py:85`)。

---

## 3. 环境(IsaacGym `tasks/grasp.py`)

### 3.1 初始化与仿真参数

- `init_configs`(FunGrasp `grasp.py:133`):读控制器/观测/奖励类型/手配置。
- 观测维度自动求和:`numObservations = Σ num_obs_dict[i] (i in observationType)`(`grasp.py:159`)。
- 动作维度:`numActions = hand_config.numActions`(inspire=13)。
- 仿真:`dt`、`decimation`(`grasp.py:192`),位置控制在 `step` 内按 decimation 线性插值下发(`grasp.py:1561`)。
- `episodeLength`:FunGrasp=40,DemoGrasp=40/50。

### 3.2 reset_idx(每次 rollout 开头对所有 env 调用)

`grasp.py:988`:
- 清空 afford/pcl/style 缓存(`grasp.py:992-996`)。
- 物体随机旋转:`resetRandomRot`(random/z/fixed),随机轴+随机角(`grasp.py:1011-1020`)。
- 物体随机 xyz:`resetPositionRange`(默认 `[[0.3,0.8],[-0.35,0.15],[0.1,0.12]]`)+ 桌高(`grasp.py:1022-1024`)。
- 机器人 dof 随机:`resetDofPosRandomInterval`。
- 可选干扰物(`useDistractorObjects`)。
- 多物体数据集:`multiObjectList` 指定 yaml,环境按对象循环分配。

### 3.3 观测构造 `compute_required_observations`(`grasp.py:1770`)

按 `observationType` 字符串包含关系拼接(每段 unscale 到 [-1,1] 或原值):

| 关键字 | 维度(inspire) | 内容 |
|---|---|---|
| armdof | 7 | 机械臂关节角 |
| handdof | 6 | 手主动关节角 |
| fulldof | 19 | 全 dof |
| eefpose | 7 | 末端位姿 |
| ftpos | 15 | 5 指尖 xyz |
| palmpose | 7 | 掌心位姿 |
| handposerror | 6 | 手关节控制误差 |
| lastact | 13 | 上一动作 |
| objxyz/objpose/objinitpose | 3/7/7 | 物体位置/位姿/初始位姿 |
| pcfeat | 64 | 点云特征(预提取) |
| refaction | 13 | 参考动作 |
| objpcl | 1536 | 512×3 物体点云 |
| **affordance** | 3 | afford 点 xyz(FunGrasp) |
| **style** | 4 | style one-hot(FunGrasp) |

- 点云:`objpcl` 把物体点云变换到世界系(`transform_obj_pcl_2_world` `grasp.py:1899`),拼进 obs 末尾;网络里用 PointNet 压成 embedding。
- 训练常用 obs(FunGrasp):`eefpose+objinitpose+objpcl+affordance+style`;DemoGrasp:`eefpose+objinitpose+objpcl`。

### 3.4 动作空间与「示教编辑」`generate_reaching_plan_idx`(`grasp.py:1305`)

策略输出动作(∈[-1,1],tanh 压过),含义:

- `action[0:3]`:手腕 **平移偏移**(× `randomizeTrackingReferenceRange[0:3]`,默认 0.05m)。
- `action[3:6]`:手腕 **旋转**(欧拉角 × range[3:6],默认 1.57rad),对 demo 手腕轨迹 **左乘旋转**(`grasp.py:1320-1326`)。
- 抬升段保持与 demo 一致的相对运动(`grasp.py:1332`)。
- 抓取手型编辑(二选一):
  - `randomizeGraspPose=True`:`action[6:6+ndof]` 直接改抓取关节角(`grasp.py:1344`)。
  - `'style' in obs`(FunGrasp 功能性):由 style label 取 `static_style` 基准手型,再用 `action` 的 delta/scale 编辑:
    - `if_use_qpos_scale`:`action[-1]` → scale∈`scale_limit`(默认[0.1,1.9])(`grasp.py:1374`)。
    - `if_use_qpos_delta`:`action[6:-1]` → delta∈`qpos_delta_scale`(`grasp.py:1379`)。
    - 合成 `style_hand_qpos = base*scale + delta`(`grasp.py:1384`)。
- 编辑完用 `batch_linear_interpolate_poses`(max_trans_step=0.04, max_rot_step=0.1)生成从当前末端到 demo 起点的 **接近计划**(`grasp.py:1428`)。

FunGrasp 动作维度(`run_demofungrasp.py:36`):基础 6,style+delta 时 +ndof,style+scale 时 +1 → inspire 功能性=13。

### 3.5 脚本控制器 `compute_reference_actions`(`grasp.py:1439`)

每个物理步:根据 `progress_buf` 判定处于 **接近段** 还是 **跟踪段**:
- 接近段:沿接近计划插值目标手腕位姿。
- 跟踪段:取编辑后 demo 的对应帧(手腕位姿 + 手型)。
- 手腕目标 → arm 控制:`armController`(默认 `pose`),经 `compute_arm_ik`(`grasp.py:1957`,阻尼最小二乘)或 delta pose(worlddpose/eedpose)转关节目标。
- 返回底层 action,喂给 `env.step`。

`armController` 取值:`qpos`/`worlddpose`/`eedpose`/`pose`(`grasp.py:151`)。

### 3.6 step(`grasp.py:1549`)

`pre_physics_step` 算目标 → decimation 次线性插值下发位置目标 → `mj/gym simulate` → `post_physics_step` 算 obs/reward/reset。返回 `(obs, rew_buf, reset_buf, extras)`。

### 3.7 状态编解码

`encode_init_state`(`grasp.py:2052`)/`decode_and_set_init_state`(`grasp.py:2075`):序列化/恢复 sim 初始状态(dof、桌/垫/物体 root state),用于视觉渲染时复现同一初始局面。

---

## 4. 奖励定义

### 4.1 DemoGrasp:纯二值成功 `reward_binary`(`tasks/reward.py:3`)

- `object_delta_z = object_pos.z - object_init.z`
- `flag = (指尖到物距 <= 0.12*num_fingers) or (掌心到物距 <= 0.15)`
- `successes = 1 iff (delta_z > 0.1) and flag`(`reward.py:39`)
- `reward = current_successes.float()`(`reward.py:49`)——**纯二值**。

### 4.2 DemoFunGrasp:`reward_resdex` + shaping(`tasks/reward.py:3`)

`reward_resdex` 内部有稠密项:
```
reward = -0.5*指尖到物距 - 掌心到物距 + 3.0*lift_object
```
但在 one-step PPO 里,**`env.step` 返回的这个稠密 reward 被丢弃**(`algo/ppo_onestep/ppo.py:441` 的 `reward` 不入 storage)。真正用的是它顺带算出的 `successes`(判据同上:`delta_z>0.1 且 flag`,`reward.py:53`)。

训练奖励在 PPO 循环里组装(`ppo.py:449` 起):
```
rews = successes.clone()                       # 基础:二值成功
+ afford_reward   (use_affordance_reward)       # exp(-afford_dist)*scale,成功且距离<clip
+ close_reward    (if_use_close_reward)         # 接近段最小 afford 距离<阈值给分
+ contact_reward  (if_use_contact_reward)       # 指尖接触模式与 style GT 一致度 exp(sim/5)*scale
+ qpos_reward     (if_use_qpos_reward)          # exp(-||编辑手型-原style手型||)*scale
```
对应实现:`calcu_affordance_rewards`(`grasp.py:2234`)、`calcu_close_rewards`(`grasp.py:2268`)、`calcu_contact_rewards`(`grasp.py:2248`)、`calcu_qpos_rewards`(`grasp.py:2063`)。afford 距离 `calcu_affordance_dist`(`grasp.py:2197`)按 `style_point_type`(mid_thumb_index / mean_contact_ft / centroid_contact_ft 等)算指尖到 afford 点的距离。

config 默认全 `False`(`tasks/task/grasp.yaml:14-45`);实际训练脚本 `scripts/train_inspire_bash.sh` 打开:`use_affordance_reward=True`(clip=4, scale=2)、`if_use_close_reward=True`(阈值0.03, scale0.5)、`if_use_qpos_reward=True`(scale0.1)。

### 4.3 metric(评测,不是奖励)

`func.metric`(如 `succ_rate+afford_dist+style_accuracy`)在 PPO 循环打印/统计:成功率、afford 距离、style 准确率(`find_similar_style_label` 余弦相似)、qpos 距离、style 多样性、contact 相似度(`ppo.py:453-505`)。

---

## 5. PPO 算法(`algo/ppo_onestep/`)

### 5.1 rollout 组织成 one-step(`ppo.py:run`)

每个 learning iteration:
1. `reset_idx(all)` → 拿 obs(`ppo.py:427`)。
2. `actor_critic(obs)` → 一个 action + logprob + value(`ppo.py:433`)。
3. `generate_reaching_plan_idx(all, actions)` 编辑 demo(`ppo.py:434`)。
4. `dones = ones`(每步后强制 reset)(`ppo.py:435`)。
5. for t in max_episode_length:`compute_reference_actions` → `step`(脚本跑完整段)(`ppo.py:439`)。
6. 在 `t == L-2` 取 `successes` 组装 `rews`(`ppo.py:447`)。
7. `storage.add_transitions(obs, states, actions, rews, dones, values, logprob, mu, sigma)`(`ppo.py:525`)。
8. `compute_returns`(GAE,gamma/lam)→ `update()`(`ppo.py:549`)。

因为 `nsteps=1` 且 `dones` 恒为 1,GAE 退化,本质是「obs→action→单步 return=成功率」的优势估计。

### 5.2 网络 `ActorCritic`(`algo/ppo_onestep/module.py:99`)

- 视觉分支(`use_pcl=is_vision=True`):obs 末尾 `pc_shape=[512,3]` 点云 → `PointNetBackbone`(`module.py:32`)→ `pc_emb_dim=128` embedding,替换回 obs。
- Actor MLP:`pi_hid_sizes=[1024,1024,512,512]`,激活 elu,输出 pre-tanh mean。
- Critic MLP:`vf_hid_sizes=[1024,1024,512,512]` → 标量 value。
- 动作分布:对角高斯 + **tanh squash**,`log_std` 为可学习参数(`module.py:155`),`init_noise_std=0.8`。log_prob 带换元雅可比修正(`module.py:84`)。
- 正交初始化(SB 风格,末层 gain 0.01/1.0)。
- 对称 actor-critic(`asymmetric=False`,critic 用 obs 而非独立 state)。

### 5.3 update(`ppo.py:635`)

标准 PPO-clip:surrogate(clip 0.2)+ `value_loss_coef=2.0`·value_loss - `ent_coef=0`·entropy。自适应 lr(`schedule=adaptive`,`desired_kl=0.016`,按 KL 在 [1e-5,1e-2] 调 lr)。`max_grad_norm=1`。

### 5.4 关键超参(`tasks/train/PPOOneStep.yaml`,两仓库几乎一致)

| 参数 | 值 |
|---|---|
| nsteps | 1 |
| noptepochs | 5 |
| nminibatches | 4 |
| max_iterations | 20000 |
| cliprange | 0.2 |
| gamma / lam | 0.96 / 0.95 |
| optim_stepsize | 3e-4(自适应) |
| desired_kl | 0.016 |
| ent_coef | 0 |
| value_loss_coef | 2.0 |
| max_grad_norm | 1 |
| init_noise_std | 0.8 |
| pi/vf_hid | [1024,1024,512,512] |
| pc_shape / pc_emb_dim | [512,3] / 128 |
| is_vision | FunGrasp True / DemoGrasp False(脚本里覆盖为 True) |
| save_interval | FunGrasp 400 / DemoGrasp 100 |
| num_envs(训练脚本) | DemoGrasp 7000 / FunGrasp inspire 1000 |

---

## 6. DemoGrasp 特有:多 embodiment 与真机数据

### 6.1 7 种手 embodiment(`tasks/hand/*.yaml`)

fr3_inspire_tac、shadow_simple、fr3_shadow、ur5_allegro、ur5_svh、fr3_panda_gripper、fr3_dclaw_gripper。每种手有各自 urdf、arm/hand dof 名、num_obs_dict、numActions、对应 `grasp_ref_*.pkl` 与 lift timestep。`train.sh` / `play_policy.sh` 逐一给出命令行(num_envs=7000 训练,175 测试)。

### 6.2 debug 模式(`run_rl_grasp.py`)

- `test_demo_replay`(`:85`):回放参考轨迹测成功率。
- `collect_real_dataset`(`:96`):用训练好的策略 rollout,`data/lerobot` 写 Lerobot 格式数据集(带相机图像),`+num_episodes` 指定成功轨迹数。渲染观测由相机采集,`render_data_type` 控制 rgb/depth/pcl。

---

## 7. DemoFunGrasp 特有

### 7.1 功能性条件:afford + style

- **style**:`static_style.npy`(`dataset_processor/inspire_static_style_cali.npy` 等)存每个 style 的基准手型 + 5 指接触标记(`[...ndof..., 5]`)。style label → one-hot 进 obs,同时决定编辑基准手型。`style_list`(inspire `[0,1,2,3]`,shadow `[0..8]`),`num_style_obs`。
- **affordance**:点云上按分数采样一个 afford 点(高度分 + 法向朝上分),其 xyz 进 obs;奖励鼓励指尖/接触指靠近该点。`affordanceType="score_based"`(`grasp.yaml:13`),`functional_generator`(`grasp.py:118`)。
- 同一物体可通过改 style/afford 产生 **多样的功能性抓取**。

### 7.2 观测/动作/奖励差异

- obs 训练用 `eefpose+objinitpose+objpcl+affordance+style`。
- 动作维度扩到 13(6 + 6 delta + 1 scale)。
- 奖励见 §4.2(afford/close/contact/qpos shaping)。

### 7.3 视觉数据集与状态策略部署(`to_vision.py`)

- `+debug=collect_dataset`(`to_vision.py:238`):部署策略,投影 afford 到图像,写 Lerobot(`lerobot_dataset`)。脚本 `scripts/collect_vision_dataset.sh`。
- `+debug=control_state_based_policy`(`to_vision.py:364`):单环境交互式部署,鼠标点选 afford,可视化 rgb。脚本 `scripts/state_based_demo.sh`。
- `README`:采集约 30k 轨迹后可用任意策略做模仿学习。

### 7.4 sim2sim:MuJoCo 部署(`sim2sim/`)

新增独立的 MuJoCo 部署栈(非 IsaacGym),把训练好的策略搬到 MuJoCo:

- `MujocoGraspEnv`(`sim2sim/grasp_env.py:16`):
  - 感知:`CameraCloudSensor`(`perception.py:38`)从相机取点云,`estimate_center` 估物体中心,`sample_affordance`(`grasp_env.py:83`)按高度+法向分采 afford 点。
  - obs 组装:`compute_observation`(`grasp_env.py:96`)= eef 位姿 + 估计物体位姿 + afford + style one-hot + 点云 xyz。
  - 编辑与计划:`generate_plan`(`grasp_env.py:142`,标准编辑)与 `generate_directed_plan`(`:146`,遥操作定向:用当前手方位作为接近方向,把 demo 抓取朝向对齐到该方向)。`_hand_plan`(`:105`)复刻 delta/scale 手型编辑(delta∈[-0.05,0.05],scale∈[0.1,1.9])。
  - 控制:`control_step`(`:194`)DLS IK(`arm_ik` `:184`)+ decimation 位置插值 + mimic 关节耦合。
  - 成功判定:`check_success`(`:213`)delta_z>0.1 且(指尖距≤0.6 或掌心距≤0.15),与训练判据同源。
- 策略侧:`sim2sim/policy.py` 的 `SimplePointNet` + `GraspActor`(`act_inference`),`load_policy` 加载 checkpoint。
- 入口:`sim2sim/run_sim2sim.py`(自动跑)、`sim2sim/run_teleop.py` 与根目录 `teleop_mujoco.py`(遥操作:拖 mocap 方块控 EEF,空格触发定向抓取)。

---

## 8. 端到端流程总览

```
[离线一次] 人手成功抓取 → retarget 到机器人手 → 转物体初始坐标系 → 分段 → grasp_ref_*.pkl
        │(每种手一条,仓库自带,无采集脚本)
        ▼
[训练] IsaacGym 并行环境
  reset_idx(随机物体/位姿/style/afford)
     → obs(eefpose+objinitpose+objpcl[+afford+style])
     → ActorCritic 输出 1 个编辑动作(EEF 变换 + 手型 delta/scale)
     → generate_reaching_plan_idx 编辑 demo
     → for t: compute_reference_actions(脚本) → step(接近→跟踪→抬升)
     → 末尾取 successes 作奖励(+ FunGrasp 的 afford/close/contact/qpos shaping)
     → one-step PPO 更新(nsteps=1, GAE, clip, 自适应lr)
        │
        ▼
[评测] test 模式跑 5 次,统计 succ_rate / afford_dist / style_accuracy ...
        │
        ▼
[数据集] 部署策略 rollout → Lerobot 格式(rgb/pcl)→ 30k 轨迹 → 模仿学习
        │
        ▼
[部署] IsaacGym 内测试 / DemoFunGrasp sim2sim MuJoCo(含遥操作定向抓取)/ 真机
```

---

## 9. 关键文件索引

| 功能 | DemoFunGrasp | DemoGrasp |
|---|---|---|
| 入口 | `run_demofungrasp.py` | `run_rl_grasp.py` |
| 环境 | `tasks/grasp.py` | `tasks/grasp.py` |
| 奖励 | `tasks/reward.py`(resdex) | `tasks/reward.py`(binary) |
| PPO | `algo/ppo_onestep/ppo.py` | `algo/ppo_onestep/ppo.py` |
| 网络 | `algo/ppo_onestep/module.py` | `algo/ppo_onestep/module.py` |
| 训练超参 | `tasks/train/PPOOneStep.yaml` | `tasks/train/PPOOneStep.yaml` |
| 任务配置 | `tasks/task/grasp.yaml` | `tasks/task/grasp.yaml` |
| 手配置 | `tasks/hand/*.yaml` | `tasks/hand/*.yaml`(7 种) |
| 示教 | `tasks/grasp_ref_{inspire,shadow}.pkl` | `tasks/grasp_ref_*.pkl`(6 种) |
| 视觉/数据集 | `to_vision.py`, `lerobot_dataset/` | `run_rl_grasp.py`(collect_real_dataset), `data/lerobot/` |
| 部署 | `sim2sim/`, `teleop_mujoco.py`, `real2sim/` | IsaacGym 内 `play_policy.sh` |
| 数据预处理 | `dataset_processor/` | — |
| 训练脚本 | `scripts/train_{inspire,shadow}_bash.sh` | `train.sh` |
| 测试脚本 | `scripts/vis_inspire_bash.sh` | `play_policy.sh` |

---

## 10. 需要注意的坑

1. **稠密 reward_resdex 的稠密项在训练中不生效**:one-step PPO 只取 `successes`,`step` 返回的 `rew_buf` 被丢弃。
2. **一条 demo 打天下**:每种手仅一条参考轨迹,泛化完全靠 RL 编辑 + 域随机化,不依赖大规模示教。
3. **原始 demo 采集方式未开源**:pkl 已是 retarget 后的机器人关节轨迹,采集设备/流程需查论文或问作者。
4. **style 与 randomizeGraspPose 互斥**(`grasp.py:1341`):功能性(style)模式下不能同时随机抓取手型。
5. **shaping 奖励默认关**:`tasks/task/grasp.yaml` 里 afford/close/contact/qpos 奖励默认 False,需在训练脚本显式打开。
