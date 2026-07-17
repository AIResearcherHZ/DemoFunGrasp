# viz_demo — 在 MuJoCo 里回放参考示教 pkl

把 `tasks/grasp_ref_*.pkl` 里的单条人类示教轨迹在 MuJoCo 窗口中回放:机器人接近物体 → 跟踪示教 → 抓取抬升。**零编辑**(不加策略动作、不做 style/afford 修改),直接把机器人驱动到 pkl 的每一帧。

支持在不同手(embodiment)间**随便换**:按手的配置(`tasks/hand/*.yaml`)通用建模,不再写死 inspire。

## 用法

在项目根目录 `/home/xhz/DemoFunGrasp` 下运行,**pkl 直接作为第一个参数**,手会按 `hand_qpos` 维度自动匹配:

```bash
# 默认 inspire pkl
python viz_demo/replay_mujoco.py

# 直接甩一个 pkl,自动识别是哪只手
python viz_demo/replay_mujoco.py tasks/grasp_ref_shadow.pkl
python viz_demo/replay_mujoco.py tasks/grasp_ref_inspire.pkl

# 换物体(assets/union_object_dataset/urdf/ 下的名字,去掉 .urdf)
python viz_demo/replay_mujoco.py tasks/grasp_ref_shadow.pkl --object 006_mustard_bottle

# 打开即自动回放一次
python viz_demo/replay_mujoco.py tasks/grasp_ref_inspire.pkl --auto

# 维度无法唯一确定时,显式指定手
python viz_demo/replay_mujoco.py my_ref.pkl --hand shadow
```

窗口内按键:
- **空格**:回放一次示教轨迹
- **退格**:复位

## 参数

- `pkl`(位置参数):参考轨迹 pkl 路径,默认 inspire ref
- `--hand`:`inspire` / `shadow`,覆盖自动识别(按 `hand_qpos` 维度推断:6→inspire,18→shadow)
- `--object`:物体名,默认 `011_banana`
- `--obj-pos`:物体放置 xyz,默认 `0.55 0.0 0.13`
- `--obj-yaw`:物体绕 z 朝向(rad),默认 0
- `--auto`:启动即自动回放一次

## 关键:hand_qpos 的关节顺序

pkl 里 `hand_qpos` 每一列对应哪个手指关节,由 **IsaacGym 生成 pkl 时的 asset dof 顺序**决定。这个顺序既不是字母序、也不是 URDF 文档序,只能由 IsaacGym 给出(env 初始化时会打印 `Active hand dof names`)。本脚本已内置两只手的**权威顺序**(`HANDS[...]["active_hand"]`):

- **inspire**(6):`index_1, little_1, middle_1, ring_1, thumb_1, thumb_2`
- **shadow**(18):`FFJ4,FFJ3,FFJ2, LFJ5,LFJ4,LFJ3,LFJ2, MFJ4,MFJ3,MFJ2, RFJ4,RFJ3,RFJ2, THJ5,THJ4,THJ3,THJ2,THJ1`

要新增一只手:先跑一次 IsaacGym 拿到该手的顺序,再填进 `HANDS`:

```bash
python run_demofungrasp.py task=grasp train=PPOOneStep hand=<HAND> num_envs=2 headless=True \
  task.env.enablePointCloud=False task.env.observationType="eefpose+objpose" \
  task.env.asset.multiObjectList="union_object_dataset/small_debug_set.yaml" \
  +debug=replay_demo 2>&1 | grep "Active hand dof names"
```

## 回放逻辑

1. 读手的 yaml(URDF、arm 关节、eef/palm/指尖、mimic 被动关节),按需通用地构建 MuJoCo 模型(浮动基座的零质量虚拟体用 `boundmass/boundinertia` 兜底)。
2. 手型轨迹直接用 pkl 原始 `hand_qpos`,按权威顺序灌到主动手关节;被动关节按 mimic 系数联动。
3. 逐帧执行:先线性/球面插值 **接近** 到示教起点,再 **跟踪** 示教手腕位姿 + 手型;手腕经阻尼最小二乘 IK 转关节目标。
4. 结束打印成功判定(物体抬升 `delta_z > 0.1` 且指尖/掌心接触)。

无头自检(不开窗口)结果:
```
inspire: object=011_banana T_ref=22 reach=7 success=True delta_z=0.225
shadow : object=011_banana T_ref=19 reach=4 success=True delta_z=0.265
```

## 说明

- 复用 `sim2sim/` 的通用建模辅助函数;沿用其物理参数(SIM_DT/decimation/接触参数)。
- 需要图形界面显示窗口;纯逻辑自检可用 `MUJOCO_GL=egl` 直接调用 `ReplayEnv` + `replay_once`。

## pkl 数据结构

| 键 | 形状 | 含义 |
|---|---|---|
| `wrist_initobj_pos` | (T,3) | 手腕位置(初始物体坐标系) |
| `wrist_quat` | (T,4) | 手腕姿态四元数 (x,y,z,w) |
| `hand_qpos` | (T,ndof) | 机器人手主动关节角(inspire=6,shadow=18) |
| `obj_initobj_pos` | (T,3) | 物体位置(初始物体坐标系) |
