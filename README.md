# DemoFunGrasp: Universal Dexterous Functional Grasping via Demonstration-Editing Reinforcement Learning



<div align="center">

<p align="center">
  <a href="https://beingbeyond.github.io/DemoFunGrasp/"><img src="https://img.shields.io/badge/🌐-Website-blue?style=flat-square"></a>
  <a href="https://arxiv.org/abs/2512.13380"><img src="https://img.shields.io/badge/📄-arXiv-red?style=flat-square"></a>
  <img src="https://img.shields.io/badge/Python-3.8-blue?style=flat-square&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/License-MIT-blue?style=flat-square">
</p>

The official pytorch implementation of: *DemoFunGrasp: Universal Dexterous Functional Grasping via Demonstration-Editing Reinforcement Learning* (CVPR 2026)

![](docs/images/overview.png)



</div>

DemoFunGrasp is a reinforcement learning framework for universal dexterous functional grasping. The learned policy generalizes to unseen combinations of objects and functional grasping conditions, and achieves zero-shot sim-to-real transfer. For the same object, the policy can produce diverse grasps by adjusting the grasping style and affordance.



## Installation

### Reinforcement Learning Environment Setup

It is recommended to use a conda environment to manage dependencies:

```bash
conda create -n demofungrasp python=3.8.19
conda activate demofungrasp

# For CUDA 11.8; you can change the version according to your GPU
pip install torch==2.3.0+cu118 torchvision==0.18.0+cu118 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118
```

Install Isaac Gym and IsaacGymEnvs:

```bash
# Download IsaacGym_Preview_4_Package.tar.gz from the NVIDIA website
tar -zxvf IsaacGym_Preview_4_Package.tar.gz
cd ./isaacgym/python
pip install -e .
cd ../../

git clone https://github.com/isaac-sim/IsaacGymEnvs.git
cd IsaacGymEnvs/
pip install -e .
```

Install other required Python packages:

```bash
pip install -r requirements.txt
```

### Download Object Dataset and Textures

Download the object assets and textures from [here](https://drive.google.com/drive/folders/1VOh1H9KMOal5S6A_NsCT1EeGLqe7rga4?usp=sharing) and unzip them into the directory `./assets/`.

### Run the Pretrained Policy

We provide checkpoints in the `checkpoint` directory for both ShadowHand and Inspire Hand. You can run:

```bash
bash script/train_inspire_bash.sh
# For shadow hand, run:
bash scripts/train_shadow_bash.sh
```

You can modify `num_envs` to fit your GPU memory and `multiObjectList` to select a subset of objects.

---

## Training and Testing

### Training

After modifying `num_envs=5000`, setting `test=False`, and changing `+run_name` to your desired experiment name, you can train your own policy by running:

```bash
bash script/train_inspire_bash.sh
```

Use TensorBoard to monitor training progress.

### Testing the State-Based Policy

You can test the trained policy and record demonstrations by running:

```bash
bash scripts/state_based_demo.sh
```

### Collecting Vision Dataset

You can collect RGB observation datasets in various formats. To collect the **Lerobot** format, run:

```bash
bash scripts/collect_vision_dataset.sh
```

After collecting a sufficient number of trajectories (e.g., 30k), you can use any policy to imitate them.

### Using Other Object Datasets

To train and test on other object datasets, you must preprocess the `.stl` or `.obj` files to generate point clouds and additional information. Run the following scripts:

```bash
# Generate point clouds and bounding boxes
python dataset_processor/generate_object_dataset_pcl.py

# Generate object list
python dataset_processor/generate_urdfs_from_meshes.py
```

---

## Acknowledgments

This repository is built upon:

* [IsaacGymEnvs](https://github.com/isaac-sim/IsaacGymEnvs)
* [UniDexGrasp](https://github.com/PKU-EPIC/UniDexGrasp)
* [DemoGrasp](https://github.com/BeingBeyond/DemoGrasp)
* [Lerobot](https://github.com/huggingface/lerobot)

---

## Citation

If you find this work useful, please consider citing:

```bibtex
@article{mao2025universal,
  title={Universal Dexterous Functional Grasping via Demonstration-Editing Reinforcement Learning},
  author={Mao, Chuan and Yuan, Haoqi and Huang, Ziye and Xu, Chaoyi and Ma, Kai and Lu, Zongqing},
  journal={arXiv preprint arXiv:2512.13380},
  year={2025}
}
```


