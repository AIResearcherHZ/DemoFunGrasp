#!/usr/bin/env python

"""
Contains utilities to process raw data format from kuavo rosbag
"""
import re
import warnings
from pathlib import Path
import gc
import pandas as pd
import numpy as np
import tqdm
import shutil
import torch
from datasets import Dataset, Features, Image, Sequence, Value
from typing import Dict

from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION
from lerobot.common.datasets.push_dataset_to_hub.utils import (
    calculate_episode_data_index,
    save_images_concurrently,
    concatenate_episodes,
    get_default_encoding
)
from lerobot.common.datasets.utils import (
    hf_transform_to_torch,
)
from lerobot.common.datasets.video_utils import (
    VideoFrame, 
    encode_video_frames
)

from scripts.common.kuavo_dataset import KuavoRosbagReader


def get_cameras(bag_data: dict) -> list[str]:
    """
    /camera/color/camera_info           : sensor_msgs/CameraInfo
    /camera/color/image_raw             : sensor_msgs/Image     
    /camera/depth/camera_info           : sensor_msgs/CameraInfo
    /camera/depth/image_rect_raw        : sensor_msgs/Image     
    """
    cameras = []
    for k in bag_data.keys():
        if 'camera' in k and len(bag_data[k]) > 0:
            cameras.append(k)
    return cameras


def check_format(raw_dir: Path) -> bool:
    """Check if the raw data format is correct for the kuavo rosbag format
    """
    assert raw_dir.exists()

    leader_file = list(raw_dir.rglob("*.bag"))
    if len(leader_file) == 0:
        raise ValueError(f"Missing bag files in '{raw_dir}'")
    print(f"Found {len(leader_file)} bag files in '{raw_dir}'")
    return True

def load_from_raw(
    raw_dir: Path,
    videos_dir: Path,
    fps: int,
    video: bool,
    episodes: list[int] | None = None,
    encoding: dict | None = None,
):
    """加载原始数据
    raw_dir: 原始数据目录
    videos_dir: 视频处理缓存目录
    fps: 帧率
    video: 是否包含视频
    episodes: 需要加载的episode
    encoding: 编码
    """
    data_dict = {}
    # TODO: load data from raw format
    
    bag_reader = KuavoRosbagReader()
    bag_files = bag_reader.list_bag_files(raw_dir)
    
    num_episodes = len(bag_files)
    ep_dicts = []
    ep_ids = episodes if episodes else range(num_episodes)
    print(f"Processing {len(bag_files)} episodes")
    
    for ep_idx in tqdm.tqdm(ep_ids):
        bag_data = bag_reader.process_rosbag(bag_files[ep_idx])
        ep_dict = {}
        num_frames = len(bag_data['action'])
        
        for img_key in get_cameras(bag_data):
            if video:
                # load all images in RAM
                images = [msg['data'] for msg in bag_data[img_key]]
                imgs_array = np.stack(images, axis=0)
                
                # save the images to disk
                tmp_imgs_dir = videos_dir / "tmp_images"
                save_images_concurrently(imgs_array, tmp_imgs_dir)
                
                # encode images to a mp4 video
                fname = f"{img_key}_episode_{ep_idx:06d}.mp4"
                video_path = videos_dir / fname
                encode_video_frames(tmp_imgs_dir, video_path, fps, **(encoding or {}))
                
                # clean temporary images directory
                shutil.rmtree(tmp_imgs_dir)
                
                # store the reference to the video frame
                ep_dict[img_key] = [
                    {"path": f"videos/{fname}", "timestamp": msg['timestamp']} for msg in bag_data[img_key]
                ]
        
            # 关节轨迹
            ep_dict["observation.state"] = np.array([msg['data'] for msg in bag_data['observation.state']])
            # 关节动作
            ep_dict["action"] = np.array([msg['data'] for msg in bag_data['action']])
            ep_dict["episode_index"] = torch.tensor([ep_idx] * num_frames)
            ep_dict["frame_index"] = torch.arange(0, num_frames, 1)
            ep_dict["timestamp"] = torch.arange(0, num_frames, 1) / fps
            
            # last step of demonstration is considered done
            ep_dicts.append(ep_dict)
        gc.collect()
        
    data_dict = concatenate_episodes(ep_dicts)
    total_frames = data_dict["frame_index"].shape[0]
    data_dict["index"] = torch.arange(0, total_frames, 1)
    
    # fix the data type
    data_dict["observation.state"] = np.array(data_dict["observation.state"])
    data_dict["action"] = np.array(data_dict["action"])
    
    return data_dict


def to_hf_dataset(data_dict, video) -> Dataset:
    """Convert the data to Hugging Face dataset format"""
    features = {}
    keys = [key for key in data_dict if "observation.camera" in key]
    for key in keys:
        if video:
            features[key] = VideoFrame()
        else:
            features[key] = Image()

    features["observation.state"] = Sequence(
        length=data_dict["observation.state"].shape[1], feature=Value(dtype="float32", id=None)
    )
    features["action"] = Sequence(
        length=data_dict["action"].shape[1], feature=Value(dtype="float32", id=None)
    )
    features["episode_index"] = Value(dtype="int64", id=None)
    features["frame_index"] = Value(dtype="int64", id=None)
    features["timestamp"] = Value(dtype="float32", id=None)
    features["index"] = Value(dtype="int64", id=None)
    # features["episode_data_index_from"] = Value(dtype="int64", id=None)
    # features["episode_data_index_to"] = Value(dtype="int64", id=None)
    
    hf_dataset = Dataset.from_dict(data_dict, features=Features(features))
    hf_dataset.set_transform(hf_transform_to_torch)
    return hf_dataset

def from_raw_to_lerobot_format(
    raw_dir: Path,
    videos_dir: Path,
    # cfg: Dict,
    fps: int | None = None,
    video: bool = True,
    episodes: list[int] | None = None,
    encoding: dict | None = None,
):
    """Convert dataset from original raw format to LeRobot format.
    """
    # sanity check
    check_format(raw_dir)
    
    if fps is None:
        fps = 30
    
    data_dict = load_from_raw(raw_dir, videos_dir, fps, video, episodes, encoding)
    hf_dataset = to_hf_dataset(data_dict, video)
    episode_data_index = calculate_episode_data_index(hf_dataset)
    info = {
        "codebase_version": CODEBASE_VERSION,
        "fps": fps,
        "video": video,
    }
    if video:
        info["encoding"] = get_default_encoding()
    
    return hf_dataset, episode_data_index, info



if __name__ == '__main__':
    bag_dir = '/Users/wason/Code/RobotEmbodiedData/lerobot/data/testcamera2/'
    video_dir = '/tmp'
    fps = 30
    video = True
    cfg = {}
    
    
    # sanity check
    check_format(Path(bag_dir))
    
    data_dict = load_from_raw(Path(bag_dir), Path(video_dir), fps, video, None, None)
    hf_dataset = to_hf_dataset(data_dict, video)
    
    # data_dict = load_from_raw(raw_dir, videos_dir, fps, video, episodes, encoding)
    # hf_dataset = to_hf_dataset(data_dict, video)
    # episode_data_index = calculate_episode_data_index(hf_dataset)



    