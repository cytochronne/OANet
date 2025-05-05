#!/usr/bin/env python3

import os
import subprocess
import argparse

def run_extraction_for_all_scenes():
    # 设置基础路径
    base_dir = "../../ETHdataset"
    
    # ETHdataset 下的所有子文件夹列表
    scenes = [
        "botanical_garden", 
        "bridge", 
        "exhibition_hall", 
        "living_room", 
        "observatory", 
        "statue",
        "boulders", 
        "door", 
        "lecture_room", 
        "lounge", 
        "old_computer", 
        "terrace_2"
    ]
    
    print(f"Starting feature extraction for {len(scenes)} scenes in ETHdataset...")
    
    # 为每个场景运行特征提取脚本
    for scene in scenes:
        scene_path = os.path.join(base_dir, scene)
        
        # 检查场景目录是否存在
        if not os.path.exists(scene_path):
            print(f"Warning: Scene directory {scene_path} does not exist. Skipping.")
            continue
            
        print(f"Processing scene: {scene}")
        
        # 构建命令行参数
        cmd = [
            "python", 
            "ETH_extract_feature.py",
            "--input_path", scene_path,
            "--img_dir", "images/dslr_images_undistorted",
            "--img_ext", ".jpg",
            "--num_kp", "1000",
            "--suffix", "sift-1000"
        ]
        
        # 运行命令
        try:
            subprocess.run(cmd, check=True)
            print(f"Successfully processed scene: {scene}")
        except subprocess.CalledProcessError as e:
            print(f"Error processing scene {scene}: {e}")
    
    print("All scenes processed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run feature extraction for all ETH scenes.')
    parser.add_argument('--base_dir', type=str, default='../../ETHdataset',
                        help='Base directory containing all scene folders (default: ../../ETHdataset)')
    
    args = parser.parse_args()
    
    # 更新基础目录（如果提供）
    base_dir = args.base_dir
    
    run_extraction_for_all_scenes()