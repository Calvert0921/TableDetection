import os
import cv2
import numpy as np
from tqdm import tqdm
import re

# 使用与原始文件相同的相机参数
# 原来 MIT 数据的相机内参
# fx, fy = 570.3422047415297, 570.3422047415297
# cx, cy = 320.0, 240.0

# ✅ 替换为 RealSense 相机参数
fx = 425.9412841796875
fy = 425.49493408203125
cx = 428.4195251464844
cy = 243.318359375

NUMPOINTS = 1024

# 从dataset_build.py复用的函数
def preprocess_pointcloud(points, num_points=1024):
    if len(points) < num_points:
        idx = np.random.choice(len(points), num_points, replace=True)
    else:
        idx = np.random.choice(len(points), num_points, replace=False)
    points = points[idx, :]

    centroid = np.mean(points, axis=0)
    points -= centroid

    furthest_dist = np.max(np.linalg.norm(points, axis=1))
    points /= furthest_dist

    return points

def depth_to_pointcloud(depth_raw):
    h, w = depth_raw.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth_raw
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    mask = (z.reshape(-1) > 0) & np.isfinite(points).all(axis=1)
    return points[mask]

def read_selected_pairs(pair_file):
    """读取selected_pairs.txt获取图像对应关系"""
    table2_pairs = []
    table3_pairs = []
    
    current_table = None
    with open(pair_file, 'r') as f:
        for line in f:
            line = line.strip()
            if "Table2 selected pairs:" in line:
                current_table = "table2"
            elif "Table3 selected pairs:" in line:
                current_table = "table3"
            elif "->" in line and current_table:
                parts = line.split(" -> ")
                if len(parts) == 2:
                    rgb_path = parts[0]
                    depth_path = parts[1]
                    if current_table == "table2":
                        table2_pairs.append((rgb_path, depth_path))
                    else:
                        table3_pairs.append((rgb_path, depth_path))
    
    return table2_pairs, table3_pairs

def extract_frame_number(filename):
    """从文件名中提取帧号，用于排序"""
    match = re.search(r'(\d+)\.', filename)
    if match:
        return int(match.group(1))
    return 0


def process_realsense_data(base_dir, output_file):
    """处理 RealSense 文件夹中的图像并转换为点云，使用时间戳匹配"""
    table_dir = os.path.join(base_dir, "table")
    no_table_dir = os.path.join(base_dir, "no_table")
    
    # 检查目录是否存在
    if not os.path.exists(table_dir) or not os.path.exists(no_table_dir):
        print(f"Error: {table_dir} or {no_table_dir} does not exist")
        return
    
    all_pointclouds = []  # 所有点云数据
    all_labels = []       # 对应的标签：1=table, 0=no_table
    frame_indices = []    # 对应的帧索引
    all_filenames = []    # 对应的文件名 - 新增
    
    # 处理包含桌子的图像
    print("Processing images with tables...")
    
    # 处理含桌子的图像
    depth_dir = os.path.join(table_dir, "depthTSDF")
    
    # 直接处理深度图，不需要匹配RGB图像
    depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith('.png')])
    
    for depth_file in tqdm(depth_files):
        # 从文件名提取时间戳作为帧索引
        frame_idx = extract_frame_number(depth_file)
        depth_path = os.path.join(depth_dir, depth_file)
        
        # 读取深度图
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0  # 深度单位转换为米
        
        # 转换为点云
        raw_points = depth_to_pointcloud(depth)
        if len(raw_points) < 10:
            print(f"Warning: Frame {frame_idx} has too few valid point cloud points")
            continue
        
        # 预处理点云
        points = preprocess_pointcloud(raw_points, num_points=NUMPOINTS)
        
        all_pointclouds.append(points)
        all_labels.append(1)  # 1表示有桌子
        frame_indices.append(frame_idx)
        all_filenames.append(f"table/{depth_file}")  # 新增：保存文件名
    
    # 处理不包含桌子的图像
    print("Processing images without tables...")
    depth_dir = os.path.join(no_table_dir, "depthTSDF")
    
    # 同样直接处理深度图
    if os.path.exists(depth_dir):
        depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith('.png')])
        
        for depth_file in tqdm(depth_files):
            # 从文件名提取时间戳作为帧索引
            frame_idx = extract_frame_number(depth_file)
            depth_path = os.path.join(depth_dir, depth_file)
            
            # 读取深度图
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
            
            # 转换为点云
            raw_points = depth_to_pointcloud(depth)
            if len(raw_points) < 10:
                print(f"Warning: Frame {frame_idx} has too few valid point cloud points")
                continue
            
            # 预处理点云
            points = preprocess_pointcloud(raw_points, num_points=NUMPOINTS)
            
            all_pointclouds.append(points)
            all_labels.append(0)  # 0表示无桌子
            frame_indices.append(frame_idx)
            all_filenames.append(f"no_table/{depth_file}")  # 新增：保存文件名
    else:
        print(f"Warning: No-table depth directory {depth_dir} not found")
    
    if not all_pointclouds:
        print("Error: No point cloud data was successfully processed")
        return
    
    # 转换为NumPy数组
    all_pointclouds = np.stack(all_pointclouds)  # [N, 1024, 3]
    all_labels = np.array(all_labels)
    all_filenames = np.array(all_filenames)  # 新增：转换为NumPy数组
    
    # 保存点云数据和标签
    np.savez_compressed(output_file,
                      pointclouds=all_pointclouds,
                      labels=all_labels,
                      frame_indices=np.array(frame_indices, dtype=np.int32),
                      filenames=all_filenames)  # 新增：保存文件名
    
    print(f"\nDataset saved as {output_file} with {len(all_pointclouds)} samples")
    print(f"Contains {np.sum(all_labels == 1)} table samples")
    print(f"Contains {np.sum(all_labels == 0)} non-table samples")


if __name__ == "__main__":
    process_realsense_data(
        base_dir="./data/RealSense",
        output_file="./datasets/realsense_pointclouds_C.npz"
    )