import os
import cv2
import numpy as np
from tqdm import tqdm
import re

# 使用与原始文件相同的相机参数
fx, fy = 570.3422047415297, 570.3422047415297
cx, cy = 320.0, 240.0

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

def process_selected_images(base_dir, output_file):
    """处理selected_images文件夹中的图像并转换为点云"""
    # 读取selected_pairs.txt
    pairs_file = os.path.join(base_dir, "selected_pairs.txt")
    if not os.path.exists(pairs_file):
        print(f"错误: {pairs_file} 不存在")
        return
    
    table2_pairs, table3_pairs = read_selected_pairs(pairs_file)
    print(f"发现 {len(table2_pairs)} 对来自table2的图像和 {len(table3_pairs)} 对来自table3的图像")
    
    all_pointclouds = []
    table_names = []  # 记录每个点云来自哪个桌子
    frame_indices = []  # 记录每个点云的帧索引
    
    # 处理table2的图像
    print("处理table2图像...")
    for i, (rgb_path, depth_path) in enumerate(tqdm(table2_pairs)):
        # 从文件路径中提取帧号
        frame_idx = extract_frame_number(os.path.basename(rgb_path))
        
        # 修正路径，适应selected_images文件夹结构
        depth_file = os.path.join(base_dir, "table2", depth_path)
        rgb_file = os.path.join(base_dir, "table2", rgb_path)
        
        if not os.path.exists(depth_file) or not os.path.exists(rgb_file):
            print(f"警告: 找不到文件 {depth_file} 或 {rgb_file}，跳过")
            continue
        
        # 读取深度图
        depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0  # 假设深度单位为毫米
        
        # 转换为点云
        raw_points = depth_to_pointcloud(depth)
        if len(raw_points) < 10:
            print(f"警告: 帧 {frame_idx} 的有效点云点数太少")
            continue
        
        # 预处理点云
        points = preprocess_pointcloud(raw_points, num_points=NUMPOINTS)
        
        all_pointclouds.append(points)
        table_names.append("table2")
        frame_indices.append(frame_idx)
    
    # 处理table3的图像
    print("处理table3图像...")
    for i, (rgb_path, depth_path) in enumerate(tqdm(table3_pairs)):
        # 从文件路径中提取帧号
        frame_idx = extract_frame_number(os.path.basename(rgb_path))
        
        # 修正路径，适应selected_images文件夹结构
        depth_file = os.path.join(base_dir, "table3", depth_path)
        rgb_file = os.path.join(base_dir, "table3", rgb_path)
        
        if not os.path.exists(depth_file) or not os.path.exists(rgb_file):
            print(f"警告: 找不到文件 {depth_file} 或 {rgb_file}，跳过")
            continue
        
        # 读取深度图
        depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0  # 假设深度单位为毫米
        
        # 转换为点云
        raw_points = depth_to_pointcloud(depth)
        if len(raw_points) < 10:
            print(f"警告: 帧 {frame_idx} 的有效点云点数太少")
            continue
        
        # 预处理点云
        points = preprocess_pointcloud(raw_points, num_points=NUMPOINTS)
        
        all_pointclouds.append(points)
        table_names.append("table3")
        frame_indices.append(frame_idx)
    
    if not all_pointclouds:
        print("错误: 没有成功处理任何点云数据")
        return
    
    # 转换为NumPy数组
    all_pointclouds = np.stack(all_pointclouds)  # [N, 1024, 3]
    
    # 由于selected_images没有标签数据，我们只保存点云数据和元信息
    np.savez_compressed(output_file,
                      pointclouds=all_pointclouds,
                      table_names=np.array(table_names, dtype=np.str_),
                      frame_indices=np.array(frame_indices, dtype=np.int32))
    
    print(f"\n数据集已保存为 {output_file}，共 {len(all_pointclouds)} 个样本")
    print(f"包含 {sum(1 for name in table_names if name == 'table2')} 个table2样本")
    print(f"包含 {sum(1 for name in table_names if name == 'table3')} 个table3样本")


if __name__ == "__main__":
    process_selected_images(
        base_dir="./data/selected_images",
        output_file="./datasets/selected_tables_pointclouds_C.npz"
    )