import os
import cv2
import numpy as np
import pickle
from tqdm import tqdm
from matplotlib import pyplot as plt

fx, fy = 570.3422047415297, 570.3422047415297
cx, cy = 320.0, 240.0

NUMPOINTS = 1024# * 4

# Processing point cloud: random sampling 1024 points + normalization
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

# Depth Map -> Point cloud
def depth_to_pointcloud(depth_raw):
    h, w = depth_raw.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth_raw
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    mask = (z.reshape(-1) > 0) & np.isfinite(points).all(axis=1)
    return points[mask]

def build_dataset(base_dir, output_file):
    sequences = [
        "mit_32_d507/d507_2/",
        "mit_76_459/76-459b/",
        "mit_76_studyroom/76-1studyroom2/",
        "mit_gym_z_squash/gym_z_squash_scan1_oct_26_2012_erika/",
        "mit_lab_hj/lab_hj_tea_nov_2_2012_scan1_erika/"
    ]
    
    all_pointclouds = []
    all_point_labels = []  # 点级别的标签
    all_frame_labels = []  # 帧级别的标签
    all_categorical_vectors = []  # 场景类型向量

    for seq_idx, seq in enumerate(sequences):
        base_path = os.path.join(base_dir, seq)

        depth_path = os.path.join(base_path, "depthTSDF") if os.path.exists(os.path.join(base_path, "depthTSDF")) else os.path.join(base_path, "depth")
        img_path = os.path.join(base_path, "image")
        label_path = os.path.join(base_path, "labels/tabletop_labels.dat")

        has_labels = os.path.exists(label_path)

        if has_labels:
            with open(label_path, 'rb') as f:
                labels_data = pickle.load(f)  # [frame][table_instance][coordinate]
        else:
            print(f"⚠ Warning: {seq} has no labels. Defaulting to label 0 for all frames.")
            labels_data = None 

        # only process jpg file
        img_files = sorted([f for f in os.listdir(img_path) if f.endswith(".jpg")])
        # only process png file
        depth_files = sorted([f for f in os.listdir(depth_path) if f.endswith(".png")])
        
        n_frames = min(len(img_files), len(depth_files))
        if n_frames == 0:
            print(f"⚠ Warning: {seq} has no valid image-depth pairs. Skipping.")
            continue

        print(f"Processing {seq}: Found {len(img_files)} jpg images, {len(depth_files)} png depth maps. Using {n_frames} frames.")

        # 确保labels_data与n_frames对齐
        if has_labels and len(labels_data) < n_frames:
            print(f"⚠ Warning: {seq} has fewer labels ({len(labels_data)}) than frames ({n_frames}). Padding with [].")
            labels_data += [[]] * (n_frames - len(labels_data))

        for i in tqdm(range(n_frames)):
            depth_file = os.path.join(depth_path, depth_files[i])
            depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
            img_file = os.path.join(img_path, img_files[i])
            img = cv2.imread(img_file)

            # 获取原始点云
            raw_points = depth_to_pointcloud(depth)
            if len(raw_points) < 10:
                continue  

            # 是否有桌子的帧级别标签
            has_table = has_labels and len(labels_data[i]) > 0
            frame_label = 1 if has_table else 0
            
            # 创建类别向量 - 包含场景类型和是否有桌子
            # 使用独热编码表示场景，[0,0,0,0,1]表示第5个场景类型
            # scene_type = np.zeros(len(sequences), dtype=np.float32)
            # scene_type[seq_idx] = 1.0
            # # 添加是否有桌子的标志
            # categorical_vector = np.concatenate([[float(has_table)], scene_type])

            categorical_vector = np.array([float(has_table)], dtype=np.float32)

            
            # 预处理点云 - 采样1024个点
            points = preprocess_pointcloud(raw_points, num_points=NUMPOINTS)
            
            # 创建点级别的标签
            point_labels = np.zeros(NUMPOINTS, dtype=np.int64)  # 默认所有点为背景(0)
            
            if has_table:
                # 使用原始深度图和多边形标签来确定点是否在桌面上
                h, w = depth.shape
                # 创建掩码图像
                mask = np.zeros((h, w), dtype=np.uint8)
                
                # 首先将所有桌面多边形合并到一个掩码图像中
                for table_idx, table_polygon in enumerate(labels_data[i]):
                    if len(table_polygon) > 0:
                        # 将多边形转换为OpenCV格式
                        x_coords = np.array(table_polygon[0], dtype=np.int32)
                        y_coords = np.array(table_polygon[1], dtype=np.int32)
                        polygon = np.vstack((x_coords, y_coords)).T  # 转换为[[x1,y1], [x2,y2], ...]格式
                        
                        # 填充多边形到掩码图像
                        cv2.fillPoly(mask, [polygon], 255)
                
                # 可视化原始图像和掩码图像
                # fig, ax = plt.subplots(1, 2, figsize=(12, 6))
                
                # # 显示原始图像
                # ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                # ax[0].set_title('Original Image')
                # ax[0].axis('off')
                
                # # 显示掩码图像
                # ax[1].imshow(mask, cmap='gray')
                # ax[1].set_title('Mask Image')
                # ax[1].axis('off')
                
                # plt.show()
                
                # 然后对所有点进行一次性处理
                for pt_idx, point in enumerate(points):
                    # 获取原始点在深度图中的坐标
                    x, y, z = point
                    # 将3D点投影回2D图像
                    u = int(x * fx / z + cx)
                    v = int(y * fy / z + cy)
                    
                    # 检查点是否在图像边界内
                    if 0 <= u < w and 0 <= v < h:
                        # 检查该点是否在掩码内
                        if mask[v, u] > 0:
                            point_labels[pt_idx] = 1  # 如果在掩码内，标记为桌面
            
            all_pointclouds.append(points)
            all_point_labels.append(point_labels)
            all_frame_labels.append(frame_label)
            all_categorical_vectors.append(categorical_vector)
        

    # 转换为NumPy数组
    all_pointclouds = np.stack(all_pointclouds)  # [N, 1024, 3]
    all_point_labels = np.stack(all_point_labels)  # [N, 1024]
    all_frame_labels = np.array(all_frame_labels, dtype=np.int64)  # [N]
    all_categorical_vectors = np.stack(all_categorical_vectors)  # [N, 5+1]

    np.savez_compressed(output_file, 
                       pointclouds=all_pointclouds, 
                       point_labels=all_point_labels,
                       frame_labels=all_frame_labels,
                       categorical_vectors=all_categorical_vectors)
    
    print(f"\n Dataset saved as {output_file}, total {len(all_frame_labels)} samples")
    
    # 打印统计信息
    table_frames = np.sum(all_frame_labels)
    print(f"Statistics: {table_frames} frames with tables ({table_frames/len(all_frame_labels)*100:.2f}%)")
    
    table_points = np.sum(all_point_labels == 1)
    total_points = all_point_labels.size
    print(f"Points statistics: {table_points} table points out of {total_points} ({table_points/total_points*100:.2f}%)")












if __name__ == "__main__":
    build_dataset(
        base_dir="./data",
        output_file="./datasets/pipelineC_dataset_mit_balanced.npz"
    )
