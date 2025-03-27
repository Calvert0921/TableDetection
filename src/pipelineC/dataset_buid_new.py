import os
import cv2
import numpy as np
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


fx, fy = 570.3422047415297, 570.3422047415297
cx, cy = 320.0, 240.0

NUMPOINTS = 1024*4

# 修改预处理函数以处理颜色信息
def preprocess_pointcloud(points, colors=None, num_points=1024):
    if len(points) < num_points:
        idx = np.random.choice(len(points), num_points, replace=True)
    else:
        idx = np.random.choice(len(points), num_points, replace=False)
    
    points = points[idx, :]
    
    # 规范化点云
    centroid = np.mean(points, axis=0)
    points -= centroid
    furthest_dist = np.max(np.linalg.norm(points, axis=1))
    points /= furthest_dist
    
    if colors is not None:
        colors = colors[idx, :]
        return points, colors
    
    return points

# 修改depth_to_pointcloud函数以接收和处理RGB图像
def depth_to_pointcloud(depth_raw, rgb_img=None):
    h, w = depth_raw.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth_raw
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    mask = (z.reshape(-1) > 0) & np.isfinite(points).all(axis=1)
    
    if rgb_img is not None:
        # 提取对应点的RGB颜色
        colors = rgb_img.reshape(-1, 3)[mask]
        return points[mask], colors
    
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
    all_image_paths = []  # 图像路径
    all_point_colors = []  # 新增：点颜色信息

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
            
            # 获取原始点云（带颜色）
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            raw_points, raw_colors = depth_to_pointcloud(depth, rgb_img)
            
            if len(raw_points) < 10:
                continue  

            # 是否有桌子的帧级别标签
            has_table = has_labels and len(labels_data[i]) > 0
            frame_label = 1 if has_table else 0

            categorical_vector = np.array([float(has_table)], dtype=np.float32)

            
            # 预处理点云 - 采样1024个点
            points = preprocess_pointcloud(raw_points, num_points=NUMPOINTS)
            
            # 创建点级别的标签
            point_labels = np.zeros(NUMPOINTS, dtype=np.int64)  # 默认所有点为背景(0)
            point_labels_raw = np.zeros(len(raw_points), dtype=np.int64)

            
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

            # 预处理点云 - 采样1024个点并保留颜色
            points, colors = preprocess_pointcloud(raw_points, raw_colors, num_points=NUMPOINTS)
             
            # 重新对采样后的点进行标注（使用最近邻）
            # 这里简化处理，重新采样
            if has_table:
                point_labels = np.zeros(NUMPOINTS, dtype=np.int64)
                for pt_idx, _ in enumerate(points):
                    # 简化标注逻辑：按比例保留标签
                    if np.random.rand() < np.mean(point_labels_raw):
                        point_labels[pt_idx] = 1
            else:
                point_labels = np.zeros(NUMPOINTS, dtype=np.int64)
            
            
            all_pointclouds.append(points)
            all_point_colors.append(colors)  # 添加颜色信息
            all_point_labels.append(point_labels)
            all_frame_labels.append(frame_label)
            all_categorical_vectors.append(categorical_vector)
            all_image_paths.append(img_file)

    # 转换为NumPy数组
    all_pointclouds = np.stack(all_pointclouds)
    all_point_colors = np.stack(all_point_colors)  # 颜色数组
    all_point_labels = np.stack(all_point_labels)

    all_frame_labels = np.array(all_frame_labels, dtype=np.int64)  # [N]
    all_categorical_vectors = np.stack(all_categorical_vectors)  # [N, 5+1]

    np.savez_compressed(output_file, 
                        pointclouds=all_pointclouds,
                        point_colors=all_point_colors,  # 添加颜色信息
                        point_labels=all_point_labels,
                        frame_labels=all_frame_labels,
                        categorical_vectors=all_categorical_vectors,
                        image_paths=np.array(all_image_paths, dtype=object))
    
    
    print(f"\n Dataset saved as {output_file}, total {len(all_frame_labels)} samples")
    
    # 打印统计信息
    table_frames = np.sum(all_frame_labels)
    print(f"Statistics: {table_frames} frames with tables ({table_frames/len(all_frame_labels)*100:.2f}%)")
    
    table_points = np.sum(all_point_labels == 1)
    total_points = all_point_labels.size
    print(f"Points statistics: {table_points} table points out of {total_points} ({table_points/total_points*100:.2f}%)")



def inspect_dataset(npz_file, num_samples=5, save_dir=None):
    """检查NPZ文件中保存的数据集
    
    Args:
        npz_file: NPZ文件路径
        num_samples: 要可视化的样本数量
        save_dir: 保存可视化结果的目录，如果为None则只显示不保存
    """
    print(f"正在加载数据集: {npz_file}")
    data = np.load(npz_file, allow_pickle=True)
    
    # 查看数据集中的键
    print(f"数据集包含的键: {list(data.keys())}")
    
    # 获取数据
    pointclouds = data['pointclouds']
    point_labels = data['point_labels']
    frame_labels = data['frame_labels']
    categorical_vectors = data['categorical_vectors']
    point_colors = data.get('point_colors', None)
    
    # 获取图像路径（如果存在）
    image_paths = data.get('image_paths', None)

    # 打印数据形状
    print(f"点云数据形状: {pointclouds.shape}")
    print(f"点标签形状: {point_labels.shape}")
    print(f"帧标签形状: {frame_labels.shape}")
    print(f"分类向量形状: {categorical_vectors.shape}")
    
    # 统计信息
    table_frames = np.sum(frame_labels)
    print(f"有桌子的帧数量: {table_frames}/{len(frame_labels)} ({table_frames/len(frame_labels)*100:.2f}%)")
    
    table_points = np.sum(point_labels == 1)
    total_points = point_labels.size
    print(f"桌面点数量: {table_points}/{total_points} ({table_points/total_points*100:.2f}%)")
    
    # 创建保存目录
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 选择有桌子的样本进行可视化
    table_indices = np.where(frame_labels == 1)[0]
    if len(table_indices) == 0:
        print("没有找到有桌子的帧！使用随机样本。")
        table_indices = np.arange(len(frame_labels))
    
    # 随机选择样本
    if len(table_indices) > num_samples:
        sample_indices = np.random.choice(table_indices, num_samples, replace=False)
    else:
        sample_indices = table_indices
        
    print(f"选择了 {len(sample_indices)} 个样本进行可视化")
    
    # 可视化选择的样本
    for idx in sample_indices:
        # 可视化点云（传入颜色信息）
        if point_colors is not None:
            fig_pointcloud = visualize_pointcloud(pointclouds[idx], point_labels[idx], idx, frame_labels[idx], colors=point_colors[idx])
        else:
            fig_pointcloud = visualize_pointcloud(pointclouds[idx], point_labels[idx], idx, frame_labels[idx])
        
        
        # 如果有图像路径，尝试加载和显示原始图像
        if image_paths is not None:
            try:
                img_path = image_paths[idx]
                img = cv2.imread(img_path)
                if img is not None:
                    # 显示原始图像
                    plt.figure(figsize=(10, 5))
                    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    plt.title(f"原始图像 - 样本 {idx}")
                    plt.axis('off')
                    
                    # 保存图像
                    if save_dir:
                        plt.savefig(os.path.join(save_dir, f"sample_{idx}_original.png"))
                    plt.show()
            except Exception as e:
                print(f"无法加载样本 {idx} 的原始图像: {e}")
        
        # 保存点云可视化
        if save_dir:
            fig_pointcloud.savefig(os.path.join(save_dir, f"sample_{idx}_pointcloud.png"))



def visualize_pointcloud(points, labels, idx, frame_label, colors=None):
    """可视化点云和点标签"""
    fig = plt.figure(figsize=(10, 8))

    # 创建两个子图：一个使用标签颜色，一个使用原始RGB颜色
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d') if colors is not None else None
    
    
    # 基于标签的颜色可视化
    label_colors = np.zeros((len(points), 3))
    label_colors[labels == 0] = [0, 0, 1]  # 背景点 - 蓝色
    label_colors[labels == 1] = [1, 0, 0]  # 桌面点 - 红色
    
    ax1.scatter(points[:, 0], points[:, 1], points[:, 2], c=label_colors, s=1)
    ax1.set_title(f"标签着色\n桌面: {np.sum(labels == 1)}, 背景: {np.sum(labels == 0)}")
    
    # 使用原始RGB颜色可视化
    if colors is not None:
        ax2.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors/255.0, s=1)
        ax2.set_title("RGB着色")
    
    # 保持坐标轴比例一致
    max_range = np.array([
        points[:, 0].max() - points[:, 0].min(),
        points[:, 1].max() - points[:, 1].min(),
        points[:, 2].max() - points[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
    
    ax1.set_xlim(mid_x - max_range, mid_x + max_range)
    ax1.set_ylim(mid_y - max_range, mid_y + max_range)
    ax1.set_zlim(mid_z - max_range, mid_z + max_range)

    ax2.set_xlim(mid_x - max_range, mid_x + max_range)
    ax2.set_ylim(mid_y - max_range, mid_y + max_range)
    ax2.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    plt.show()
    return fig







if __name__ == "__main__":
    build_dataset(
        base_dir="./data",
        output_file="./datasets/pipelineC_dataset_mit_balanced.npz"
    )
    npz_file = "./datasets/pipelineC_dataset_mit_balanced.npz"
    inspect_dataset(npz_file, num_samples=3)