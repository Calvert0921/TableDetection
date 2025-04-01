# Point cloud table detection prediction script - for datasets without labels

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from model import DGCNN_seg
import argparse
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

THRESHOLD = 0.02  # 1%阈值

class CustomPointCloudDataset(Dataset):
    """Point cloud dataset with filenames"""
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.pointclouds = data['pointclouds'].astype(np.float32)  # [N, 1024, 3]
        
        # 读取文件名
        self.filenames = data['filenames'] if 'filenames' in data else np.array(['unknown'] * len(self.pointclouds))
        self.frame_indices = data['frame_indices'] if 'frame_indices' in data else np.arange(len(self.pointclouds))
        
        # 如果有标签，也读取标签
        if 'labels' in data:
            self.labels = data['labels'] 
            self.has_labels = True
        else:
            self.has_labels = False
        
        # 创建默认分类向量(模型需要)
        self.categorical_vectors = np.ones((len(self.pointclouds), 1), dtype=np.float32)

    def __len__(self):
        return len(self.pointclouds)

    def __getitem__(self, idx):
        # 返回点云、类别向量、文件名和帧索引
        return self.pointclouds[idx], self.categorical_vectors[idx], self.filenames[idx], self.frame_indices[idx]
    

def predict_point_labels(model, dataloader, device):
    """Predict tabletop points in the point cloud"""
    model.eval()
    all_point_preds = []
    all_table_names = []
    all_frame_indices = []
    
    with torch.no_grad():
        for points, cat_vec, table_names, frame_indices in tqdm(dataloader, desc="Predicting"):
            batch_size = points.size(0)
            points = points.to(device)
            cat_vec = cat_vec.to(device)
            
            # Forward pass
            outputs = model(points, cat_vec)  # [B, 2, 1024]
            outputs = outputs.permute(0, 2, 1)  # [B, 1024, 2]
            point_preds = outputs.argmax(dim=2).cpu().numpy()  # [B, 1024]
            
            # Collect predictions and metadata
            for i in range(batch_size):
                all_point_preds.append(point_preds[i])
                all_table_names.append(table_names[i])
                all_frame_indices.append(frame_indices[i])

    return all_point_preds, all_table_names, all_frame_indices


def visualize_prediction(points, point_preds, save_dir, file_prefix, idx):
    """Visualize point cloud prediction results - table detected only when >1% points are table points"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    fig = plt.figure(figsize=(10, 8))
    
    # Draw prediction labels - single subplot
    ax = fig.add_subplot(111, projection='3d')
    
    # Red for table surface, blue for background
    colors = np.array(['blue', 'red'])
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors[point_preds], s=5)
    
    # Calculate table point ratio
    table_ratio = np.mean(point_preds)
    
    # Table detected only when >1% points are table points
    has_table = table_ratio > THRESHOLD  # 修改为使用1%阈值
    table_presence = "Table Detected" if has_table else "No Table"
    
    # Show both the binary result and the ratio for reference
    ax.set_title(f'Prediction: {table_presence} (Table Points: {table_ratio:.2%})')
    ax.set_axis_off()
    
    filename = f'{file_prefix}_{idx}.png'
    plt.savefig(os.path.join(save_dir, filename), bbox_inches='tight', dpi=150)
    plt.close()
    
    # Return binary table presence instead of ratio
    return has_table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="best_model_pipelineC.pth", help="Pretrained model path")
    parser.add_argument('--data_npz', type=str, default="./datasets/realsense_pointclouds_C.npz", help="Point cloud data path")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size")
    parser.add_argument('--vis_dir', type=str, default="./predictions", help="Visualization output directory")
    parser.add_argument('--threshold', type=float, default=0.3, help="Threshold for table detection")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    dataset = CustomPointCloudDataset(args.data_npz)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    print(f"Loaded {len(dataset)} point cloud samples")

    # Load model
    model = DGCNN_seg(categorical_dim=1).to(device)  # Note categorical_dim=1
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print(f"Loaded model: {args.model_path}")

    # Clear visualization directory
    if os.path.exists(args.vis_dir):
        for file in os.listdir(args.vis_dir):
            file_path = os.path.join(args.vis_dir, file)
            if os.path.isfile(file_path):
                os.unlink(file_path)
    else:
        os.makedirs(args.vis_dir)

    # Predict
    point_preds, table_names, frame_indices = predict_point_labels(model, dataloader, device)
    
    # Save prediction results
    results_file = os.path.join(args.vis_dir, "prediction_results.txt")
    
    with open(results_file, 'w') as f:
        # 修改列标题以匹配实际写入的内容
        f.write("Index\tFilename\tFrame\tTable Detection\tTable Point Ratio\n")
        
        # Visualize and analyze each point cloud
        print("\nGenerating prediction visualizations...")
        for i, (points, preds, filename, frame_idx) in enumerate(zip(dataset.pointclouds, point_preds, table_names, frame_indices)):
            # 提取更干净的文件名用于显示
            clean_filename = filename.replace('\\', '_').split('/')[-1] if isinstance(filename, str) else f"sample_{i}"
            
            # 计算桌面点的比例
            table_ratio = np.mean(preds)
            
            # 计算桌面点的比例
            table_ratio = np.mean(preds)
            
            # 使用1%阈值判断是否有桌子
            has_table = table_ratio > THRESHOLD  
            
            # 使用原始文件名作为可视化文件的前缀
            file_prefix = f"{i:03d}_{clean_filename.split('.')[0]}"
            
            # 可视化
            visualize_prediction(points, preds, args.vis_dir, file_prefix, i)
            
            # 写入结果，同时保存表面比例和二值结果
            result_text = "Yes" if has_table else "No"
            f.write(f"{i}\t{filename}\t{frame_idx}\t{result_text}\t{table_ratio:.4f}\n")

    # 附加计算整体统计信息
    num_tables = sum(1 for preds in point_preds if np.mean(preds) > THRESHOLD)
    print(f"\nDetection Summary:")
    print(f"- Total samples: {len(point_preds)}")
    print(f"- Tables detected: {num_tables} ({num_tables/len(point_preds)*100:.1f}%)")
    print(f"- No tables detected: {len(point_preds) - num_tables} ({(len(point_preds) - num_tables)/len(point_preds)*100:.1f}%)")
    print(f"\nResults saved to: {results_file}")
    print(f"Visualization images saved to: {args.vis_dir}")