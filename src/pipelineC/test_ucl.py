# Point cloud table detection prediction script - for datasets without labels

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from model import DGCNN_seg
import argparse
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

class CustomPointCloudDataset(Dataset):
    """Point cloud dataset without labels"""
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.pointclouds = data['pointclouds'].astype(np.float32)  # [N, 1024, 3]
        
        # Extract metadata (if exists)
        self.table_names = data['table_names'] if 'table_names' in data else np.array(['unknown'] * len(self.pointclouds))
        self.frame_indices = data['frame_indices'] if 'frame_indices' in data else np.arange(len(self.pointclouds))
        
        # Create default category vector (required by the model)
        self.categorical_vectors = np.ones((len(self.pointclouds), 1), dtype=np.float32)

    def __len__(self):
        return len(self.pointclouds)

    def __getitem__(self, idx):
        return self.pointclouds[idx], self.categorical_vectors[idx], self.table_names[idx], self.frame_indices[idx]

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
    """Visualize point cloud prediction results"""
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
    table_confidence = "High" if table_ratio > 0.4 else "Medium" if table_ratio > 0.2 else "Low"
    
    ax.set_title(f'Prediction - Table Points: {table_ratio:.2%} (Confidence: {table_confidence})')
    ax.set_axis_off()
    
    filename = f'{file_prefix}_{idx}.png'
    plt.savefig(os.path.join(save_dir, filename), bbox_inches='tight', dpi=150)
    plt.close()
    
    return table_ratio

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="best_model_pipelineC.pth", help="Pretrained model path")
    parser.add_argument('--data_npz', type=str, default="./datasets/selected_tables_pointclouds_C.npz", help="Point cloud data path")
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

    # Predict
    point_preds, table_names, frame_indices = predict_point_labels(model, dataloader, device)
    
    # Save prediction results
    results_file = os.path.join(args.vis_dir, "prediction_results.txt")
    os.makedirs(args.vis_dir, exist_ok=True)
    
    with open(results_file, 'w') as f:
        f.write("Index\tTable Name\tFrame Index\tTable Point Ratio\tResult\n")
        
        # Visualize and analyze each point cloud
        print("\nGenerating prediction visualizations...")
        for i, (points, preds, table_name, frame_idx) in enumerate(zip(dataset.pointclouds, point_preds, table_names, frame_indices)):
            # Generate visualization filename prefix
            file_prefix = f"{table_name}_frame{frame_idx}"
            
            # Visualize and get table point ratio
            table_ratio = visualize_prediction(points, preds, args.vis_dir, file_prefix, i)
            
            # Determine if table is present
            has_table = "Yes" if table_ratio > args.threshold else "No"
            
            # Write results
            f.write(f"{i}\t{table_name}\t{frame_idx}\t{table_ratio:.4f}\t{has_table}\n")
    
    # Output statistics
    detected_tables = sum(1 for preds in point_preds if np.mean(preds) > args.threshold)
    print(f"\nPrediction complete! Detected {detected_tables} tables in {len(point_preds)} samples ({detected_tables/len(point_preds)*100:.1f}%)")
    print(f"Results saved to: {results_file}")
    print(f"Visualization images saved to: {args.vis_dir}")