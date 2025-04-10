# Point cloud semantic segmentation model testing script

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from model import DGCNN_seg
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, jaccard_score
import argparse
import matplotlib.pyplot as plt
import os

class PointCloudDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.pointclouds = data['pointclouds'].astype(np.float32)       # [N, 1024, 3]
        self.point_labels = data['point_labels'].astype(np.int64)       # [N, 1024]
        self.frame_labels = data['frame_labels'].astype(np.int64)       # [N]
        self.categorical_vectors = data['categorical_vectors'].astype(np.float32) # [N, 1]

    def __len__(self):
        return len(self.frame_labels)

    def __getitem__(self, idx):
        return self.pointclouds[idx], self.categorical_vectors[idx], self.point_labels[idx], self.frame_labels[idx]

def evaluate_model(model, dataloader, device):
    model.eval()
    all_point_preds = []
    all_point_labels = []
    all_frame_preds = []
    all_frame_labels = []
    
    with torch.no_grad():
        for points, cat_vec, point_labels, frame_labels in dataloader:
            batch_size = points.size(0)
            points = points.to(device)
            cat_vec = cat_vec.to(device)
            
            # Forward pass
            outputs = model(points, cat_vec)  # [B, 2, 1024]
            
            # Transpose output to match evaluation expected format
            outputs = outputs.permute(0, 2, 1)  # [B, 1024, 2]
            
            # Point-level prediction
            point_preds = outputs.argmax(dim=2).cpu().numpy()  # [B, 1024]
            
            # Frame-level prediction - based on majority voting from point predictions
            frame_preds = []
            for i in range(batch_size):
                # If more than 30% of points are predicted as table, consider frame as containing table
                table_ratio = point_preds[i].mean()
                frame_pred = 1 if table_ratio > 0.3 else 0
                frame_preds.append(frame_pred)
            
            all_point_preds.extend(point_preds.reshape(-1))
            all_point_labels.extend(point_labels.numpy().reshape(-1))
            all_frame_preds.extend(frame_preds)
            all_frame_labels.extend(frame_labels.numpy())

    return (
        np.array(all_point_labels), 
        np.array(all_point_preds),
        np.array(all_frame_labels),
        np.array(all_frame_preds)
    )

def visualize_results(points, point_labels, point_preds, save_dir, idx):
    """Visualize point cloud segmentation results"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    fig = plt.figure(figsize=(12, 5))
    
    # Draw ground truth labels
    ax1 = fig.add_subplot(121, projection='3d')
    colors = np.array(['blue', 'red'])
    ax1.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors[point_labels], s=5)
    ax1.set_title('Ground Truth')
    ax1.set_axis_off()
    
    # Draw prediction labels
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors[point_preds], s=5)
    ax2.set_title('Prediction')
    ax2.set_axis_off()
    
    plt.savefig(os.path.join(save_dir, f'result_{idx}.png'), bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="best_model_pipelineC.pth")
    parser.add_argument('--test_npz', type=str, default="./datasets/pipelineC_dataset_harvard_test.npz")
    # selected_tables_pointclouds_C.npz
    # parser.add_argument('--test_npz', type=str, default="./datasets/selected_tables_pointclouds_C.npz")
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--vis_dir', type=str, default="./visualizations")
    parser.add_argument('--vis_samples', type=int, default=10, help="Number of samples to visualize")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load test dataset
    test_dataset = PointCloudDataset(args.test_npz)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Load model
    model = DGCNN_seg(categorical_dim=1).to(device)  # Note categorical_dim=1
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print(f"Loaded model from {args.model_path}")

    # Evaluate
    point_true, point_pred, frame_true, frame_pred = evaluate_model(model, test_loader, device)

    # Point-level evaluation
    point_acc = accuracy_score(point_true, point_pred)
    iou = jaccard_score(point_true, point_pred, average='macro')
    
    print(f"\n === Point-level Evaluation ===")
    print(f" Point Accuracy: {point_acc:.4f}")
    print(f" IoU Score: {iou:.4f}")
    print("\n Point Classification Report:")
    print(classification_report(point_true, point_pred, target_names=["Background", "Table"]))
    print(" Point Confusion Matrix:")
    print(confusion_matrix(point_true, point_pred))
    
    # Frame-level evaluation
    # frame_acc = accuracy_score(frame_true, frame_pred)
    # print(f"\n === Frame-level Evaluation ===")
    # print(f" Frame Accuracy: {frame_acc:.4f}")
    # print("\n Frame Classification Report:")
    # print(classification_report(frame_true, frame_pred, target_names=["No Table", "Has Table"]))
    # print(" Frame Confusion Matrix:")
    # print(confusion_matrix(frame_true, frame_pred))
    
    # Visualize some sample results
    if args.vis_samples > 0:
        print(f"\nGenerating visualization for {args.vis_samples} samples...")
        vis_indices = np.random.choice(len(test_dataset), min(args.vis_samples, len(test_dataset)), replace=False)
        for i, idx in enumerate(vis_indices):
            points, cat_vec, point_labels, _ = test_dataset[idx]
            points_tensor = torch.from_numpy(points).unsqueeze(0).to(device)
            cat_vec_tensor = torch.from_numpy(cat_vec).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(points_tensor, cat_vec_tensor)
                outputs = outputs.permute(0, 2, 1)
                point_preds = outputs[0].argmax(dim=1).cpu().numpy()
            
            visualize_results(points, point_labels, point_preds, args.vis_dir, i)