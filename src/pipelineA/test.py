# src/pipelineA/test.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from model import DGCNN_cls
from sklearn.metrics import classification_report, confusion_matrix
import argparse

class PointCloudDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.pointclouds = data['pointclouds'].astype(np.float32)
        self.labels = data['labels'].astype(np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.pointclouds[idx], self.labels[idx]

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for points, labels in dataloader:
            points = points.to(device)
            outputs = model(points)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="weights/best_model_pipelineA.pth")
    # parser.add_argument('--test_npz', type=str, default="datasets/pipelineA_dataset_harvard_all.npz")
    parser.add_argument('--test_npz', type=str, default="datasets/pipelineA_RealSense_dataset.npz")
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load train dataset
    test_dataset = PointCloudDataset(args.test_npz)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # load model
    model = DGCNN_cls().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print(f"Loaded model from {args.model_path}")

    # evaluate
    y_true, y_pred = evaluate_model(model, test_loader, device)

    acc = (y_true == y_pred).mean()
    print(f"\n Test Accuracy: {acc:.4f}")
    print("\n Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["No Table", "Table"]))

    print(" Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
