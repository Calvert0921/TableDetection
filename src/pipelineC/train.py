# src/pipelineA/train.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from model import DGCNN_seg
import torch.nn.functional as F


class PointCloudDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.pointclouds = data['pointclouds'].astype(np.float32)       # [N, 1024, 3]
        self.point_labels = data['point_labels'].astype(np.int64)       # [N, 1024]
        self.frame_labels = data['frame_labels'].astype(np.int64)       # [N]
        self.categorical_vectors = data['categorical_vectors'].astype(np.float32) # [N, 6]

    def __len__(self):
        return len(self.frame_labels)

    def __getitem__(self, idx):
        return self.pointclouds[idx], self.categorical_vectors[idx], self.point_labels[idx]


def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss, correct_points, total_points = 0.0, 0, 0
    
    for points, cat_vec, point_labels in train_loader:
        batch_size = points.size(0)
        points = points.to(device)
        cat_vec = cat_vec.to(device)
        point_labels = point_labels.to(device)  # [B, 1024]
        
        # 前向传播
        outputs = model(points, cat_vec)  # [B, 2, 1024]
        
        # 转置输出以匹配CrossEntropyLoss的期望格式 [B, 1024, 2]
        outputs = outputs.permute(0, 2, 1)
        
        # 计算损失 - 对每个点的预测进行评估
        loss = criterion(outputs.reshape(-1, 2), point_labels.reshape(-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * batch_size
        
        # 计算点级别的准确率
        preds = outputs.argmax(dim=2)  # [B, 1024]
        correct_points += (preds == point_labels).sum().item()
        total_points += points.size(0) * points.size(1)
    
    avg_loss = total_loss / len(train_loader.dataset)
    point_accuracy = correct_points / total_points
    return avg_loss, point_accuracy

def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss, correct_points, total_points = 0.0, 0, 0
    
    with torch.no_grad():
        for points, cat_vec, point_labels in val_loader:
            batch_size = points.size(0)
            points = points.to(device)
            cat_vec = cat_vec.to(device)
            point_labels = point_labels.to(device)  # [B, 1024]
            
            # 前向传播
            outputs = model(points, cat_vec)  # [B, 2, 1024]
            
            # 转置输出以匹配CrossEntropyLoss的期望格式
            outputs = outputs.permute(0, 2, 1)  # [B, 1024, 2]
            
            # 计算损失
            loss = criterion(outputs.reshape(-1, 2), point_labels.reshape(-1))
            
            total_loss += loss.item() * batch_size
            
            # 计算点级别的准确率
            preds = outputs.argmax(dim=2)  # [B, 1024]
            correct_points += (preds == point_labels).sum().item()
            total_points += points.size(0) * points.size(1)
    
    avg_loss = total_loss / len(val_loader.dataset)
    point_accuracy = correct_points / total_points
    return avg_loss, point_accuracy

# 或使用Focal Loss处理极度不平衡数据
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

if __name__ == "__main__":
    npz_path = "./datasets/pipelineC_dataset_mit_balanced.npz"
    save_path = "./best_model_pipelineC.pth"
    batch_size = 64
    epochs = 50
    lr = 0.001

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = PointCloudDataset(npz_path)
    train_len = int(0.8 * len(dataset))
    val_len = len(dataset) - train_len
    train_set, val_set = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model = DGCNN_seg().to(device)
    # criterion = nn.CrossEntropyLoss()

    # 添加类别权重反映数据不平衡情况
    class_weights = torch.tensor([1.0, 10.0], device=device)  # 背景:桌面 = 1:10
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    # criterion = FocalLoss(gamma=2)


    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    best_val_loss = np.inf
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f}, Point Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Point Acc: {val_acc:.4f}")

        # if val_acc > best_val_acc or val_loss < best_val_loss:
        if True:
            best_val_acc = val_acc
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f" Saved best model with point accuracy: {val_acc:.4f}")

    print("Training completed. Best validation point accuracy:", best_val_acc)