# src/pipelineA/train.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from model import DGCNN_cls

class PointCloudDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.pointclouds = data['pointclouds'].astype(np.float32)  # [N, 1024, 3]
        self.labels = data['labels'].astype(np.int64)              # [N]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.pointclouds[idx], self.labels[idx]

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss, correct = 0.0, 0
    for points, labels in train_loader:
        points, labels = points.to(device), labels.to(device)
        outputs = model(points)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * points.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()

    avg_loss = total_loss / len(train_loader.dataset)
    accuracy = correct / len(train_loader.dataset)
    return avg_loss, accuracy

def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        for points, labels in val_loader:
            points, labels = points.to(device), labels.to(device)
            outputs = model(points)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * points.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()

    avg_loss = total_loss / len(val_loader.dataset)
    accuracy = correct / len(val_loader.dataset)
    return avg_loss, accuracy

if __name__ == "__main__":
    npz_path = "/home/hank/TableDetection/datasets/pipelineA_dataset_mit_balanced.npz"
    save_path = "./best_model_pipelineA.pth"
    batch_size = 8
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

    model = DGCNN_cls().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f" Saved best model with accuracy: {val_acc:.4f}")

    print("Training completed. Best validation accuracy:", best_val_acc)
