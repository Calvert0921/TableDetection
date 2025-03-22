import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import models, transforms
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from dataloader_tableClassifier import get_dataloader_table, TableClassificationDataset

# ----- Model Definition: Table Classifier using Pretrained ResNet50 -----
class TableClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(TableClassifier, self).__init__()
        # Load pretrained EfficientNet-B0
        self.effnet = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        
        # Modify the first convolution layer to accept 1-channel input
        # EfficientNet-B0 first layer is at effnet.features[0][0] 
        # (a Conv2d with in_channels=3).
        conv1 = self.effnet.features[0][0]
        new_conv = nn.Conv2d(
            in_channels=1, 
            out_channels=conv1.out_channels, 
            kernel_size=conv1.kernel_size, 
            stride=conv1.stride, 
            padding=conv1.padding, 
            bias=False
        )
        nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')
        self.effnet.features[0][0] = new_conv

        # Change final classification layer
        # By default, EfficientNet-B0 has self.effnet.classifier = Sequential(...)
        # The last layer is Linear(in_features=1280, out_features=1000)
        in_feats = self.effnet.classifier[1].in_features
        self.effnet.classifier[1] = nn.Linear(in_feats, num_classes)
    
    def forward(self, x):
        return self.effnet(x)

# ----- Training Function -----
def train_model(model, dataloader, criterion, optimizer, num_epochs=10, device='cuda'):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        total = 0
        correct = 0
        epoch_start = time.time()
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            total += labels.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        print(f"Epoch {epoch+1}/{num_epochs}: Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Time: {time.time()-epoch_start:.2f}s")
    return model

# ----- Main Training Script -----
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Big data directory and folder lists
    big_data_dir = "../../data"

    # Specify which subfolders to use for each split
    train_folders = ["mit_32_d507/d507_2", "mit_76_459/76-459b", "mit_76_studyroom/76-1studyroom2", 
                     "mit_gym_z_squash/gym_z_squash_scan1_oct_26_2012_erika", "mit_lab_hj/lab_hj_tea_nov_2_2012_scan1_erika"]
    val_folders   = ["harvard_c5/hv_c5_1", "harvard_c6/hv_c6_1"]
    test_folders  = ["harvard_c5/hv_c5_1", "harvard_c6/hv_c6_1", "harvard_c11/hv_c11_2", "harvard_tea_2/hv_tea2_2"]

    # Define an image transform for the classifier.
    transform_classifier = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    # Get dataloader using the given depth images
    dataloader_true = get_dataloader_table(big_data_dir, train_folders, use_pred=False, transform=transform_classifier, batch_size=4, shuffle=True, num_workers=4)
    dataloader_pred = get_dataloader_table(big_data_dir, train_folders, use_pred=True, transform=transform_classifier, batch_size=4, shuffle=True, num_workers=4)
    
    # Create dataset objects for true and predicted depth images.
    dataset_true = TableClassificationDataset(big_data_dir, train_folders, use_pred=False, transform=transform_classifier)
    dataset_pred = TableClassificationDataset(big_data_dir, train_folders, use_pred=True, transform=transform_classifier)

    # Combine the two dataset objects.
    combined_dataset = ConcatDataset([dataset_true, dataset_pred])

    # Create a single DataLoader from the combined dataset.
    combined_dataloader = DataLoader(combined_dataset, batch_size=4, shuffle=True, num_workers=4)

    # Instantiate the model, loss function, and optimizer.
    model = TableClassifier(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Train the model.
    num_epochs = 30
    trained_model = train_model(model, combined_dataloader, criterion, optimizer, num_epochs, device)

    # Save the trained model.
    torch.save(trained_model.state_dict(), "table_classifier.pth")
    print("Model training complete and saved as 'table_classifier.pth'.")
