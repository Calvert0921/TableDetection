import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from dataloader_depthEst import get_dataloaders
import warnings

# Ignore warning from import model
warnings.filterwarnings("ignore", category=FutureWarning)

def main():
    # Path to the big data directory containing all subfolders
    big_data_dir = "../../data"  # Adjust this path if needed

    # Specify which subfolders to use for each split
    train_folders = ["mit_32_d507/d507_2", "mit_76_459/76-459b", "mit_76_studyroom/76-1studyroom2", 
                     "mit_gym_z_squash/gym_z_squash_scan1_oct_26_2012_erika", "mit_lab_hj/lab_hj_tea_nov_2_2012_scan1_erika"]
    val_folders   = ["harvard_c5/hv_c5_1", "harvard_c6/hv_c6_1"]
    test_folders  = ["harvard_c5/hv_c5_1", "harvard_c6/hv_c6_1", "harvard_c11/hv_c11_2", "harvard_tea_2/hv_tea2_2"]

    # Define transforms for the RGB images. MiDaS typically uses 384x384 inputs.
    rgb_transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Create DataLoaders for each dataset split
    train_loader, val_loader, test_loader = get_dataloaders(big_data_dir, train_folders, val_folders, test_folders, rgb_transform)

    # Device configuration: use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pre-trained MiDaS model (using MiDaS_small for efficiency)
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS")
    midas.to(device)
    midas.train()  # Set model to training mode for fine-tuning

    # Define loss function and optimizer
    criterion = nn.L1Loss()
    optimizer = optim.Adam(midas.parameters(), lr=1e-4)

    num_epochs = 20

    for epoch in range(num_epochs):
        midas.train()
        running_loss = 0.0
        for i, (rgb, depth) in enumerate(train_loader):
            rgb = rgb.to(device)
            depth = depth.to(device)
            
            optimizer.zero_grad()
            # Forward pass: predict depth from RGB input
            pred_depth = midas(rgb)
            
            # Check and adjust predicted depth dimensions
            if pred_depth.dim() == 3:
                pred_depth = pred_depth.unsqueeze(1)  # Now shape is (N, 1, H, W)
            
            # MiDaS output resolution may differ from the ground truth; resize accordingly.
            pred_depth_resized = nn.functional.interpolate(
                pred_depth, size=depth.shape[-2:], mode="bilinear", align_corners=False
            )
            
            # Compute L1 loss between prediction and ground truth
            loss = criterion(pred_depth_resized, depth)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Average Training Loss: {avg_train_loss:.4f}")

        # Validation phase (optional)
        midas.eval()
        val_loss = 0.0
        with torch.no_grad():
            for rgb, depth in val_loader:
                rgb = rgb.to(device)
                depth = depth.to(device)
                pred_depth = midas(rgb)
                if pred_depth.dim() == 3:
                    pred_depth = pred_depth.unsqueeze(1)
                pred_depth_resized = nn.functional.interpolate(
                    pred_depth, size=depth.shape[-2:], mode="bilinear", align_corners=False
                )
                loss = criterion(pred_depth_resized, depth)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Validation Loss: {avg_val_loss:.4f}")
        midas.train()  # Return to training mode for the next epoch

    # Save the fine-tuned model
    torch.save(midas.state_dict(), "midas_finetuned.pth")
    print("Training complete. Model saved as midas_finetuned.pth")
    
    # Testing phase
    midas.eval()
    test_loss = 0.0
    with torch.no_grad():
        for rgb, depth in test_loader:
            rgb = rgb.to(device)
            depth = depth.to(device)
            pred_depth = midas(rgb)
            if pred_depth.dim() == 3:
                pred_depth = pred_depth.unsqueeze(1)
            pred_depth_resized = nn.functional.interpolate(
                pred_depth, size=depth.shape[-2:], mode="bilinear", align_corners=False
            )
            loss = criterion(pred_depth_resized, depth)
            test_loss += loss.item()
    avg_test_loss = test_loss / len(test_loader)
    print(f"Testing Loss: {avg_test_loss:.4f}")

if __name__ == "__main__":
    main()