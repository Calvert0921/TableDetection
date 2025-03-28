import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import warnings
from utils import convert_depth_to_8bit_depth

# Ignore warning from import model
warnings.filterwarnings("ignore", category=FutureWarning)

# Simple dataset for a single test folder (assumes "image" and "depthTSDF" subfolders)
class SingleFolderDepthDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        """
        Args:
            folder_path (str): Path to a single dataset folder containing 'image' and 'depthTSDF'
            transform: Transformations to apply to the RGB image.
        """
        self.transform = transform
        self.samples = []
        image_dir = os.path.join(folder_path, "image")
        if folder_path == "../../data/harvard_tea_2/hv_tea2_2":
                depth_dir = os.path.join(folder_path, "depth")
        else:
            depth_dir = os.path.join(folder_path, "depthTSDF")
        
        # List image and depth files (assumes they are sorted in the same order)
        image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        depth_files = sorted([f for f in os.listdir(depth_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        min_len = min(len(image_files), len(depth_files))
        if len(image_files) != len(depth_files):
            print(f"Warning: Different number of images ({len(image_files)}) and depth files ({len(depth_files)}). Pairing first {min_len} files.")
        for i in range(min_len):
            img_path = os.path.join(image_dir, image_files[i])
            depth_path = os.path.join(depth_dir, depth_files[i])
            self.samples.append((img_path, depth_path))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, depth_path = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        depth = Image.open(depth_path).convert("I")  # Grayscale for depth
        
        if self.transform:
            image = self.transform(image)
        depth = transforms.ToTensor()(depth)  # Convert depth image to tensor [1, H, W]
        return image, depth

def compute_metrics(pred_depth, depth):
    # Convert to 8 bit depth image
    pred_depth, depth = convert_depth_to_8bit_depth(pred_depth), convert_depth_to_8bit_depth(depth)
    
    # Convert the PIL images back to tensors for PyTorch operations
    pred_depth = transforms.ToTensor()(pred_depth).float()  # Convert to tensor and ensure float type
    depth = transforms.ToTensor()(depth).float()  # Convert to tensor and ensure float type
    
    # Compute the Mean Absolute Error (MAE)
    mae = torch.mean(torch.abs(pred_depth - depth))

    # Compute the Root Mean Squared Error (RMSE)
    rmse = torch.sqrt(torch.mean((pred_depth - depth) ** 2))

    # Compute the Mean Squared Error (MSE)
    mse = torch.mean((pred_depth - depth) ** 2)

    return mae.item(), rmse.item(), mse.item()

def main():
    # Set paths
    # mit_folders_base = ["../../data/mit_32_d507/d507_2", "../../data/mit_76_459/76-459b", "../../data/mit_76_studyroom/76-1studyroom2", 
    #                     "../../data/mit_gym_z_squash/gym_z_squash_scan1_oct_26_2012_erika", "../../data/mit_lab_hj/lab_hj_tea_nov_2_2012_scan1_erika"]
    harvard_folders_base = [
        "../../data/harvard_c5/hv_c5_1", 
        "../../data/harvard_c6/hv_c6_1", 
        "../../data/harvard_c11/hv_c11_2", 
        "../../data/harvard_tea_2/hv_tea2_2",
    ]
    # ucl_folders_base = ["../../data/RealSense/table", "../../data/RealSense/no_table"]
    model_path = "best_midas_finetuned.pth"  # Path to your saved model

    # Define transforms for the RGB image
    rgb_transform = transforms.Compose([ 
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the pre-trained MiDaS model (MiDaS_small for efficiency)
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS")
    midas.load_state_dict(torch.load(model_path, map_location=device))
    midas.to(device)
    midas.eval()

    all_mae, all_rmse, all_mse = [], [], []

    for test_folder_base in harvard_folders_base:
        output_dir = os.path.join(test_folder_base, "depthPred")
        
        # Create output folder if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create the test dataset and DataLoader (batch size 1 for visualization)
        test_dataset = SingleFolderDepthDataset(test_folder_base, transform=rgb_transform)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        # Iterate through all samples in the test dataset
        with torch.no_grad():
            for idx, (rgb, depth) in enumerate(test_loader):
                rgb = rgb.to(device)
                depth = depth.to(device)

                # Predict depth from the RGB image
                pred_depth = midas(rgb)
                # If output is [N, H, W], add a channel dimension
                if pred_depth.dim() == 3:
                    pred_depth = pred_depth.unsqueeze(1)
                
                # Resize predicted depth to match ground truth dimensions
                pred_depth_resized = nn.functional.interpolate(
                    pred_depth, size=depth.shape[-2:], mode="bilinear", align_corners=False
                )

                # Compute the metrics for the current sample
                mae, rmse, mse = compute_metrics(pred_depth_resized, depth)

                # Store metrics for averaging later
                all_mae.append(mae)
                all_rmse.append(rmse)
                all_mse.append(mse)

                # Convert tensors to numpy arrays for display
                pred_depth_np = pred_depth_resized.squeeze().cpu().numpy()
                depth_np = depth.squeeze().cpu().numpy()

                # Save the predicted depth image using PIL
                pred_depth_np = pred_depth_np.astype(np.uint16)
                output_path = os.path.join(output_dir, f"pred_{idx+1:03d}.png")
                Image.fromarray(pred_depth_np).save(output_path)
                print(f"Saved prediction {idx+1} to {output_path}")

    # Calculate the average metrics over all the test samples
    avg_mae = np.mean(all_mae)
    avg_rmse = np.mean(all_rmse)
    avg_mse = np.mean(all_mse)

    # Print the overall performance metrics
    print(f"\nAverage Performance on Harvard Set:")
    print(f"Mean Absolute Error (MAE): {avg_mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {avg_rmse:.4f}")
    print(f"Mean Squared Error (MSE): {avg_mse:.4f}")

if __name__ == "__main__":
    main()
