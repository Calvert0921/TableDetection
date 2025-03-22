import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

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

def main():
    # Set paths
    test_folder = "../../data/harvard_c11/hv_c11_2"  # Example test folder; adjust as needed
    model_path = "midas_finetuned.pth"  # Path to your saved model

    # Define transforms for the RGB image (should match training)
    rgb_transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Create the test dataset and DataLoader (batch size 1 for visualization)
    test_dataset = SingleFolderDepthDataset(test_folder, transform=rgb_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the pre-trained MiDaS model (MiDaS_small for efficiency)
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    midas.load_state_dict(torch.load(model_path, map_location=device))
    midas.to(device)
    midas.eval()

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

            # Convert tensors to numpy arrays for display
            pred_depth_np = pred_depth_resized.squeeze().cpu().numpy()
            depth_np = depth.squeeze().cpu().numpy()

            # Display the predicted depth and the ground truth depth side by side
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(pred_depth_np, cmap="gray")
            plt.title("Predicted Depth")
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.imshow(depth_np, cmap="gray")
            plt.title("Ground Truth Depth")
            plt.axis("off")
            
            plt.suptitle(f"Sample {idx+1} of {len(test_dataset)}")
            plt.show()  # Close window to proceed to the next image

if __name__ == "__main__":
    main()
