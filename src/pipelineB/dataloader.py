import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class MultiFolderDepthDataset(Dataset):
    def __init__(self, big_data_dir, folder_list, transform=None):
        """
        Args:
            big_data_dir (str): Root directory containing all dataset subfolders.
            folder_list (list): List of subfolder names to include.
            transform: Transformations to apply to the RGB images.
            
        Note:
            This implementation pairs images and depth maps by their sorted order.
        """
        self.samples = []
        self.transform = transform

        for folder in folder_list:
            folder_path = os.path.join(big_data_dir, folder)
            image_dir = os.path.join(folder_path, "image")
            if folder == "harvard_tea_2/hv_tea2_2":
                depth_dir = os.path.join(folder_path, "depth")
            else:
                depth_dir = os.path.join(folder_path, "depthTSDF")
            if not os.path.exists(image_dir) or not os.path.exists(depth_dir):
                print(f"Warning: {folder} missing required subfolders. Skipping.")
                continue

            # List image files (adjust extensions as needed)
            image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            depth_files = sorted([f for f in os.listdir(depth_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            
            # Pair images and depth files by their order in the sorted list
            min_len = min(len(image_files), len(depth_files))
            if len(image_files) != len(depth_files):
                print(f"Warning: {folder} has a different number of images and depth files. Pairing first {min_len} files.")
            for i in range(min_len):
                img_path = os.path.join(image_dir, image_files[i])
                depth_path = os.path.join(depth_dir, depth_files[i])
                self.samples.append((img_path, depth_path))
                
        if not self.samples:
            raise RuntimeError("No image-depth pairs found in the provided folders.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, depth_path = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        depth = Image.open(depth_path).convert("I")  # Convert depth image to grayscale

        if self.transform:
            image = self.transform(image)
        depth = transforms.ToTensor()(depth)  # Convert depth to a tensor with shape [1, H, W]
        return image, depth

def get_dataloaders(big_data_dir, train_folders, val_folders, test_folders, transform, batch_size=4, num_workers=4):
    # Create dataset objects for each split
    train_dataset = MultiFolderDepthDataset(big_data_dir, train_folders, transform=transform)
    val_dataset = MultiFolderDepthDataset(big_data_dir, val_folders, transform=transform)
    test_dataset = MultiFolderDepthDataset(big_data_dir, test_folders, transform=transform)
    
    # Create DataLoaders for each split
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader