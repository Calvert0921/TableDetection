import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class TableClassificationDataset(Dataset):
    def __init__(self, big_data_dir, folder_list, use_pred, transform=None):
        """
        Args:
            big_data_dir (str): Root directory containing all dataset subfolders.
            folder_list (list): List of folder paths. Each folder is expected to have a subfolder
                                containing depth images (default "depth").
            transform (callable, optional): Transformations to apply to the depth images.
        """
        self.samples = []  # List of tuples: (depth_image_path, label)
        self.transform = transform

        for folder in folder_list:
            # Construct the path to the depth images in the current folder.
            folder_path = os.path.join(big_data_dir, folder)
            if use_pred:
                depth_folder = os.path.join(folder_path, "depthPred")
            elif folder == "harvard_tea_2/hv_tea2_2":
                depth_folder = os.path.join(folder_path, "depth")
            else:
                depth_folder = os.path.join(folder_path, "depthTSDF")
            if not os.path.exists(depth_folder):
                print(f"Warning: Depth subfolder {depth_folder} does not exist. Skipping folder {folder}.")
                continue

            # Gather depth image filenames (assumes valid image extensions).
            valid_exts = ('.png', '.jpg', '.jpeg')
            depth_files = sorted([f for f in os.listdir(depth_folder) if f.lower().endswith(valid_exts)])

            # Check if a label file exists in the current folder.
            label_file_path = os.path.join(folder_path, "labels/tabletop_labels.dat")
            if os.path.exists(label_file_path):
                # Load the label data (assuming JSON format).
                with open(label_file_path, 'rb') as f:
                    labels_data = pickle.load(f)
                    f.close()
                # Expect labels_data to be a list of polygon lists—one per image.
                if len(labels_data) != len(depth_files):
                    print(f"Warning: In folder {folder}, number of depth images ({len(depth_files)}) and "
                          f"labels ({len(labels_data)}) differ. Using first {min(len(depth_files), len(labels_data))} samples.")
                num_samples = min(len(depth_files), len(labels_data))
                labels_data = labels_data[:num_samples]
            else:
                # No label file: mark all images as having no table (empty polygon list → label 0).
                labels_data = [[] for _ in range(len(depth_files))]
                num_samples = len(depth_files)

            # Pair each depth image with its binary label (1 if polygon list non-empty, else 0).
            for i in range(num_samples):
                depth_img_path = os.path.join(depth_folder, depth_files[i])
                label = 1 if (isinstance(labels_data[i], list) and len(labels_data[i]) > 0) else 0
                self.samples.append((depth_img_path, label))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        depth_img_path, label = self.samples[idx]
        # Load the image in 16-bit mode.
        depth_img = Image.open(depth_img_path).convert("I;16")
        # Apply any provided transforms. If your transform includes ToTensor(),
        # it will convert the image to an integer tensor.
        if self.transform:
            depth_img = self.transform(depth_img)
        else:
            depth_img = transforms.ToTensor()(depth_img)
        # Convert to float and scale from [0, 65535] to [0, 1]
        depth_img = depth_img.float() / 65535.0
        return depth_img, torch.tensor(label, dtype=torch.long)


def get_dataloader_table(big_data_dir, folder_list, use_pred=False, transform=None, batch_size=4, shuffle=True, num_workers=4):
    """
    Returns a DataLoader for the combined dataset from multiple folders.
    
    Args:
        folder_list (list): List of folder paths.
        transform (callable, optional): Transformations to apply to depth images.
        batch_size (int): Batch size for DataLoader.
        shuffle (bool): Whether to shuffle the dataset.
        num_workers (int): Number of subprocesses for data loading.
    """
    dataset = TableClassificationDataset(big_data_dir, folder_list, use_pred, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

if __name__ == "__main__":
    big_data_dir = "../../data"

    # Specify which subfolders to use for each split
    train_folders = ["mit_32_d507/d507_2", "mit_76_459/76-459b", "mit_76_studyroom/76-1studyroom2", 
                     "mit_gym_z_squash/gym_z_squash_scan1_oct_26_2012_erika", "mit_lab_hj/lab_hj_tea_nov_2_2012_scan1_erika"]
    val_folders   = ["harvard_c5/hv_c5_1", "harvard_c6/hv_c6_1"]
    test_folders  = ["harvard_c5/hv_c5_1", "harvard_c6/hv_c6_1", "harvard_c11/hv_c11_2", "harvard_tea_2/hv_tea2_2"]
    
    # Define an image transform. For example, we resize to 224x224 and convert to tensor.
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    # Get DataLoaders for true and predicted depth datasets.
    dataloader_true = get_dataloader_table(big_data_dir, train_folders, use_pred=False, transform=transform, batch_size=4, shuffle=True, num_workers=4)
    dataloader_pred = get_dataloader_table(big_data_dir, train_folders, use_pred=True, transform=transform, batch_size=4, shuffle=True, num_workers=4)
    
    # Example: iterate over one batch from each DataLoader.
    for x, y in dataloader_true:
        print("True depth batch - x shape:", x.shape, "labels:", y)
        break
        
    for x, y in dataloader_pred:
        print("Predicted depth batch - x shape:", x.shape, "labels:", y)
        break
