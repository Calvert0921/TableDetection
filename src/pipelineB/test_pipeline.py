import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report
from train_tableClassifier import TableClassifier
import warnings
from utils import convert_depth_to_8bit_classify  # Assuming this function is available

# Ignore warning from import model
warnings.filterwarnings("ignore", category=FutureWarning)

# Define a variable to indicate wrong images based on filename.
# Any image with this exact filename will be skipped.
WRONG_IMAGE_VAR = "0000006-000000187873.jpg"

# --------------------------
# Dataset for Testing Pipeline
# --------------------------
class TestPipelineDataset(Dataset):
    """
    Loads RGB images and corresponding labels (derived from polygon lists or based on folder name)
    from a list of test folders. Assumes each folder has an "image" subfolder and, if available, a 
    label file at "labels/tabletop_labels.dat".
    
    - For folders with a label file, the label is determined as 1 if the polygon list is non-empty, else 0.
    - For folders without a label file:
        * If the folder name contains "table2" (case-insensitive), all images are labeled 1.
        * If the folder name contains "table3" (case-insensitive), all images are labeled 0.
    This version also removes any sample whose image filename matches WRONG_IMAGE_VAR.
    """
    def __init__(self, big_data_dir, folder_list, transform=None):
        self.samples = []  # List of (rgb_image_path, label)
        self.transform = transform
        
        for folder in folder_list:
            folder_path = os.path.join(big_data_dir, folder)
            image_folder = os.path.join(folder_path, "image")
            if not os.path.exists(image_folder):
                print(f"Warning: image folder {image_folder} does not exist. Skipping folder {folder}.")
                continue
            valid_exts = ('.png', '.jpg', '.jpeg')
            rgb_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(valid_exts)])
            
            # Load label file if it exists; otherwise, assign labels based on folder name.
            label_file_path = os.path.join(folder_path, "labels/tabletop_labels.dat")
            if os.path.exists(label_file_path):
                with open(label_file_path, 'rb') as f:
                    labels_data = pickle.load(f)
                # Align the number of images and labels.
                num_samples = min(len(rgb_files), len(labels_data))
                rgb_files = rgb_files[:num_samples]
                labels_data = labels_data[:num_samples]
            else:
                # If there is no label file, assign labels based on folder name.
                if "table" in folder.lower():
                    labels_data = [1] * len(rgb_files)
                elif "no_table" in folder.lower():
                    labels_data = [0] * len(rgb_files)
                else:
                    # Default behavior: assign empty list so that later logic assigns label 0.
                    labels_data = [[] for _ in range(len(rgb_files))]
            
            # Determine indices to keep: skip any image file whose name matches WRONG_IMAGE_VAR.
            valid_indices = [
                i for i, file in enumerate(rgb_files)
                if WRONG_IMAGE_VAR.lower() not in file.lower()
            ]
            rgb_files = [rgb_files[i] for i in valid_indices]
            labels_data = [labels_data[i] for i in valid_indices]
            
            # For each remaining image, determine the label:
            # If the label data is a list, label = 1 if non-empty, else 0.
            # Otherwise, use the provided integer label.
            for i, file in enumerate(rgb_files):
                img_path = os.path.join(image_folder, file)
                if isinstance(labels_data[i], list):
                    label = 1 if len(labels_data[i]) > 0 else 0
                else:
                    label = labels_data[i]
                self.samples.append((img_path, label))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)

# --------------------------
# Main Testing Pipeline
# --------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define test folders.
    # You can mix folders with label files and folders without label files.
    big_data_dir = "../../data"
    # test_folders = [
    #     "harvard_c5/hv_c5_1", 
    #     "harvard_c6/hv_c6_1", 
    #     "harvard_c11/hv_c11_2", 
    #     "harvard_tea_2/hv_tea2_2",
    # ]
    
    test_folders = ["RealSense/table", "RealSense/no_table"]
    
    # Transform for RGB images for depth estimation (MiDaS expects 384x384)
    transform_midas = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
    # For classifier input, predicted depth images will be resized to 224x224.
    classifier_input_size = (224, 224)
    
    # Create test dataset and DataLoader.
    test_dataset = TestPipelineDataset(big_data_dir, test_folders, transform=transform_midas)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # --------------------------
    # Load the Depth Estimator (MiDaS)
    # --------------------------
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS")
    midas_weights_path = "best_midas_finetuned.pth"
    midas.load_state_dict(torch.load(midas_weights_path, map_location=device))
    midas.to(device)
    midas.eval()
    
    # --------------------------
    # Load the Table Classifier
    # --------------------------
    classifier = TableClassifier(num_classes=2)
    classifier_weights_path = "table_classifier.pth"
    classifier.load_state_dict(torch.load(classifier_weights_path, map_location=device))
    classifier.to(device)
    classifier.eval()
    
    # --------------------------
    # Testing Pipeline: Predict and Compute Metrics
    # --------------------------
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for rgb, label in test_loader:
            rgb = rgb.to(device)
            label = label.to(device)
            
            # Predict depth using MiDaS.
            pred_depth = midas(rgb)  # Expected shape: [1, H, W]
            if pred_depth.dim() == 3:
                pred_depth = pred_depth.unsqueeze(1)  # Now shape: [1, 1, H, W]
            
            # Resize predicted depth to classifier input size (224x224).
            pred_depth_resized = F.interpolate(pred_depth, size=classifier_input_size, mode="bilinear", align_corners=False)
            
            # Convert the depth image to 8-bit for classification.
            pred_depth_resized = convert_depth_to_8bit_classify(pred_depth_resized.squeeze())
    
            # Convert the image to tensor (necessary for classification)
            pred_depth_resized_tensor = transforms.ToTensor()(pred_depth_resized).unsqueeze(0).to(device)
            
            # Predict label using the classifier.
            outputs = classifier(pred_depth_resized_tensor)
            _, preds = torch.max(outputs, 1)
            
            all_labels.append(label.item())
            all_preds.append(preds.item())
    
    # Compute overall accuracy.
    total_samples = len(all_labels)
    correct_preds = sum([1 for l, p in zip(all_labels, all_preds) if l == p])
    accuracy = correct_preds / total_samples if total_samples > 0 else 0
    print(f"Test Accuracy on Harvard and New Test folders: {accuracy*100:.2f}%")
    
    # Compute confusion matrix and classification report.
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=["No Table", "Table"])
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(report)

if __name__ == "__main__":
    main()
