import os
import cv2
import numpy as np
import pickle
from tqdm import tqdm

fx, fy = 570.3422047415297, 570.3422047415297
cx, cy = 320.0, 240.0

# Processing point cloud: random sampling 1024 points + normalization
def preprocess_pointcloud(points, num_points=1024):
    if len(points) < num_points:
        idx = np.random.choice(len(points), num_points, replace=True)
    else:
        idx = np.random.choice(len(points), num_points, replace=False)
    points = points[idx, :]

    centroid = np.mean(points, axis=0)
    points -= centroid

    furthest_dist = np.max(np.linalg.norm(points, axis=1))
    points /= furthest_dist

    return points

# Depth Map -> Point cloud
def depth_to_pointcloud(depth_raw):
    h, w = depth_raw.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth_raw
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    mask = (z.reshape(-1) > 0) & np.isfinite(points).all(axis=1)
    return points[mask]

# build dataset
def build_dataset(base_dir, output_file):
    sequences = [
        "harvard_c5/hv_c5_1/",
        "harvard_c6/hv_c6_1/",
        "harvard_c11/hv_c11_2/",
        "harvard_tea_2/hv_tea2_2/"
    ]

    all_pointclouds = []
    all_labels = []

    for seq in sequences:
        base_path = os.path.join(base_dir, seq)

        # solve `hv_tea2_2/` different situation
        if "hv_tea2_2" in seq:
            depth_path = os.path.join(base_path, "depth")  
        else:
            depth_path = os.path.join(base_path, "depthTSDF") if os.path.exists(os.path.join(base_path, "depthTSDF")) else os.path.join(base_path, "depth")

        img_path = os.path.join(base_path, "image")
        label_path = os.path.join(base_path, "labels/tabletop_labels.dat")

        has_labels = os.path.exists(label_path)

        if has_labels:
            with open(label_path, 'rb') as f:
                labels_data = pickle.load(f)
        else:
            print(f"⚠ Warning: {seq} has no labels. Defaulting to label 0 for all frames.")
            labels_data = None  

        # only process jpg file
        img_files = sorted([f for f in os.listdir(img_path) if f.endswith(".jpg")])
        # only process png file
        depth_files = sorted([f for f in os.listdir(depth_path) if f.endswith(".png")])
        
        n_frames = min(len(img_files), len(depth_files))
        if n_frames == 0:
            print(f"⚠ Warning: {seq} has no valid image-depth pairs. Skipping.")
            continue

        print(f"Processing {seq}: Found {len(img_files)} jpg images, {len(depth_files)} png depth maps. Using {n_frames} frames.")

        # make sure aligning labels_data with n_frames 
        if has_labels and len(labels_data) < n_frames:
            print(f"⚠ Warning: {seq} has fewer labels ({len(labels_data)}) than frames ({n_frames}). Padding with 0.")
            labels_data += [[]] * (n_frames - len(labels_data))

        for i in tqdm(range(n_frames)):
            depth_file = os.path.join(depth_path, depth_files[i])
            depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0

            points = depth_to_pointcloud(depth)
            if len(points) < 10:
                continue  

            points = preprocess_pointcloud(points, num_points=1024)

            label = 1 if (has_labels and len(labels_data[i]) > 0) else 0

            all_pointclouds.append(points)
            all_labels.append(label)

    # thransfer to NumPy array
    all_pointclouds = np.stack(all_pointclouds)  # shape: [N, 1024, 3]
    all_labels = np.array(all_labels, dtype=np.int64)

    np.savez_compressed(output_file, pointclouds=all_pointclouds, labels=all_labels)
    print(f"\n Dataset saved as {output_file}, total {len(all_labels)} samples")

if __name__ == "__main__":
    output_file = "datasets/pipelineA_dataset_harvard_all.npz"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    build_dataset(
        base_dir="data/",
        output_file=output_file
    )

