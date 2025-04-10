import os
import cv2
import numpy as np
from tqdm import tqdm

fx = 425.9412841796875
fy = 425.49493408203125
cx = 428.4195251464844
cy = 243.318359375

def depth_to_pointcloud(depth_raw):
    h, w = depth_raw.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth_raw
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    mask = (z.reshape(-1) > 0) & np.isfinite(points).all(axis=1)
    return points[mask]

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

def build_manual_labeled_dataset(base_dir, output_file):
    all_pointclouds = []
    all_labels = []

    label_map = {
        "table": 1, 
        "no_table": 0  
    }

    for folder_name, label in label_map.items():
        folder_path = os.path.join(base_dir, folder_name)
        depth_folder = os.path.join(folder_path, "depthTSDF")

        if not os.path.exists(depth_folder):
            print(f"NO {depth_folder}, skip")
            continue

        depth_files = sorted([f for f in os.listdir(depth_folder) if f.endswith(".png")])
        print(f"solve {depth_folder}, total {len(depth_files)} figures, label = {label}")

        for depth_file in tqdm(depth_files):
            depth_path = os.path.join(depth_folder, depth_file)
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if depth is None:
                print(f"read fail: {depth_path}")
                continue

            depth = depth.astype(np.float32) / 1000.0 
            points = depth_to_pointcloud(depth)

            if len(points) < 10:
                continue

            points = preprocess_pointcloud(points, num_points=1024)
            all_pointclouds.append(points)
            all_labels.append(label)

    if not all_pointclouds:
        print("stop saving")
        return

    all_pointclouds = np.stack(all_pointclouds)
    all_labels = np.array(all_labels, dtype=np.int64)

    np.savez_compressed(output_file, pointclouds=all_pointclouds, labels=all_labels)
    print(f"\n data saved in {output_file}，共 {len(all_labels)} ")

if __name__ == "__main__":
    output_file = "datasets/pipelineA_RealSense_dataset.npz"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    build_manual_labeled_dataset(
        base_dir="data/RealSense",
        output_file=output_file
    )

