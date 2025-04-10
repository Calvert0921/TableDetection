import os
import cv2
import numpy as np
from tqdm import tqdm
import re

# Using the same camera parameters as the original file
# Original MIT camera intrinsics
# fx, fy = 570.3422047415297, 570.3422047415297
# cx, cy = 320.0, 240.0

# ✅ Replaced with RealSense camera parameters
fx = 425.9412841796875
fy = 425.49493408203125
cx = 428.4195251464844
cy = 243.318359375

NUMPOINTS = 1024

# Functions reused from dataset_build.py
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

def depth_to_pointcloud(depth_raw):
    h, w = depth_raw.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth_raw
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    mask = (z.reshape(-1) > 0) & np.isfinite(points).all(axis=1)
    return points[mask]

def extract_frame_number(filename):
    """Extract frame number from filename, used for sorting"""
    match = re.search(r'(\d+)\.', filename)
    if match:
        return int(match.group(1))
    return 0


def process_realsense_data(base_dir, output_file):
    """Process images from the RealSense folder and convert to point clouds, using timestamp matching"""
    table_dir = os.path.join(base_dir, "table")
    no_table_dir = os.path.join(base_dir, "no_table")
    
    # Check if directories exist
    if not os.path.exists(table_dir) or not os.path.exists(no_table_dir):
        print(f"Error: {table_dir} or {no_table_dir} does not exist")
        return
    
    all_pointclouds = []  # All point cloud data
    all_labels = []       # Corresponding labels: 1=table, 0=no_table
    frame_indices = []    # Corresponding frame indices
    all_filenames = []    # Corresponding filenames - added
    
    # Process images with tables
    print("Processing images with tables...")
    
    # Process images containing tables
    depth_dir = os.path.join(table_dir, "depthTSDF")
    
    # Directly process depth maps, no need to match RGB images
    depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith('.png')])
    
    for depth_file in tqdm(depth_files):
        # Extract timestamp from filename as frame index
        frame_idx = extract_frame_number(depth_file)
        depth_path = os.path.join(depth_dir, depth_file)
        
        # Read depth map
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0  # Convert depth unit to meters
        
        # Convert to point cloud
        raw_points = depth_to_pointcloud(depth)
        if len(raw_points) < 10:
            print(f"Warning: Frame {frame_idx} has too few valid point cloud points")
            continue
        
        # Preprocess point cloud
        points = preprocess_pointcloud(raw_points, num_points=NUMPOINTS)
        
        all_pointclouds.append(points)
        all_labels.append(1)  # 1 means table present
        frame_indices.append(frame_idx)
        all_filenames.append(f"table/{depth_file}")  # Added: save filename
    
    # Process images without tables
    print("Processing images without tables...")
    depth_dir = os.path.join(no_table_dir, "depthTSDF")
    
    # Similarly process depth maps directly
    if os.path.exists(depth_dir):
        depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith('.png')])
        
        for depth_file in tqdm(depth_files):
            # Extract timestamp from filename as frame index
            frame_idx = extract_frame_number(depth_file)
            depth_path = os.path.join(depth_dir, depth_file)
            
            # Read depth map
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
            
            # Convert to point cloud
            raw_points = depth_to_pointcloud(depth)
            if len(raw_points) < 10:
                print(f"Warning: Frame {frame_idx} has too few valid point cloud points")
                continue
            
            # Preprocess point cloud
            points = preprocess_pointcloud(raw_points, num_points=NUMPOINTS)
            
            all_pointclouds.append(points)
            all_labels.append(0)  # 0 means no table
            frame_indices.append(frame_idx)
            all_filenames.append(f"no_table/{depth_file}")  # Added: save filename
    else:
        print(f"Warning: No-table depth directory {depth_dir} not found")
    
    if not all_pointclouds:
        print("Error: No point cloud data was successfully processed")
        return
    
    # Convert to NumPy arrays
    all_pointclouds = np.stack(all_pointclouds)  # [N, 1024, 3]
    all_labels = np.array(all_labels)
    all_filenames = np.array(all_filenames)  # Added: convert to NumPy array
    
    # Save point cloud data and labels
    np.savez_compressed(output_file,
                      pointclouds=all_pointclouds,
                      labels=all_labels,
                      frame_indices=np.array(frame_indices, dtype=np.int32),
                      filenames=all_filenames)  # Added: save filenames
    
    print(f"\nDataset saved as {output_file} with {len(all_pointclouds)} samples")
    print(f"Contains {np.sum(all_labels == 1)} table samples")
    print(f"Contains {np.sum(all_labels == 0)} non-table samples")


if __name__ == "__main__":
    process_realsense_data(
        base_dir="./data/RealSense",
        output_file="./datasets/realsense_pointclouds_C.npz"
    )