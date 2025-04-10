import os
import cv2
import numpy as np
import pickle
from tqdm import tqdm

fx, fy = 570.3422047415297, 570.3422047415297
cx, cy = 320.0, 240.0

NUMPOINTS = 1024 * 4

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

def build_test_dataset(base_dir, output_file):
    sequences = [
        "harvard_c5/hv_c5_1/",
        "harvard_c6/hv_c6_1/",
        "harvard_c11/hv_c11_2/",
        "harvard_tea_2/hv_tea2_2/"
    ]

    all_pointclouds = []
    all_point_labels = []  # point-level labels
    all_frame_labels = []  # frame-level labels
    all_categorical_vectors = []  # category vectors

    for seq_idx, seq in enumerate(sequences):
        base_path = os.path.join(base_dir, seq)

        # Handle different folder structures
        if "hv_tea2_2" in seq:
            depth_path = os.path.join(base_path, "depth")  
        else:
            depth_path = os.path.join(base_path, "depthTSDF") if os.path.exists(os.path.join(base_path, "depthTSDF")) else os.path.join(base_path, "depth")

        img_path = os.path.join(base_path, "image")
        label_path = os.path.join(base_path, "labels/tabletop_labels.dat")

        has_labels = os.path.exists(label_path)

        if has_labels:
            with open(label_path, 'rb') as f:
                labels_data = pickle.load(f)  # [frame][table_instance][coordinate]
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

        # Ensure labels_data aligns with n_frames
        if has_labels and len(labels_data) < n_frames:
            print(f"⚠ Warning: {seq} has fewer labels ({len(labels_data)}) than frames ({n_frames}). Padding with [].")
            labels_data += [[]] * (n_frames - len(labels_data))

        for i in tqdm(range(n_frames)):
            depth_file = os.path.join(depth_path, depth_files[i])
            depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
            img_file = os.path.join(img_path, img_files[i])
            img = cv2.imread(img_file)

            # Get original point cloud
            raw_points = depth_to_pointcloud(depth)
            if len(raw_points) < 10:
                continue  

            # Frame-level label for table presence
            has_table = has_labels and len(labels_data[i]) > 0
            frame_label = 1 if has_table else 0
            
            # Create category vector - contains only table presence
            categorical_vector = np.array([float(has_table)], dtype=np.float32)
            
            # Preprocess point cloud - sample 1024 points
            points = preprocess_pointcloud(raw_points, num_points=NUMPOINTS)
            
            # Create point-level labels
            point_labels = np.zeros(NUMPOINTS, dtype=np.int64)  # Default all points as background (0)
            
            if has_table:
                # Use original depth map and polygon labels to determine if points are on the tabletop
                h, w = depth.shape
                # Create mask image
                mask = np.zeros((h, w), dtype=np.uint8)
                
                # First merge all table polygons into one mask image
                for table_idx, table_polygon in enumerate(labels_data[i]):
                    if len(table_polygon) > 0:
                        # Convert polygon to OpenCV format
                        x_coords = np.array(table_polygon[0], dtype=np.int32)
                        y_coords = np.array(table_polygon[1], dtype=np.int32)
                        polygon = np.vstack((x_coords, y_coords)).T  # Convert to [[x1,y1], [x2,y2], ...] format
                        
                        # Fill polygon into mask image
                        cv2.fillPoly(mask, [polygon], 255)
                
                # Then process all points at once
                for pt_idx, point in enumerate(points):
                    # Get coordinates of original point in depth map
                    x, y, z = point
                    # Project 3D point back to 2D image
                    u = int(x * fx / z + cx)
                    v = int(y * fy / z + cy)
                    
                    # Check if point is within image boundaries
                    if 0 <= u < w and 0 <= v < h:
                        # Check if the point is inside the mask
                        if mask[v, u] > 0:
                            point_labels[pt_idx] = 1  # If inside mask, mark as tabletop
            
            all_pointclouds.append(points)
            all_point_labels.append(point_labels)
            all_frame_labels.append(frame_label)
            all_categorical_vectors.append(categorical_vector)

    # Convert to NumPy arrays
    all_pointclouds = np.stack(all_pointclouds)  # [N, 1024, 3]
    all_point_labels = np.stack(all_point_labels)  # [N, 1024]
    all_frame_labels = np.array(all_frame_labels, dtype=np.int64)  # [N]
    all_categorical_vectors = np.stack(all_categorical_vectors)  # [N, 1]

    np.savez_compressed(output_file, 
                       pointclouds=all_pointclouds, 
                       point_labels=all_point_labels,
                       frame_labels=all_frame_labels,
                       categorical_vectors=all_categorical_vectors)
    
    print(f"\n Dataset saved as {output_file}, total {len(all_frame_labels)} samples")
    
    # Print statistics
    table_frames = np.sum(all_frame_labels)
    print(f"Statistics: {table_frames} frames with tables ({table_frames/len(all_frame_labels)*100:.2f}%)")
    
    table_points = np.sum(all_point_labels == 1)
    total_points = all_point_labels.size
    print(f"Points statistics: {table_points} table points out of {total_points} ({table_points/total_points*100:.2f}%)")



if __name__ == "__main__":
    build_test_dataset(
        base_dir="./data",
        output_file="./datasets/pipelineC_dataset_harvard_test.npz"
    )