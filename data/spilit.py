import os
import random
import shutil

# 定义路径
base_path = '/media/hw3579/KINGSTON 2T/table'
table2_path = os.path.join(base_path, 'table2')
table3_path = os.path.join(base_path, 'table3')
output_path = os.path.join(base_path, 'selected_images')

# 创建输出目录结构
os.makedirs(os.path.join(output_path, 'table2', 'rgb'), exist_ok=True)
os.makedirs(os.path.join(output_path, 'table2', 'depth_aligned'), exist_ok=True)
os.makedirs(os.path.join(output_path, 'table3', 'rgb'), exist_ok=True)
os.makedirs(os.path.join(output_path, 'table3', 'depth_aligned'), exist_ok=True)

def read_associations(file_path):
    """读取associations.txt文件并解析对应关系"""
    pairs = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 4:
                rgb_time = parts[0]
                rgb_file = parts[1]
                depth_time = parts[2]
                depth_file = parts[3]
                
                # 修改深度图路径从depth到depth_aligned
                depth_aligned_file = depth_file.replace('depth/', 'depth_aligned/')
                
                pairs.append((rgb_file, depth_aligned_file))
    return pairs

def select_and_copy_images(table_path, output_table_path, num_samples=25):
    """随机选择并复制图像对"""
    assoc_file = os.path.join(table_path, 'associations.txt')
    pairs = read_associations(assoc_file)
    
    # 随机选择25对
    selected_pairs = random.sample(pairs, num_samples)
    
    # 复制文件
    for rgb_file, depth_aligned_file in selected_pairs:
        src_rgb = os.path.join(table_path, rgb_file)
        src_depth = os.path.join(table_path, depth_aligned_file)
        
        dst_rgb = os.path.join(output_table_path, rgb_file)
        dst_depth = os.path.join(output_table_path, depth_aligned_file)
        
        # 确保目标目录存在
        os.makedirs(os.path.dirname(dst_rgb), exist_ok=True)
        os.makedirs(os.path.dirname(dst_depth), exist_ok=True)
        
        # 复制文件
        shutil.copy2(src_rgb, dst_rgb)
        shutil.copy2(src_depth, dst_depth)
        
        print(f"已复制: {os.path.basename(src_rgb)} -> {os.path.basename(src_depth)}")
    
    return selected_pairs

# 执行抽取和复制操作
print("从table2中抽取25对图像...")
table2_pairs = select_and_copy_images(table2_path, os.path.join(output_path, 'table2'))

print("\n从table3中抽取25对图像...")
table3_pairs = select_and_copy_images(table3_path, os.path.join(output_path, 'table3'))

# 保存选择结果到文件
with open(os.path.join(output_path, 'selected_pairs.txt'), 'w') as f:
    f.write("Table2 selected pairs:\n")
    for rgb, depth in table2_pairs:
        f.write(f"{rgb} -> {depth}\n")
    
    f.write("\nTable3 selected pairs:\n")
    for rgb, depth in table3_pairs:
        f.write(f"{rgb} -> {depth}\n")

print(f"\n操作完成! 所选图像对保存在 {output_path}")
print(f"选择结果已保存到 {os.path.join(output_path, 'selected_pairs.txt')}")