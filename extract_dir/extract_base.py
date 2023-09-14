# 提取所有的raw_ops下的api

import os
import shutil

# 源文件夹和目标文件夹路径
source_folder = "/home/cc/Workspace/tfconstraint/base_api"
destination_folder = "/home/cc/Workspace/tfconstraint/extract_dir/extract"

# 确保目标文件夹存在
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# 遍历源文件夹中的所有文件
for filename in os.listdir(source_folder):
    # 检查文件名是否符合条件
    if filename.startswith("api_def_") and filename.endswith(".pbtxt"):
        # 创建新的文件名
        new_filename = "raw_ops." + filename[len("api_def_"):]
        
        # 构建源文件和目标文件的完整路径
        source_filepath = os.path.join(source_folder, filename)
        destination_filepath = os.path.join(destination_folder, new_filename)
        
        # 复制并重命名文件
        shutil.copyfile(source_filepath, destination_filepath)
        print(f"Copied {filename} to {new_filename}")
