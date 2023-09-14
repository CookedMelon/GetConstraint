import os
import shutil
folder_path = '/home/cc/Workspace/tfconstraint/extract_dir/extract'  # 你的文件夹路径

def process_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    new_lines = lines[1:]  # 去除第一行
    new_name=''
    for line in new_lines:
        if '"' in line:
            new_name = line.split('"')[1]  # 提取双引号中的内容
            break
    print('new_name',new_name)
    new_file_path = os.path.join(os.path.dirname(folder_path), new_name+'.py')
    print('new_file_path',new_file_path)
    with open(new_file_path, 'w') as f:
        f.writelines(new_lines)

    os.remove(file_path)  # 删除原文件

def main():

    for filename in os.listdir(folder_path):
        if filename.startswith('11_'):
            file_path = os.path.join(folder_path, filename)
            print(file_path)
            process_file(file_path)

if __name__ == '__main__':
    main()
