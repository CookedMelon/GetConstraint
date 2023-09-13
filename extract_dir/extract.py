# 功能：从文件中提取出@tf_export标签的内容，并保存在api对应名字的文件
import re
import random
import string
import os

def list_files(startpath):
    file_list = []
    for root, dirs, files in os.walk(startpath):
        for file in files:
            if file.endswith('.py'):
                file_list.append(os.path.join(root, file))
    return file_list

def is_empty_or_specific_whitespace(s):
    return not s.strip(" \t\n\r")
def generate_random_string(length):
    letters_and_digits = string.ascii_letters + string.digits
    return ''.join(random.choice(letters_and_digits) for i in range(length))
def extract_key_from_line(line):
    key=None
    if line.startswith("@tf_export(\""):
        # 提取引号内的内容作为字典的键
        match = re.search(r'"(.*?)"', line)
        key = match.group(1)
        # 如果key中出现_则不考虑
        if '__' in key:
            key = None
    elif line.startswith("@keras_export(\""):
        # 提取引号内的内容作为字典的键
        match = re.search(r'"(.*?)"', line)
        key = match.group(1)
        # 如果key中出现_则不考虑
        if '__' in key:
            key = None
    
    # 针对首行为@tf_export(，但没在该行找到引号的，生成随机key
    elif line=="@tf_export(":
        # 生成随机值
        key = '11_tf_export'+generate_random_string(10)
    elif line=="@keras_export(":
        # 生成随机值
        key = '11_keras_export'+generate_random_string(10)
    return key
def read_tf_export(file_path):
    result_dict = {}
    lastkey=None
    key = None  # 用于存储当前@tf_export标签的键
    with open(file_path, 'r') as f:
        keep=0
        for line in f:

            line = line.rstrip()  # 去掉行尾的空白字符
            tempkey=extract_key_from_line(line)
            if tempkey:
               key=tempkey 
            if key and key != lastkey:
                result_dict[key] = []
                if key.startswith("11_"):
                    result_dict[key].append('"'+file_path+'"')
                result_dict[key].append(line)
                lastkey=key
                keep=1
            elif key is not None:
                if is_empty_or_specific_whitespace(line):
                    continue
                # 检查行是否以空格或制表符开始
                if line.startswith(" ") or line.startswith("\t"):
                    result_dict[key].append(line)
                    if not key.startswith('11_'):
                        keep=0
                else:
                    if keep==1:
                      result_dict[key].append(line)
                    else:  
                      key = extract_key_from_line(line)  # 结束当前键的记录
                      if key:
                            result_dict[key] = []
                            if key.startswith("11_"):
                                result_dict[key].append('"'+file_path+'"')
                            result_dict[key].append(line)
                            lastkey=key
                            keep=1
    return result_dict

# # 使用示例
# 遍历/home/cc/Workspace/tfconstraint/ops下的所有文件名以及子文件名
startpath = '/home/cc/Workspace/tfconstraint/python'
startpath2 = '/home/cc/Workspace/tfconstraint/keras'
file_list=list_files(startpath)
file_list+=list_files(startpath2)
# file_list=["/home/cc/Workspace/tfconstraint/python/ops/nn_ops.py"]
print(file_list)
# for file_path in file_list:
#     if "nn_ops.py" in file_path:
#         print("find",file_path)
for file_path in file_list:
    result = read_tf_export(file_path)
    for key, value in result.items():
        print(f"{key}: {value}")
        # 将value以文件形式保存进文件./extract/{key}.py下
        with open(f"./extract_dir/extract/{key}.py", 'w') as f:
        # with open(f"./extract_dir/temp/{key}.py", 'w') as f:
            for line in value:
                f.write(line + '\n')
