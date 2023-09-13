def is_empty_or_specific_whitespace(s):
    return not s.strip(" \t\n\r")
def read_tf_export(file_path):
    result_dict = {}
    key = None  # 用于存储当前@tf_export标签的键
    with open(file_path, 'r') as f:
        keep=0
        for line in f:
            line = line.rstrip()  # 去掉行尾的空白字符
            if line.startswith("@tf_export(\""):
                # 提取括号内的内容作为字典的键
                key = line[len("@tf_export("):-1].strip('")')
                # 如果key中出现_则不考虑
                if '_' in key:
                    key = None
                    continue
                result_dict[key] = []
                result_dict[key].append(line)
                keep=1
            elif line=="@tf_export(":
                
            elif key is not None:
                if is_empty_or_specific_whitespace(line):
                    continue
                # 检查行是否以空格或制表符开始
                if line.startswith(" ") or line.startswith("\t"):
                    result_dict[key].append(line)
                    keep=0
                else:
                    if keep==1:
                      result_dict[key].append(line)
                    else:  
                      key = None  # 结束当前键的记录
    return result_dict

# # 使用示例
file_path = "./nn_ops.py"  # 替换为你的文件路径
result = read_tf_export(file_path)
for key, value in result.items():
    print(f"{key}: {value}")
    # 将value以文件形式保存进文件./extract/{key}.py下
    with open(f"./extract/{key}.py", 'w') as f:
        for line in value:
            f.write(line + '\n')
    # input("pause")

# print(is_empty_or_specific_whitespace('\n '))