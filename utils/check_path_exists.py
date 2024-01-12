import os

import os


def check_path_exists(path):
    return os.path.exists(path)


# 获取当前工作目录
current_directory = os.getcwd()

# 去掉 utils 目录
parent_directory = os.path.dirname(current_directory)

# 相对路径示例
relative_directory_path = 'dataset\\data_to_json'

# 构建相对路径
full_path = os.path.join(parent_directory, relative_directory_path)

# 检查目录是否存在
if check_path_exists(full_path):
    print(f"The directory '{full_path}' exists.")
else:
    print(f"The directory '{full_path}' does not exist.")

# # 检查文件是否存在
# file_path = 'your_file_path'
# if check_path_exists(file_path):
#     print(f"The file '{file_path}' exists.")
# else:
#     print(f"The file '{file_path}' does not exist.")
