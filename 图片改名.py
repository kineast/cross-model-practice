import os
import re


def rename_files(folder_path):
    # 获取文件夹中的所有文件
    files = os.listdir(folder_path)

    # 定义匹配文件名的正则表达式
    pattern = re.compile(r'(\d+)_[a-f0-9]+\.jpg')

    for file_name in files:
        # 使用正则表达式匹配文件名
        match = pattern.match(file_name)

        if match:
            # 获取匹配到的部分
            original_name = match.group(1)

            # 构建新的文件名
            new_name = f"{original_name}.jpg"

            # 构建完整的文件路径
            old_path = os.path.join(folder_path, file_name)
            new_path = os.path.join(folder_path, new_name)

            # 重命名文件
            os.rename(old_path, new_path)
            print(f"已将文件 {file_name} 重命名为 {new_name}")


# 调用函数，传入文件夹路径
folder_path = "D:\\pycharmcode\\multi-model改\\origin\\Flicker8k_Dataset\\"
rename_files(folder_path)
